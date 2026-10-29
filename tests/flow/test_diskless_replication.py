# flow-matrix: spawn
import os

from common import *
from time import sleep, time

GRAPH_ID = "diskless"

# Pairs of (:N)-[:R]->(:M); with VKEY_MAX_ENTITY_COUNT 10 the graph spans
# hundreds of RDB keys once GRAPH.DEBUG AUX START splits it, so a delayed save
# leaves a wide window to cut it.
PAIRS = 2000

UDF_LIB = "dl"

# Small single-key graphs, scattered through the RDB among the big graph's
# slices: whenever a load is cut, some of them have been decoded.
SMALL_GRAPHS = [f"small_{i}" for i in range(40)]

# Written to the replica's log by Redis itself.
DISKLESS_LOAD = "receiving streamed RDB from master with EOF to parser"
SKIPPED = "Skipping diskless-load"
LOAD_STARTED = "MASTER <-> REPLICA sync: Loading DB in memory"
LOAD_FAILED = "Failed trying to load the MASTER synchronization DB from socket"
SWAPPED = "MASTER <-> REPLICA sync: Swapping active DB with loaded DB"
DISCARDED_TEMP = "MASTER <-> REPLICA sync: Discarding temporary DB"
SYNC_DONE = "MASTER <-> REPLICA sync: Finished with success"


def _udf(version):
    return f"function f() {{ return {version}; }}\nfalkor.register('f', f);"


class testDisklessReplication(FlowTestsBase):
    """A replica with `repl-diskless-load` decodes the primary's RDB straight
    off the socket.

    `on-empty-db` decodes into the live (empty) database, so a broken link
    stops the load mid-graph and what it had decoded must go with it: the
    retry decodes the same graph again, from its first key.

    `swapdb` decodes into a temporary database while the live one stays in
    service -- with `async_loading`, to clients the replica keeps answering
    during the load. What the load brings (graphs, UDFs) must not replace the
    live ones until Redis swaps the databases, and must vanish if it fails."""

    def __init__(self):
        # replication timing doesn't play well with sanitizers
        if SANITIZER:
            Environment.skip(None)

        # Detaches, flushes and re-attaches the replica, and slows the
        # primary's saves; on CI's shared services container that would
        # reconfigure replication for every other class in the cell. The
        # `flow-matrix: spawn` marker above routes this file to a private one.
        if os.getenv("FALKORDB_USE_SERVICE"):
            Environment.skip(None)

        self.env, self.db = Env(env='oss', useSlaves=True,
                                moduleArgs='VKEY_MAX_ENTITY_COUNT 10')
        if self.env.replica_log_path is None:
            # Every assertion about *how* the replica loaded reads its log.
            Environment.skip(None)

        self.master = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        info = self.replica.info("replication")
        self.master_host = info["master_host"]
        self.master_port = info["master_port"]

        self.master.config_set("repl-diskless-sync", "yes")
        self.master.config_set("repl-diskless-sync-delay", "0")
        # Redis 8 streams the RDB over a dedicated channel that CLIENT KILL
        # TYPE replica leaves alone; keep it on the main link so the test can
        # cut it. Older servers have no such option.
        for con in (self.master, self.replica):
            try:
                con.config_set("repl-rdb-channel", "no")
            except Exception:
                pass

        self.graph = self.db.select_graph(GRAPH_ID)
        self.graph.query(
            f"UNWIND range(1, {PAIRS}) AS i CREATE (:N {{v: i}})-[:R]->(:M {{v: i}})")
        for name in SMALL_GRAPHS:
            self.db.select_graph(name).query("CREATE ()")
        self.master.execute_command("GRAPH.UDF", "LOAD", "REPLACE", UDF_LIB, _udf(1))

    # -- helpers -------------------------------------------------------------

    def _replica_log(self):
        with open(self.env.replica_log_path) as f:
            return f.read()

    def _wait_for_log(self, needle, since, timeout=30):
        deadline = time() + timeout
        while time() < deadline:
            log = self._replica_log()[since:]
            if needle in log:
                return log
            sleep(0.05)
        self.env.assertTrue(False, message=f"replica never logged {needle!r}")
        return self._replica_log()[since:]

    def _wait_for_sync(self, timeout=30):
        deadline = time() + timeout
        while time() < deadline:
            info = self.replica.info("replication")
            if info.get("master_link_status") == "up" and \
                    self.replica.info("persistence")["loading"] == 0:
                self.master.execute_command("WAIT", 1, 5000)
                return
            sleep(0.1)
        self.env.assertTrue(False, message="replica never finished syncing")

    def _detach(self):
        # Detach from a replica that has caught up, so what it keeps is the
        # primary's current state rather than wherever its stream had got to.
        self._wait_for_sync()
        self.replica.replicaof("no", "one")

    def _attach(self):
        """Re-attach the replica; return the log offset its sync starts at.
        Having been a primary since, it cannot partially resync: this is a
        full sync, and -- its replication id now differs from the primary's --
        a blocking one, not an async load."""
        since = len(self._replica_log())
        self.replica.replicaof(self.master_host, self.master_port)
        return since

    def _contents(self, con):
        g = Graph(con, GRAPH_ID)
        return [
            g.ro_query("MATCH (n:N) RETURN count(n), sum(n.v)").result_set,
            g.ro_query("MATCH (m:M) RETURN count(m), sum(m.v)").result_set,
            g.ro_query("MATCH (n:N)-[:R]->(m:M) WHERE n.v = m.v RETURN count(*)").result_set,
        ]

    def _state(self, con):
        """Everything a client sees: the graphs, their contents, the UDFs."""
        graphs = sorted(con.execute_command("GRAPH.LIST"))
        counts = {
            name: Graph(con, name).ro_query("MATCH (n) RETURN count(n)").result_set
            for name in graphs if name != GRAPH_ID
        }
        udf = Graph(con, GRAPH_ID).ro_query(f"RETURN {UDF_LIB}.f()").result_set
        return [graphs, counts, self._contents(con), udf]

    def _change_master(self, version):
        """Give the primary a snapshot the replica's differs from in every way
        it can: a graph modified, one added, one removed, the UDF replaced."""
        self.graph.query(f"CREATE (:N {{v: {version}}})")
        self.db.select_graph(f"added_{version}").query("CREATE ()")
        for name in self.master.execute_command("GRAPH.LIST"):
            if name.startswith("added_") and name != f"added_{version}":
                self.db.select_graph(name).delete()
        self.master.execute_command("GRAPH.UDF", "LOAD", "REPLACE", UDF_LIB, _udf(version))

    # -- on-empty-db ---------------------------------------------------------

    def test01_full_sync_loads_from_socket(self):
        self.replica.config_set("repl-diskless-load", "on-empty-db")
        self._detach()
        self.replica.flushall()
        since = self._attach()
        self._wait_for_sync()

        log = self._wait_for_log(SYNC_DONE, since)
        self.env.assertFalse(SKIPPED in log, message="replica refused to load off the socket")
        self.env.assertTrue(DISKLESS_LOAD in log, message="replica did not load off the socket")
        self.env.assertEqual(self._state(self.replica), self._state(self.master))

    def test02_aborted_multi_key_load_is_discarded(self):
        self.replica.config_set("repl-diskless-load", "on-empty-db")
        # Hold the graph's virtual keys in the primary's keyspace, so the
        # sync's RDB carries it as one graphdata key plus its graphmeta slices
        # and the replica decodes it through the multi-key path.
        self.env.assertEqual(self.master.execute_command("GRAPH.DEBUG", "AUX", "START"), 1)
        try:
            self.master.config_set("rdb-key-save-delay", "10000")  # 10ms per key
            self._detach()
            self.replica.flushall()
            since = self._attach()

            # Cut the link once the replica is decoding: a diskless load only
            # serves clients every 2MB it reads, so watch its log rather than
            # asking it.
            self._wait_for_log(LOAD_STARTED, since)
            sleep(0.5)
            self.env.assertEqual(
                self.master.execute_command("CLIENT", "KILL", "TYPE", "replica"), 1)
            # The next sync forks a fresh child; let it run at full speed.
            self.master.config_set("rdb-key-save-delay", "0")

            log = self._wait_for_log(LOAD_FAILED, since)
            self._wait_for_sync()
            log = self._wait_for_log(SYNC_DONE, since + log.index(LOAD_FAILED))
            self.env.assertTrue(DISKLESS_LOAD in log, message="retry did not load off the socket")

            self.env.assertEqual(self._state(self.replica), self._state(self.master))
        finally:
            self.master.config_set("rdb-key-save-delay", "0")
            self.master.execute_command("GRAPH.DEBUG", "AUX", "END")

    # -- swapdb --------------------------------------------------------------

    def test03_swapdb_replaces_graphs_and_udfs(self):
        self.replica.config_set("repl-diskless-load", "swapdb")
        self._detach()
        self._change_master(3)
        since = self._attach()
        self._wait_for_sync()

        log = self._wait_for_log(SYNC_DONE, since)
        self.env.assertFalse(SKIPPED in log, message="replica refused to load off the socket")
        self.env.assertTrue(DISKLESS_LOAD in log, message="replica did not load off the socket")
        self.env.assertTrue(SWAPPED in log, message="replica did not load into a temporary db")
        self.env.assertEqual(self._state(self.replica), self._state(self.master))

    def test04_swapdb_aborted_load_keeps_live_data(self):
        self.replica.config_set("repl-diskless-load", "swapdb")
        self._detach()
        live = self._state(self.replica)
        self._change_master(4)
        self.env.assertNotEqual(self._state(self.master), live)

        self.env.assertEqual(self.master.execute_command("GRAPH.DEBUG", "AUX", "START"), 1)
        try:
            self.master.config_set("rdb-key-save-delay", "10000")  # 10ms per key
            since = self._attach()

            # Cut it well into the transfer, so the temporary database holds
            # graphs that share a name with live ones.
            self._wait_for_log(LOAD_STARTED, since)
            sleep(2.5)
            # Hold the retry back: the primary waits this long before forking
            # the next sync, which keeps the replica on its live database --
            # and answering -- in between.
            self.master.config_set("repl-diskless-sync-delay", "5")
            self.master.config_set("rdb-key-save-delay", "0")
            self.env.assertEqual(
                self.master.execute_command("CLIENT", "KILL", "TYPE", "replica"), 1)

            log = self._wait_for_log(DISCARDED_TEMP, since)
            self.env.assertTrue(LOAD_FAILED in log)
            # The temporary database is gone; the live one and its UDFs are
            # exactly what they were before the load began.
            self.env.assertEqual(self._state(self.replica), live)

            # ...and its graphs are still the registered ones: promoted, the
            # replica can write to every one of them (a writer aborts on a
            # graph the module no longer tracks). Detach now, not once synced:
            # the primary is still holding the retry back.
            self.replica.replicaof("no", "one")
            for name in live[0]:
                Graph(self.replica, name).query("CREATE (:Written)")

            self.master.config_set("repl-diskless-sync-delay", "0")
            self._attach()
            self._wait_for_sync()
            self._wait_for_log(SYNC_DONE, since + log.index(DISCARDED_TEMP))
            self.env.assertEqual(self._state(self.replica), self._state(self.master))
        finally:
            self.master.config_set("repl-diskless-sync-delay", "0")
            self.master.config_set("rdb-key-save-delay", "0")
            self.master.execute_command("GRAPH.DEBUG", "AUX", "END")

    def test05_async_load_serves_previous_snapshot(self):
        self.replica.config_set("repl-diskless-load", "swapdb")
        # Serve clients every KB the load reads rather than every 2MB, so the
        # replica answers throughout this (small) snapshot's load.
        self.replica.config_set("loading-process-events-interval-bytes", "1024")
        self._wait_for_sync()
        live = self._state(self.replica)

        # An async load needs a full sync that keeps the replication id: the
        # replica reconnects to the same primary, whose backlog no longer
        # reaches back to its offset. MULTI makes the cut, the new snapshot and
        # the overflow atomic, so the replica cannot reconnect in between.
        self.master.config_set("repl-backlog-size", "16384")
        self.master.config_set("rdb-key-save-delay", "10000")  # 10ms per key
        since = len(self._replica_log())
        try:
            pipe = self.master.pipeline(transaction=True)
            pipe.execute_command("CLIENT", "KILL", "TYPE", "replica")
            pipe.execute_command("GRAPH.QUERY", GRAPH_ID, "CREATE (:N {v: 5})")
            pipe.execute_command("GRAPH.QUERY", "added_5", "CREATE ()")
            pipe.execute_command("GRAPH.DELETE", "added_4")
            pipe.execute_command("GRAPH.UDF", "LOAD", "REPLACE", UDF_LIB, _udf(5))
            pipe.execute_command("GRAPH.DEBUG", "AUX", "START")
            pipe.set("backlog_overflow", "x" * 65536)
            pipe.execute()

            # Sample what clients see while the replica is async loading. A
            # sample takes several queries; it counts only if the load was
            # still running after the last of them.
            def async_loading():
                return self.replica.info("persistence").get("async_loading") == 1

            seen = []
            deadline = time() + 30
            while time() < deadline:
                if async_loading():
                    state = self._state(self.replica)
                    if async_loading():
                        seen.append(state)
                elif seen:
                    break
                sleep(0.05)

            self.env.assertTrue(len(seen) > 0, message="replica never async loaded")
            for state in seen:
                self.env.assertEqual(state, live)

            self._wait_for_sync()
            log = self._wait_for_log(SYNC_DONE, since)
            self.env.assertTrue(SWAPPED in log, message="replica did not load into a temporary db")
            self.env.assertEqual(self._state(self.replica), self._state(self.master))
        finally:
            self.master.config_set("rdb-key-save-delay", "0")
            self.master.execute_command("GRAPH.DEBUG", "AUX", "END")
