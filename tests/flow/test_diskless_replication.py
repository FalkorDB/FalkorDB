# flow-matrix: spawn
import os

from common import *
from index_utils import *
from constraint_utils import *
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
            except ResponseError:
                pass

        self.graph = self.db.select_graph(GRAPH_ID)
        self.graph.query(
            f"UNWIND range(1, {PAIRS}) AS i CREATE (:N {{v: i}})-[:R]->(:M {{v: i}})")
        self.graph.create_node_range_index("N", "v")
        wait_for_indices_to_sync(self.graph)
        # A unique and a mandatory constraint: a snapshot carries them, and a
        # replica promoted after a load must enforce them.
        create_unique_node_constraint(self.graph, "N", "v", sync=True)
        create_mandatory_node_constraint(self.graph, "N", "v", sync=True)
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
        return [graphs, counts, self._contents(con), udf, self._index_reads(con),
                self._constraints(con)]

    def _constraints(self, con):
        """The constraints each graph of the snapshot carries."""
        return [self._constraints_of(con, name)
                for name in sorted(con.execute_command("GRAPH.LIST"))
                if name not in SMALL_GRAPHS]  # none of them has a constraint

    def _constraints_of(self, con, name):
        res = Graph(con, name).ro_query(
            "CALL db.constraints() YIELD type, label, properties, entitytype, status "
            "RETURN type, label, properties, entitytype, status")
        return [name, sorted(res.result_set)]

    def _assert_enforced(self, con, name):
        """A promoted replica enforces the snapshot's constraints on `name`."""
        g = Graph(con, name)
        if name == GRAPH_ID:
            for q in ["CREATE (:N {v: 1})", "CREATE (:N)"]:  # unique / mandatory
                try:
                    g.query(q)
                    self.env.assertTrue(False, message=f"{q} was accepted")
                except ResponseError:
                    pass  # rejected, as it must be
        elif name.startswith("added_"):
            try:
                g.query("CREATE (:A)")  # mandatory on A.v
                self.env.assertTrue(False, message="CREATE (:A) was accepted")
            except ResponseError:
                pass

    def _index_reads(self, con):
        """The indexes of the graphs a snapshot carries, and what they answer."""
        out = []
        for name in sorted(con.execute_command("GRAPH.LIST")):
            g = Graph(con, name)
            indexes = g.ro_query("CALL db.indexes() YIELD label, status RETURN label, status")
            out.append([name, indexes.result_set])
        for name, query in [(GRAPH_ID, "MATCH (n:N) WHERE n.v = 7 RETURN count(n)"),
                            (GRAPH_ID, "MATCH (n:N) WHERE n.v > 1990 RETURN count(n)")]:
            out.append(Graph(con, name).ro_query(query).result_set)
        for name in sorted(con.execute_command("GRAPH.LIST")):
            if name.startswith("added_"):
                query = "MATCH (a:A) WHERE a.v = 1 RETURN count(a)"
                out.append(Graph(con, name).ro_query(query).result_set)
        return out

    def _quick_state(self, con):
        """The part of `_state` a few queries read: what a sample taken inside a
        short async-load window can afford."""
        udf = Graph(con, GRAPH_ID).ro_query(f"RETURN {UDF_LIB}.f()").result_set
        return [sorted(con.execute_command("GRAPH.LIST")), self._contents(con), udf,
                self._constraints_of(con, GRAPH_ID)]

    def _change_master(self, version):
        """Give the primary a snapshot the replica's differs from in every way
        it can: a graph modified, one added, one removed, the UDF replaced."""
        # MERGE: unique on v, and a version may be applied twice
        self.graph.query(f"MERGE (:N {{v: {PAIRS + version}}})")
        added = self.db.select_graph(f"added_{version}")
        added.query("CREATE (:A {v: 1})")
        try:
            added.create_node_range_index("A", "v")
        except ResponseError as e:
            if "already" not in str(e):
                raise  # only a version changed twice may find it there
        wait_for_indices_to_sync(added)
        try:
            create_mandatory_node_constraint(added, "A", "v", sync=True)
        except ResponseError as e:
            if "already" not in str(e):
                raise  # only a version changed twice may find it there
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
        self.env.assertNotContains(SKIPPED, log, message="replica refused to load off the socket")
        self.env.assertContains(DISKLESS_LOAD, log, message="replica did not load off the socket")
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
            self.env.assertContains(DISKLESS_LOAD, log, message="retry did not load off the socket")

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
        self.env.assertNotContains(SKIPPED, log, message="replica refused to load off the socket")
        self.env.assertContains(DISKLESS_LOAD, log, message="replica did not load off the socket")
        self.env.assertContains(SWAPPED, log, message="replica did not load into a temporary db")
        self.env.assertEqual(self._state(self.replica), self._state(self.master))
        # promoted, the replica enforces what the swap brought in
        self.replica.replicaof("no", "one")
        for name in (GRAPH_ID, "added_3"):
            self._assert_enforced(self.replica, name)
        self._attach()
        self._wait_for_sync()

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
            self.env.assertContains(LOAD_FAILED, log)
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
            # An indexed graph takes writes through its index too.
            g = Graph(self.replica, GRAPH_ID)
            g.query("CREATE (:N {v: 99999})")
            self.env.assertEqual(
                g.query("MATCH (n:N) WHERE n.v = 99999 RETURN count(n)").result_set, [[1]])
            # ...and the constraints of the live snapshot still hold
            self._assert_enforced(self.replica, GRAPH_ID)

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
        # Start from a snapshot with the graph this test deletes, whichever tests
        # ran before.
        self._change_master(4)
        self._wait_for_sync()
        live = self._state(self.replica)
        quick_live = self._quick_state(self.replica)

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
            pipe.execute_command("GRAPH.QUERY", GRAPH_ID, f"CREATE (:N {{v: {PAIRS + 5}}})")
            pipe.execute_command("GRAPH.QUERY", "added_5", "CREATE (:A {v: 1})")
            pipe.execute_command("GRAPH.QUERY", "added_5", "CREATE INDEX FOR (a:A) ON (a.v)")
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

            seen, seen_full = [], []
            deadline = time() + 30
            while time() < deadline:
                if async_loading():
                    state = self._quick_state(self.replica)
                    if async_loading():
                        seen.append(state)
                        # the whole of it, when the window is wide enough
                        full = self._state(self.replica)
                        if async_loading():
                            seen_full.append(full)
                elif seen:
                    break
                sleep(0.05)

            self.env.assertGreater(len(seen), 0, message="replica never async loaded")
            for state in seen:
                self.env.assertEqual(state, quick_live)
            for state in seen_full:
                self.env.assertEqual(state, live)

            self._wait_for_sync()
            log = self._wait_for_log(SYNC_DONE, since)
            self.env.assertContains(SWAPPED, log, message="replica did not load into a temporary db")
            wait_for_indices_to_sync(Graph(self.master, "added_5"))
            wait_for_indices_to_sync(Graph(self.replica, "added_5"))
            self.env.assertEqual(self._state(self.replica), self._state(self.master))
        finally:
            self.master.config_set("rdb-key-save-delay", "0")
            self.master.execute_command("GRAPH.DEBUG", "AUX", "END")

    def test06_restore_during_async_load_survives_an_abort(self):
        # A writable replica takes client commands while it async-loads, and a
        # RESTORE of a graph key goes through the same load callback as the
        # replication stream. That key lives in the live database, so an
        # aborted load must leave it registered: the graph the client restored
        # takes writes afterwards.
        self.replica.config_set("repl-diskless-load", "swapdb")
        self.replica.config_set("replica-read-only", "no")
        self.replica.config_set("loading-process-events-interval-bytes", "1024")
        self.db.select_graph("restore_src").query("CREATE (:S)")
        payload = self.master.dump("restore_src")
        self._change_master(6)
        self._wait_for_sync()

        self.master.config_set("repl-backlog-size", "16384")
        self.master.config_set("rdb-key-save-delay", "10000")  # 10ms per key
        since = len(self._replica_log())
        try:
            pipe = self.master.pipeline(transaction=True)
            pipe.execute_command("CLIENT", "KILL", "TYPE", "replica")
            pipe.execute_command("GRAPH.QUERY", GRAPH_ID, f"MERGE (:N {{v: {PAIRS + 7}}})")
            pipe.execute_command("GRAPH.DEBUG", "AUX", "START")
            pipe.set("backlog_overflow", "x" * 65536)
            pipe.execute()

            deadline = time() + 30
            while time() < deadline and \
                    self.replica.info("persistence").get("async_loading") != 1:
                sleep(0.02)
            self.env.assertEqual(self.replica.info("persistence").get("async_loading"), 1)
            self.env.assertTrue(self.replica.restore("rx", 0, payload))

            # Cut the load, and hold the retry back so the replica stays on
            # the live database.
            self.master.config_set("repl-diskless-sync-delay", "5")
            self.master.config_set("rdb-key-save-delay", "0")
            self.master.execute_command("CLIENT", "KILL", "TYPE", "replica")
            log = self._wait_for_log(DISCARDED_TEMP, since)
            self.env.assertContains(LOAD_FAILED, log)

            g = Graph(self.replica, "rx")
            self.env.assertEqual(g.ro_query("MATCH (n:S) RETURN count(n)").result_set, [[1]])
            g.query("CREATE (:W)")  # a writer aborts on an unregistered graph
            self.env.assertEqual(g.ro_query("MATCH (n:W) RETURN count(n)").result_set, [[1]])
        finally:
            self.replica.delete("rx")
            self.replica.config_set("replica-read-only", "yes")
            self.master.config_set("repl-diskless-sync-delay", "0")
            self.master.config_set("rdb-key-save-delay", "0")
            self.master.execute_command("GRAPH.DEBUG", "AUX", "END")

