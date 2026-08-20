from common import *
import os
import time

GRAPH_ID = "replica_divergence"

# verifies that when a replicated GRAPH.EFFECT fails to apply on a replica
# because the replica has diverged from the master (e.g. an entity the
# effect targets no longer exists locally), the replica:
#   1. logs the divergence
#   2. forces a full resync with the master (not a partial PSYNC CONTINUE)
#   3. ends up with a dataset identical to the master's again


class testReplicaDivergence():
    def __init__(self):
        # replication timing doesn't play well with Valgrind/sanitizers
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env(env='oss', useSlaves=True,
                                 enableDebugCommand=True)

    def test_forced_full_resync_on_effect_divergence(self):
        env = self.env
        master = env.getConnection()
        replica = env.getSlaveConnection()

        master_graph = Graph(master, GRAPH_ID)
        replica_graph = Graph(replica, GRAPH_ID)

        # force effects-based replication, so a small single-node delete
        # replicates as GRAPH.EFFECT rather than the raw GRAPH.QUERY
        self.db.config_set("EFFECTS_THRESHOLD", 0)

        # create a small graph and let the replica catch up
        master_graph.query("CREATE (:P {v: 1}), (:P {v: 2}), (:P {v: 3})")
        master.execute_command("WAIT", "1", "0")

        # pick a node we'll remove directly on the replica only, simulating
        # real divergence (e.g. a bug, manual intervention, or an earlier
        # undetected inconsistency) while the master keeps it
        res = master_graph.query(
            "MATCH (n:P) RETURN id(n) ORDER BY id(n) LIMIT 1")
        target_id = res.result_set[0][0]

        # allow direct writes on the replica and remove the node there only
        replica.config_set("slave-read-only", "no")
        replica_graph.query(f"MATCH (n) WHERE id(n) = {target_id} DELETE n")
        replica.config_set("slave-read-only", "yes")

        # sanity check: replica and master have diverged
        # (GRAPH.RO_QUERY, since the replica is read-only again by now)
        replica_count = replica_graph.ro_query(
            "MATCH (n:P) RETURN count(n)").result_set[0][0]
        env.assertEquals(replica_count, 2)

        # snapshot the master's full-sync counter before triggering the
        # divergence, so we can later confirm a FULLRESYNC actually
        # happened, not merely a reconnect that continues the stream
        sync_full_before = master.info()["sync_full"]

        # on the master, delete the very same node - this replicates as a
        # GRAPH.EFFECT DELETE_NODE effect targeting an id that no longer
        # exists on the replica
        master_graph.query(f"MATCH (n) WHERE id(n) = {target_id} DELETE n")

        # wait for the replica to detect the divergence, force a
        # REPLICAOF NO ONE / REPLICAOF cycle, and complete a fresh full
        # resync with the master
        deadline = time.time() + 30
        resynced = False
        while time.time() < deadline:
            info = master.info()
            if (info["sync_full"] > sync_full_before and
                    info["connected_slaves"] >= 1):
                resynced = True
                break
            time.sleep(0.5)

        env.assertTrue(resynced)

        # the replica's dataset should now match the master exactly again,
        # the earlier direct deletion included, since the full resync
        # replaced its entire dataset
        master.execute_command("WAIT", "1", "5000")

        expected = master_graph.query(
            "MATCH (n:P) RETURN id(n) ORDER BY id(n)").result_set
        actual = replica_graph.ro_query(
            "MATCH (n:P) RETURN id(n) ORDER BY id(n)").result_set
        env.assertEquals(expected, actual)


# a divergence hit while replaying the local AOF (rather than while
# consuming the live replication stream) can't be fixed by a resync: the
# divergence is baked into this instance's own persisted state, and the
# replication subsystem isn't even running yet at that point. This verifies
# that case makes the replica bail out (log + exit) instead of attempting -
# and failing at - a REPLICAOF NO ONE / REPLICAOF cycle.
#
# lives in its own class/Env: this test kills the replica process, which
# would break testReplicaDivergence's tests if they shared one.
class testAOFDivergence():
    def __init__(self):
        # replication timing doesn't play well with Valgrind/sanitizers
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env(env='oss', useSlaves=True,
                                 enableDebugCommand=True)

    def test_bail_on_aof_divergence(self):
        env = self.env
        replica = env.getSlaveConnection()
        replica_graph = Graph(replica, GRAPH_ID + "_aof")

        # force effects-based replication, so writes land in the AOF as
        # GRAPH.EFFECT commands
        replica.execute_command("GRAPH.CONFIG", "SET",
                                 "EFFECTS_THRESHOLD", "0")

        # enable AOF on the replica and wait for the initial rewrite
        # (triggered by turning appendonly on) to finish
        replica.config_set("appendonly", "yes")
        # fsync every write, so the incr AOF file's on-disk size right
        # after a query's reply is deterministic - no reliance on timing
        replica.config_set("appendfsync", "always")
        deadline = time.time() + 30
        while time.time() < deadline:
            info = replica.info("persistence")
            if (info.get("aof_enabled") == 1 and
                    info.get("aof_rewrite_in_progress") == 0):
                break
            time.sleep(0.2)

        # locate the current incremental AOF file via the manifest, so we
        # don't have to guess Redis' file naming/rewrite behavior. Manifest
        # lines are alternating key/value tokens, e.g.:
        #   file appendonly.aof.1.incr.aof seq 1 type i startoffset 0 ...
        aof_dir = os.path.join(replica.config_get("dir")["dir"],
                                replica.config_get("appenddirname")["appenddirname"])
        manifest_name = replica.config_get("appendfilename")["appendfilename"] + ".manifest"
        incr_name = None
        with open(os.path.join(aof_dir, manifest_name)) as f:
            for line in f:
                tokens = line.split()
                fields = dict(zip(tokens[0::2], tokens[1::2]))
                if fields.get("type") == "i":
                    incr_name = fields["file"]
        env.assertIsNotNone(incr_name)
        incr_path = os.path.join(aof_dir, incr_name)

        # allow direct writes on the replica
        replica.config_set("slave-read-only", "no")

        # create a node - appended to the AOF as a GRAPH.EFFECT CREATE_NODE
        res = replica_graph.query("CREATE (n:P {v: 1}) RETURN id(n)")
        target_id = res.result_set[0][0]

        # delete it - appended to the AOF as a GRAPH.EFFECT DELETE_NODE
        # targeting target_id; this succeeds, since the node still exists
        size_before = os.path.getsize(incr_path)
        replica_graph.query(f"MATCH (n) WHERE id(n) = {target_id} DELETE n")
        size_after = os.path.getsize(incr_path)

        # duplicate the raw bytes of that DELETE_NODE command onto the end
        # of the AOF file: replaying the file will now apply the delete
        # twice against the same id, and the second application will find
        # no such node locally - i.e. a divergence, manufactured entirely
        # at the file level without needing to hand-encode an effect
        with open(incr_path, "rb") as f:
            f.seek(size_before)
            dup_bytes = f.read(size_after - size_before)
        env.assertGreater(len(dup_bytes), 0)
        with open(incr_path, "ab") as f:
            f.write(dup_bytes)

        # replay the (now doctored) AOF in place; this should hit the
        # duplicated DELETE_NODE, detect the divergence, and - because
        # we're loading, not consuming a live replication stream - bail
        # out immediately rather than attempting a resync
        try:
            replica.execute_command("DEBUG", "LOADAOF")
        except Exception:
            pass

        # the replica process should have terminated rather than hang
        # around attempting REPLICAOF. this alone is a meaningful check,
        # not just "did it crash": a real, reachable master is configured
        # here, so under the pre-fix behavior the deferred REPLICAOF cycle
        # would have succeeded and the replica would still be alive - it
        # would NOT have exited. exit code 1 (a clean exit() call, not a
        # signal - Popen.wait() would return a negative number for a
        # crash) is specifically what our bail-during-loading path does.
        slave_process = env.envRunner.slaveProcess
        exit_code = slave_process.wait(timeout=30)
        env.envRunner.slaveProcess = None
        env.assertEquals(exit_code, 1)

        # confirm it bailed for the right reason, when a log file is
        # available (RLTest may run with output capturing disabled, e.g.
        # under -s/--no-output-catch, in which case there's no file to
        # check and the exit-code assertion above already made the point)
        log_name = env.envRunner._getFileName("slave", ".log")
        try:
            with open(os.path.join(env.logDir, log_name)) as f:
                log = f.read()
        except FileNotFoundError:
            return

        env.assertContains("while loading from disk", log)
        env.assertNotContains("Scheduling a forced full resync", log)


# verifies that after a replica completes a FULL SYNC its graph is identical
# to the master's - including the order in which entities are returned.
#
# a graph's matrices keep pending changes in delta-plus / delta-minus that are
# not necessarily merged into the main matrix. during a full sync the master
# forks, encodes its live matrices (M / DP / DM, separately) to an RDB, and the
# replica decodes them. the decoder must restore the M / DP / DM split as-is,
# and rebuild each matrix's transpose from all three - rather than merging the
# deltas into M (the old Graph_ApplyAllPending behaviour).
#
# a merge is logically equivalent but reorders a DeltaMatrixIterator, so an
# order-sensitive query (a label scan such as MATCH (n:L) RETURN n, or any
# LIMIT query) could return entities in a different order on the replica than
# on the master. a missing / stale transpose would additionally break reverse
# traversals on the replica.
#
# lives in its own class/Env: it cycles REPLICAOF on the replica to force a
# fresh full sync, which would interfere with the tests above.
class testFullSyncMatrixConsistency():
    def __init__(self):
        # replication timing doesn't play well with Valgrind/sanitizers
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env(env='oss', useSlaves=True,
                                 enableDebugCommand=True)

    def test_full_sync_preserves_matrix_order(self):
        env = self.env
        master = env.getConnection()
        replica = env.getSlaveConnection()

        graph_id = GRAPH_ID + "_fullsync"
        master_graph = Graph(master, graph_id)
        replica_graph = Graph(replica, graph_id)

        # remember where the replica replicates from, then detach it so the
        # graph reaches it via a fresh FULL SYNC - not the incremental
        # replication stream - once we re-attach
        repl_info = replica.info()
        master_host = repl_info["master_host"]
        master_port = repl_info["master_port"]
        replica.execute_command("REPLICAOF", "NO", "ONE")

        # build a graph on the master with pending deltas: create a batch of
        # labeled, connected nodes then delete a portion, leaving entries in
        # the matrices' delta-minus and gaps in the datablocks
        master_graph.query(
            "UNWIND range(0, 999) AS i CREATE (:L {v: i})-[:R]->(:M {v: i})")
        master_graph.query("MATCH (:L)-[e:R]->(:M) WHERE e.v % 3 = 0 DELETE e")
        master_graph.query("MATCH (n:M) WHERE n.v % 5 = 0 DELETE n")

        # snapshot the master's full-sync counter so we can confirm a
        # FULLRESYNC actually happened (not a partial PSYNC CONTINUE)
        sync_full_before = master.info()["sync_full"]

        # re-attach the replica -> triggers a full resync: the master forks,
        # encodes its live matrices to an RDB, and the replica loads it through
        # the exact decode path this branch changed
        replica.execute_command("REPLICAOF", master_host, master_port)

        # wait for the full sync to complete
        deadline = time.time() + 60
        synced = False
        while time.time() < deadline:
            if (master.info()["sync_full"] > sync_full_before and
                    replica.info()["master_link_status"] == "up"):
                synced = True
                break
            time.sleep(0.5)
        env.assertTrue(synced)

        # make sure the replica has fully caught up
        master.execute_command("WAIT", "1", "10000")

        # every query - label scans plus forward and reverse traversals - must
        # return identical, identically-ordered results on master and replica.
        # intentionally no ORDER BY: the result order reflects the underlying
        # matrix iteration order, which is what a merge-on-decode would change
        queries = [
            "MATCH (n:L) RETURN n.v",                        # label scan
            "MATCH (n:M) RETURN n.v",                        # label scan
            "MATCH (n:L)-[e:R]->(m:M) RETURN n.v, e.v, m.v", # forward traverse
            "MATCH (m:M)<-[e:R]-(n:L) RETURN m.v, e.v, n.v", # reverse (transpose)
        ]
        for q in queries:
            expected = master_graph.query(q).result_set
            actual = replica_graph.ro_query(q).result_set
            env.assertEquals(actual, expected)

