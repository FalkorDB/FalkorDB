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

    # same divergence contract, but exercised through EFFECT_UPDATE_EDGE
    # rather than EFFECT_DELETE_EDGE
    #
    # EFFECT_CREATE_EDGE doesn't encode an edge id - the replica allocates
    # one itself off its own free list - while UPDATE/DELETE address an edge
    # by explicit id. so master and replica must keep allocating identical
    # ids, and once they don't, an UPDATE_EDGE can name an id that sits on
    # the replica's deleted list. Graph_GetEdge then yields a NULL
    # attribute-set, which AttributeSet_Update dereferences unconditionally
    #
    # DELETE_EDGE has always reported that cleanly; UPDATE_EDGE used to
    # guard it with ASSERT() alone, which compiles to nothing in a release
    # build, so the same divergence segfaulted the whole replica process
    def test_forced_full_resync_on_edge_update_divergence(self):
        env = self.env
        master = env.getConnection()
        replica = env.getSlaveConnection()

        graph_id = GRAPH_ID + "_edge_update"
        master_graph = Graph(master, graph_id)
        replica_graph = Graph(replica, graph_id)

        # force effects-based replication, so the SET below replicates as
        # GRAPH.EFFECT UPDATE_EDGE rather than the raw GRAPH.QUERY
        self.db.config_set("EFFECTS_THRESHOLD", 0)

        # create a couple of edges carrying a property, let the replica sync
        master_graph.query(
            "CREATE (a:P {v: 1})-[:R {seen: 1}]->(b:P {v: 2}),"
            "       (b)-[:R {seen: 2}]->(:P {v: 3})")
        master.execute_command("WAIT", "1", "0")

        # pick the edge we'll remove on the replica only
        res = master_graph.query(
            "MATCH ()-[r:R]->() RETURN id(r) ORDER BY id(r) LIMIT 1")
        target_id = res.result_set[0][0]

        # allow direct writes on the replica and delete the edge there only,
        # putting that id on the replica's free list while the master keeps
        # the edge live - exactly the state the crash needs
        replica.config_set("slave-read-only", "no")
        replica_graph.query(
            f"MATCH ()-[r]->() WHERE id(r) = {target_id} DELETE r")
        replica.config_set("slave-read-only", "yes")

        # sanity check: replica and master have diverged
        replica_count = replica_graph.ro_query(
            "MATCH ()-[r:R]->() RETURN count(r)").result_set[0][0]
        env.assertEquals(replica_count, 1)

        sync_full_before = master.info()["sync_full"]

        # update that same edge on the master
        # the new value must differ from the current one, a no-op SET emits
        # no effect at all
        master_graph.query(
            f"MATCH ()-[r]->() WHERE id(r) = {target_id} SET r.seen = 42")

        # the replica must detect the divergence and force a full resync
        # rather than dereference a NULL attribute-set and die
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

        # replica is still alive and serving, and its dataset matches the
        # master's again - including the updated property
        master.execute_command("WAIT", "1", "5000")

        expected = master_graph.query(
            "MATCH ()-[r:R]->() RETURN id(r), r.seen ORDER BY id(r)").result_set
        actual = replica_graph.ro_query(
            "MATCH ()-[r:R]->() RETURN id(r), r.seen ORDER BY id(r)").result_set
        env.assertEquals(expected, actual)

        # confirm it was our UPDATE_EDGE guard that reported the divergence,
        # when a log file is available (RLTest may run with output capturing
        # disabled, in which case the resync assertion above made the point)
        log_name = env.envRunner._getFileName("slave", ".log")
        try:
            with open(os.path.join(env.logDir, log_name)) as f:
                log = f.read()
        except FileNotFoundError:
            return

        env.assertContains("UPDATE_EDGE references edge", log)


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

