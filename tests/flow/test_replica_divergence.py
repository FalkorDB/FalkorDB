from common import *
import os
import re
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



# A refusal raised PART WAY THROUGH a payload still reaches the divergence
# guard, and the replica still converges.
#
# WHY THIS IS NOT A COPY OF testReplicaDivergence ABOVE. Until effects v3
# decoded one record at a time, a payload was decoded whole and then applied, so
# a refusal happened before the graph was touched at all. Streaming decode
# applies records 1..k and only then raises the refusal, which is a state that
# could not previously exist. The guard turns it into a forced resync that
# overwrites the graph wholesale, so the partial state is real but transient -
# apply returning false is a report, not a rollback.
#
# THE ASSERTION THAT MAKES THIS TEST MEAN ANYTHING IS `k >= 1`. A prefix refused
# BEFORE any record applies raises sync_full and converges exactly like one
# refused after - a resync converges by definition, whatever it is repairing. So
# sync_full and the state comparison cannot tell the new behaviour from the old,
# and without k this class would silently re-prove what the first class in this
# file already proves. The log is the only place the distinction survives,
# because the partial state exists solely in the window the resync erases.
class testMidStreamRefusal():
    def __init__(self):
        # replication timing doesn't play well with Valgrind/sanitizers
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        # Passed as moduleArgs rather than set at runtime: EFFECTS_VERSION is
        # load-time only, and C still emits v2 by default
        # (EFFECTS_VERSION_EMIT is 2 in effects.h, deliberately, until every
        # reader understands v3). Without it this class would exercise the v2
        # path, which decodes and applies in one pass and has no mid-stream
        # refusal to find - it would pass while testing nothing it claims to.
        self.env, self.db = Env(env='oss', useSlaves=True,
                                 enableDebugCommand=True,
                                 moduleArgs='EFFECTS_THRESHOLD 0 EFFECTS_VERSION 3')

    @staticmethod
    def _replica_log(env):
        """Every slave log in this env's log dir, concatenated.

        Read by scanning rather than through RLTest's _getFileName: the slave's
        role string has moved between RLTest versions, and a helper that
        guessed it wrong would return an empty log, which reads as "the line is
        not there" rather than as "I looked in the wrong file".
        """
        parts = []
        for name in sorted(os.listdir(env.logDir)):
            if "slave" in name and name.endswith(".log"):
                with open(os.path.join(env.logDir, name),
                          errors="replace") as f:
                    parts.append(f.read())
        return "\n".join(parts)

    def test_partial_apply_still_reaches_the_guard(self):
        env      = self.env
        master   = env.getConnection()
        replica  = env.getSlaveConnection()
        graph_id = GRAPH_ID + "_midstream"

        master_graph  = Graph(master, graph_id)
        replica_graph = Graph(replica, graph_id)

        # WAIT IS NOT THE GUARD HERE, and reaching for it is the trap. A write
        # issued before the replica finishes its initial sync is not propagated
        # as a command at all - it is folded into the full-sync RDB, and the
        # replica ends up holding the data with an empty command stream. WAIT
        # blocks until the replica is online and acks, so it returns
        # successfully after that fold has already happened.
        #
        # This class cannot survive that: it asserts the payload arrived as a
        # GRAPH.EFFECT and was refused part way through. A folded write produces
        # no refusal line, so the failure would be real rather than silent - but
        # it would point at the applier instead of at the setup.
        # redis-py parses the slave0 line into a dict, so this reads the
        # 'state' key rather than matching "state=online" as a substring -
        # a substring test against a dict silently never matches, which is a
        # guard that cannot fire
        deadline = time.time() + 30
        while time.time() < deadline:
            slave0 = master.info("replication").get("slave0")
            state = (slave0.get("state") if isinstance(slave0, dict)
                     else str(slave0 or ""))
            if state == "online":
                break
            time.sleep(0.2)
        else:
            raise Exception("replica never reached state=online")

        master_graph.query("CREATE (:P {v: 1}), (:P {v: 2}), (:P {v: 3})")
        master.execute_command("WAIT", "1", "0")

        res = master_graph.query(
            "MATCH (n:P) RETURN id(n) ORDER BY id(n)")
        target_id = res.result_set[0][0]   # removed on the replica below
        keep_id   = res.result_set[1][0]   # survives on both, so SET applies

        # diverge the replica: remove one node there and nowhere else
        replica.config_set("slave-read-only", "no")
        replica_graph.query(f"MATCH (n) WHERE id(n) = {target_id} DELETE n")
        replica.config_set("slave-read-only", "yes")

        env.assertEquals(
            replica_graph.ro_query("MATCH (n:P) RETURN count(n)").result_set[0][0], 2)

        # count the refusals already in the log, so the one this test causes is
        # distinguishable from any the earlier classes left behind
        before_refusals = len(re.findall(
            r"after applying (\d+) record\(s\)", self._replica_log(env)))

        sync_full_before = master.info()["sync_full"]

        # ONE query carrying TWO record groups, ordered so that a record
        # applies before the one that fails.
        #
        # v3 sorts groups by opcode ascending (_cmp_group compares opcode
        # first), and UPDATE_NODE is 1 while DELETE_NODE is 5. So the property
        # set lands on the replica, and the delete - targeting the id the replica
        # no longer has - is refused after it. That ordering is a property of
        # the format rather than of this query, which is why k is predictable
        # at all.
        #
        # SET rather than CREATE, and the reason is specific to this class's
        # setup. `... DELETE n CREATE (:Q)` does not work here because the
        # created node REUSES THE DELETED NODE'S ID - measured: deleting id 0
        # and creating in the same query yields id 0 again. Sorted by opcode
        # that payload is CREATE_NODE[0] then DELETE_NODE[0], and this class
        # has already removed id 0 from the replica, so the create fills the
        # hole the delete made, both records apply cleanly, and there is no
        # refusal to measure. On an undiverged replica the same payload is
        # refused at the create instead, because id 0 still exists there -
        # reported upstream separately, and not something this class relies on.
        #
        # SET touches a node present on both, so it applies; the DELETE of the
        # id only the master still has is refused after it.
        master_graph.query(
            f"MATCH (a:P) WHERE id(a) = {keep_id} SET a.v = 99 "
            f"WITH count(a) AS _ "
            f"MATCH (b) WHERE id(b) = {target_id} DELETE b")

        # (1) THE NEW PROPERTY: refused, with at least one record already
        # applied. Polled because the log is written by the replica.
        applied = None
        deadline = time.time() + 30
        while time.time() < deadline:
            hits = re.findall(r"after applying (\d+) record\(s\)",
                              self._replica_log(env))
            if len(hits) > before_refusals:
                applied = int(hits[before_refusals])
                break
            time.sleep(0.2)

        env.assertIsNotNone(
            applied,
            message="no 'after applying N record(s)' line appeared in the "
                    "replica log; the payload was not refused, or the refusal "
                    "did not reach the logging path")

        if applied is not None:
            env.assertTrue(
                applied >= 1,
                message=f"refused after applying {applied} records. This class "
                        f"exists for the case where records applied BEFORE the "
                        f"refusal; at 0 it is testing the same thing as "
                        f"test_forced_full_resync_on_effect_divergence and "
                        f"should be fixed rather than accepted")

        # (2) the guard fired: a genuine FULLRESYNC, not a reconnect that
        # continued the stream
        resynced = False
        deadline = time.time() + 60
        while time.time() < deadline:
            info = master.info()
            if (info["sync_full"] > sync_full_before and
                    info.get("connected_slaves", 0) > 0):
                resynced = True
                break
            time.sleep(0.5)

        env.assertTrue(resynced,
                       message="sync_full did not rise: the refusal did not "
                               "reach DivergenceGuard_OnFailure")

        # (3) the repair worked. Deliberately last and deliberately weakest:
        # a resync converges whether or not the mechanism under test did
        # anything, so this confirms the outcome and cannot establish the cause
        expected = master_graph.query(
            "MATCH (n) RETURN labels(n), n.v ORDER BY id(n)").result_set
        actual = replica_graph.ro_query(
            "MATCH (n) RETURN labels(n), n.v ORDER BY id(n)").result_set
        env.assertEquals(expected, actual)
