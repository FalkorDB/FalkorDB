import os
import time
from common import *
from graph_utils import graph_eq

# ---------------------------------------------------------------------------
# Regression: GRAPH.EFFECT crash on replica when a referenced relation/label
# schema is missing locally (master/replica schema desync).
#
# Customer-reported crash chain (FalkorDB v4.18.1):
#   replica receives GRAPH.EFFECT for an edge whose relation type does not
#   exist in the replica's schema -> GraphHub_CreateEdges calls
#   GraphContext_GetSchemaByID which OOB-reads the schemas array and
#   returns garbage/NULL -> Schema_HasIndices(NULL) dereferences NULL ->
#   SIGSEGV at (nil) (the symbolicator pointed at the nearest exported
#   symbol, QGEdge_RelationID, in the stripped release build).
#
# Fix (PR #1907):
#   - GraphContext_GetSchemaByID now bounds-checks `id` and returns NULL
#     for out-of-range IDs.
#   - GraphHub_CreateEdges and ApplyLabels detect a NULL schema, log a
#     warning, and call exit(1) so Redis triggers a full RDB resync on
#     restart - matching the existing ValidateVersion pattern in
#     Effects_Apply (effects_apply.c:654-656).
#
# The tests below reproduce the crash through the public Cypher API and
# real master/replica replication - no direct GRAPH.EFFECT calls.
# ---------------------------------------------------------------------------


def _server_alive(conn):
    try:
        time.sleep(0.5)
        conn.ping()
        return True
    except Exception:
        return False


def _replica_log_has_desync_warning(runner):
    """Return True if the replica log contains the PR #1907 clean-exit warning.

    The warning is emitted by GraphHub_CreateEdges / ApplyLabels before
    calling exit(1) when an OOB schema ID is detected:
        "... schema desync detected, aborting"

    When the server crashes via SIGSEGV instead, the log contains
    "REDIS BUG REPORT START" / "Segmentation fault" instead.
    Returns None when the log file cannot be found (non-fatal).
    """
    if runner.dbDirPath is None:
        return None
    # _getFileName(role, suffix): role='slave' matches the else branch
    log_path = os.path.join(runner.dbDirPath,
                            runner._getFileName('slave', '.log'))
    try:
        with open(log_path) as f:
            content = f.read()
        return "schema desync detected, aborting" in content
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Crash reproduction via real replication (PR #1907 defensive hardening).
#
# The synthetic-payload approach (direct GRAPH.EFFECT calls) is replaced by
# an end-to-end replication scenario that never calls GRAPH.EFFECT from
# test code - the server sends GRAPH.EFFECT automatically when
# EFFECTS_THRESHOLD=0, matching the customer's production configuration.
#
# Scenario - edge relation OOB via replication:
#
#   1. Both master and replica build an identical graph with N edge schemas
#      (R0..R(N-1) at IDs 0..N-1) through normal GRAPH.EFFECT replication.
#
#   2. The replica's copy of the graph is deleted locally (replica-read-only
#      disabled briefly, then re-enabled). Master still has [R0..R(N-1)].
#
#   3. Master creates one more edge with a brand-new relation type R_NEW.
#      Because EFFECTS_THRESHOLD=0 the server sends:
#
#        EFFECT_CREATE_NODE / EFFECT_CREATE_NODE (two anonymous endpoints)
#        EFFECT_ADD_SCHEMA(R_NEW)           <- master assigns id=N locally
#        EFFECT_CREATE_EDGE(relation_id=N)  <- references master's id for R_NEW
#
#   4. The replica receives the stream.  cmd_effect.c creates a fresh empty
#      graph (GraphContext_Retrieve with shouldCreate=true).
#      EFFECT_ADD_SCHEMA(R_NEW) assigns id=0 (first entry in the empty
#      schemas array).
#      EFFECT_CREATE_EDGE(relation_id=N):
#        GraphHub_CreateEdges calls GraphContext_GetSchemaByID(gc, N, EDGE).
#        N >= 1 (the replica's schema array has exactly one entry) ->
#        bounds check returns NULL -> exit(1) + warning log.
#
# Without PR #1907 bounds check: OOB array access -> garbage schema pointer
#   -> Schema_HasIndices(NULL/garbage) -> SIGSEGV.
# With PR #1907: clean exit(1), Redis triggers a full RDB resync on restart.
# ---------------------------------------------------------------------------
class testEffectSchemaDesyncViaReplication():
    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True)
        self.master  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        # Force all writes through GRAPH.EFFECT - matches customer deployment
        self.db.config_set("EFFECTS_THRESHOLD", 0)
        self.master.wait(1, 0)

    def test01_oob_relation_triggers_clean_exit(self):
        """
        Trigger OOB relation-id via real GRAPH.EFFECT replication.

        Three edge schemas are synced to both sides, the replica's graph is
        deleted, then master creates R_NEW (local id=3).  The EFFECT stream
        carries relation_id=3 which is out of range on the empty replica
        (R_NEW would be id=0 there) -> bounds check -> exit(1).

        Without PR #1907: SIGSEGV.
        With PR #1907: clean exit(1) + "schema desync detected" warning log.
        """
        GRAPH_ID = "desync_edge_crash"
        N = 3  # number of pre-existing edge schemas to create first
        master_g = Graph(self.master, GRAPH_ID)

        # ------------------------------------------------------------------
        # Step 1: create N edge schemas on BOTH master and replica via
        # GRAPH.EFFECT replication.  After this both sides are identical.
        # ------------------------------------------------------------------
        for i in range(N):
            master_g.query("CREATE ()-[:R%d]->()" % i)
        self.master.wait(1, 2000)

        r_types = master_g.ro_query("CALL db.relationshipTypes()").result_set
        self.env.assertEquals(len(r_types), N)

        # ------------------------------------------------------------------
        # Step 2: delete the graph from the REPLICA ONLY to create a schema
        # desync: master has N schemas, replica has an empty / no graph.
        # ------------------------------------------------------------------
        self.replica.execute_command("CONFIG", "SET", "replica-read-only", "no")
        self.replica.execute_command("DEL", GRAPH_ID)
        self.replica.execute_command("CONFIG", "SET", "replica-read-only", "yes")

        # ------------------------------------------------------------------
        # Step 3: create a new edge with a brand-new relation type.
        # EFFECTS_THRESHOLD=0 makes the master send GRAPH.EFFECT:
        #   EFFECT_ADD_SCHEMA(R_NEW)           <- id=N on master
        #   EFFECT_CREATE_EDGE(relation_id=N)  <- OOB on empty replica
        # ------------------------------------------------------------------
        res = master_g.query("CREATE ()-[:R_NEW]->()")
        self.env.assertEquals(res.relationships_created, 1)

        # ------------------------------------------------------------------
        # Step 4: replica must be dead.
        #   _server_alive() already waits 0.5s for the EFFECT to propagate.
        #   exit(1) with PR #1907; SIGSEGV without it.
        # ------------------------------------------------------------------
        self.env.assertFalse(_server_alive(self.replica))

        # ------------------------------------------------------------------
        # Step 5: verify the replica exited cleanly (PR #1907 fix) rather
        # than crashing with a SIGSEGV.  Check the replica log for the
        # "schema desync detected, aborting" warning written before exit(1).
        # (Non-fatal if the log file is not available in this environment.)
        # ------------------------------------------------------------------
        clean = _replica_log_has_desync_warning(self.env.envRunner)
        if clean is not None:
            self.env.assertTrue(clean)


# ---------------------------------------------------------------------------
# Real master/replica integration test - reproduces the customer-reported
# desync chain organically (no synthetic GRAPH.EFFECT payloads).
#
# Two desync mechanisms were identified in v4.18.1:
#
# ── Mechanism 1: RedisModule_Yield interleaving (PRIMARY crash path) ────────
#
#   In v4.18.1, _QueryCtx_ThreadSafeContextLock calls RedisModule_Yield
#   (YIELD_FLAG_CLIENTS) whenever the query runs without a blocked client
#   (i.e. every replicated GRAPH.QUERY on the replica's main thread).
#   The Yield fires BETWEEN the READ-lock release and the WRITE-lock
#   acquisition, opening a window where the Redis event loop can process
#   the *next* command from the replication socket.
#
#   If that next command is a GRAPH.EFFECT that references a schema just
#   about to be committed by the still-in-flight GRAPH.QUERY, the replica
#   crashes:
#
#     Replica main thread:
#       GRAPH.QUERY "CREATE ()-[:R1]->()"   ← fast (<threshold) → GRAPH.QUERY
#         QueryCtx_AcquireWriteLock():
#           Graph_ReleaseLock(READ)          ← R1 not yet committed
#           _QueryCtx_ThreadSafeContextLock:
#             RedisModule_Yield()            ← event loop runs here
#               → processes replication socket
#               → GRAPH.EFFECT [EFFECT_CREATE_EDGE(r_id=0)]
#                   GraphHub_CreateEdges(r=0)
#                   GraphContext_GetSchemaByID(gc,0,SCHEMA_EDGE) → NULL
#                   Schema_HasIndices(NULL) → SIGSEGV   ← crash
#           Graph_AcquireWriteLock()
#           _CommitEdgesBlueprint()          ← creates R1 (too late)
#
#   This race is closed by PR #1877 (already in master), which restricts
#   the Yield to AOF-loading paths only. Because the timing is
#   sub-millisecond, this race cannot be triggered deterministically from
#   a Python test; it is documented here for completeness.
#
# ── Mechanism 2: Failed-query orphan schema (before PR #1815) ───────────────
#
#   A write query introduces a NEW schema as part of its plan but later
#   raises an error (e.g. division by zero in a projection that runs
#   AFTER the CREATE clause).
#
#   Pre-PR-#1815: the rollback path did NOT undo the schema addition, and
#   ResultSet_Clear suppressed any replication. The new schema became an
#   "orphan" present on the master only.
#
#   A subsequent successful write query referencing a different brand-new
#   relation type was replicated as GRAPH.EFFECT (EFFECTS_THRESHOLD=0):
#
#     EFFECT_ADD_SCHEMA(<new relation>) → replica appends, gets the next
#       sequential id locally (one behind master's because of the orphan).
#     EFFECT_CREATE_EDGE(relation_id=<master's id>) → master's id is one
#       ahead of the replica's; GraphContext_GetSchemaByID OOB-reads the
#       schemas array → Schema_HasIndices(garbage) → SIGSEGV.
#
#   PR #1815 (f2a8531c3, in v4.18.0+) makes failing queries roll back the
#   schema atomically, so the orphan can no longer form.
#
# ── Defensive hardening (this PR #1907) ─────────────────────────────────────
#
#   Regardless of *how* desync occurs, GraphContext_GetSchemaByID had no
#   bounds check on the schema ID. This PR adds:
#     - bounds check in GraphContext_GetSchemaByID (returns NULL for OOB id)
#     - NULL check in GraphHub_CreateEdges BEFORE Graph_CreateEdges
#     - NULL check in ApplyLabels
#   On NULL: log a warning and exit(1), triggering a full RDB resync.
#   This matches the existing ValidateVersion + exit(1) pattern in
#   Effects_Apply (effects_apply.c:654-656).
#
# The integration tests below drive Mechanism 2 through the public Cypher API.
# They pass on this branch. Reverting PR #1815 causes the replica to SIGSEGV.
# ---------------------------------------------------------------------------
class testEffectSchemaDesyncReplica():
    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True)
        self.master  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        # force every write to be replicated as GRAPH.EFFECT so that a
        # schema-id mismatch between master and replica becomes observable
        # exactly as in the customer's deployment
        self.db.config_set("EFFECTS_THRESHOLD", 0)
        # let the replica catch up on the config change
        self.master.wait(1, 0)

    def __del__(self):
        try:
            self.replica.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # helper: run a query on master and wait for the replica to ack it
    # ------------------------------------------------------------------
    def _run_and_replicate(self, g, q):
        res = g.query(q)
        # wait up to 2s for at least one replica to ack
        self.master.wait(1, 2000)
        return res

    def test01_failing_schema_query_then_effect_edge(self):
        """
        End-to-end reproduction of the v4.18.1 desync -> EFFECT crash chain.
        The test must remain green: with PR #1815 the orphan schema is
        rolled back on master, so the subsequent EFFECT-replicated edge
        carries an in-range relation id and the replica stays alive.
        """

        graph_id = "effect_desync_repro"
        master_g  = Graph(self.master,  graph_id)
        replica_g = Graph(self.replica, graph_id)

        # ----------------------------------------------------------------
        # step 1 - failing query that would introduce edge schema R_BAD on
        # master only, prior to PR #1815. The "/0" raises ArithmeticError
        # AFTER the CREATE clause has registered the relation type.
        # ----------------------------------------------------------------
        try:
            master_g.query(
                "CREATE ()-[:R_BAD]->() WITH 1 AS x RETURN x / 0"
            )
            # the projection must raise; if we get here the test premise
            # is broken (the query accidentally succeeded)
            self.env.assertTrue(False)
        except Exception:
            pass

        # ----------------------------------------------------------------
        # step 2 - successful write that introduces a different brand-new
        # relation type. With EFFECTS_THRESHOLD=0 this is replicated as
        # GRAPH.EFFECT, carrying EFFECT_ADD_SCHEMA(R_GOOD) followed by
        # EFFECT_CREATE_EDGE(relation_id=<master id for R_GOOD>, ...).
        #
        # Pre-PR-#1815: master id for R_GOOD == 1 (orphan R_BAD took id 0),
        # replica appends and assigns id 0 locally -> EFFECT_CREATE_EDGE
        # references a non-existent relation id 1 on the replica, crashing
        # GraphHub_CreateEdges.
        # ----------------------------------------------------------------
        res = self._run_and_replicate(
            master_g,
            "CREATE ()-[:R_GOOD]->()"
        )
        self.env.assertEquals(res.relationships_created, 1)

        # ----------------------------------------------------------------
        # step 3 - replica must still be alive and consistent. If the
        # replica had segfaulted, ping() would raise / the connection
        # would be dead.
        # ----------------------------------------------------------------
        self.env.assertTrue(_server_alive(self.replica))

        # cross-check: master and replica agree on the schema view
        m_rels = master_g.ro_query("CALL db.relationshipTypes()").result_set
        r_rels = replica_g.ro_query("CALL db.relationshipTypes()").result_set
        self.env.assertEquals(m_rels, r_rels)

        # cross-check: graph contents agree
        self.env.assertTrue(graph_eq(master_g, replica_g))

    def test02_repeated_failures_keep_replica_in_sync(self):
        """
        Stress the schema-rollback path by alternating failing schema-
        introducing queries with successful EFFECT-replicated writes.
        Every cycle adds a fresh failing query (different label name) so
        the rollback path is exercised against a non-empty schema array,
        which is exactly the situation that produced an off-by-one
        relation id on the customer's deployment.
        """

        graph_id = "effect_desync_repeat"
        master_g  = Graph(self.master,  graph_id)
        replica_g = Graph(self.replica, graph_id)

        # seed both sides with one successful relation so the schema array
        # is non-empty before we start poking at it
        self._run_and_replicate(master_g, "CREATE ()-[:SEED]->()")
        self.env.assertTrue(_server_alive(self.replica))

        for i in range(5):
            # failing schema-introducing query - must roll back cleanly
            try:
                master_g.query(
                    "CREATE ()-[:R_FAIL_%d]->() WITH 1 AS x RETURN x / 0" % i
                )
                self.env.assertTrue(False)
            except Exception:
                pass

            # successful EFFECT-replicated write with a fresh relation
            res = self._run_and_replicate(
                master_g,
                "CREATE ()-[:R_OK_%d]->()" % i
            )
            self.env.assertEquals(res.relationships_created, 1)

            # replica must remain alive after every cycle
            self.env.assertTrue(_server_alive(self.replica))

        # final consistency check
        m_rels = sorted(master_g.ro_query(
            "CALL db.relationshipTypes()").result_set)
        r_rels = sorted(replica_g.ro_query(
            "CALL db.relationshipTypes()").result_set)
        self.env.assertEquals(m_rels, r_rels)
        self.env.assertTrue(graph_eq(master_g, replica_g))
