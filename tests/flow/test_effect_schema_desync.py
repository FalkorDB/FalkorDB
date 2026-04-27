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
#     warning, and call _exit(1) so Redis triggers a full RDB resync on
#     restart - matching the existing ValidateVersion pattern in
#     Effects_Apply (effects_apply.c).
#
# ============================================================================
# IMPORTANT: With current master, NO realistic user operation can produce a
# master/replica schema-id divergence anymore.  Both historical mechanisms are
# closed:
#
#   * Mechanism 1 - RedisModule_Yield interleaving (PRIMARY cause of the
#     customer's v4.18.1 crashes).  Closed by PR #1877 (39460fe8c) which
#     restricts Yield to AOF/RDB loading only.  The race required
#     sub-millisecond timing and the now-removed Yield call; it cannot be
#     reproduced from a Python test on current master.
#
#   * Mechanism 2 - Failed-query orphan schema (pre-v4.18.0).  Closed by
#     PR #1815 (f2a8531c3) which makes failing queries atomically roll back
#     any schemas they introduced.  Reproducing the desync requires
#     reverting that PR.
#
# This file therefore tests:
#
#   1. testEffectSchemaDesyncReplica - the PR #1815 regression scenario.
#      Uses ONLY the public Cypher API and real master/replica replication.
#      Passes today; reverting PR #1815 causes the replica to SIGSEGV
#      (without PR #1907) or to exit(1) cleanly (with PR #1907).  Either
#      way the test catches a regression in PR #1815's rollback path.
#
#   2. testEffectReplicationStability - positive smoke test that exercises
#      the GRAPH.EFFECT path through normal Cypher writes and verifies
#      that PR #1907's bounds-check does not produce false positives in
#      day-to-day operation.
#
# The bounds-check itself (GraphContext_GetSchemaByID) is exercised by the
# unit test tests/unit/test_graphcontext_schema_bounds.c which directly
# constructs a GraphContext and calls the function with in-range, out-of-
# range, and negative IDs.  That is the only way to exercise the defensive
# code path without resorting to direct GRAPH.EFFECT injection or to
# reverting the upstream fixes.
# ---------------------------------------------------------------------------


def _server_alive(conn):
    """Return True if a redis connection still responds to PING."""
    try:
        time.sleep(0.5)
        conn.ping()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Real master/replica integration test - reproduces the customer-reported
# desync chain organically (no synthetic GRAPH.EFFECT payloads, no replica-
# side mutation).  This is a PR #1815 regression test: with #1815 the
# rollback prevents the desync, the replica stays alive and graphs match.
# Reverting #1815 (without PR #1907) causes the replica to SIGSEGV; with
# PR #1907 the replica exits cleanly with the desync warning.
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
            # best-effort teardown; replica may already be terminated
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
        With PR #1815 the orphan schema is rolled back on master, so the
        subsequent EFFECT-replicated edge carries an in-range relation id
        and the replica stays alive.
        """

        graph_id = "effect_desync_repro"
        master_g  = Graph(self.master,  graph_id)
        replica_g = Graph(self.replica, graph_id)

        # ----------------------------------------------------------------
        # step 1 - failing query that would introduce edge schema R_BAD on
        # master only, prior to PR #1815. The "/0" raises ArithmeticError
        # AFTER the CREATE clause has registered the relation type.
        # ----------------------------------------------------------------
        query_raised = False
        try:
            master_g.query(
                "CREATE ()-[:R_BAD]->() WITH 1 AS x RETURN x / 0"
            )
        except Exception:
            # expected: division by zero raises after CREATE registers R_BAD
            query_raised = True
        self.env.assertTrue(query_raised)

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
        # step 3 - replica must still be alive and consistent.
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
        the rollback path is exercised against a non-empty schema array.
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
            query_raised = False
            try:
                master_g.query(
                    "CREATE ()-[:R_FAIL_%d]->() WITH 1 AS x RETURN x / 0" % i
                )
            except Exception:
                # expected: division by zero raises after CREATE registers type
                query_raised = True
            self.env.assertTrue(query_raised)

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


# ---------------------------------------------------------------------------
# Positive smoke test: PR #1907 added a bounds check inside the hot path
# of every replicated edge / SET LABELS effect.  This test verifies that
# normal master/replica operation - which now flows through that bounds
# check on every effect application - still works correctly and does not
# produce false positives (i.e. the replica never trips the desync warning
# during normal operation, regardless of the EFFECTS_THRESHOLD value).
# ---------------------------------------------------------------------------
class testEffectReplicationStability():
    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True)
        self.master  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()

    def __del__(self):
        try:
            self.replica.shutdown()
        except Exception:
            # best-effort teardown; replica may already be terminated
            pass

    def _wait_replica(self):
        # wait up to 2s for at least one replica to ack
        self.master.wait(1, 2000)

    def test01_normal_writes_do_not_trip_bounds_check_effect_path(self):
        """
        Force every write through GRAPH.EFFECT (EFFECTS_THRESHOLD=0).  Every
        replicated effect goes through the new GraphHub_CreateEdges /
        ApplyLabels NULL-schema check.  Verify that normal operations do
        NOT trigger the check (no replica exit, graphs stay equal).
        """

        self.db.config_set("EFFECTS_THRESHOLD", 0)
        self.master.wait(1, 0)

        graph_id = "stability_effect"
        master_g  = Graph(self.master,  graph_id)
        replica_g = Graph(self.replica, graph_id)

        # mix of node creations (introduce labels), edge creations
        # (introduce relation types), label additions and label removals -
        # every code path that calls GraphContext_GetSchemaByID on the
        # replica's effects-application thread.
        master_g.query("CREATE (:A {n: 1})")
        master_g.query("CREATE (:B {n: 2})")
        master_g.query("CREATE (:A:B {n: 3})")
        master_g.query(
            "MATCH (a:A {n: 1}), (b:B {n: 2}) CREATE (a)-[:R1]->(b)"
        )
        master_g.query(
            "MATCH (a:A {n: 1}), (b:B {n: 2}) CREATE (a)-[:R2 {w: 1.5}]->(b)"
        )
        # add and remove labels - exercises ApplyLabels and ApplyRemoveLabels
        master_g.query("MATCH (n:A) SET n:NEWLBL")
        master_g.query("MATCH (n:NEWLBL) REMOVE n:NEWLBL")
        self._wait_replica()

        # replica must be alive
        self.env.assertTrue(_server_alive(self.replica))
        # graphs must be equal
        self.env.assertTrue(graph_eq(master_g, replica_g))

    def test02_normal_writes_do_not_trip_bounds_check_query_path(self):
        """
        Same as test01 but with EFFECTS_THRESHOLD high enough that writes
        replicate as GRAPH.QUERY (re-executed on replica).  Acts as a
        control - the bounds check should not be exercised here at all,
        but the test still validates that PR #1907 did not break the
        normal QUERY-replicated write path.
        """

        self.db.config_set("EFFECTS_THRESHOLD", 999999)
        self.master.wait(1, 0)

        graph_id = "stability_query"
        master_g  = Graph(self.master,  graph_id)
        replica_g = Graph(self.replica, graph_id)

        master_g.query("CREATE (:A)-[:R1]->(:B)")
        master_g.query("CREATE (:C)-[:R2]->(:D)")
        master_g.query("MATCH (n:A) SET n:EXTRA")
        self._wait_replica()

        self.env.assertTrue(_server_alive(self.replica))
        self.env.assertTrue(graph_eq(master_g, replica_g))
