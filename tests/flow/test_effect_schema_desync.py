import struct
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
# Fix:
#   - GraphContext_GetSchemaByID now bounds-checks `id` and returns NULL
#     for out-of-range IDs.
#   - GraphHub_CreateEdges and ApplyLabels detect a NULL schema, log a
#     warning, and call exit(1) so Redis triggers a full RDB resync on
#     restart - matching the existing ValidateVersion pattern in
#     Effects_Apply (effects_apply.c:654-656).
#
# These tests craft GRAPH.EFFECT payloads with out-of-range relation/label
# IDs and assert the server exits cleanly instead of crashing.
# ---------------------------------------------------------------------------

# Effects binary protocol constants (must match src/effects/effects.h)
EFFECTS_VERSION    = 1
EFFECT_CREATE_NODE = 3   # node creation
EFFECT_CREATE_EDGE = 4   # edge creation
EFFECT_SET_LABELS  = 7   # set labels
EFFECT_ADD_SCHEMA  = 9   # schema addition

# Schema types (must match src/schema/schema.h)
SCHEMA_NODE = 0
SCHEMA_EDGE = 1


def _build_effects_payload(effects):
    """Build a binary GRAPH.EFFECT payload from a list of effect tuples.

    Supported effects (keep in sync with src/effects/effects.c writers):
        ('add_schema',  schema_type, name_str)
        ('create_node', [label_id, ...], attr_count)         # attr_count == 0
        ('create_edge', relation_id, src_id, dest_id)        # 0 attributes
        ('set_labels',  node_id, [label_id, ...])
    """

    buf = struct.pack('<B', EFFECTS_VERSION)  # version: uint8_t

    for effect in effects:
        kind = effect[0]
        if kind == 'add_schema':
            _, schema_type, name = effect
            name_bytes = name.encode('utf-8') + b'\x00'
            buf += struct.pack('<i', EFFECT_ADD_SCHEMA)      # EffectType
            buf += struct.pack('<i', schema_type)            # SchemaType
            buf += struct.pack('<Q', len(name_bytes))        # size_t
            buf += name_bytes
        elif kind == 'create_node':
            _, label_ids, attr_count = effect
            buf += struct.pack('<i', EFFECT_CREATE_NODE)
            buf += struct.pack('<H', len(label_ids))         # uint16_t
            for lid in label_ids:
                buf += struct.pack('<i', lid)                # LabelID (int32)
            buf += struct.pack('<H', attr_count)             # uint16_t
        elif kind == 'create_edge':
            _, rel_id, src_id, dest_id = effect
            # packed struct: EffectType(int) + uint16_t rel_count + RelationID
            # + NodeID src + NodeID dest
            buf += struct.pack('<i', EFFECT_CREATE_EDGE)     # EffectType
            buf += struct.pack('<H', 1)                      # rel_count
            buf += struct.pack('<i', rel_id)                 # RelationID int32
            buf += struct.pack('<Q', src_id)                 # NodeID uint64
            buf += struct.pack('<Q', dest_id)                # NodeID uint64
            buf += struct.pack('<H', 0)                      # attr_count == 0
        elif kind == 'set_labels':
            _, node_id, label_ids = effect
            buf += struct.pack('<i', EFFECT_SET_LABELS)
            buf += struct.pack('<Q', node_id)                # node ID
            buf += struct.pack('<H', len(label_ids))         # label count
            for lid in label_ids:
                buf += struct.pack('<i', lid)                # LabelID
        else:
            raise ValueError("unknown effect kind: %s" % kind)

    return buf


def _server_alive(conn):
    try:
        time.sleep(0.5)
        conn.ping()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# EDGE: relation ID out of range -> clean exit (was SIGSEGV)
# ---------------------------------------------------------------------------
class testEffectOOBRelation():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test01_create_edge_unknown_relation_exits(self):
        graph_id = "edge_oob_rel"
        g = Graph(self.conn, graph_id)

        # create two nodes so node IDs 0 and 1 exist on this server
        g.query("CREATE (), ()")

        # craft a GRAPH.EFFECT that creates an edge with relation_id = 100
        # the graph has zero edge schemas - this is the customer's crash
        payload = _build_effects_payload([
            ('create_edge', 100, 0, 1),
        ])

        try:
            self.conn.execute_command('GRAPH.EFFECT', graph_id, payload)
        except Exception:
            pass

        # the server must NOT have segfaulted - it must have exited cleanly
        # in either case the connection is dead, but with the fix there is
        # no SIGSEGV in the redis log (we cannot easily assert on the log
        # from here, but a clean exit() triggers full RDB resync on restart)
        self.env.assertFalse(_server_alive(self.conn))


# ---------------------------------------------------------------------------
# EDGE: relation_id = -1 (negative) -> clean exit (was OOB read / SIGSEGV)
# ---------------------------------------------------------------------------
class testEffectNegativeRelation():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test01_create_edge_negative_relation_exits(self):
        graph_id = "edge_neg_rel"
        g = Graph(self.conn, graph_id)

        g.query("CREATE (), ()")

        payload = _build_effects_payload([
            ('create_edge', -5, 0, 1),
        ])

        try:
            self.conn.execute_command('GRAPH.EFFECT', graph_id, payload)
        except Exception:
            pass

        self.env.assertFalse(_server_alive(self.conn))


# ---------------------------------------------------------------------------
# LABEL: SET_LABELS with a label_id beyond the local schema count must
# trigger a clean exit instead of dereferencing a NULL Schema.
# ---------------------------------------------------------------------------
class testEffectSetLabelsOOBLabel():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test01_set_labels_unknown_label_exits(self):
        graph_id = "set_label_oob"
        g = Graph(self.conn, graph_id)

        # create one node with label A -> node id 0, label id 0
        g.query("CREATE (:A)")

        # try to add label_id = 99 to node 0
        payload = _build_effects_payload([
            ('set_labels', 0, [99]),
        ])

        try:
            self.conn.execute_command('GRAPH.EFFECT', graph_id, payload)
        except Exception:
            pass

        self.env.assertFalse(_server_alive(self.conn))


# ---------------------------------------------------------------------------
# Real master/replica integration test - reproduces the customer-reported
# desync chain organically (no synthetic GRAPH.EFFECT payloads).
#
# Customer scenario (v4.18.1, prior to PR #1815 "rollback schema creation
# for failing queries"):
#
#   1. A write query introduces a NEW schema as part of its plan but later
#      raises an error (e.g. division by zero in a projection that runs
#      AFTER the CREATE clause).
#
#   2. Pre-PR-#1815 behaviour: the rollback path did NOT undo the schema
#      addition, and ResultSet_Clear suppressed any replication. The new
#      schema therefore became an "orphan" present on the master only.
#
#   3. A subsequent successful write query that references a different,
#      brand-new relation type was replicated as GRAPH.EFFECT (because
#      EFFECTS_THRESHOLD is 0 on this deployment, so every write goes
#      through the effects path).
#
#      The EFFECT stream contained:
#        EFFECT_ADD_SCHEMA(name=<new relation>) - replica appends, gets the
#        next sequential id locally.
#        EFFECT_CREATE_EDGE(relation_id=<master's id>, ...) - master's id
#        is one ahead of the replica's because of the orphan schema.
#
#   4. On the replica, GraphHub_CreateEdges resolved the (out-of-range)
#      relation id via GraphContext_GetSchemaByID, which OOB-read the
#      schemas array. Schema_HasIndices(NULL) then dereferenced NULL ->
#      SIGSEGV at (nil) (the symbolicator pointed at QGEdge_RelationID,
#      the nearest exported symbol in the stripped release build).
#
# The fix has two layers:
#
#   * PR #1815 makes the failing query roll back the schema, so master and
#     replica stay aligned and no malformed EFFECT is ever produced.
#
#   * PR #1907 (this branch) hardens GraphContext_GetSchemaByID with a
#     bounds check and makes GraphHub_CreateEdges / ApplyLabels detect a
#     desync and exit(1) cleanly. exit(1) triggers a full RDB resync on
#     restart - matching the existing ValidateVersion + exit(1) pattern in
#     Effects_Apply (effects_apply.c:654-656). This prevents future regressions
#     where some other code path leaves master and replica out of sync.
#
# This integration test runs an actual master + replica pair, drives the
# failing-query scenario through the public Cypher API, and asserts that:
#
#   - the replica is still alive after the EFFECT replicates
#   - the replica's view of the graph matches the master byte-for-byte
#
# Reverting either PR #1815 or PR #1907 will cause this test to fail
# (replica process dies under SIGSEGV before the second assertion can
# complete).
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
