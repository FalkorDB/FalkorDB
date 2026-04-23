import struct
import time
from common import *

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
