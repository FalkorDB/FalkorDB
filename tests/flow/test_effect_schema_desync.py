import struct
import time
from common import *

# Effects binary protocol constants (must match src/effects/effects.h)
EFFECTS_VERSION    = 1
EFFECT_CREATE_NODE = 3   # node creation
EFFECT_ADD_SCHEMA  = 9   # schema addition

# Schema types (must match src/schema/schema.h)
SCHEMA_NODE = 0


def _build_effects_payload(effects):
    """Build a binary GRAPH.EFFECT payload from a list of effect tuples.

    Each effect is a tuple:
        ('add_schema', schema_type, name_str)
        ('create_node', [label_id, ...], attr_count)  # attr_count must be 0
    """

    buf = struct.pack('<B', EFFECTS_VERSION)  # version: uint8_t

    for effect in effects:
        if effect[0] == 'add_schema':
            _, schema_type, name = effect
            name_bytes = name.encode('utf-8') + b'\x00'  # null-terminated
            buf += struct.pack('<i', EFFECT_ADD_SCHEMA)      # EffectType
            buf += struct.pack('<i', schema_type)            # SchemaType
            buf += struct.pack('<Q', len(name_bytes))        # size_t (8 bytes)
            buf += name_bytes                                # schema name
        elif effect[0] == 'create_node':
            _, label_ids, attr_count = effect
            buf += struct.pack('<i', EFFECT_CREATE_NODE)     # EffectType
            buf += struct.pack('<H', len(label_ids))         # label_count: uint16_t
            for lid in label_ids:
                buf += struct.pack('<i', lid)                # LabelID: int32
            buf += struct.pack('<H', attr_count)             # attr_count: uint16_t

    return buf


# ---------------------------------------------------------------------------
# Test: GRAPH.EFFECT with out-of-bounds label ID triggers clean exit(1)
#
# Root cause: Graph_GetLabelMatrix accesses g->labels[label_idx] without
# a runtime bounds check (the ASSERT is compiled out in release builds).
# When label_idx >= label_count the Delta_Matrix* is NULL/garbage and the
# subsequent Delta_Matrix_nrows call dereferences it -> SIGSEGV.
#
# The fix adds a runtime bounds check that calls exit(1) with a warning log
# when a schema desync is detected, matching the existing ValidateVersion
# pattern in Effects_Apply (effects_apply.c:654-656). On a replica, this
# forces a full RDB resync from the master on restart.
#
# See:
#   src/graph/graph.c  Graph_GetLabelMatrix
#   src/graph/graph.c  Graph_GetRelationMatrix
#   src/effects/effects_apply.c  ValidateVersion
# ---------------------------------------------------------------------------


class testEffectOOBLabel():
    """GRAPH.EFFECT referencing a non-existent label triggers a clean exit.

    The graph has zero node-schemas.  A crafted GRAPH.EFFECT tries to
    create a node with label_id = 100.  Before the fix this was a SIGSEGV;
    after the fix the server exits cleanly with a warning log.
    """

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test01_oob_label_id_causes_exit(self):
        graph_id = "oob_label"
        g = Graph(self.conn, graph_id)

        # ensure the graph exists (no node schemas yet)
        g.query("RETURN 1")

        # craft payload: create node with label_id=100, 0 attributes
        payload = _build_effects_payload([
            ('create_node', [100], 0),
        ])

        try:
            self.conn.execute_command('GRAPH.EFFECT', graph_id, payload)
        except Exception:
            pass

        # the server should have exited — verify it is no longer responsive
        server_alive = True
        try:
            time.sleep(0.5)
            self.conn.ping()
        except Exception:
            server_alive = False

        self.env.assertFalse(server_alive)


class testEffectLabelExceedsSchemaCount():
    """Label ID beyond current schema count must trigger clean exit.

    Create label A (id=0) via a normal query.  Then send a GRAPH.EFFECT
    that tries to create a node with label_id=5 (out-of-range).
    """

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test01_label_id_exceeds_count(self):
        graph_id = "label_exceeds"
        g = Graph(self.conn, graph_id)

        # create label A -> label id 0
        g.query("CREATE (:A)")

        # craft payload: create node with label_id=5 (only 0 exists)
        payload = _build_effects_payload([
            ('create_node', [5], 0),
        ])

        try:
            self.conn.execute_command('GRAPH.EFFECT', graph_id, payload)
        except Exception:
            pass

        # server should have exited
        server_alive = True
        try:
            time.sleep(0.5)
            self.conn.ping()
        except Exception:
            server_alive = False

        self.env.assertFalse(server_alive)


class testEffectSchemaAddThenOOBCreate():
    """EFFECT_ADD_SCHEMA + EFFECT_CREATE_NODE with mismatched ID.

    Simulates the production crash scenario: the effects buffer contains
    EFFECT_ADD_SCHEMA for "Product" followed by EFFECT_CREATE_NODE with
    a label_id that is valid on the master but out-of-bounds on a replica
    whose schema state has diverged.
    """

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test01_schema_add_then_oob_create(self):
        graph_id = "schema_desync"
        g = Graph(self.conn, graph_id)

        # ensure graph exists
        g.query("RETURN 1")

        # add schema "Product" (will get id = 0, the first label)
        # then create node with label_id = 50 (way out of range)
        # this simulates the master having many more schemas than the replica
        payload = _build_effects_payload([
            ('add_schema', SCHEMA_NODE, 'Product'),
            ('create_node', [50], 0),
        ])

        try:
            self.conn.execute_command('GRAPH.EFFECT', graph_id, payload)
        except Exception:
            pass

        # server should have exited
        server_alive = True
        try:
            time.sleep(0.5)
            self.conn.ping()
        except Exception:
            server_alive = False

        self.env.assertFalse(server_alive)
