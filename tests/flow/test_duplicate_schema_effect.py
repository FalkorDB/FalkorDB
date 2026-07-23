import struct
from common import *

# Effects binary protocol constants (must match src/effects/effects.h)
EFFECTS_VERSION = 1
EFFECT_ADD_SCHEMA = 9

# Schema types (must match src/schema/schema.h)
SCHEMA_NODE = 0


def _build_add_schema_effect(schema_type, schema_name):
    payload = struct.pack("<B", EFFECTS_VERSION)  # version: uint8_t
    payload += struct.pack("<i", EFFECT_ADD_SCHEMA)  # EffectType
    payload += struct.pack("<i", schema_type)  # SchemaType

    name = schema_name.encode("utf-8") + b"\x00"
    payload += struct.pack("<Q", len(name))  # size_t on 64-bit
    payload += name
    return payload


class testDuplicateSchemaEffect():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test01_duplicate_node_schema_labels_via_effect(self):
        graph_id = "duplicate_schema_effect"
        self.conn.delete(graph_id, f"telemetry{{{graph_id}}}")
        g = Graph(self.conn, graph_id)

        g.query(
            "CREATE (:navTreeNode {nodeKey:'a'})-[:R]->(:site {nodeKey:'b'})"
        )
        g.query("CREATE INDEX FOR (n:navTreeNode) ON (n.nodeKey)")

        self.conn.execute_command(
            "GRAPH.EFFECT", graph_id,
            _build_add_schema_effect(SCHEMA_NODE, "navTreeNode")
        )
        self.conn.execute_command(
            "GRAPH.EFFECT", graph_id,
            _build_add_schema_effect(SCHEMA_NODE, "site")
        )

        dup_count = g.query(
            "CALL db.labels() YIELD label RETURN count(label) - count(DISTINCT label)"
        ).result_set
        # Expected behavior: duplicated schema additions should not create
        # duplicate labels in graph schema.
        self.env.assertEqual(dup_count, [[0]])
