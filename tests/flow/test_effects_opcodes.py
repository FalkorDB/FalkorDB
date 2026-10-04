"""Does every opcode replicate at all, in both payload versions?

One class, run twice per test: once with the primary emitting v2 and once v3.
The assertions are identical because the question is -- an effect shipped, and
the replica agrees. What differs is the format that carried it, which
`assert_wire_version` pins so a passing run cannot be a run of the wrong one.
"""

from common import *
from index_utils import *
from effects_common import _EffectsBase


class test01_Opcodes(_EffectsBase):
    GRAPH_ID = "effects_opcodes"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def _indices(self):
        create_node_range_index(self.master_graph, "L", "a", "b", "c", sync=True)
        create_edge_range_index(self.master_graph, "R", "a", "b", "c", sync=True)

    def test01_add_schema(self):
        def body(v):
            self.query_and_sync("CREATE (:New_Label_%d)" % v)
            window = self.assert_effect_emitted()
            self.assert_wire_version(window)
            self.assert_graph_eq()
        self.for_each_version(body)

    def test02_add_attribute(self):
        def body(v):
            self.query_and_sync("CREATE (:L {new_attr_%d: 1})" % v)
            window = self.assert_effect_emitted()
            self.assert_wire_version(window)
            self.assert_graph_eq()
        self.for_each_version(body)

    def test03_create_node(self):
        def body(v):
            self.query_and_sync("CREATE (:L {a: 1, b: 2, c: 3})")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test04_create_edge(self):
        def body(v):
            self.query_and_sync("CREATE (:L {a: 1})-[:R {a: 1}]->(:L {a: 2})")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test05_update_node(self):
        def body(v):
            self.query_and_sync("CREATE (:L {a: 1})")
            self.assert_effect_emitted()
            self.query_and_sync("MATCH (n:L {a: 1}) SET n.b = 7")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test06_update_edge(self):
        def body(v):
            self.query_and_sync("CREATE (:L)-[:R {a: 1}]->(:L)")
            self.assert_effect_emitted()
            self.query_and_sync("MATCH ()-[e:R {a: 1}]->() SET e.b = 7")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test07_set_and_remove_labels(self):
        def body(v):
            self.query_and_sync("CREATE (:L {a: 99})")
            self.assert_effect_emitted()
            self.query_and_sync("MATCH (n:L {a: 99}) SET n:Extra")
            self.assert_wire_version(self.assert_effect_emitted())
            self.query_and_sync("MATCH (n:Extra) REMOVE n:Extra")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test08_delete_edge(self):
        def body(v):
            self.query_and_sync("CREATE (:L)-[:R {del: 1}]->(:L)")
            self.assert_effect_emitted()
            self.query_and_sync("MATCH ()-[e:R {del: 1}]->() DELETE e")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test09_delete_node(self):
        def body(v):
            self.query_and_sync("CREATE (:L {del: 2})")
            self.assert_effect_emitted()
            self.query_and_sync("MATCH (n:L {del: 2}) DELETE n")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)
