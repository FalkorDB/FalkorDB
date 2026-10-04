"""Does it hold at scale, and under operations nobody chose by hand?

Two different questions. The scale tests assert that a large buffer survives
being built, shipped and applied. The random-ops test asserts that a sequence
nobody designed leaves the two graphs agreeing -- it is the one test here that
can find a shape the hand-written cases do not contain.
"""

from common import *
from effects_common import _EffectsBase


class test01_Scale(_EffectsBase):
    GRAPH_ID = "effects_batch_scale"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def test01_large_create(self):
        def body(v):
            res = self.query_and_sync(
                "UNWIND range(0, 20000) AS i CREATE (:B {i: i})")
            self.env.assertEquals(res.nodes_created, 20001)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test02_large_update_then_delete(self):
        def body(v):
            self.query_and_sync("UNWIND range(0, 5000) AS i CREATE (:U {i: i})")
            self.assert_effect_emitted()
            self.query_and_sync("MATCH (u:U) SET u.j = u.i * 2")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
            self.query_and_sync("MATCH (u:U) DELETE u")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)


class test02_RandomOps(_EffectsBase):
    GRAPH_ID = "effects_batch_random"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def test01_random_graph_ops(self):
        from random_graph import (create_random_schema, create_random_graph,
                                  run_random_graph_ops, ALL_OPS)

        def body(v):
            nodes, edges = create_random_schema()
            create_random_graph(self.master_graph, nodes, edges)
            self.master.execute_command("WAIT", "1", "0")
            self.assert_graph_eq()

            run_random_graph_ops(self.master_graph, nodes, edges, ALL_OPS)
            self.master.execute_command("WAIT", "1", "0")
            self.assert_graph_eq()
        self.for_each_version(body)
