"""What must one buffer survive -- merges, and statements touching many entities.

A MERGE is the interesting shape because one statement both creates and updates,
so a single buffer carries records of more than one opcode and the replica has to
apply them in an order that leaves the same graph.
"""

import random
from common import *
from effects_common import _EffectsBase


class test01_Merge(_EffectsBase):
    GRAPH_ID = "effects_shapes_merge"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def test01_merge_node_create_then_match(self):
        def body(v):
            res = self.query_and_sync(
                "MERGE (n:A {v:'red'}) ON MATCH SET n.v='green' "
                "ON CREATE SET n.v='blue'")
            self.env.assertEquals(res.nodes_created, 1)
            self.env.assertEquals(res.properties_set, 2)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()

            # second time the MERGE matches rather than creates
            res = self.query_and_sync(
                "MERGE (n:A {v:'blue'}) ON MATCH SET n.v='green' "
                "ON CREATE SET n.v='red'")
            self.env.assertEquals(res.properties_set, 1)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test02_merge_edge_create_then_match(self):
        def body(v):
            res = self.query_and_sync(
                "MERGE (n:A {v:'red'}) MERGE (n)-[e:R {v:'red'}]->(n) "
                "ON MATCH SET e.v='green' ON CREATE SET e.v='blue'")
            self.env.assertEquals(res.relationships_created, 1)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()

            res = self.query_and_sync(
                "MERGE (n:A {v:'red'}) MERGE (n)-[e:R {v:'blue'}]->(n) "
                "ON MATCH SET e.v='green' ON CREATE SET e.v='red'")
            self.env.assertEquals(res.properties_set, 1)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)


class test02_ManyEntities(_EffectsBase):
    GRAPH_ID = "effects_shapes_many"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def test01_multiple_nodes_create_and_delete(self):
        def body(v):
            lbls  = ["L0", "L1", "L2", "L3"]
            nodes = ["(:%s)" % random.choice(lbls) for _ in range(2048)]
            res = self.query_and_sync("CREATE " + ",".join(nodes))
            self.env.assertEquals(res.nodes_created, 2048)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()

            res = self.query_and_sync("MATCH (n) DELETE n")
            self.env.assertEquals(res.nodes_deleted, 2048)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()

            q = "MATCH (n) RETURN count(n)"
            self.env.assertEquals(
                self.master_graph.query(q).result_set[0][0], 0)
            self.env.assertEquals(
                self.replica_graph.ro_query(q).result_set[0][0], 0)
        self.for_each_version(body)

    def test02_multiple_edges_create_and_delete(self):
        def body(v):
            self.query_and_sync(
                "UNWIND range(0, 511) AS i CREATE (:S {i: i})")
            self.assert_effect_emitted()
            res = self.query_and_sync(
                "MATCH (a:S), (b:S) WHERE a.i = b.i - 1 CREATE (a)-[:E]->(b)")
            self.env.assertTrue(res.relationships_created > 0)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()

            res = self.query_and_sync("MATCH ()-[e:E]->() DELETE e")
            self.env.assertTrue(res.relationships_deleted > 0)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test03_mixed_entities_in_one_statement(self):
        def body(v):
            res = self.query_and_sync(
                "CREATE (a:M {x: 1})-[:K {w: 2}]->(b:M {x: 3}), (c:M {x: 4})")
            self.env.assertEquals(res.nodes_created, 3)
            self.env.assertEquals(res.relationships_created, 1)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)
