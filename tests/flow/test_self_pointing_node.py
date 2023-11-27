from common import *

GRAPH_ID = "self_pointing_node"

class testSelfPointingNode(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()
   
    def populate_graph(self):
        # Construct a graph with the form:
        # (v1)-[:e]->(v1)
        self.graph.query("CREATE (n:L), (n)-[:e]->(n)")

    # Test patterns that traverse 1 edge.
    def test_self_pointing_node(self):
        # Conditional traversal with label
        query = """MATCH (a)-[:e]->(a) RETURN a"""
        result_a = self.graph.query(query)
        plan_a = str(self.graph.explain(query))

        query = """MATCH (a:L)-[:e]->(a) RETURN a"""
        result_b = self.graph.query(query)
        plan_b = str(self.graph.explain(query))

        query = """MATCH (a)-[:e]->(a:L) RETURN a"""
        result_c = self.graph.query(query)
        plan_c = str(self.graph.explain(query))

        query = """MATCH (a)-[]->(a) RETURN a"""
        result_d = self.graph.query(query)
        plan_d = str(self.graph.explain(query))

        query = """MATCH (a:L)-[]->(a) RETURN a"""
        result_e = self.graph.query(query)
        plan_e = str(self.graph.explain(query))

        query = """MATCH (a)-[]->(a:L) RETURN a"""
        result_f = self.graph.query(query)
        plan_f = str(self.graph.explain(query))

        self.env.assertEquals(len(result_a.result_set), 1)
        n = result_a.result_set[0][0]
        self.env.assertEquals(n.id, 0)

        self.env.assertEquals(result_b.result_set, result_a.result_set)
        self.env.assertEquals(result_c.result_set, result_a.result_set)
        self.env.assertEquals(result_d.result_set, result_a.result_set)
        self.env.assertEquals(result_e.result_set, result_a.result_set)
        self.env.assertEquals(result_f.result_set, result_a.result_set)

        self.env.assertIn("Expand Into", plan_a)
        self.env.assertIn("Expand Into", plan_b)
        self.env.assertIn("Expand Into", plan_c)
        self.env.assertIn("Expand Into", plan_d)
        self.env.assertIn("Expand Into", plan_e)
        self.env.assertIn("Expand Into", plan_f)
