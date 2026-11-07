from common import *

GRAPH_ID = "self_pointing_node"

class testSelfPointingNode(FlowTestsBase):
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env()
        cls.graph = cls.db.select_graph(GRAPH_ID)
        cls.populate_graph()
   
    @classmethod
    def populate_graph(cls):
        # Construct a graph with the form:
        # (v1)-[:e]->(v1)
        cls.graph.query("CREATE (n:L), (n)-[:e]->(n)")

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

        assert len(result_a.result_set) == 1
        n = result_a.result_set[0][0]
        assert n.id == 0

        assert result_b.result_set == result_a.result_set
        assert result_c.result_set == result_a.result_set
        assert result_d.result_set == result_a.result_set
        assert result_e.result_set == result_a.result_set
        assert result_f.result_set == result_a.result_set

        assert "Expand Into" in plan_a
        assert "Expand Into" in plan_b
        assert "Expand Into" in plan_c
        assert "Expand Into" in plan_d
        assert "Expand Into" in plan_e
        assert "Expand Into" in plan_f
