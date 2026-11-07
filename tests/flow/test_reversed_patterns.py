from common import *

GRAPH_ID = "reversed_patterns"

class testReversedPatterns(FlowTestsBase):
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env()
        cls.redis_con = cls.env.getConnection()
        cls.graph = cls.db.select_graph(GRAPH_ID)
        cls.populate_graph()

    @classmethod
    def populate_graph(cls):
        if GRAPH_ID not in cls.db.list_graphs():
            # Create entities
            cls.graph.query("CREATE (:L {name:'SRC'})-[:E]->(:L {name:'DEST'})")

    # Verify that edges are not modified after entity deletion
    def test01_reversed_pattern(self):
        leftToRight = """MATCH (a:L)-[b]->(c:L) RETURN a, TYPE(b), c"""
        rightToLeft = """MATCH (c:L)<-[b]-(a:L) RETURN a, TYPE(b), c"""
        leftToRightResult = self.graph.query(leftToRight)
        rightToLeftResult = self.graph.query(rightToLeft)
        assert leftToRightResult.result_set == rightToLeftResult.result_set
