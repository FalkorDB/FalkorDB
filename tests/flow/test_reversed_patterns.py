from common import *

GRAPH_ID = "reversed_patterns"

class testReversedPatterns(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        if GRAPH_ID not in self.db.list_graphs():
            # Create entities
            self.graph.query("CREATE (:L {name:'SRC'})-[:E]->(:L {name:'DEST'})")

    # Verify that edges are not modified after entity deletion
    def test01_reversed_pattern(self):
        leftToRight = """MATCH (a:L)-[b]->(c:L) RETURN a, TYPE(b), c"""
        rightToLeft = """MATCH (c:L)<-[b]-(a:L) RETURN a, TYPE(b), c"""
        leftToRightResult = self.graph.query(leftToRight)
        rightToLeftResult = self.graph.query(rightToLeft)
        self.env.assertEquals(leftToRightResult.result_set, rightToLeftResult.result_set)
