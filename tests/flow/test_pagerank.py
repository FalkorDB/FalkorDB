from common import *

GRAPH_ID = "pagerank"


class testPagerankFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test_pagerank_no_label_no_relation(self):
        self.env.cmd('flushall')
        q = "CREATE (a:L0 {v:0})-[:R0]->(b:L1 {v:1})-[:R1]->(c:L2 {v:2})"
        self.graph.query(q)
        q = """CALL algo.pageRank(NULL, NULL) YIELD node, score RETURN node.v, score"""
        resultset = self.graph.query(q).result_set

        self.env.assertEqual(len(resultset), 3)
        self.env.assertEqual(resultset[0][0], 2)
        self.env.assertAlmostEqual(resultset[0][1], 0.609753012657166, 0.0001)
        self.env.assertEqual(resultset[1][0], 1)
        self.env.assertAlmostEqual(resultset[1][1], 0.286585807800293 , 0.0001)
        self.env.assertEqual(resultset[2][0], 0)
        self.env.assertAlmostEqual(resultset[2][1], 0.103661172091961, 0.0001)

    def test_pagerank_no_label(self):
        self.env.cmd('flushall')
        q = "CREATE (a:L0 {v:0})-[:R]->(b:L1 {v:1})-[:R0]->(c:L2 {v:2})"
        self.graph.query(q)
        q = """CALL algo.pageRank(NULL, 'R') YIELD node, score RETURN node.v, score"""
        resultset = self.graph.query(q).result_set

        self.env.assertEqual(len(resultset), 3)
        self.env.assertEqual(resultset[0][0], 1)
        self.env.assertAlmostEqual(resultset[0][1], 0.660703718662262, 0.0001)
        self.env.assertEqual(resultset[1][0], 0)
        self.env.assertAlmostEqual(resultset[1][1], 0.169648125767708, 0.0001)
        self.env.assertEqual(resultset[2][0], 2)
        self.env.assertAlmostEqual(resultset[2][1], 0.169648125767708, 0.0001)

    def test_pagerank_no_relation(self):
        self.env.cmd('flushall')
        q = "CREATE (a:L {v:0})-[:R]->(b:L {v:1})-[:R0]->(c:L2 {v:2})"
        self.graph.query(q)
        q = """CALL algo.pageRank('L', NULL) YIELD node, score RETURN node.v, score"""
        resultset = self.graph.query(q).result_set

        self.env.assertEqual(len(resultset), 2)
        self.env.assertEqual(resultset[0][0], 1)
        self.env.assertAlmostEqual(resultset[0][1], 0.777813196182251, 0.0001)
        self.env.assertEqual(resultset[1][0], 0)
        self.env.assertAlmostEqual(resultset[1][1], 0.22218681871891, 0.0001)

    def test_pagerank_no_connections(self):
        # Pagerank only considers connections where both ends
        # are of the same type, as such it might happen that pagerank
        # is executed against an empty matrix.
        queries = [
            # No data, no edges with both src and dest of type 'L'
            "CREATE (a {v:1})-[:R]->(b {v:2})",
            "CREATE (a:L {v:1})-[:R]->(b {v:2})",
            "CREATE (a {v:1})-[:R]->(b:L {v:2})",
            "CREATE (a:L {v:1})-[:R]->(b {v:2})-[:R]->(c:L {v:3})",
        ]

        for q in queries:
            self.env.cmd('flushall')
            self.graph.query(q)
            q = """CALL algo.pageRank('L', 'R') YIELD node, score RETURN node.v, score"""
            resultset = self.graph.query(q).result_set

            self.env.assertEqual(len(resultset), 0)

    def test_pagerank(self):
        # Pagerank considers only nodes of given label and
        # relation of given relationship type
        # multiple edges between two nodes are considered as a single connection.
        queries = [
            # Single Label, single connection.
            "CREATE (a:L {v:1})-[:R]->(b:L {v:2})",
            # Single Label, multi connection.
            "CREATE (a:L {v:1})-[:R]->(b:L {v:2}), (a)-[:R]->(b)",
            # Multi Label, single connection.
            "CREATE (a:L {v:1})-[:R]->(b:L {v:2}), (:X)-[:R]->(:X)",
            # Multi Label, multi connection.
            "CREATE (a:L {v:1})-[:R]->(b:L {v:2}), (a)-[:R]->(b), (:X)-[:R]->(:X)"
        ]

        for q in queries:
            self.env.cmd('flushall')
            self.graph.query(q)
            q = """CALL algo.pageRank('L', 'R') YIELD node, score RETURN node.v, score"""
            resultset = self.graph.query(q).result_set

            # 2) 1) 1) (integer) 2
            # 2) "0.777813196182251"
            # 2) 1) (integer) 1
            # 2) "0.22218681871891"
            self.env.assertEqual(len(resultset), 2)
            self.env.assertEqual(resultset[0][0], 2)
            self.env.assertAlmostEqual(resultset[0][1], 0.777813196182251, 0.0001)
            self.env.assertEqual(resultset[1][0], 1)
            self.env.assertAlmostEqual(resultset[1][1], 0.22218681871891, 0.0001)
