from common import *

nodes        =  []
GRAPH_ID     =  "shortest_path"

class testShortestPath(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        # Construct a graph with the form:
        # (a {v:1})-[:E]->(b {v:2})-[:E]->(c {v:3})-[:E]->(d {v:4}),
        # (a)-[:E]->(e {v:5})-[:E2]->(d)

        q = """
        CREATE (a:L {v:1})
        CREATE (b:L {v:2})
        CREATE (c:L {v:3})
        CREATE (d:L {v:4})
        CREATE (e:L {v:5})
        CREATE (a)-[:E]->(b)
        CREATE (b)-[:E]->(c)
        CREATE (c)-[:E]->(d)
        CREATE (a)-[:E]->(e)
        CREATE (e)-[:E2]->(d)

        RETURN a, b, c, d, e
        """

        res = self.graph.query(q).result_set

        global nodes
        nodes.append(res[0][0])
        nodes.append(res[0][1])
        nodes.append(res[0][2])
        nodes.append(res[0][3])
        nodes.append(res[0][4])

    def test01_invalid_shortest_paths(self):
        query = """MATCH (a {v: 1}), (b {v: 4}), p = shortestPath((a)-[*]->(b)) RETURN p"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("FalkorDB currently only supports shortestPaths in WITH or RETURN clauses", str(e))

        query = """MATCH (a {v: 1}), (b {v: 4}) RETURN shortestPath((a)-[*2..]->(b))"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("shortestPath does not support a minimal length different from 0 or 1", str(e))

        query = """MATCH (a {v: 1}), (b {v: 4}) RETURN shortestPath((a)-[]->()-[*]->(b))"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("shortestPath requires a path containing a single relationship", str(e))

        query = """MATCH (a {v: 1}), (b {v: 4}) RETURN shortestPath((a)-[* {weight: 4}]->(b))"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("filters on relationships in shortestPath", str(e))

        # Try iterating over an invalid relationship type
        query = """MATCH (a {v: 1}), (b {v: 4}) RETURN shortestPath((a)-[:FAKE*]->(b))"""
        actual_result = self.graph.query(query)
        # No results should be found
        expected_result = [[None]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test02_simple_shortest_path(self):
        query = """MATCH (a {v: 1}), (d {v: 4})
                   WITH shortestPath((a)-[*]->(d)) AS p
                   UNWIND nodes(p) AS n
                   RETURN n.v"""
        actual_result = self.graph.query(query)

        # The shorter 2-hop traversal should be found
        # (a)-[]->(e)-[]->(d)
        expected_result = [[1], [5], [4]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Verify that a right-to-left traversal produces the same results
        query = """MATCH (a {v: 1}), (b {v: 4})
                   WITH shortestPath((b)<-[*]-(a)) AS p
                   UNWIND nodes(p) AS n
                   RETURN n.v"""
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test03_shortest_path_multiple_results(self):
        # Traverse from all source nodes to the destination node
        query = """MATCH (a), (b {v: 4}) WITH a, shortestPath((a)-[*]->(b)) AS p RETURN a, nodes(p) ORDER BY a"""
        actual_result = self.graph.query(query)
        expected_result = [[nodes[0], [nodes[0], nodes[4], nodes[3]]],
                           [nodes[1], [nodes[1], nodes[2], nodes[3]]],
                           [nodes[2], [nodes[2], nodes[3]]],
                           [nodes[3], None],
                           [nodes[4], [nodes[4], nodes[3]]]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test04_max_hops(self):
        # Traverse from all source nodes to the destination node if there is a single-hop path
        query = """MATCH (a), (b {v: 4})
                   WITH a, shortestPath((a)-[*..1]->(b)) AS p
                   RETURN a, nodes(p) ORDER BY a"""
        actual_result = self.graph.query(query)
        expected_result = [[nodes[0], None],
                           [nodes[1], None],
                           [nodes[2], [nodes[2], nodes[3]]],
                           [nodes[3], None],
                           [nodes[4], [nodes[4], nodes[3]]]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test05_min_hops(self):
        # Traverse from all source nodes to the destination node with a minimum hop value of 0.
        # This will produce the same results as the above query with the exception of
        # the src == dest case, in which case that node is returned.
        query = """MATCH (a), (b {v: 4}) WITH a, shortestPath((a)-[*0..]->(b)) AS p RETURN a, nodes(p) ORDER BY a"""
        actual_result = self.graph.query(query)
        expected_result = [[nodes[0], [nodes[0], nodes[4], nodes[3]]],
                           [nodes[1], [nodes[1], nodes[2], nodes[3]]],
                           [nodes[2], [nodes[2], nodes[3]]],
                           [nodes[3], [nodes[3]]],
                           [nodes[4], [nodes[4], nodes[3]]]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test06_restricted_reltypes(self):
        # Traverse both relationship types
        query = """MATCH (a {v: 1}), (b {v: 4}) WITH shortestPath((a)-[:E|:E2*]->(b)) AS p UNWIND nodes(p) AS n RETURN n.v"""
        actual_result = self.graph.query(query)
        # The shorter 2-hop traversal should be found
        expected_result = [[1], [5], [4]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Only traverse edges of type E
        query = """MATCH (a {v: 1}), (b {v: 4}) WITH shortestPath((a)-[:E*]->(b)) AS p UNWIND nodes(p) AS n RETURN n.v"""
        actual_result = self.graph.query(query)
        # The longer traversal will be found
        expected_result = [[1], [2], [3], [4]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test07_shortestPath_in_filter(self):
        # Traverse both relationship types
        query = """MATCH (a {v: 1}), (b {v: 4}) WHERE length(shortestPath((a)-[:E|:E2*]->(b))) > 0 RETURN a.v, b.v"""
        actual_result = self.graph.query(query)
        # The shorter 2-hop traversal should be found
        expected_result = [[1, 4]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Traverse both relationship types
        query = """MATCH (a {v: 1}), (b {v: 4}) WHERE length(shortestPath((a)-[:E|:E2*]->())) > 0 RETURN a.v, b.v"""
        try:
            self.graph.query(query)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("A shortestPath requires bound nodes", str(e))

        # shortestPath requires bound nodes
        queries = [
            """MATCH (a {v: 1}), (b {v: 4}) WHERE length(shortestPath((a)-[:E|:E2*]->())) > 0 RETURN a.v, b.v""",
            """MATCH (a {v: 1}), (b {v: 4}) WHERE length(shortestPath(()-[:E|:E2*]->(b))) > 0 RETURN a.v, b.v""",
            """MATCH (a {v: 1}), (b {v: 4}) WHERE length(shortestPath(()-[:E|:E2*]->())) > 0 RETURN a.v, b.v""",
            """MATCH (a {v: 1}), (b {v: 4}) WHERE length(shortestPath((z)-[:E|:E2*]->(x))) > 0 RETURN a.v, b.v""",
            """MATCH (a {v: 1}), (b {v: 4}) WHERE length(shortestPath(({v:1})-[:E|:E2*]->())) > 0 RETURN a.v, b.v"""
        ]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertIn("A shortestPath requires bound nodes", str(e))

