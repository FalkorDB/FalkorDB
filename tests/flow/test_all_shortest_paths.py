from common import *

class testAllShortestPaths():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("all_shortest_paths")
        self.cyclic_graph = self.db.select_graph("all_shortest_paths_cyclic")
        self.populate_graph()
        self.populate_cyclic_graph()

    def populate_graph(self):
        # Construct a graph with the form:
        # (v1)-[:E]->(v2)-[:E]->(v3)-[:E]->(v4)
        # (v1)-[:E]->(v5)-[:E2]->(v4)
        # (v2)-[:E2]->(v4)

        self.v1 = Node(alias="v1", labels="L", properties={"v": 1})
        self.v2 = Node(alias="v2", labels="L", properties={"v": 2})
        self.v3 = Node(alias="v3", labels="L", properties={"v": 3})
        self.v4 = Node(alias="v4", labels="L", properties={"v": 4})
        self.v5 = Node(alias="v5", labels="L", properties={"v": 5})

        e0 = Edge(self.v1, "E", self.v2, properties={"weight": 1})
        e1 = Edge(self.v2, "E", self.v3, properties={"weight": 1})
        e2 = Edge(self.v3, "E", self.v4, properties={"weight": 1})
        e3 = Edge(self.v1, "E", self.v5, properties={"weight": 1})
        e4 = Edge(self.v5, "E2", self.v4, properties={"weight": 1})
        e5 = Edge(self.v2, "E2", self.v4, properties={"weight": 2})

        query = f"""CREATE {self.v1}, {self.v2}, {self.v3}, {self.v4}, {self.v5},
        {e0}, {e1}, {e2}, {e3}, {e4}, {e5}"""
        self.graph.query(query)

    def populate_cyclic_graph(self):
        # Construct a graph with the form:
        # (v1)-[:E]->(v2)-[:E]->(v3)-[:E]->(v4)
        # (v1)-[:E]->(v5)-[:E2]->(v4)
        # (v2)-[:E2]->(v4)
        # (v3)-[:E2]->(v1)
        # (v4)-[:E2]->(v1)

        self.cyclic_v1 = Node(alias="v1", labels="L", properties={"v": 1})
        self.cyclic_v2 = Node(alias="v2", labels="L", properties={"v": 2})
        self.cyclic_v3 = Node(alias="v3", labels="L", properties={"v": 3})
        self.cyclic_v4 = Node(alias="v4", labels="L", properties={"v": 4})
        self.cyclic_v5 = Node(alias="v5", labels="L", properties={"v": 5})

        e0 = Edge(self.cyclic_v1, "E",  self.cyclic_v2, properties={"weight": 1})
        e1 = Edge(self.cyclic_v2, "E",  self.cyclic_v3, properties={"weight": 1})
        e2 = Edge(self.cyclic_v3, "E",  self.cyclic_v4, properties={"weight": 1})
        e3 = Edge(self.cyclic_v1, "E",  self.cyclic_v5, properties={"weight": 1})
        e4 = Edge(self.cyclic_v5, "E2", self.cyclic_v4, properties={"weight": 1})
        e5 = Edge(self.cyclic_v2, "E2", self.cyclic_v4, properties={"weight": 2})
        e6 = Edge(self.cyclic_v3, "E2", self.cyclic_v1, properties={"weight": 2})
        e7 = Edge(self.cyclic_v4, "E2", self.cyclic_v1, properties={"weight": 2})

        query = f"""CREATE {self.cyclic_v1}, {self.cyclic_v2}, {self.cyclic_v3}, {self.cyclic_v4}, {self.cyclic_v5},
        {e0}, {e1}, {e2}, {e3}, {e4}, {e5}, {e6}, {e7}"""
        self.cyclic_graph.query(query)

    def test01_invalid_shortest_paths(self):
        # running against following graph
        #
        # (v1)-[:E]->(v2)-[:E]->(v3)-[:E]->(v4)
        # (v1)-[:E]->(v5)-[:E2]->(v4)
        # (v2)-[:E2]->(v4)

        # Test unbound variables
        query = """MATCH (v1 {v: 1}), (v4 {v: 4}), p = allShortestPaths((v1)-[*]->(v4))
                   RETURN p"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("Source and destination must already be resolved to call allShortestPaths", str(e))

        # Test non-node endpoint
        query = """MATCH (v1 {v: 1}) WITH v1, 'stringval' AS v4
                   MATCH p = allShortestPaths((v1)-[*]->(v4))
                   RETURN p"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("encountered unexpected type in Record; expected Node", str(e))

        # Test NULL endpoint
        query = """MATCH (v1 {v: 1}) OPTIONAL MATCH (v9 {v: v9}) WITH v1, v9 MATCH p = allShortestPaths((v1)-[*]->(v9)) RETURN p"""
        actual_result = self.graph.query(query)
        expected_result = []
        self.env.assertEqual(actual_result.result_set, expected_result)

        query = """MATCH (v1 {v: 1}), (v4 {v: 4})
                   WITH v1, v4
                   MATCH p = allShortestPaths((v1)-[*3..]->(v4))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""

        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("allShortestPaths(...) does not support a minimal length different from 1", str(e))

        # Test unsupported in return
        query = """MATCH (a),(b)
                   RETURN allShortestPaths((a)-[*]->(b))"""

        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("FalkorDB support allShortestPaths only in match clauses", str(e))
        
        # Test unsupported in where
        query = """MATCH (a),(b)
                   WHERE size(allShortestPaths((a)-[*]->(b))) > 0
                   RETURN a, b"""

        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("FalkorDB support allShortestPaths only in match clauses", str(e))

    def test02_all_shortest_paths(self):
        # running against following graph
        #
        # (v1)-[:E]->(v2)-[:E]->(v3)-[:E]->(v4)
        # (v1)-[:E]->(v5)-[:E2]->(v4)
        # (v2)-[:E2]->(v4)

        query = """MATCH (v1 {v: 1}), (v4 {v: 4})
                   WITH v1, v4
                   MATCH p = allShortestPaths((v1)-[*]->(v4))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""

        actual_result = self.graph.query(query)
        # The 2 2-hop traversals should be found
        expected_result = [[[self.v1, self.v2, self.v4]],
                           [[self.v1, self.v5, self.v4]]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Verify that a right-to-left traversal produces the same results
        query = """MATCH (v1 {v: 1}), (v4 {v: 4})
                   WITH v1, v4
                   MATCH p = allShortestPaths((v4)<-[*]-(v1))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""

        actual_result = self.graph.query(query)
        expected_result = [[[self.v4, self.v2, self.v1]],
                           [[self.v4, self.v5, self.v1]]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        query = """MATCH (v1 {v: 1}), (v4 {v: 4})
                   WITH v1, v4
                   MATCH p = allShortestPaths((v1)-[*]-(v4))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""

        actual_result = self.graph.query(query)
        expected_result = [[[self.v1, self.v2, self.v4]],
                           [[self.v1, self.v5, self.v4]]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test03_all_shortest_multiple_traversals(self):
        # running against following graph
        #
        # (v1)-[:E]->(v2)-[:E]->(v3)-[:E]->(v4)
        # (v1)-[:E]->(v5)-[:E2]->(v4)
        # (v2)-[:E2]->(v4)

        query = """MATCH (v1 {v: 1}), (v4 {v: 4})
                   WITH v1, v4
                   MATCH p = allShortestPaths((v1)-[:E]->(:L)-[*..]->(v4))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""

        actual_result = self.graph.query(query)
        # The single 3-hop traversal should be found
        expected_result = [[[self.v1, self.v2, self.v4]], 
                           [[self.v1, self.v5, self.v4]]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Verify that a right-to-left traversal produces the same results
        query = """MATCH (v1 {v: 1}), (v4 {v: 4})
                   WITH v1, v4
                   MATCH p = allShortestPaths((v4)<-[*..]-(:L)<-[:E]-(v1))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""

        actual_result = self.graph.query(query)
        expected_result = [[[self.v4, self.v2, self.v1]], 
                           [[self.v4, self.v5, self.v1]]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test04_all_shortest_edge_filter(self):
        # running against following graph
        #
        # (v1)-[:E {weight:1}]->(v2)-[:E {weight:1}]->(v3)-[:E {weight:1}]->(v4)
        # (v1)-[:E {weight:1}]->(v5)-[:E2 {weight:1}]->(v4)
        # (v2)-[:E2 {weight:2}]->(v4)

        query = """MATCH (v1 {v: 1}), (v4 {v: 4})
                   WITH v1, v4
                   MATCH p = allShortestPaths((v1)-[* {weight: 1}]->(v4))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""
        actual_result = self.graph.query(query)
        # The 1 2-hop traversal with all edge weights of 1 should be found
        expected_result = [[[self.v1, self.v5, self.v4]]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Verify that a right-to-left traversal produces the same results
        query = """MATCH (v1 {v: 1}), (v4 {v: 4})
                   WITH v1, v4
                   MATCH p = allShortestPaths((v4)<-[* {weight: 1}]-(v1))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""

        actual_result = self.graph.query(query)
        expected_result = [[[self.v4, self.v5, self.v1]]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test05_all_shortest_no_results(self):
        # running against following graph
        #
        # (v1)-[:E]->(v2)-[:E]->(v3)-[:E]->(v4)
        # (v1)-[:E]->(v5)-[:E2]->(v4)
        # (v2)-[:E2]->(v4)

        query = """MATCH (v5 {v: 5}), (v3 {v: 3})
                   WITH v5, v3
                   MATCH p = allShortestPaths((v5)-[*]->(v3))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""

        actual_result = self.graph.query(query)
        # No results should be found
        expected_result = []
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Verify that a right-to-left traversal produces the same results
        query = """MATCH (v5 {v: 5}), (v3 {v: 3})
                   WITH v5, v3
                   MATCH p = allShortestPaths((v3)<-[*]-(v5))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""

        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test06_all_shortest_cycle(self):
        # running against following graph
        #
        # (v1)-[:E]->(v2)-[:E]->(v3)-[:E]->(v4)
        # (v1)-[:E]->(v5)-[:E2]->(v4)
        # (v2)-[:E2]->(v4)
        # (v3)-[:E2]->(v1)
        # (v4)-[:E2]->(v1)

        query = """MATCH (v1 {v: 1})
                   WITH v1
                   MATCH p = allShortestPaths((v1)-[*]->(v1))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""

        actual_result = self.cyclic_graph.query(query)
        # 3 paths should be found
        expected_result = [[[self.cyclic_v1, self.cyclic_v3, self.cyclic_v2, self.cyclic_v1]],
                           [[self.cyclic_v1, self.cyclic_v4, self.cyclic_v2, self.cyclic_v1]],
                           [[self.cyclic_v1, self.cyclic_v4, self.cyclic_v5, self.cyclic_v1]]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Verify that a right-to-left traversal produces the same results
        query = """MATCH (v1 {v: 1})
                   WITH v1
                   MATCH p = allShortestPaths((v1)<-[*]-(v1))
                   RETURN nodes(p) AS nodes
                   ORDER BY nodes"""

        actual_result = self.cyclic_graph.query(query)
        self.env.assertEqual(actual_result.result_set, expected_result)
