from common import *


class testBFS(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("proc_bfs")
        self.populate_graph()

    def populate_graph(self):
        # Construct a graph with the form:
        # (a)-[:E1]->(b:B)-[:E1]->(c), (b)-[:E2]->(d)-[:E1]->(e)

        q = """
        CREATE (a {v: 'a'})
        CREATE (b {v: 'b'})
        CREATE (c {v: 'c'})
        CREATE (d {v: 'd'})
        CREATE (e {v: 'e'})
        CREATE (a)-[ab:E1 {v:'b'}]->(b)
        CREATE (b)-[bc:E1 {v:'c'}]->(c)
        CREATE (b)-[bd:E2 {v:'d'}]->(d)
        CREATE (d)-[de:E1 {v:'e'}]->(e)
        RETURN a, b, c, d, e, ab, bc, bd, de"""

        res = self.graph.query(q).result_set

    # Verify that the contents of two arrays are equal without respect to order.
    def compare_unsorted_arrays(self, a, b):
        self.env.assertEquals(len(a), len(b))
        for elem in a:
            # Each element in a should appear in b exactly once.
            self.env.assertEquals(b.count(elem), 1)

    def test_invalid_invocation(self):
        invalid_queries = [

            # array relationship type
            """MATCH (a {v: 'a'}) CALL algo.bfs(a, -1, {'E1'}) YIELD nodes""",         

            # unexpected extra parameters
            """MATCH (a {v: 'a'}) CALL algo.bfs(a, -1, NULL, 'extra') YIELD nodes""",         

            # invalid configuration type
            """MATCH (a {v: 'a'}) CALL algo.bfs('invalid') YIELD nodes""",

            # integer values in relationshipType
            # FIXME should throw error but does not:
            # """MATCH (a {v: 'a'}) CALL algo.bfs(a, -1, 0) YIELD nodes""",

            # non-existent yield field
            """MATCH (a {v: 'a'}) CALL algo.bfs(null) YIELD nodes, invalidField""",

            # max depth is expected to an integer
            # FIXME should throw error but does not:
            # """MATCH (a {v: 'a'}) CALL algo.bfs(a, 'hello', NULL)""",
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except:
                pass
    # Test BFS from a single source without specifying a relationship type.
    def test01_bfs_single_source_all_reltypes(self):
        # Test BFS algorithm for node collection.
        # The results array must be sorted, since the order is non-deterministic (due to creations occurring in any order).
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, NULL) YIELD nodes
                   UNWIND nodes AS n
                   WITH n.v AS v
                   ORDER BY n.v
                   RETURN COLLECT(v)"""

        actual_result = self.graph.query(query)
        expected_result = [[['b', 'c', 'd', 'e']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, NULL) YIELD nodes
                   RETURN nodes"""
        self.compare_unsorted_arrays(actual_result.result_set[0][0], expected_result[0][0])

        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, NULL) YIELD nodes
                   UNWIND nodes AS n
                   WITH a, n.v AS v
                   ORDER BY n.v
                   RETURN a.v, COLLECT(v)"""

        actual_result = self.graph.query(query)
        expected_result = [['a', ['b', 'c', 'd', 'e']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test BFS algorithm for node and edge collection.
        # Parity between nodes and edges can be validated by testing the properties of each for equality,
        # as edges have the same property as their destination node.
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, NULL) YIELD nodes, edges
                   RETURN [n IN nodes | n.v], [e IN edges | e.v]"""

        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], actual_result.result_set[0][1])

    # Test BFS from a single source traversing a single relationship type.
    def test02_bfs_single_source_restricted_reltype(self):
        # Test BFS algorithm for node collection.
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, 'E1') YIELD nodes
                   RETURN [n IN nodes | n.v]"""

        actual_result = self.graph.query(query)
        expected_result = ['b', 'c']
        self.compare_unsorted_arrays(actual_result.result_set[0][0], expected_result)

        # Test BFS algorithm for node and edge collection.
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, 'E1') YIELD nodes, edges
                   RETURN [n IN nodes | n.v], [e IN edges | e.v]"""

        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], actual_result.result_set[0][1])

    # Test BFS from all sources traversing a single relationship type, ignoring 0-hop paths.
    def test03_bfs_all_sources_restricted_reltype(self):
        # We only expect to see 'd' as a source node, as it is connected as a destination by an 'E2' edge.
        query = """MATCH (a)
                   CALL algo.BFS(a, -1, 'E1') YIELD nodes
                   RETURN a.v, [n IN nodes | n.v]
                   ORDER BY a.v"""

        actual_result = self.graph.query(query)
        expected_result = [['a', ['b', 'c']],
                           ['b', ['c']],
                           ['d', ['e']]]
        for idx, row in enumerate(actual_result.result_set):
            self.env.assertEquals(row[0], expected_result[idx][0])
            self.compare_unsorted_arrays(row[1], expected_result[idx][1])

        # Test BFS path-tracking algorithm
        query = """MATCH (a)
                   CALL algo.BFS(a, -1, 'E1') YIELD nodes, edges
                   RETURN a.v, [n IN nodes | n.v], [e IN edges | e.v]
                   ORDER BY a.v"""

        actual_result = self.graph.query(query)
        expected_result = [['a', ['b', 'c'], ['b', 'c']],
                           ['b', ['c'], ['c']],
                           ['d', ['e'], ['e']]]
        for idx, row in enumerate(actual_result.result_set):
            self.env.assertEquals(row[0], expected_result[idx][0])
            self.compare_unsorted_arrays(row[1], expected_result[idx][1])
            self.compare_unsorted_arrays(row[2], expected_result[idx][2])

    # Test BFS from a single source with a maximum depth.
    def test04_bfs_single_source_max_depth(self):
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, 1, NULL) YIELD nodes
                   RETURN [n IN nodes | n.v]"""

        actual_result = self.graph.query(query)
        expected_result = [[['b']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (a {v: 'a'}) CALL algo.BFS(a, 1, NULL) YIELD nodes, edges RETURN [n IN nodes | n.v], [e IN edges | e.v]"""
        actual_result = self.graph.query(query)
        expected_result = [[['b'], ['b']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Test BFS from all sources with a maximum depeth.
    def test05_bfs_all_sources_max_depth(self):
        query = """MATCH (a)
                   CALL algo.BFS(a, 1, NULL) YIELD nodes
                   RETURN a.v, [n IN nodes | n.v]
                   ORDER BY a.v"""

        actual_result = self.graph.query(query)
        expected_result = [['a', ['b']],
                           ['b', ['c', 'd']],
                           ['d', ['e']]]
        for idx, row in enumerate(actual_result.result_set):
            self.env.assertEquals(row[0], expected_result[idx][0])
            self.compare_unsorted_arrays(row[1], expected_result[idx][1])

        query = """MATCH (a)
                   CALL algo.BFS(a, 1, NULL) YIELD nodes, edges
                   RETURN a.v, [n IN nodes | n.v], [e IN edges | e.v]
                   ORDER BY a.v"""

        actual_result = self.graph.query(query)
        expected_result = [['a', ['b'], ['b']],
                           ['b', ['c', 'd'], ['c', 'd']],
                           ['d', ['e'], ['e']]]
        for idx, row in enumerate(actual_result.result_set):
            self.env.assertEquals(row[0], expected_result[idx][0])
            self.compare_unsorted_arrays(row[1], expected_result[idx][1])
            self.compare_unsorted_arrays(row[2], expected_result[idx][2])

    def test06_bfs_no_results(self):
        empty_result_set = []
        # Missing relationship type
        query = """MATCH (a)
                   CALL algo.BFS(a, -1, 'NONE_EXISTING_RELATION') YIELD nodes"""

        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, empty_result_set)

        # Leaf node
        query = """MATCH (leaf {v:'e'})
                   CALL algo.BFS(leaf, -1, NULL) YIELD nodes"""

        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, empty_result_set)

        # NULL node
        query = """OPTIONAL MATCH (n:NONE_EXISTING_LABEL)
                   CALL algo.BFS(n, -1, NULL) YIELD nodes"""

        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, empty_result_set)

    # test a query which calls BFS multiple times in the same scope
    def test07_multiple_bfs_calls(self):
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, 1, NULL) YIELD nodes as n1
                   MATCH (b {v: 'd'})
                   CALL algo.BFS(b, 1, NULL) YIELD nodes as n2
                   RETURN [n IN n1 | n.v] AS x, [n IN n2 | n.v] AS y
                   ORDER BY x"""

        actual_result = self.graph.query(query)
        expected_result = [[['b'], ['e']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

