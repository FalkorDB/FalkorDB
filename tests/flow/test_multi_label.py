from common import *
from index_utils import *

graph = None
GRAPH_ID = "multi_label"


class testMultiLabel():
    def __init__(self):
        global graph
        self.env = Env(decodeResponses=True)
        self.redis_con = self.env.getConnection()
        graph = Graph(self.redis_con, GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        # Construct a graph with the form:
        # (v1:L0:L1)-[:E]->(v2:L1)-[:E]->(v3:L1:L2)
        q = "CREATE (v1:L0:L1 {v: 1})-[:E]->(v2:L1 {v: 2})-[:E]->(v3:L1:L2 {v: 3})"
        graph.query(q)

    # Validate basic multi-label scans.
    def test01_multilabel_scan(self):
        # Issue a query that matches the single (:L0) node.
        query = "MATCH (a:L0) RETURN LABELS(a)"
        query_result = graph.query(query)
        expected_result = [[['L0','L1']]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Issue a query that matches the single (:L0:L1) node.
        query = "MATCH (a:L0:L1) RETURN LABELS(a)"
        query_result = graph.query(query)
        expected_result = [[['L0','L1']]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Issue a query that matches the single (:L1:L0) node.
        query = "MATCH (a:L1:L0) RETURN LABELS(a)"
        query_result = graph.query(query)
        expected_result = [[['L0','L1']]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Issue a query that matches all 3 nodes with the label :L1.
        query = "MATCH (a:L1) RETURN LABELS(a) ORDER BY LABELS(a)"
        query_result = graph.query(query)
        expected_result = [[['L0','L1']], [['L1']], [['L1','L2']]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Issue a query that matches no nodes, as the (:L0:L2) label disjunction is not present.
        query = "MATCH (a:L0:L2) RETURN LABELS(a)"
        query_result = graph.query(query)
        expected_result = []
        self.env.assertEquals(query_result.result_set, expected_result)

    # Validate basic multi-label traversals.
    def test02_multilabel_traversal(self):
        # (v1:L0:L1)-[:E]->(v2:L1)-[:E]->(v3:L1:L2)
        queries = [
            "MATCH (a:L0:L1)-[]->(b) RETURN LABELS(a), LABELS(b)",
            "MATCH (a:L0:L1)-[:E]->(b) RETURN LABELS(a), LABELS(b)",
            "MATCH (a:L0:L1)-[:E]->(b:L1) RETURN LABELS(a), LABELS(b)",
            "MATCH (b:L1) WITH (b) MATCH (a:L0:L1)-[:E]->(b) RETURN LABELS(a), LABELS(b)",
            "MATCH (b:L1) WITH (b) MATCH (a:L0)-[:E]->(b) RETURN LABELS(a), LABELS(b)",
        ]

        expected_result = [[['L0','L1'], ['L1']]]

        for q in queries:
            query_result = graph.query(q)
            self.env.assertEquals(query_result.result_set, expected_result)

    # Validate that the graph properly handles label counts greater than its default.
    def test03_large_label_count(self):
        # Introduce a node with enough labels to force graph resizes.
        labels = ['L' + str(x) for x in range(10, 28)]
        query = "CREATE (n :" + ':'.join(labels) + ") RETURN LABELS(n)"
        query_result = graph.query(query)
        expected_result = [[labels]]
        self.env.assertEquals(query_result.result_set, expected_result)

    def test04_label_scan_optimization(self):
        # create graph with 10 A nodes, 100 B nodes and 1000 C nodes
        query = "UNWIND range(0, 10) AS x CREATE (:A)"
        graph.query(query)

        query = "UNWIND range(0, 100) AS x CREATE (:B)"
        graph.query(query)

        query = "UNWIND range(0, 1000) AS x CREATE (:C)"
        graph.query(query)

        labels = ['A', 'B', 'C']
        queries = [
            "MERGE (n:{ls}) RETURN n",
            "MATCH (n:{ls}) RETURN n",
            "MATCH (n:{ls})-[e:R]->(m) RETURN n",
            "MATCH (n:{ls})<-[e:R]-(m) RETURN n",
            "MATCH (n:{ls})-[e:R]-(m) RETURN n",
            "MATCH (n:{ls}) WHERE n.v = 1 RETURN n",
            "MATCH (n:{ls})-[e:R]->(m) WHERE n.v = 1 RETURN n",
            "MATCH (n:{ls})<-[e:R]-(m) WHERE n.v = 1 RETURN n",
            "MATCH (n:{ls})-[e:R]-(m) WHERE n.v = 1 RETURN n",
            "MATCH (a) WITH a AS a MATCH (n:{ls}) RETURN n",
            "CREATE (a) WITH a AS a MATCH (n:{ls}) RETURN n",
            "MERGE (a) WITH a AS a MATCH (n:{ls}) RETURN n"
            ]

        import itertools
        permutations = list(itertools.permutations(labels))
        for permutation in permutations:
            for query in queries:
                query = query.format(ls=':'.join(permutation))
                plan = graph.execution_plan(query)
                self.env.assertContains("Node By Label Scan | (n:A)", plan)

    # Validate behavior of index scans on multi-labeled nodes
    def test05_index_scan(self):
        query_result = create_node_range_index(graph, 'L1', 'v', sync=True)
        self.env.assertEquals(query_result.indices_created, 1)

        # Query the explicitly created index
        query = """MATCH (a:L1) WHERE a.v > 0 RETURN a.v ORDER BY a.v"""
        plan = graph.execution_plan(query)
        self.env.assertContains("Index Scan", plan)
        query_result = graph.query(query)
        expected_result = [[1],
                           [2],
                           [3]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Query the explicitly created index on a multi-labeled node
        queries = [
            "MATCH (a:L1:L2) WHERE a.v > 0 RETURN a.v ORDER BY a.v",
            "MATCH (a:L2:L1) WHERE a.v > 0 RETURN a.v ORDER BY a.v"
        ]

        for q in queries:
            plan = graph.execution_plan(q)
            self.env.assertContains("Index Scan", plan)
            query_result = graph.query(q)
            expected_result = [[3]]
            self.env.assertEquals(query_result.result_set, expected_result)

    # Validate the creation of multi-labeled nodes with the MERGE clause
    def test06_multi_label_merge(self):
        query = """MERGE (a:L2:L3 {v: 4}) RETURN labels(a)"""
        query_result = graph.query(query)
        expected_result = [[["L2", "L3"]]]
        self.env.assertEquals(query_result.nodes_created, 1)
        self.env.assertEquals(query_result.labels_added, 1)
        self.env.assertEquals(query_result.result_set, expected_result)

        # Repetition of the query should not create a new node
        query_result = graph.query(query)
        self.env.assertEquals(query_result.nodes_created, 0)
        self.env.assertEquals(query_result.labels_added, 0)
        self.env.assertEquals(query_result.result_set, expected_result)

    # Validate that OPTIONAL MATCH enforces multi-label constraints
    def test07_multi_label_optional_match(self):
        # traverse to a multi-label destination in an OPTIONAL MATCH
        query = """MATCH (a:L1)
                   OPTIONAL MATCH (a)-[]->(b:L2:L1)
                   RETURN labels(a) AS la, labels(b) AS lb
                   ORDER BY la, lb"""
        query_result = graph.query(query)
        expected_result = [[["L0", "L1"], None],
                           [["L1"], ["L1", "L2"]],
                           [["L1", "L2"], None]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # specify an additional label for the source node in the OPTIONAL MATCH
        query = """MATCH (a:L0)
                   OPTIONAL MATCH (a:L1)-[]->(b:L1)
                   RETURN labels(a) AS la, labels(b) AS lb
                   ORDER BY la, lb"""
        query_result = graph.query(query)
        expected_result = [[["L0", "L1"], ["L1"]]]
        self.env.assertEquals(query_result.result_set, expected_result)

    # Validate multi-labeled sources and destinations in variable-length traversals
    def test08_multi_label_variable_length_traversal(self):
        query = """MATCH (a {v: 1})-[*]->(b:L1:L2 {v: 3}) RETURN labels(a), labels(b)"""
        query_result = graph.query(query)
        expected_result = [[["L0", "L1"], ["L1", "L2"]]]
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """MATCH (a:L0 {v: 1})-[*]->(b:L1:L2 {v: 3}) RETURN labels(a), labels(b)"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """MATCH (a:L0 {v: 1})-[*]->(b:L1:L2 {v: 3}) RETURN labels(a), labels(b)"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """MATCH (a:L0:L1 {v: 1})-[*]->(b:L2 {v: 3}) RETURN labels(a), labels(b)"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """MATCH (a:L0:L1 {v: 1})-[*]->(b {v: 3}) RETURN labels(a), labels(b)"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """MATCH (a:L0)-[*]->(b:L1:L2 {v: 3}) RETURN labels(a), labels(b)"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """MATCH (a:L0:L1)-[*]->(b:L2 {v: 3}) RETURN labels(a), labels(b)"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """MATCH (a:L0:L1)-[*]->(b {v: 3}) RETURN labels(a), labels(b)"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.result_set, expected_result)

    def test10_test_delete_label(self):
        graph = Graph(self.redis_con, 'delete_multi_label')

        query = """CREATE (a:L1) RETURN labels(a)"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.labels_added, 1)
        self.env.assertEquals(query_result.nodes_created, 1)
        self.env.assertEquals(query_result.result_set[0][0], ["L1"])

        query = """CREATE (a:L2) RETURN labels(a)"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.labels_added, 1)
        self.env.assertEquals(query_result.nodes_created, 1)
        self.env.assertEquals(query_result.result_set[0][0], ["L2"])

        query = """CREATE (a:L3) RETURN labels(a)"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.labels_added, 1)
        self.env.assertEquals(query_result.nodes_created, 1)
        self.env.assertEquals(query_result.result_set[0][0], ["L3"])

        query = """MATCH (a) DELETE a"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.nodes_deleted, 3)

        query = """CREATE (a:L4) RETURN labels(a)"""
        query_result = graph.query(query)
        self.env.assertEquals(query_result.labels_added, 1)
        self.env.assertEquals(query_result.nodes_created, 1)
        self.env.assertEquals(query_result.result_set[0][0], ["L4"])
