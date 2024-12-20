from common import *

GRAPH_ID = "degree"


class testDegree():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

        self.build_graph()

    def build_graph(self):

        # graph's structure
        #
        #      out-degree / in-degree
        # a0   2            2
        # a1   2            2
        # b    5            3
        # c    1            1
        # d    0            1

        self.graph.query("""
            CREATE
            (a0:A {name:'a0'}),
            (a1:A {name:'a1'}),
            (b:B  {name:'b'}),
            (c:C  {name:'c'}),
            (d:D  {name:'d'}),

            (a0)-[:R]->(a1),
            (a0)-[:R]->(b),

            (a1)-[:R]->(a0),
            (a1)-[:R]->(b),

            (b)-[:S]->(a0),
            (b)-[:S]->(a1),
            (b)-[:S]->(b),
            (b)-[:S]->(c),
            (b)-[:S]->(d),

            (c)-[:R]->(d)
        """)

    def test_invalid_invocation(self):
        queries = ["CALL algo.degree()",                              # missing mandatory argument
                   "CALL algo.degree(1)",                             # wrong argument
                   "CALL algo.degree({dir: 'outgoing', unknown: 4})", # unexpected key
                   "CALL algo.degree({dir: '4})"                      # invalid direction value
        ]

        for q in queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except Exception:
                pass

    def test_all_nodes_all_edges_degree(self):

        # by default:
        # 1. all nodes are considered
        # 2. all edges are considered
        # 3. computing out-degree
        q = """CALL algo.degree({}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        # graph's structure
        #
        #      out-degree / in-degree
        # a0   2            2
        # a1   2            2
        # b    5            3
        # c    2            1
        # d    0            2

        self.env.assertEqual(result_set[0][0], 'a0')
        self.env.assertEqual(result_set[0][1], 2)

        self.env.assertEqual(result_set[1][0], 'a1')
        self.env.assertEqual(result_set[1][1], 2)

        self.env.assertEqual(result_set[2][0], 'b')
        self.env.assertEqual(result_set[2][1], 5)

        self.env.assertEqual(result_set[3][0], 'c')
        self.env.assertEqual(result_set[3][1], 1) # multi-edge

        #self.env.assertEqual(result_set[4][0], 'd')
        #self.env.assertEqual(result_set[4][1], 0)

    def test_all_nodes_all_edges_outdegree(self):
        q = """CALL algo.degree({dir:'outgoing'}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        # graph's structure
        #
        #      out-degree / in-degree
        # a0   2            2
        # a1   2            2
        # b    5            3
        # c    2            1
        # d    0            2

        self.env.assertEqual(result_set[0][0], 'a0')
        self.env.assertEqual(result_set[0][1], 2)

        self.env.assertEqual(result_set[1][0], 'a1')
        self.env.assertEqual(result_set[1][1], 2)

        self.env.assertEqual(result_set[2][0], 'b')
        self.env.assertEqual(result_set[2][1], 5)

        self.env.assertEqual(result_set[3][0], 'c')
        self.env.assertEqual(result_set[3][1], 1) # multi-edge

        # 0 is implicit
        #self.env.assertEqual(result_set[4][0], 'd') 
        #self.env.assertEqual(result_set[4][1], 0)

    def test_all_nodes_all_edges_incoming(self):
        q = """CALL algo.degree({dir:'incoming'}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        # graph's structure
        #
        #      out-degree / in-degree
        # a0   2            2
        # a1   2            2
        # b    5            3
        # c    2            1
        # d    0            2

        self.env.assertEqual(result_set[0][0], 'a0')
        self.env.assertEqual(result_set[0][1], 2)

        self.env.assertEqual(result_set[1][0], 'a1')
        self.env.assertEqual(result_set[1][1], 2)

        self.env.assertEqual(result_set[2][0], 'b')
        self.env.assertEqual(result_set[2][1], 3)

        self.env.assertEqual(result_set[3][0], 'c')
        self.env.assertEqual(result_set[3][1], 1)

        self.env.assertEqual(result_set[4][0], 'd')
        self.env.assertEqual(result_set[4][1], 2)

    def test_relations_specific_outgoing_edges(self):
        # consider only outgoing edges with source nodes of a specific type

        q = """CALL algo.degree({relation:'S'}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 1)

        self.env.assertEqual(result_set[0][0], 'b')
        self.env.assertEqual(result_set[0][1], 5)

    def test_relations_specific_incoming_edges(self):
        # consider only outgoing edges with source nodes of a specific type

        q = """CALL algo.degree({relation:'S', dir:'incoming'}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 5)

        self.env.assertEqual(result_set[0][0], 'a0')
        self.env.assertEqual(result_set[0][1], 1)

        self.env.assertEqual(result_set[1][0], 'a1')
        self.env.assertEqual(result_set[1][1], 1)

        self.env.assertEqual(result_set[2][0], 'b')
        self.env.assertEqual(result_set[2][1], 1)

        self.env.assertEqual(result_set[3][0], 'c')
        self.env.assertEqual(result_set[3][1], 1)

        self.env.assertEqual(result_set[4][0], 'd')
        self.env.assertEqual(result_set[4][1], 1)

    def test_labeled_outgoing_srcs(self):
        # consider only outgoing edges with source nodes of a specific type

        q = """CALL algo.degree({source:'A'}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 2)

        self.env.assertEqual(result_set[0][0], 'a0')
        self.env.assertEqual(result_set[0][1], 2)

        self.env.assertEqual(result_set[1][0], 'a1')
        self.env.assertEqual(result_set[1][1], 2)

    def test_labeled_outgoing_dest(self):
        # consider only outgoing edges with destination nodes of a specific type

        q = """CALL algo.degree({destination:'A'}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 3)

        self.env.assertEqual(result_set[0][0], 'a0')
        self.env.assertEqual(result_set[0][1], 1)

        self.env.assertEqual(result_set[1][0], 'a1')
        self.env.assertEqual(result_set[1][1], 1)

        self.env.assertEqual(result_set[2][0], 'b')
        self.env.assertEqual(result_set[2][1], 2)

    def test_labeled_outgoing_src_and_dest(self):
        # consider only outgoing edges with source and destination nodes of
        # a specific type

        q = """CALL algo.degree({source: 'B', destination:'A'}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 1)

        self.env.assertEqual(result_set[0][0], 'b')
        self.env.assertEqual(result_set[0][1], 2)

    def test_labeled_incoming_srcs(self):
        # consider only incoming edges with source nodes of a specific type

        q = """CALL algo.degree({source:'A', dir:'incoming'}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 2)

        self.env.assertEqual(result_set[0][0], 'a0')
        self.env.assertEqual(result_set[0][1], 2)

        self.env.assertEqual(result_set[1][0], 'a1')
        self.env.assertEqual(result_set[1][1], 2)

    def test_labeled_incoming_dest(self):
        # consider only incoming edges with destination nodes of a specific type

        q = """CALL algo.degree({destination:'A', dir: 'incoming'}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 3)

        self.env.assertEqual(result_set[0][0], 'a0')
        self.env.assertEqual(result_set[0][1], 1)

        self.env.assertEqual(result_set[1][0], 'a1')
        self.env.assertEqual(result_set[1][1], 1)

        self.env.assertEqual(result_set[2][0], 'b')
        self.env.assertEqual(result_set[2][1], 2)

    def test_labeled_incoming_src_and_dest(self):
        # consider only incoming edges with source and destination nodes of
        # a specific type

        q = """CALL algo.degree({source: 'B', destination:'A', dir: 'incoming'})
               YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 1)

        self.env.assertEqual(result_set[0][0], 'b')
        self.env.assertEqual(result_set[0][1], 2)

    def test_none_existing_labels_relations(self):
        # specify none existing labels and relationship types

        queries = [
            "CALL algo.degree({source: 'Z'})",
            "CALL algo.degree({destination: 'Z'})",
            "CALL algo.degree({source: 'Z', destination: 'Z'})",
            "CALL algo.degree({relation: 'Z'})",
            "CALL algo.degree({source: 'Z', relation: 'Z'})",
            "CALL algo.degree({destination: 'Z', relation: 'Z'})",
            "CALL algo.degree({source: 'Z', destination: 'Z', relation: 'Z'})",

            "CALL algo.degree({source: 'Z', dir:'incoming'})",
            "CALL algo.degree({destination: 'Z', dir:'incoming'})",
            "CALL algo.degree({source: 'Z', destination: 'Z', dir: 'incoming'})",
            "CALL algo.degree({relation: 'Z', dir: 'incoming'})",
            "CALL algo.degree({source: 'Z', relation: 'Z', dir: 'incoming'})",
            "CALL algo.degree({destination: 'Z', relation: 'Z', dir: 'incoming'})",
            "CALL algo.degree({source: 'Z', destination: 'Z', relation: 'Z', dir: 'incoming'})"
        ]

        for q in queries:
            result_set = self.graph.query(q).result_set
            self.env.assertEqual(len(result_set), 0)

    def test_tensors(self):
        # try running algo.degree against a graph containing tensors
        # at the moment this is not supported and will emit an exception

        self.env, self.db = Env()
        g = self.db.select_graph("degree_tensors")
        q = "CREATE (a:A)-[:R]->(b:B), (a)-[:R]->(b)"
        g.query(q)

        queries = [
            "CALL algo.degree({})",
            "CALL algo.degree({relation: 'R'})"
        ]

        for q in queries:
            try:
                g.query(q)
                self.env.assertFalse(True)
            except Exception:
                pass
