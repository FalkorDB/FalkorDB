from common import *

GRAPH_ID = "degree"
GRAPH_ID_TENSORS = "degree_tensors"

class testDegree(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.tensorGraph = self.db.select_graph(GRAPH_ID_TENSORS)
        self.build_graphs()

    def build_graphs(self):

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

        # build a graph with tensors
        #    a0  a1  a2  a3 b0  b1  b2
        #a0  0   0   0   0  5   1   0
        #a1  0   0   2   0  1   1   1
        #a2  1   1   1   1  1   1   3

        self.tensorGraph.query("""
            CREATE
            (a0:A {name:'a0'}),
            (a1:A {name:'a1'}),
            (a2:A {name:'a2'}),
            (a3:A {name:'a3'}),
            (b0:B {name:'b0'}),
            (b1:B {name:'b1'}),
            (b2:B {name:'b2'}),


            (a0)-[:R]->(b0),
            (a0)-[:R]->(b0),
            (a0)-[:R]->(b0),
            (a0)-[:S]->(b0),
            (a0)-[:S]->(b0),
            (a0)-[:R]->(b1),

            (a1)-[:R]->(b0),
            (a1)-[:R]->(b1),
            (a1)-[:R]->(b2),
            (a1)-[:R]->(a2),
            (a1)-[:R]->(a2),
            (a1)-[:S]->(a2),

            (a2)-[:R]->(a0),
            (a2)-[:R]->(a1),
            (a2)-[:R]->(a3),
            (a2)-[:R]->(b0),
            (a2)-[:R]->(b1),
            (a2)-[:R]->(b2),
            (a2)-[:S]->(b2),
            (a2)-[:S]->(b2)
        """)

    def test_invalid_invocation(self):
        queries = ["CALL algo.degree()",                              # missing mandatory argument
                   "CALL algo.degree(1)",                             # wrong argument
                   "CALL algo.degree({dir: 'outgoing', unknown: 4})", # unexpected key
                   "CALL algo.degree({dir: '4'})"                      # invalid direction value
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

        # 0 is implicit
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

        q = """CALL algo.degree({relationshipTypes: ['S']}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 5)
        for name, deg in result_set:
            if name != 'b':
                self.env.assertEqual(deg, 0)
            else:
                self.env.assertEqual(result_set[2][1], 5)

    def test_relations_specific_incoming_edges(self):
        # consider only outgoing edges with source nodes of a specific type

        q = """CALL algo.degree({relationshipTypes:['S'], dir:'incoming'}) 
               YIELD node, degree
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

        q = """CALL algo.degree({srcLabels:['A']}) YIELD node, degree
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

        q = """CALL algo.degree({destLabels:['A']}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 5)

        self.env.assertEqual(result_set[0][0], 'a0')
        self.env.assertEqual(result_set[0][1], 1)

        self.env.assertEqual(result_set[1][0], 'a1')
        self.env.assertEqual(result_set[1][1], 1)

        self.env.assertEqual(result_set[2][0], 'b')
        self.env.assertEqual(result_set[2][1], 2)

        self.env.assertEqual(result_set[3][0], 'c')
        self.env.assertEqual(result_set[3][1], 0)

        self.env.assertEqual(result_set[4][0], 'd')
        self.env.assertEqual(result_set[4][1], 0)

    def test_labeled_outgoing_src_and_dest(self):
        # consider only outgoing edges with source and destination nodes of
        # a specific type

        q = """CALL algo.degree({srcLabels: ['B'], destLabels:['A']}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 1)

        self.env.assertEqual(result_set[0][0], 'b')
        self.env.assertEqual(result_set[0][1], 2)

    def test_labeled_incoming_srcs(self):
        # consider only incoming edges with source nodes of a specific type

        q = """CALL algo.degree({srcLabels:['A'], dir:'incoming'}) YIELD node, degree
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

        q = """CALL algo.degree({destLabels:['A'], dir: 'incoming'}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 5)

        self.env.assertEqual(result_set[0][0], 'a0')
        self.env.assertEqual(result_set[0][1], 1)

        self.env.assertEqual(result_set[1][0], 'a1')
        self.env.assertEqual(result_set[1][1], 1)

        self.env.assertEqual(result_set[2][0], 'b')
        self.env.assertEqual(result_set[2][1], 2)

        self.env.assertEqual(result_set[3][0], 'c')
        self.env.assertEqual(result_set[3][1], 0)

        self.env.assertEqual(result_set[4][0], 'd')
        self.env.assertEqual(result_set[4][1], 0)

    def test_labeled_incoming_src_and_dest(self):
        # consider only incoming edges with source and destination nodes of
        # a specific type

        q = """CALL algo.degree({srcLabels: ['B'], destLabels: ['A'], dir: 'incoming'})
               YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        self.env.assertEqual(len(result_set), 1)

        self.env.assertEqual(result_set[0][0], 'b')
        self.env.assertEqual(result_set[0][1], 2)


    # FIXME 
    # def test_none_existing_labels_relations(self):
    #     # specify none existing labels and relationship types

    #     queries = [
    #         "CALL algo.degree({srcLabels: ['Z']})",
    #         "CALL algo.degree({destLabels: ['Z']})",
    #         "CALL algo.degree({srcLabels: ['Z'], destLabels: ['Z']})",
    #         "CALL algo.degree({relationshipTypes: ['Z']})",
    #         "CALL algo.degree({srcLabels: ['Z'], relationshipTypes: ['Z']})",
    #         "CALL algo.degree({destLabels: ['Z'], relationshipTypes: ['Z']})",
    #         "CALL algo.degree({srcLabels: ['Z'], destLabels: ['Z'], relationshipTypes: ['Z']})",

    #         "CALL algo.degree({srcLabels: ['Z'], dir:'incoming'})",
    #         "CALL algo.degree({destLabels: ['Z'], dir:'incoming'})",
    #         "CALL algo.degree({srcLabels: ['Z'], destLabels: ['Z'], dir: 'incoming'})",
    #         "CALL algo.degree({relationshipTypes: ['Z'], dir: 'incoming'})",
    #         "CALL algo.degree({srcLabels: ['Z'], relationshipTypes: ['Z'], dir: 'incoming'})",
    #         "CALL algo.degree({destLabels: ['Z'], relationshipTypes: ['Z'], dir: 'incoming'})",
    #         "CALL algo.degree({srcLabels: ['Z'], destLabels: ['Z'], relationshipTypes: ['Z'], dir: 'incoming'})"
    #     ]

    #     for q in queries:
    #         result_set = self.graph.query(q).result_set
    #         self.env.assertEqual(len(result_set), 0)

    def test_tensors_all_outgoing(self):
        # try running algo.degree against a graph containing tensors

        q = """CALL algo.degree({}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 6],
            ['a1', 6],
            ['a2', 8],
            ['a3', 0],
            ['b0', 0],
            ['b1', 0],
            ['b2', 0]
        ]
        self.env.assertEqual(result_set, ans_set)

    def test_tensors_all_incoming(self):
        # try running algo.degree against a graph containing tensors

        q = """CALL algo.degree({dir:'incoming'}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 1],
            ['a1', 1],
            ['a2', 3],
            ['a3', 1],
            ['b0', 7],
            ['b1', 3],
            ['b2', 4]
        ]
        self.env.assertEqual(result_set, ans_set)

    def test_tensors_relationships(self):
        # try running algo.degree against a graph containing tensors

        q = """CALL algo.degree({relationshipTypes:['R']}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 4],
            ['a1', 5],
            ['a2', 6],
            ['a3', 0],
            ['b0', 0],
            ['b1', 0],
            ['b2', 0]
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degree({relationshipTypes:['R'], dir:'incoming'}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 1],
            ['a1', 1],
            ['a2', 2],
            ['a3', 1],
            ['b0', 5],
            ['b1', 3],
            ['b2', 2]
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degree({relationshipTypes:['S'], dir:'outgoing'}) 
                YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 2],
            ['a1', 1],
            ['a2', 2],
            ['a3', 0],
            ['b0', 0],
            ['b1', 0],
            ['b2', 0]
        ]
        self.env.assertEqual(result_set, ans_set)
        q = """CALL algo.degree({relationshipTypes:['S'], dir:'incoming'}) 
                YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 0],
            ['a1', 0],
            ['a2', 1],
            ['a3', 0],
            ['b0', 2],
            ['b1', 0],
            ['b2', 2]
        ]
        self.env.assertEqual(result_set, ans_set)
    def test_tensors_nodeTypes(self):
        # try running algo.degree against a graph containing tensors

        q = """CALL algo.degree({srcLabels:['A']}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 6],
            ['a1', 6],
            ['a2', 8],
            ['a3', 0],
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degree({srcLabels:['A'], dir:'incoming'}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 1],
            ['a1', 1],
            ['a2', 3],
            ['a3', 1],
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degree({srcLabels:['B'], dir:'outgoing'}) 
                YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['b0', 0],
            ['b1', 0],
            ['b2', 0]
        ]
        self.env.assertEqual(result_set, ans_set)
        q = """CALL algo.degree({srcLabels:['B'], dir:'incoming'}) 
                YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['b0', 7],
            ['b1', 3],
            ['b2', 4]
        ]
        self.env.assertEqual(result_set, ans_set)
        q = """CALL algo.degree({srcLabels:['B'], destLabels:['A'], dir:'incoming'}) 
                YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        self.env.assertEqual(result_set, ans_set)
        self.env.assertEqual(result_set, ans_set)
        q = """CALL algo.degree({srcLabels:['A'], destLabels:['B']}) 
                YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 6],
            ['a1', 3],
            ['a2', 5],
            ['a3', 0]
        ]
        self.env.assertEqual(result_set, ans_set)

    def test_tensors_type_and_label(self):
        # try running algo.degree against a graph containing tensors

        q = """CALL algo.degree({srcLabels:['A'],relationshipTypes:['R']}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 4],
            ['a1', 5],
            ['a2', 6],
            ['a3', 0],
        ]
        self.env.assertEqual(result_set, ans_set)
        q = """CALL algo.degree({srcLabels:['A'], destLabels: ['B'], relationshipTypes:['R']}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 4],
            ['a1', 3],
            ['a2', 3],
            ['a3', 0],
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degree({srcLabels:['A'], destLabels: ['B'], relationshipTypes:['S']}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 2],
            ['a1', 0],
            ['a2', 2],
            ['a3', 0],
        ]
        self.env.assertEqual(result_set, ans_set)