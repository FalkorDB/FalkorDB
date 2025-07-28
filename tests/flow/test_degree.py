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

        # R relationships:
        #   a0   <--->   a1
        #    '---> b <---'
        #   c ---> d
        #
        # S relationships:
        #   b ---> a0
        #   |---> a1
        #   |---> b
        #   |---> c
        #   '---> d

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

        # R relationships:
        #  a2 ---> a0 -x3-> b0
        #  |^-x2-   '---> b1
        #  |     \
        #  |---> a1
        #  |------|--> b0
        #  |------|--> b1
        #  |------'--> b2
        #  '-> a3

        # S relationships:
        #  a0 -x2-> b0
        #  a1 --> a2 -x2-> b2

        # build a graph with tensors
        #    a0  a1  a2  a3 b0  b1  b2
        #a0  0   0   0   0  5   1   0
        #a1  0   0   2   0  1   1   1
        #a2  1   1   1   1  1   1   3

        #      in  out
        #  a0   1   6
        #  a1   1   6
        #  a2   3   8
        #  a3   1   0
        #  b0   7   0
        #  b1   3   0
        #  b2   4   0
        self.tensorGraph.query("""
            CREATE
            (a0:A {name:'a0'}),
            (a1:A {name:'a1'}),
            (a2:A {name:'a2'}),
            (a3:A {name:'a3'}),
            (b0:B {name:'b0'}),
            (b1:B {name:'b1'}),
            (b2:B {name:'b2'}),


            (a0)-[:R {cost: 12}]->(b0),
            (a0)-[:R {cost: 2}]->(b0),
            (a0)-[:R {cost: 4}]->(b0),
            (a0)-[:S]->(b0),
            (a0)-[:S]->(b0),
            (a0)-[:R {cost: 7}]->(b1),

            (a1)-[:R {cost: 2}]->(b0),
            (a1)-[:R {cost: 3}]->(b1),
            (a1)-[:R {cost: 4}]->(b2),
            (a1)-[:R {cost: 5}]->(a2),
            (a1)-[:S]->(a2),
            (a1)-[:R {cost: 6}]->(a2),

            (a2)-[:R {cost: 4}]->(a0),
            (a2)-[:R {cost: 5}]->(a1),
            (a2)-[:R {cost: 21}]->(a3),
            (a2)-[:R {cost: 8}]->(b0),
            (a2)-[:R {cost: 0}]->(b1),
            (a2)-[:R {cost: -1}]->(b2),
            (a2)-[:S]->(b2),
            (a2)-[:S]->(b2)
        """)

    def test_invalid_invocation(self):
        queries = [
                   # wrong argument
                   "CALL algo.degree(1)",

                   # unexpected key
                   "CALL algo.degree({dir: 'outgoing', unknown: 4})",

                   # invalid direction value
                   "CALL algo.degree({dir: '4'})",

                   # invalid direction value
                   "CALL algo.degree({dir: 4})",

                   # second argument
                   "CALL algo.degree({dir: 'outgoing'}, 0)",

                   # srcLabels should be a string array
                   "CALL algo.degree({dir: 'outgoing', srcLabels = [4, 1]})",
                   "CALL algo.degree({dir: 'outgoing', srcLabels = 3.14})",
                   "CALL algo.degree({dir: 'incoming', srcLabels = 'A'})",

                   # destLabels should be a string array
                   "CALL algo.degree({dir: 'incoming', destLabels = [4,1]})",
                   "CALL algo.degree({dir: 'incoming', destLabels = 3.14})",
                   "CALL algo.degree({dir: 'incoming', destLabels = 'A'})",

                   # relationshipTypes should be a string array
                   "CALL algo.degree({relationshipTypes = [4,1]})",
                   "CALL algo.degree({relationshipTypes = 3.14})",
                   "CALL algo.degree({relationshipTypes = 'A'})",

                   # fake direction
                   "CALL algo.degree({dir: 'wasd'})",
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
        q = """CALL algo.degree() YIELD node, degree
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
        ans_set = [
            ['a0', 2],
            ['a1', 2],
            ['b', 5],
            ['c', 1],
            ['d', 0]
        ]
        self.env.assertEqual(result_set, ans_set)

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
        # c    1            1
        # d    0            2
        ans_set = [
            ['a0', 2],
            ['a1', 2],
            ['b', 5],
            ['c', 1],
            ['d', 0]
        ]
        self.env.assertEqual(result_set, ans_set)

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
        # c    1            1
        # d    0            2
        ans_set = [
            ['a0', 2],
            ['a1', 2],
            ['b', 3],
            ['c', 1],
            ['d', 2]
        ]
        self.env.assertEqual(result_set, ans_set)

    def test_relations_specific_outgoing_edges(self):
        # consider only outgoing edges with source nodes of a specific type

        q = """CALL algo.degree({relationshipTypes: ['S']}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 0],
            ['a1', 0],
            ['b', 5],
            ['c', 0],
            ['d', 0]
        ]
        self.env.assertEqual(result_set, ans_set)

    def test_relations_specific_incoming_edges(self):
        # consider only outgoing edges with source nodes of a specific type

        q = """CALL algo.degree({relationshipTypes:['S'], dir:'incoming'}) 
               YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 1],
            ['a1', 1],
            ['b', 1],
            ['c', 1],
            ['d', 1]
        ]
        self.env.assertEqual(result_set, ans_set)

    def test_labeled_outgoing_srcs(self):
        # consider only outgoing edges with source nodes of a specific type

        q = """CALL algo.degree({srcLabels:['A']}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 2],
            ['a1', 2]
        ]
        self.env.assertEqual(result_set, ans_set)

    def test_labeled_outgoing_dest(self):
        # consider only outgoing edges with destination nodes of a specific type

        q = """CALL algo.degree({destLabels:['A']}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 1],
            ['a1', 1],
            ['b', 2],
            ['c', 0],
            ['d', 0],
        ]
        self.env.assertEqual(result_set, ans_set)

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
        ans_set = [
            ['a0', 2],
            ['a1', 2]
        ]
        self.env.assertEqual(result_set, ans_set)

    def test_labeled_incoming_dest(self):
        # consider only incoming edges with destination nodes of a specific type

        q = """CALL algo.degree({destLabels:['A'], dir: 'incoming'}) YIELD node, degree
               RETURN node.name, degree
               ORDER BY node.name"""

        result_set = self.graph.query(q).result_set

        ans_set = [
            ['a0', 1],
            ['a1', 1],
            ['b', 2],
            ['c', 0],
            ['d', 0],
        ]
        self.env.assertEqual(result_set, ans_set)

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


    def test_non_existing_labels_relations(self):
        # specify none existing labels and relationship types

        # expect no answer back
        queries_empty = [
            "CALL algo.degree({srcLabels: ['Z']}) YIELD degree",
            "CALL algo.degree({srcLabels: ['Z'], destLabels: ['Z']}) "
            "YIELD degree",
            "CALL algo.degree({srcLabels: ['Z'], relationshipTypes: ['Z']}) "
            "YIELD degree",
            "CALL algo.degree({srcLabels: ['Z'], destLabels: ['Z'], "
            "relationshipTypes: ['Z']}) YIELD degree",

            "CALL algo.degree({srcLabels: ['Z'], dir:'incoming'}) "
            "YIELD degree",
            "CALL algo.degree({srcLabels: ['Z'], destLabels: ['Z'], "
            "dir: 'incoming'}) YIELD degree",
            "CALL algo.degree({srcLabels: ['Z'], relationshipTypes: ['Z'], "
            "dir: 'incoming'}) YIELD degree",
            "CALL algo.degree({srcLabels: ['Z'], destLabels: ['Z'], "
            "relationshipTypes: ['Z'], dir: 'incoming'}) YIELD degree",

            "CALL algo.degree({srcLabels: ['Z'], dir:'both'}) "
            "YIELD degree",
            "CALL algo.degree({srcLabels: ['Z'], destLabels: ['Z'], "
            "dir: 'both'}) YIELD degree",
            "CALL algo.degree({srcLabels: ['Z'], relationshipTypes: ['Z'], "
            "dir: 'both'}) YIELD degree",
            "CALL algo.degree({srcLabels: ['Z'], destLabels: ['Z'], "
            "relationshipTypes: ['Z'], dir: 'both'}) YIELD degree"
        ]

        for q in queries_empty:
            try:
                result_set = self.graph.query(q).result_set
                print(q)
                self.env.assertFalse(True)
            except:
                pass

        # expect explicit zeros back
        queries_zero = [
            "CALL algo.degree({destLabels: ['Z']}) YIELD degree",
            "CALL algo.degree({relationshipTypes: ['Z']}) YIELD degree",
            "CALL algo.degree({destLabels: ['Z'], relationshipTypes: ['Z']}) "
            "YIELD degree",

            "CALL algo.degree({destLabels: ['Z'], dir:'incoming'}) "
            "YIELD degree",
            "CALL algo.degree({relationshipTypes: ['Z'], dir: 'incoming'}) "
            "YIELD degree",
            "CALL algo.degree({destLabels: ['Z'], relationshipTypes: ['Z'], "
            "dir: 'incoming'}) YIELD degree",

            "CALL algo.degree({destLabels: ['Z'], dir:'both'}) "
            "YIELD degree",
            "CALL algo.degree({relationshipTypes: ['Z'], dir: 'both'}) "
            "YIELD degree",
            "CALL algo.degree({destLabels: ['Z'], relationshipTypes: ['Z'], "
            "dir: 'both'}) YIELD degree"
        ]

        for q in queries_empty:
            try:
                result_set = self.graph.query(q).result_set
                self.env.assertFalse(True)
            except:
                pass

    def test_tensors_all_outgoing(self):
        # try running algo.degree against a graph containing tensors

        q = """CALL algo.degree() YIELD node, degree
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
    def test_tensors_all_both(self):
        # try running algo.degree against a graph containing tensors

        q = """CALL algo.degree({dir:'both'}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = [
            ['a0', 7],
            ['a1', 7],
            ['a2', 11],
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
        ans_set_in = [
            ['a0', 4],
            ['a1', 5],
            ['a2', 6],
            ['a3', 0],
            ['b0', 0],
            ['b1', 0],
            ['b2', 0]
        ]
        self.env.assertEqual(result_set, ans_set_in)

        q = """CALL algo.degree({relationshipTypes:['R'], dir:'incoming'}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set_out = [
            ['a0', 1],
            ['a1', 1],
            ['a2', 2],
            ['a3', 1],
            ['b0', 5],
            ['b1', 3],
            ['b2', 2]
        ]
        self.env.assertEqual(result_set, ans_set_out)

        q = """CALL algo.degree({relationshipTypes:['R'], dir:'both'}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = ans_set_in
        for i, n in enumerate(ans_set_out):
            ans_set[i][1] += n[1]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degree({relationshipTypes:['S'], dir:'outgoing'}) 
                YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set_in = [
            ['a0', 2],
            ['a1', 1],
            ['a2', 2],
            ['a3', 0],
            ['b0', 0],
            ['b1', 0],
            ['b2', 0]
        ]
        self.env.assertEqual(result_set, ans_set_in)
        q = """CALL algo.degree({relationshipTypes:['S'], dir:'incoming'}) 
                YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set_out = [
            ['a0', 0],
            ['a1', 0],
            ['a2', 1],
            ['a3', 0],
            ['b0', 2],
            ['b1', 0],
            ['b2', 2]
        ]
        self.env.assertEqual(result_set, ans_set_out)

        q = """CALL algo.degree({relationshipTypes:['S'], dir:'both'}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set = ans_set_in
        for i, n in enumerate(ans_set_out):
            ans_set[i][1] += n[1]
        self.env.assertEqual(result_set, ans_set)

    def test_tensors_nodeTypes(self):
        # try running algo.degree against a graph containing tensors

        q = """CALL algo.degree({srcLabels:['A']}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set_in = [
            ['a0', 6],
            ['a1', 6],
            ['a2', 8],
            ['a3', 0],
        ]
        self.env.assertEqual(result_set, ans_set_in)

        q = """CALL algo.degree({srcLabels:['A'], dir:'incoming'}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set_out = [
            ['a0', 1],
            ['a1', 1],
            ['a2', 3],
            ['a3', 1],
        ]
        self.env.assertEqual(result_set, ans_set_out)
        
        q = """CALL algo.degree({srcLabels:['A'], dir:'both'}) YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        ans_set_out = [
            ['a0', 1],
            ['a1', 1],
            ['a2', 3],
            ['a3', 1],
        ]
        ans_set = ans_set_in
        for i, n in enumerate(ans_set_out):
            ans_set[i][1] += n[1]
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
        q = """CALL algo.degree({srcLabels:['B'], dir:'both'}) 
                YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
        self.env.assertEqual(result_set, ans_set)
        q = """CALL algo.degree({srcLabels:['B'], destLabels:['A'], dir:'incoming'}) 
                YIELD node, degree
                RETURN node.name, degree
                ORDER BY node.name"""
        result_set = self.tensorGraph.query(q).result_set
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
