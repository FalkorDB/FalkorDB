from common import *

GRAPH_ID = "degree_weighted"

class testDegreeWeighted(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.build_graphs()

    def build_graphs(self):

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

        #     weight
        #       in     out
        #  a0  1.01   26.04
        #  a1  6.78   27.11
        #  a2  10.67  35.69
        #  a3  9.12   0.0  
        #  b0  24.63  0.0  
        #  b1  23.41  0.0  
        #  b2  13.22  0.0  

        #      cost
        #      in  out
        #  a0  4   25
        #  a1  5   20
        #  a2  11  37
        #  a3  21  0  
        #  b0  28  0  
        #  b1  10  0  
        #  b2  3   0  
        self.graph.query("""
            CREATE
            (a0:A {name:'a0'}),
            (a1:A {name:'a1'}),
            (a2:A {name:'a2'}),
            (a3:A {name:'a3'}),
            (b0:B {name:'b0'}),
            (b1:B {name:'b1'}),
            (b2:B {name:'b2'}),

            (a0)-[:R {cost: 12, weight: 8.73}]->(b0),
            (a0)-[:R {cost: 2, weight: 1.25}]->(b0),
            (a0)-[:R {cost: 4, weight: 5.61}]->(b0),
            (a0)-[:S {weight: 0.98}]->(b0),
            (a0)-[:S {weight: 3.42}]->(b0),
            (a0)-[:R {cost: 7, weight: 6.05}]->(b1),

            (a1)-[:R {cost: 2, weight: 2.19}]->(b0),
            (a1)-[:R {cost: 3, weight: 9.37}]->(b1),
            (a1)-[:R {cost: 4, weight: 4.88}]->(b2),
            (a1)-[:R {cost: 5, weight: 7.02}]->(a2),
            (a1)-[:S {weight: 0.55}]->(a2),
            (a1)-[:R {cost: 6, weight: 3.10}]->(a2),

            (a2)-[:R {cost: 4, weight: 1.01}]->(a0),
            (a2)-[:R {cost: 5, weight: 6.78}]->(a1),
            (a2)-[:R {cost: 21, weight: 9.12}]->(a3),
            (a2)-[:R {cost: 8, weight: 2.45}]->(b0),
            (a2)-[:R {cost: 0, weight: 7.99}]->(b1),
            (a2)-[:R {cost: -1, weight: 0.33}]->(b2),
            (a2)-[:S {weight: 5.14}]->(b2),
            (a2)-[:S {weight: 2.87}]->(b2)
        """)

    def test_invalid_invocation(self):
        queries = [
                   # wrong argument
                   "CALL algo.degreeWeight(1)",

                   # unexpected key
                   "CALL algo.degreeWeight({dir: 'outgoing', unknown: 4})",

                   # invalid direction value
                   "CALL algo.degreeWeight({dir: '4'})",

                   # invalid direction value
                   "CALL algo.degreeWeight({dir: 4})",

                   # second argument
                   "CALL algo.degreeWeight({dir: 'outgoing'}, 0)",

                   # srcLabels should be a string array
                   "CALL algo.degreeWeight({dir: 'outgoing', srcLabels = [4, 1]})",
                   "CALL algo.degreeWeight({dir: 'outgoing', srcLabels = 3.14})",
                   "CALL algo.degreeWeight({dir: 'incoming', srcLabels = 'A'})",

                   # destLabels should be a string array
                   "CALL algo.degreeWeight({dir: 'incoming', destLabels = [4,1]})",
                   "CALL algo.degreeWeight({dir: 'incoming', destLabels = 3.14})",
                   "CALL algo.degreeWeight({dir: 'incoming', destLabels = 'A'})",

                   # relationshipTypes should be a string array
                   "CALL algo.degreeWeight({relationshipTypes = [4,1]})",
                   "CALL algo.degreeWeight({relationshipTypes = 3.14})",
                   "CALL algo.degreeWeight({relationshipTypes = 'A'})",

                   # fake direction
                   "CALL algo.degreeWeight({dir: 'wasd'})",
                   
                   # fake attribute
                   "CALL algo.degreeWeight({weightAttribute: 'fake'})"
        ]

        for q in queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except Exception:
                pass

    def test_tensors_weight_all(self):
        q = """CALL algo.degreeWeight({weightAttribute: 'cost'}) 
                YIELD node, weight
                RETURN node.name, weight
                ORDER BY node.name"""
        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 25],
            ['a1', 20],
            ['a2', 37],
            ['a3', 0],
            ['b0', 0],
            ['b1', 0],
            ['b2', 0]
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degreeWeight({weightAttribute: 'weight'}) 
                YIELD node, weight
                RETURN node.name, weight
                ORDER BY node.name"""
        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 26.04],
            ['a1', 27.11],
            ['a2', 35.69],
            ['a3', 0.0],
            ['b0', 0.0],
            ['b1', 0.0],
            ['b2', 0.0]
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degreeWeight({weightAttribute: 'cost', dir: 'incoming'}) 
                YIELD node, weight
                RETURN node.name, weight
                ORDER BY node.name"""
        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 4],
            ['a1', 5],
            ['a2', 11],
            ['a3', 21],
            ['b0', 28],
            ['b1', 10],
            ['b2', 3]
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degreeWeight({weightAttribute: 'weight', dir: 'incoming'}) 
                YIELD node, weight
                RETURN node.name, weight
                ORDER BY node.name"""
        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 1.01],
            ['a1', 6.78],
            ['a2', 10.67],
            ['a3', 9.12],
            ['b0', 24.63],
            ['b1', 23.41],
            ['b2', 13.22]
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degreeWeight({weightAttribute: 'cost', dir: 'both'}) 
                YIELD node, weight
                RETURN node.name, weight
                ORDER BY node.name"""
        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 25 + 4],
            ['a1', 20 + 5],
            ['a2', 37 + 11],
            ['a3', 0  + 21],
            ['b0', 0  + 28],
            ['b1', 0  + 10],
            ['b2', 0  + 3]
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degreeWeight({weightAttribute: 'weight', dir: 'both'}) 
                YIELD node, weight
                RETURN node.name, weight
                ORDER BY node.name"""
        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 26.04 + 1.01],
            ['a1', 27.11 + 6.78],
            ['a2', 35.69 + 10.67],
            ['a3', 0.0   + 9.12],
            ['b0', 0.0   + 24.63],
            ['b1', 0.0   + 23.41],
            ['b2', 0.0   + 13.22]
        ]
        self.env.assertEqual(result_set, ans_set)

    def test_tensors_weight_labels(self):
        q = """CALL algo.degreeWeight({srcLabels:['A'], destLabels: ['B'], 
                relationshipTypes:['R'], weightAttribute: 'cost'}) 
                YIELD node, weight
                RETURN node.name, weight
                ORDER BY node.name"""
        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 25],
            ['a1', 9],
            ['a2', 7],
            ['a3', 0],
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degreeWeight({srcLabels:['A'], destLabels: ['B'], 
                weightAttribute: 'cost'}) 
                YIELD node, weight
                RETURN node.name, weight
                ORDER BY node.name"""
        result_set = self.graph.query(q).result_set
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degreeWeight({srcLabels:['A'], destLabels: ['B'], 
                relationshipTypes:['R'], weightAttribute: 'cost', 
                dir: 'incoming'}) 
                YIELD node, weight
                RETURN node.name, weight
                ORDER BY node.name"""
        result_set = self.graph.query(q).result_set
        ans_set = [
            ['a0', 0],
            ['a1', 0],
            ['a2', 0],
            ['a3', 0],
        ]
        self.env.assertEqual(result_set, ans_set)

        q = """CALL algo.degreeWeight({srcLabels:['A'], destLabels: ['B'], 
                relationshipTypes:['S'], weightAttribute: 'cost'}) 
                YIELD node, weight
                RETURN node.name, weight
                ORDER BY node.name"""
        result_set = self.graph.query(q).result_set
        self.env.assertEqual(result_set, ans_set)

    def test_add_and_remove_edge(self):
        # Add new edges
        self.graph.query("""
            MATCH (a0:A {name:'a0'}), (b0:B {name:'b0'})
            CREATE (a0)-[:TEMP {weight: 2.96, cost: 10}]->(b0),
                   (a0)-[:TEMP {weight: 3.14, cost: 6}]->(b0)
        """)

        # Check weight after adding edges
        result_set = self.graph.query("""
            CALL algo.degreeWeight({weightAttribute: 'cost'})
            YIELD node, weight 
            RETURN node.name, weight
            ORDER BY node.name
        """).result_set
        ans_set = [
            ['a0', 41],
            ['a1', 20],
            ['a2', 37],
            ['a3', 0],
            ['b0', 0],
            ['b1', 0],
            ['b2', 0]
        ]
        self.env.assertEqual(result_set, ans_set)

        # Check weight after adding edges
        result_set = self.graph.query("""
            CALL algo.degreeWeight({weightAttribute: 'weight'})
            YIELD node, weight 
            RETURN node.name, weight
            ORDER BY node.name
        """).result_set
        ans_set = [
            ['a0', 32.14],
            ['a1', 27.11],
            ['a2', 35.69],
            ['a3', 0.0],
            ['b0', 0.0],
            ['b1', 0.0],
            ['b2', 0.0]
        ]
        self.env.assertEqual(result_set, ans_set)
        # Delete newly added edges
        self.graph.query("""
            MATCH (a0:A {name:'a0'})-[r]->(b0:B {name:'b0'})
            WHERE type(r) = 'TEMP'
            DELETE r
        """)

        # Check weight after deleting edges
        result_set = self.graph.query("""
            CALL algo.degreeWeight({weightAttribute: 'cost', relationshipTypes:['TYPE'], dir: 'both'})
            YIELD node, weight
            RETURN weight
        """).result_set
        ans_set = [[0]]*7
        self.env.assertEqual(result_set, ans_set)