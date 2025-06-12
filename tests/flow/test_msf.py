from common import *
import random

GRAPH_ID = "MSF"
GRAPH_ID_RAND = "MSF_rand"

class testMSF(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.randomGraph = self.db.select_graph(GRAPH_ID_RAND)
        self.generate_random_graph_cypher()

    def generate_random_graph_cypher(self, num_nodes=20, num_edges=140, numlabels=4):
        random.seed(1111)
        # 1. Create Nodes
        query_parts = [f"UNWIND range({num_nodes * j}, {num_nodes * (j + 1) - 1}) AS i "
                        f"CREATE (n:{chr(ord('A') + j)} {{id: i}})"
                        for j in range(numlabels)]

        # Create a list of node IDs for random selection
        node_ids = list(range(num_nodes * numlabels))

        # 2. ensure components are connected.
        for j in range(numlabels):
            query_parts.append(f"UNWIND range({num_nodes * j + 1}, {num_nodes * (j + 1) - 1}) AS i "
                f"MATCH (a {{id: {num_nodes * j}}}), (b {{id: i}})"
                "CREATE (a)-[:R {weight: 100}]->(b)")
            

        # 3. Create Relationships with random weights
        relationship_creation_statements = []
        
        
        
        for _ in range(num_edges):
            # Pick two random nodes (can be the same for self-loops)
            source_id = random.choice(node_ids)
            target_id = random.choice(node_ids)
            if(source_id == target_id): continue
            # Generate a random double weight
            weight = random.uniform(-100.0, 100.0)
        
            # Cypher statement to create the relationship
            # We find the nodes by their 'id' property and then create the relationship.
            relationship_creation_statements.append(
                    f'MATCH (a {{id: {source_id}}}), (b {{id: {target_id}}})'
                    f'CREATE (a)-[:R {{weight: {weight}}}]->(b)'
                )
        query_parts.extend(relationship_creation_statements)
        
        for q in query_parts:
            self.randomGraph.query(q)
    def find_min_edge (self, src, dest):
        res = self.randomGraph.query(f"MATCH (:{src})-[r]->(:{dest}) RETURN MIN(r.weight) as minW").result_set
        res2 = self.randomGraph.query(f"MATCH (:{dest})-[r]->(:{src}) RETURN MIN(r.weight) as minW").result_set
        if res[0][0] is None:
            return res2[0][0]
        elif res2[0][0] is None:
            return res[0][0]
        else:
            return min(res[0][0], res2[0][0])

    def test_invalid_invocation(self):
        invalid_queries = [
                """CALL algo.MSF({nodeLabels: 'Person'})""",         # non-array nodeLabels parameter
                """CALL algo.MSF({relationshipTypes: 'KNOWS'})""",   # non-array relationshipTypes parameter
                """CALL algo.MSF({invalidParam: 'value'})""",        # unexpected extra parameters
                """CALL algo.MSF('invalid')""",                      # invalid configuration type
                """CALL algo.MSF({nodeLabels: [1, 2, 3]})""",        # integer values in nodeLabels array
                """CALL algo.MSF({relationshipTypes: [1, 2, 3]})""", # integer values in relationshipTypes array
                """CALL algo.MSF(null) YIELD node, invalidField""",  # non-existent yield field

                """CALL algo.MSF({nodeLabels: ['Person'],
                               relationshipTypes: ['KNOWS'],
                               invalidParam: 'value'})""",           # unexpected extra parameters
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except redis.exceptions.ResponseError as e:
                pass
        self.graph.delete()

    def test_msf_on_empty_graph(self):
        """Test MSF algorithm behavior on an empty graph"""
        
        # run MSF on empty graph
        result = self.graph.query("CALL algo.MSF(null)")
        # if we reach here, the algorithm didn't throw an exception
        
        # check if it returned empty results (acceptable behavior)
        self.env.assertEqual(len(result.result_set), 0)
        self.graph.delete()

    def test_msf_on_unlabeled_graph(self):
        """Test MSF algorithm on unlabeled nodes with multiple connected components"""
        # Create an unlabeled graph with multiple connected components
        # - Component 1: nodes 1-2-3-1 forming a cycle
        # - Component 2: nodes 4-5 connected
        # - Component 3: isolated node 6
        self.graph.query("""
            CREATE
            (n1 {id: 1}),
            (n2 {id: 2}),
            (n3 {id: 3}),
            (n4 {id: 4}),
            (n5 {id: 5}),
            (n6 {id: 6}),
            (n1)-[:R]->(n2),
            (n2)-[:R]->(n3),
            (n3)-[:R]->(n1),
            (n4)-[:R]->(n5)
        """)

        # Run MSF algorithm
        result = self.graph.query("""CALL algo.MSF({relationshipTypes: ['R']}) yield edge""")
        result_set = result.result_set

        # We should have exactly 3 different edges 
        self.env.assertEqual(len(result_set), 3)
        self.graph.delete()

    def test_msf_on_multilable(self):
        """Test MSF algorithm on multilabled nodes with multiple connected components"""

        # Create an unlabeled graph with multiple connected components
        # - Component 1: nodes 1-2 are connected with multiple edges
        self.graph.query("""
            CREATE
            (n1 {id: 1}),
            (n2 {id: 2}),
            (n3 {id: 3}),
            (n4 {id: 4}),
            (n5 {id: 5}),
            (n6 {id: 6}),
            (n1)-[:Car {distance: 11.2, cost: 23}]->(n2),
            (n1)-[:Walk {distance: 9.24321, cost: 7}]->(n2),
            (n1)-[:Plane {distance: 123.2490, cost: 300}]->(n2),
            (n1)-[:Boat {distance: .75, cost: 100}]->(n2),
            (n1)<-[:Swim {distance: .5, cost: 12}]-(n2),
            (n1)<-[:Kayak {distance: .68, cost: 4}]-(n2)
        """)
        # Run MSF algorithm
        result = self.graph.query("""CALL algo.MSF({weightAttribute: 'cost'}) 
            yield edge, weight return type(edge), weight""")
        result_set = result.result_set
        self.env.assertEqual(len(result_set), 1)
        # Check the edge and weight
        edge, weight = result_set[0]
        self.env.assertEqual(edge, 'Kayak')
        self.env.assertEqual(weight, 4)
        self.graph.delete()


    def test_msf_with_multiedge(self):
        """Test WCC algorithm with different relationship type filters"""
        # Create an unlabeled graph with two relationship types
        # Relationship type A:
        # - n1-A->n2 (x3 edges)
        # Relationship type B:
        # - n1-B->n2 (x2 edges)
        # - n2-B->n3 (x1 edge w/o score attributes)
        # - n3-B->n4 (x2 edges)
        self.graph.query("""
            CREATE
            (n1 {id: 1}),
            (n2 {id: 2}),
            (n3 {id: 3}),
            (n4 {id: 4}),
            (n5 {id: 5}),
            (n6 {id: 6}),
            (n7 {id: 7}),
            (n1)-[:A {score: 789134, msf_ans: 0}]->(n2),
            (n1)-[:A {score: 5352, msf_ans: 0}]->(n2),
            (n1)-[:A {score: 1234, msf_ans: 1}]->(n2),
            (n1)-[:B {score: 123456, msf_ans: 0}]->(n2),
            (n1)-[:B {score: 1000, msf_ans: 2}]->(n2),
            (n2)-[:B {msf_ans: 0}]->(n3),
            (n3)-[:B {score: 8991234, msf_ans: 0}]->(n4),
            (n3)-[:B {score: 7654, msf_ans: 2}]->(n4)
        """)
        # Run MSF algorithm with relationship type A
        result = self.graph.query("""
            CALL algo.MSF({relationshipTypes: ['A'], weightAttribute: 'score'}) 
            YIELD edge RETURN edge.msf_ans
            """)
        result_set = result.result_set
        self.env.assertEqual(len(result_set), 1)
        for edge in result_set:
            self.env.assertEqual(edge[0], 1)
        # Run MSF algorithm with relationship type B
        result = self.graph.query("""
            CALL algo.MSF({relationshipTypes: ['B'], weightAttribute: 'score'}) 
            YIELD edge RETURN edge.msf_ans
            """)
        result_set = result.result_set
        self.env.assertEqual(len(result_set), 2)
        for edge in result_set:
            self.env.assertEqual(edge[0], 2)
        
        # Run MSF algorithm with both relationships
        result = self.graph.query("""
            CALL algo.MSF({weightAttribute: 'score'}) 
            YIELD edge RETURN edge.msf_ans
            """)
        result_set = result.result_set
        self.env.assertEqual(len(result_set), 2)
        for edge in result_set:
            self.env.assertEqual(edge[0], 2)
        self.graph.delete()


    def test_msf_rand_labels(self):
        result_set = self.randomGraph.query("""
            CALL algo.MSF({nodeLabels: ['A', 'B'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("A", "B")
        ok = False
        for w in result_set:
            ok = ok or w[0] == minEdge
        self.env.assertTrue(ok or (minEdge is None))

        result_set = self.randomGraph.query("""
            CALL algo.MSF({nodeLabels: ['A', 'C'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("A", "C")
        ok = False
        for w in result_set:
            ok = ok or w[0] == minEdge
        self.env.assertTrue(ok or (minEdge is None))

        result_set = self.randomGraph.query("""
            CALL algo.MSF({nodeLabels: ['A', 'D'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("A", "D")
        ok = False
        for w in result_set:
            ok = ok or w[0] == minEdge
        self.env.assertTrue(ok or (minEdge is None))

        result_set = self.randomGraph.query("""
            CALL algo.MSF({nodeLabels: ['B', 'C'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("B", "C")
        ok = False
        for w in result_set:
            ok = ok or w[0] == minEdge
        self.env.assertTrue(ok or (minEdge is None))
        
        result_set = self.randomGraph.query("""
            CALL algo.MSF({nodeLabels: ['B', 'D'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("B", "D")
        ok = False
        for w in result_set:
            ok = ok or w[0] == minEdge
        self.env.assertTrue(ok or (minEdge is None))

        result_set = self.randomGraph.query("""
            CALL algo.MSF({nodeLabels: ['C', 'D'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("C", "D")
        ok = False
        for w in result_set:
            ok = ok or w[0] == minEdge
        self.env.assertTrue(ok or (minEdge is None))

