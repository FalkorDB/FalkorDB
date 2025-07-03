from common import *
import random

GRAPH_ID = "MST"
GRAPH_ID_RAND = "MST_rand"

class testMST(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.randomGraph = self.db.select_graph(GRAPH_ID_RAND)
        self.generate_random_graph_cypher()

    def tearDown(self):
        self.graph.delete()

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

    # find the minumum edge weight between 'src' node and 'dest' node
    def find_min_edge(self, src, dest):
        q = f"""
        OPTIONAL MATCH (:{src})-[r0]->(:{dest})
        WITH MIN(r0.weight) as min_r0
        OPTIONAL MATCH (:{dest})-[r1]->(:{src})
        WITH MIN(r1.weight)) as min_r1
        WITH [min_r0, min_r1] AS mins
        UNWIND mins AS x
        RETURN MIN(x) as minW
        """

        return self.randomGraph.query(q).result_set[0][0]

    # find the maximum edge weight between 'src' node and 'dest' node
    def find_max_edge(self, src, dest):
        q = f"""
        OPTIONAL MATCH (:{src})-[r0]->(:{dest})
        WITH MAX(r0.weight) as min_r0
        OPTIONAL MATCH (:{dest})-[r1]->(:{src})
        WITH MAX(r1.weight)) as min_r1
        WITH [min_r0, min_r1] AS mins
        UNWIND mins AS x
        RETURN MAX(x) as minW
        """

        return self.randomGraph.query(q).result_set[0][0]

    def test_invalid_invocation(self):
        invalid_queries = [
                """CALL algo.MST({nodeLabels: 'Person'})""",         # non-array nodeLabels parameter
                """CALL algo.MST({relationshipTypes: 'KNOWS'})""",   # non-array relationshipTypes parameter
                """CALL algo.MST({invalidParam: 'value'})""",        # unexpected extra parameters
                """CALL algo.MST('invalid')""",                      # invalid configuration type
                """CALL algo.MST({nodeLabels: [1, 2, 3]})""",        # integer values in nodeLabels array
                """CALL algo.MST({relationshipTypes: [1, 2, 3]})""", # integer values in relationshipTypes array
                """CALL algo.MST(null) YIELD node, invalidField""",  # non-existent yield field

                """CALL algo.MST({nodeLabels: ['Person'],
                               relationshipTypes: ['KNOWS'],
                               invalidParam: 'value'})""",           # unexpected extra parameters
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except redis.exceptions.ResponseError as e:
                pass

    def test_mst_on_empty_graph(self):
        """Test MST algorithm behavior on an empty graph"""
        
        # run MST on empty graph
        result = self.graph.query("CALL algo.MST()")

        # if we reach here, the algorithm didn't throw an exception
        # check if it returned empty results (acceptable behavior)
        self.env.assertEqual(len(result.result_set), 0)

    def test_mst_on_missing_attributes(self):
        """Test MST algorithm on unlabeled nodes with multiple connected components"""
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
            (n1)-[:R {cost: .2198}]->(n2),
            (n1)<-[:R]-(n2),
            (n1)<-[:R {cost: 3.14159}]-(n2),
            (n2)-[:R {cost: .02189}]->(n3),
            (n3)-[:R {cost: .993284}]->(n1),
            (n4)-[:R {cost: 2.71828}]->(n5)
        """)

        # Run MST algorithm
        result = self.graph.query("""
            CALL algo.MST({relationshipTypes: ['R'], weightAttribute: 'cost'}) 
            YIELD weight RETURN weight ORDER BY weight""")
        result_set = result.result_set

        ans_set = [[0.02189], [0.2198], [2.71828]]
        self.env.assertEqual(ans_set, result_set)

    def test_mst_on_missing_attributes_max(self):
        """Test MST algorithm on unlabeled nodes with multiple connected components"""
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
            (n1)-[:R {cost: .2198}]->(n2),
            (n1)<-[:R {}]-(n2),
            (n1)<-[:R {cost: 3.14159}]-(n2),
            (n2)-[:R {cost: .02189}]->(n3),
            (n3)-[:R {cost: .993284}]->(n1),
            (n4)-[:R {cost: 2.71828}]->(n5)
        """)

        # Run MST algorithm
        result = self.graph.query("""
            CALL algo.MST({relationshipTypes: ['R'], weightAttribute: 'cost', objective: 'max'})
            YIELD weight RETURN weight ORDER BY weight""")
        result_set = result.result_set
        ans_set = [[.993284], [2.71828], [3.14159]]
        self.env.assertEqual(ans_set, result_set)

    def test_mst_on_unlabeled_graph(self):
        """Test MST algorithm on unlabeled nodes with multiple connected components"""
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

        # Run MST algorithm
        result = self.graph.query("""CALL algo.MST({relationshipTypes: ['R']}) yield edge""")
        result_set = result.result_set

        # We should have exactly 3 different edges 
        self.env.assertEqual(len(result_set), 3)

    def test_mst_on_multilable(self):
        """Test MST algorithm on multilabled nodes with multiple connected components"""

        # Create an unlabeled graph with multiple connected components
        # - Component 1: nodes 1-2 are connected with multiple edges
        self.graph.query("""
            CREATE
            (n1 {id: 1}),
            (n2 {id: 2}),
            (n1)-[:Car {distance: 11.2, cost: 23}]->(n2),
            (n1)-[:Walk {distance: 9.24321, cost: 7}]->(n2),
            (n1)-[:Plane {distance: 123.2490, cost: 300}]->(n2),
            (n1)-[:Boat {distance: .75, cost: 100}]->(n2),
            (n1)<-[:Swim {distance: .5, cost: 12}]-(n2),
            (n1)<-[:Kayak {distance: .68, cost: 4}]-(n2)
        """)

        # Run MST algorithm
        result = self.graph.query("""CALL algo.MST({weightAttribute: 'cost'}) 
            yield edge, weight return type(edge), weight""")
        result_set = result.result_set
        self.env.assertEqual(len(result_set), 1)

        # Check the edge and weight
        edge, weight = result_set[0]
        self.env.assertEqual(edge, 'Kayak')
        self.env.assertEqual(weight, 4)

    def test_mst_on_multilable_max(self):
        """Test MST algorithm on multilabled nodes with multiple connected components"""

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

        # Run MST algorithm
        result = self.graph.query("""
            CALL algo.MST({weightAttribute: 'cost', objective: 'max'}) 
            yield edge, weight return type(edge), weight""")

        result_set = result.result_set
        self.env.assertEqual(len(result_set), 1)

        # Check the edge and weight
        edge, weight = result_set[0]
        self.env.assertEqual(edge, 'Plane')
        self.env.assertEqual(weight, 300)

    def test_mst_with_multiedge(self):
        """Test MST algorithm with different relationship type filters"""
        # Create an unlabeled graph with two relationship types
        # Relationship type A:
        # - n1-A->n2 (x3 edges)
        # Relationship type B:
        # - n1-B->n2 (x2 edges)
        # - n2-B->n3 (x1 edge without score attributes)
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
            (n1)-[:A {score: 789134, mst_ans: 0}]->(n2),
            (n1)-[:A {score: 5352, mst_ans: 0}]->(n2),
            (n1)-[:A {score: 1234, mst_ans: 1}]->(n2),
            (n1)-[:B {score: 123456, mst_ans: 0}]->(n2),
            (n1)-[:B {score: 1000, mst_ans: 2}]->(n2),
            (n2)-[:B {mst_ans: 0}]->(n3),
            (n3)-[:B {score: 8991234, mst_ans: 0}]->(n4),
            (n3)-[:B {score: 7654, mst_ans: 2}]->(n4)
        """)

        # Run MST algorithm with relationship type A
        result = self.graph.query("""
            CALL algo.MST({relationshipTypes: ['A'], weightAttribute: 'score'}) 
            YIELD edge RETURN edge.mst_ans
            """)

        result_set = result.result_set
        self.env.assertEqual(len(result_set), 1)
        for edge in result_set:
            self.env.assertEqual(edge[0], 1)

        # Run MST algorithm with relationship type B
        result = self.graph.query("""
            CALL algo.MST({relationshipTypes: ['B'], weightAttribute: 'score'}) 
            YIELD edge RETURN edge.mst_ans
            """)

        result_set = result.result_set
        self.env.assertEqual(len(result_set), 2)
        for edge in result_set:
            self.env.assertEqual(edge[0], 2)
        
        # Run MST algorithm with both relationships
        result = self.graph.query("""
            CALL algo.MST({weightAttribute: 'score'}) 
            YIELD edge RETURN edge.mst_ans
            """)

        result_set = result.result_set
        self.env.assertEqual(len(result_set), 2)
        for edge in result_set:
            self.env.assertEqual(edge[0], 2)

    def test_mst_rand_labels(self):
        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['A', 'B'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("A", "B")

        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge], result_set)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['A', 'C'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("A", "C")
        
        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge], result_set)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['A', 'D'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("A", "D")
        
        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge], result_set)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['B', 'C'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("B", "C")
        
        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge], result_set)
        
        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['B', 'D'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("B", "D")
        
        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge], result_set)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['C', 'D'], weightAttribute: 'weight'}) 
            YIELD weight
            """).result_set
        minEdge = self.find_min_edge("C", "D")
        
        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge], result_set)

    def test_mst_rand_labels_max(self):
        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['A', 'B'], weightAttribute: 'weight', objective: 'maximize'}) 
            YIELD weight
            """).result_set
        maxEdge = self.find_max_edge("A", "B")

        if(maxEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([maxEdge], result_set)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['A', 'C'], weightAttribute: 'weight', objective: 'maximize'}) 
            YIELD weight
            """).result_set
        maxEdge = self.find_max_edge("A", "C")
        
        if(maxEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([maxEdge], result_set)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['A', 'D'], weightAttribute: 'weight', objective: 'maximize'}) 
            YIELD weight
            """).result_set
        maxEdge = self.find_max_edge("A", "D")
        
        if(maxEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([maxEdge], result_set)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['B', 'C'], weightAttribute: 'weight', objective: 'maximize'}) 
            YIELD weight
            """).result_set
        maxEdge = self.find_max_edge("B", "C")
        
        if(maxEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([maxEdge], result_set)
        
        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['B', 'D'], weightAttribute: 'weight', objective: 'maximize'}) 
            YIELD weight
            """).result_set
        maxEdge = self.find_max_edge("B", "D")
        
        if(maxEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([maxEdge], result_set)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['C', 'D'], weightAttribute: 'weight', objective: 'maximize'}) 
            YIELD weight
            """).result_set
        maxEdge = self.find_max_edge("C", "D")
        
        if(maxEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([maxEdge], result_set)

