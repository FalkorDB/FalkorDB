from common import *
from random_graph import create_random_graph
import random
from math import nan 

GRAPH_ID = "MST"
GRAPH_ID_RAND = "MST_rand"

class testMST(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.randomGraph = self.db.select_graph(GRAPH_ID_RAND)
        self.generate_random_graph()

    def tearDown(self):
        self.graph.query("MATCH (a) RETURN count(a)")
        self.graph.delete()

    def generate_random_graph(self):
        # nodes of four different labels, each with 20 nodes
        nodes =[{"count": 20, "properties": 3, "labels": [l]} for l in "ABCD"]
        
        # add many edges between nodes of the same label
        edges = [{"type": "R", "source": l, "target": l, "count": 100} 
                    for l in range(4)]

        # ensure there are edges between nodes of different labels
        for l1 in range(4):
            for l2 in range(4):
                    edges.append(
                        {"type": "S", "source": l1, "target": l2, "count": 10})

        # call random graph
        create_random_graph(self.randomGraph, nodes, edges)

        # ensure that each label is connected
        for label in "ABCD":
            q = f"""MATCH (s:{label}) 
                WITH s LIMIT 1 
                MATCH (n:{label}) 
                CREATE (n)-[:R]->(s)"""
            
            self.randomGraph.query(q)
        
        # set the wieghts 
        q = "MATCH ()-[r]->() SET r.weight = rand() * 100, r.cost = rand() * 10"
        self.randomGraph.query(q)

    # find the minumum edge weight between 'src' node and 'dest' node
    def find_min_edge(self, src, dest):
        q = f"""
        OPTIONAL MATCH (:{src})-[r0]->(:{dest})
        WITH MIN(r0.weight) as min_r0
        OPTIONAL MATCH (:{dest})-[r1]->(:{src})
        WITH min_r0, MIN(r1.weight) as min_r1
        WITH [min_r0, min_r1] AS mins
        UNWIND mins AS x
        RETURN MIN(x) as minW
        """

        return self.randomGraph.query(q).result_set[0][0]

    # find the maximum edge weight between 'src' node and 'dest' node
    def find_max_edge(self, src, dest):
        q = f"""
        OPTIONAL MATCH (:{src})-[r0]->(:{dest})
        WITH MAX(r0.weight) as max_r0
        OPTIONAL MATCH (:{dest})-[r1]->(:{src})
        WITH max_r0, MAX(r1.weight) as max_r1
        WITH [max_r0, max_r1] AS maxs
        UNWIND maxs AS x
        RETURN MAX(x) as maxW
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
                """CALL algo.MST({relationshipTypes: ['FAKE']})""",  # fake values in relationshipTypes array
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
        """Test MST algorithm on multi edges with missing parameters"""
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

        # The MST should include the edges with the lowest costs
        # The edge without a cost attribute is only used if it is the only edge
        # connecting to seperate components.
        # The expected edges are:
        #   .02189 (n2 to n3)
        #   .2198 (n1 to n2)
        #   2.71828 (n4 to n5)
        ans_set = [[0.02189], [0.2198], [2.71828]]
        self.env.assertEqual(ans_set, result_set)

    def test_mst_on_missing_attributes_max(self):
        """Test MST algorithm on multi edges with missing parameters"""
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
            CALL algo.MST({relationshipTypes: ['R'], weightAttribute: 'cost', objective: 'maximize'})
            YIELD weight RETURN weight ORDER BY weight""")
        result_set = result.result_set

        # The MST should include the edges with the highest costs
        # The edge without a cost attribute is only used if it is the only edge
        # connecting to separate components.
        # The expected edges are:
        #   .993284 (n3 to n1)
        #   2.71828 (n4 to n5)
        #   3.14159 (n2 to n1)

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
            (n1)-[:R {id: 1}]->(n2),
            (n2)-[:R {id: 2}]->(n3),
            (n3)-[:R {id: 3}]->(n1),
            (n4)-[:R {id: 4}]->(n5)
        """)

        # Run MST algorithm
        result = self.graph.query(""" CALL algo.MST({relationshipTypes: ['R']}) 
            YIELD edge RETURN edge.id""")
        result_set = result.result_set

        # We should have exactly 3 different edges 
        self.env.assertEqual(len(result_set), 3)
        self.env.assertContains([4], result_set)
        
        # two edges in the cycle should be present
        cycle_count = 0
        for edge in result_set:
            if edge[0] in [1, 2, 3]:
                cycle_count += 1
        self.env.assertEqual(cycle_count, 2)


    def test_mst_on_multitypes(self):
        """Test MST algorithm on nodes connected by multiple relationships"""

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

    def test_mst_on_multitypes_max(self):
        """Test MST algorithm on nodes connected by multiple relationships"""

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
        result = self.graph.query("""
            CALL algo.MST({weightAttribute: 'cost', objective: 'maximize'}) 
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
            (n1)-[:A {score: 789134, mst_ans: 0}]->(n2),
            (n1)-[:A {score: 5352, mst_ans: 0}]->(n2),
            (n1)-[:A {score: 1234, mst_ans: 1}]->(n2),
            (n1)-[:B {score: 123456, mst_ans: 0}]->(n2),
            (n1)-[:B {score: 1000, mst_ans: 2}]->(n2),
            (n2)-[:B {mst_ans: 2}]->(n3),
            (n3)-[:B {score: 8991234, mst_ans: 0}]->(n4),
            (n3)-[:B {score: 7654, mst_ans: 2}]->(n4)
        """)

        # Run MST algorithm with relationship type A
        result = self.graph.query("""
            CALL algo.MST({relationshipTypes: ['A'], weightAttribute: 'score'}) 
            YIELD edge, weight RETURN edge.mst_ans, weight
            ORDER BY weight
            """)

        result_set = result.result_set
        ans_set = [[1, 1234.0]]
        self.env.assertEqual(ans_set, result_set)

        # Run MST algorithm with relationship type B
        result = self.graph.query("""
            CALL algo.MST({relationshipTypes: ['B'], weightAttribute: 'score'}) 
            YIELD edge, weight RETURN edge.mst_ans, weight
            ORDER BY weight
            """)

        result_set = result.result_set
        ans_set = [[2, 1000.0], [2, 7654.0], [2, None]]
        self.env.assertEqual(ans_set, result_set)
        
        # Run MST algorithm with both relationships
        result = self.graph.query("""
            CALL algo.MST({weightAttribute: 'score'}) 
            YIELD edge, weight RETURN edge.mst_ans, weight
            ORDER BY weight
            """)

        result_set = result.result_set
        ans_set = [[2, 1000.0], [2, 7654.0], [2, None]]
        self.env.assertEqual(ans_set, result_set)

         # Run MST algorithm with both relationships (max)
        result = self.graph.query("""
            CALL algo.MST({weightAttribute: 'score', objective: 'maximize'}) 
            YIELD edge, weight
            RETURN edge.mst_ans, weight
            ORDER BY weight
            """)

        result_set = result.result_set
        ans_set = [[0, 789134.0], [0, 8991234.0], [2, None]]
        self.env.assertEqual(ans_set, result_set)

        # set the edge 2->3 to have a non-numeric score
        q = "MATCH ({id:2})-[b:B]->({id:3}) SET b.score = 'Hello'"
        self.graph.query(q)

        # Run MST algorithm with relationship type B
        result = self.graph.query("""
            CALL algo.MST({relationshipTypes: ['B'], weightAttribute: 'score'}) 
            YIELD edge, weight 
            RETURN edge.mst_ans, weight
            ORDER BY weight
            """)

        result_set = result.result_set
        ans_set = [[2, 1000.0], [2, 7654.0], [2, None]]
        self.env.assertEqual(ans_set, result_set)

    def test_mst_rand_labels(self):
        """Test MST algorithm on random graph with multiple labels"""
        # randomGraph contains four groups of nodes each of which are connected
        # each of these groups are connected to each other
        # test checks that one property of MST is satisfied:
        # an edge that bridges a partition of the graph must be the minimum 
        # edge which bridges that partition.
        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['A', 'B'], weightAttribute: 'weight'}) 
            YIELD edge, weight RETURN edge.weight, weight
            """).result_set
        minEdge = self.find_min_edge("A", "B")

        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            # connected component with 40 nodes: must have 39 edges.
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge, minEdge], result_set)

        for w1, w2 in result_set:
            # check that the edge given has the correct weight
            self.env.assertEqual(w1, w2)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['A', 'C'], weightAttribute: 'weight'}) 
            YIELD edge, weight RETURN edge.weight, weight
            """).result_set
        minEdge = self.find_min_edge("A", "C")
        
        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            # connected component with 40 nodes: must have 39 edges.
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge, minEdge], result_set)

        for w1, w2 in result_set:
            # check that the edge given has the correct weight
            self.env.assertEqual(w1, w2)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['A', 'D'], weightAttribute: 'weight'}) 
            YIELD edge, weight RETURN edge.weight, weight
            """).result_set
        minEdge = self.find_min_edge("A", "D")
        
        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            # connected component with 40 nodes: must have 39 edges.
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge, minEdge], result_set)

        for w1, w2 in result_set:
            # check that the edge given has the correct weight
            self.env.assertEqual(w1, w2)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['B', 'C'], weightAttribute: 'weight'}) 
            YIELD edge, weight RETURN edge.weight, weight
            """).result_set
        minEdge = self.find_min_edge("B", "C")
        
        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            # connected component with 40 nodes: must have 39 edges.
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge, minEdge], result_set)

        for w1, w2 in result_set:
            # check that the edge given has the correct weight
            self.env.assertEqual(w1, w2)
        
        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['B', 'D'], weightAttribute: 'weight'}) 
            YIELD edge, weight RETURN edge.weight, weight
            """).result_set
        minEdge = self.find_min_edge("B", "D")
        
        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            # connected component with 40 nodes: must have 39 edges.
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge, minEdge], result_set)

        for w1, w2 in result_set:
            # check that the edge given has the correct weight
            self.env.assertEqual(w1, w2)

        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['C', 'D'], weightAttribute: 'weight'}) 
            YIELD edge, weight RETURN edge.weight, weight
            """).result_set
        minEdge = self.find_min_edge("C", "D")
        
        if(minEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            # connected component with 40 nodes: must have 39 edges.
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([minEdge, minEdge], result_set)

        for w1, w2 in result_set:
            # check that the edge given has the correct weight
            self.env.assertEqual(w1, w2)

    def test_mst_rand_labels_max(self):
        """Test MST algorithm on random graph with multiple labels"""
        # randomGraph contains four groups of nodes each of which are connected
        # each of these groups are connected to each other
        # test checks that one property of MST is satisfied:
        # an edge that bridges a partition of the graph must be the minimum 
        # edge which bridges that partition.
        result_set = self.randomGraph.query("""
            CALL algo.MST({nodeLabels: ['A', 'B'], weightAttribute: 'weight', objective: 'maximize'}) 
            YIELD weight
            """).result_set
        maxEdge = self.find_max_edge("A", "B")

        if(maxEdge is None): #components are disconnected. 
            #Should return two spanning trees with 19 edges each.
            self.env.assertEqual(len(result_set), 38)
        else:
            # connected component with 40 nodes: must have 39 edges.
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
            # connected component with 40 nodes: must have 39 edges.
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
            # connected component with 40 nodes: must have 39 edges.
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
            # connected component with 40 nodes: must have 39 edges.
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
            # connected component with 40 nodes: must have 39 edges.
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
            # connected component with 40 nodes: must have 39 edges.
            self.env.assertEqual(len(result_set), 39)
            self.env.assertIn([maxEdge], result_set)

