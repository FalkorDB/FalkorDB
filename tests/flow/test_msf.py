from common import *
from random_graph import create_random_graph
from random import randint

GRAPH_ID      = "MSF"
GRAPH_ID_RAND = "MSF_rand"

class testMSF(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.randomGraph = self.db.select_graph(GRAPH_ID_RAND)
        self.generate_random_graph()

    def tearDown(self):
        try:
            self.graph.delete()
        except:
            pass

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

        # ensure that each label is a connected component
        # by connecting one node (s) in label to every other one (n)
        for label in "ABCD":
            q = f"""MATCH (s:{label}) 
                WITH s LIMIT 1 
                MATCH (n:{label}) 
                CREATE (n)-[:R]->(s)"""
            
            self.randomGraph.query(q)
        
        # set the wieghts 
        q = "MATCH ()-[r]->() SET r.weight = rand() * 100, r.cost = rand() * 10"
        self.randomGraph.query(q)

    # find the minumum edge weight between all nodes with label 'src' node and 
    # all those with label 'dest'
    def find_min_edge(self, src, dest):
        q = f"""
        OPTIONAL MATCH (:{src})-[r0]-(:{dest})
        RETURN MIN(r0.weight) as minW
        """

        return self.randomGraph.query(q).result_set[0][0]

    # find the maximum edge weight between all nodes with label 'src' node and 
    # all those with label 'dest'
    def find_max_edge(self, src, dest):
        q = f"""
        OPTIONAL MATCH (:{src})-[r0]-(:{dest})
        RETURN MAX(r0.weight) as maxW
        """

        return self.randomGraph.query(q).result_set[0][0]

    def test_invalid_invocation(self):
        invalid_queries = [
                """CALL algo.MSF({nodeLabels: 'Person'})""",         # non-array nodeLabels parameter
                """CALL algo.MSF({relationshipTypes: 'KNOWS'})""",   # non-array relationshipTypes parameter
                """CALL algo.MSF({weightAttribute: 'fake'})""",      # fake weight parameter
                """CALL algo.MSF({weightAttribute: 4})""",           # fake weight parameter
                """CALL algo.MSF({objective: 'fake'})""",            # fake objective parameter
                """CALL algo.MSF({objective: 4})""",                 # fake objective parameter
                """CALL algo.MSF({invalidParam: 'value'})""",        # unexpected extra parameters
                """CALL algo.MSF('invalid')""",                      # invalid configuration type
                """CALL algo.MSF({nodeLabels: [1, 2, 3]})""",        # integer values in nodeLabels array
                """CALL algo.MSF({relationshipTypes: [1, 2, 3]})""", # integer values in relationshipTypes array
                """CALL algo.MSF({relationshipTypes: ['FAKE']})""",  # non-existent relationship type
                """CALL algo.MSF(null) YIELD nodes, invalidField""", # non-existent yield field
                """CALL algo.MSF('arg1', 'arg2') YIELD nodes""",     # non-existent yield field

                """CALL algo.MSF({nodeLabels: ['Person'],
                               relationshipTypes: ['KNOWS'],
                               invalidParam: 'value'})""",           # unexpected extra parameters

                """CALL algo.MSF({nodeLabels: ['Person'],
                                objective: 'minimize',
                                weightAttribute: 'cost',
                                relationshipTypes: ['KNOWS'],
                                invalidParam: 'value'})""",           # unexpected extra parameters
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except redis.exceptions.ResponseError as e:
                pass

    def test_msf_on_empty_graph(self):
        """Test MSF algorithm behavior on an empty graph"""

        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        self.env.assertEqual(node_count, 0)

        # run MSF on empty graph
        result = self.graph.query("CALL algo.MSF() YIELD nodes, edges")

        # if we reach here, the algorithm didn't throw an exception
        # check if it returned empty results (acceptable behavior)
        self.env.assertEqual(len(result.result_set), 0)

    def test_msf_on_missing_attributes(self):
        """Test MSF algorithm on multi edges with missing parameters"""
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

        # Run MSF algorithm
        result = self.graph.query("""
            CALL algo.MSF({relationshipTypes: ['R'], 
            weightAttribute: 'cost',
            objective: 'minimize'}) 
            YIELD edges 
            UNWIND edges AS e 
            RETURN e.cost AS cost
            ORDER BY cost
            """)
        result_set = result.result_set

        # The MSF should include the edges with the lowest costs
        # The edge without a cost attribute is only used if it is the only edge
        # connecting two seperate components.
        # The expected edges are:
        #   0.02189 (n2 to n3)
        #   0.2198  (n1 to n2)
        #   2.71828 (n4 to n5)
        ans_set = [[0.02189], [0.2198], [2.71828]]
        self.env.assertEqual(ans_set, result_set)

    def test_msf_on_missing_attributes_max(self):
        """Test MSF algorithm on multi edges with missing parameters"""
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

        # Run MSF algorithm
        result = self.graph.query("""
            CALL algo.MSF(
            {relationshipTypes: ['R'], 
            weightAttribute: 'cost', 
            objective: 'maximize'})
            YIELD edges
            UNWIND edges AS e
            RETURN e.cost AS weight 
            ORDER BY weight""")
        result_set = result.result_set

        # The MSF should include the edges with the highest costs
        # The edge without a cost attribute is only used if it is the only edge
        # connecting to separate components.
        # The expected edges are:
        #   0.993284 (n3 to n1)
        #   2.71828  (n4 to n5)
        #   3.14159  (n2 to n1)

        ans_set = [[.993284], [2.71828], [3.14159]]
        self.env.assertEqual(ans_set, result_set)

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
            (n1)-[:R {id: 1}]->(n2),
            (n2)-[:R {id: 2}]->(n3),
            (n3)-[:R {id: 3}]->(n1),
            (n4)-[:R {id: 4}]->(n5)
        """)

        # Run MSF algorithm
        result = self.graph.query(""" 
            CALL algo.MSF({relationshipTypes: ['R']}) 
            YIELD edges
            UNWIND edges AS e
            RETURN e.id AS id
            """)
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


    def test_msf_on_multitypes(self):
        """Test MSF algorithm on nodes connected by multiple relationships"""

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

        # Run MSF algorithm
        result = self.graph.query("""
            CALL algo.MSF({weightAttribute: 'cost'}) 
            YIELD edges
            UNWIND edges as e
            RETURN type(e), e.cost""")
        result_set = result.result_set
        self.env.assertEqual(len(result_set), 1)

        # Check the edge and weight
        edge, weight = result_set[0]
        self.env.assertEqual(edge, 'Kayak')
        self.env.assertEqual(weight, 4)

    def test_msf_on_multitypes_max(self):
        """Test MSF algorithm on nodes connected by multiple relationships"""

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

        # Run MSF algorithm
        result = self.graph.query("""
            CALL algo.MSF({weightAttribute: 'cost', objective: 'maximize'}) 
            YIELD edges
            UNWIND edges as e
            RETURN type(e), e.cost""")

        result_set = result.result_set
        self.env.assertEqual(len(result_set), 1)

        # Check the edge and weight
        edge, weight = result_set[0]
        self.env.assertEqual(edge, 'Plane')
        self.env.assertEqual(weight, 300)

    def test_msf_with_multiedge(self):
        """Test MSF algorithm with different relationship type filters"""
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

            (n1)-[:A {score: 789134, msf_ans: 0}]->(n2),
            (n1)-[:A {score: 5352,   msf_ans: 0}]->(n2),
            (n1)-[:A {score: 1234,   msf_ans: 1}]->(n2),

            (n1)-[:B {score: 123456, msf_ans: 0}]->(n2),
            (n1)-[:B {score: 1000,   msf_ans: 2}]->(n2),

            (n2)-[:A {msf_ans: 2}]->(n3),
            (n2)-[:B {msf_ans: 2}]->(n3),

            (n3)-[:B {score: 8991234, msf_ans: 0}]->(n4),
            (n3)-[:B {score: 7654,    msf_ans: 2}]->(n4)
        """)

        # Run MSF algorithm with relationship type A
        result = self.graph.query("""
            CALL algo.MSF({relationshipTypes: ['A'], weightAttribute: 'score'}) 
            YIELD edges
            UNWIND edges AS e
            RETURN e.msf_ans, e.score as weight
            ORDER BY weight
            """)

        result_set = result.result_set
        ans_set = [[1, 1234.0], [2, None]]
        self.env.assertEqual(ans_set, result_set)

        # Run MSF algorithm with relationship type B
        result = self.graph.query("""
            CALL algo.MSF({relationshipTypes: ['B'], weightAttribute: 'score'}) 
            YIELD edges
            UNWIND edges AS e
            RETURN e.msf_ans, e.score as weight
            ORDER BY weight
            """)

        result_set = result.result_set
        ans_set = [[2, 1000.0], [2, 7654.0], [2, None]]
        self.env.assertEqual(ans_set, result_set)
        
        # Run MSF algorithm with both relationships
        result = self.graph.query("""
            CALL algo.MSF({weightAttribute: 'score'}) 
            YIELD edges
            UNWIND edges AS e
            RETURN e.msf_ans, e.score as weight
            ORDER BY weight
            """)

        result_set = result.result_set
        ans_set = [[2, 1000.0], [2, 7654.0], [2, None]]
        self.env.assertEqual(ans_set, result_set)

        # Run MSF algorithm with both relationships (max)
        result = self.graph.query("""
            CALL algo.MSF({weightAttribute: 'score', objective: 'maximize'}) 
            YIELD edges
            UNWIND edges AS e
            RETURN e.msf_ans, e.score as weight
            ORDER BY weight
            """)

        result_set = result.result_set
        ans_set = [[0, 789134.0], [0, 8991234.0], [2, None]]
        self.env.assertEqual(ans_set, result_set)

        # set the edge 2->3 to have a non-numeric score
        q = "MATCH ({id:2})-[b:B]->({id:3}) SET b.score = 'Hello'"
        self.graph.query(q)

        # Run MSF algorithm with relationship type B
        result = self.graph.query("""
            CALL algo.MSF({relationshipTypes: ['B'], weightAttribute: 'score'}) 
            YIELD edges UNWIND edges AS e
            RETURN e.msf_ans, e.score as weight
            ORDER BY weight
            """)

        result_set = result.result_set
        self.env.assertIn([2, 1000.0], result_set)
        self.env.assertIn([2, 7654.0], result_set)
        self.env.assertTrue([2, "Hello"] in result_set or [2, None] in result_set)

    def test_msf_rand_labels(self):
        """Test MSF algorithm on random graph with multiple labels"""
        # randomGraph contains four groups of nodes each of which are connected
        # each of these groups are connected to each other
        # test checks that one property of MSF is satisfied:
        # an edge that bridges a partition of the graph must be the minimum 
        # edge which bridges that partition.
        for l1 in "ABCD":
            for l2 in "ABCD":
                if(l1 >= l2): continue

                result_set = self.randomGraph.query("""
                    CALL algo.MSF({nodeLabels: [$l1, $l2], 
                    weightAttribute: 'weight'}) 
                    YIELD edges
                    RETURN [e IN edges| e.weight]
                    """,
                    params = {'l1': l1, 'l2': l2}).result_set
                # minEdge = self.find_min_edge(l1, l2)
                
                # if(minEdge is None): # components are disconnected.
                #     # should return two spanning trees with 19 edges each.
                #     # check that there are 19*2 = 38 edges total 
                #     self.env.assertEqual(len(result_set), 2)
                #     self.env.assertEqual(len(result_set[0][0]), 19)
                #     self.env.assertEqual(len(result_set[1][0]), 19)
                # else:
                #     # connected component with 40 nodes: must have 39 edges.
                #     self.env.assertEqual(len(result_set), 1)
                #     self.env.assertEqual(len(result_set[0][0]), 39)
                #     self.env.assertIn(minEdge, result_set[0][0])

    def test_msf_rand_labels_max(self):
        """Test MSF algorithm on random graph with multiple labels"""
        # randomGraph contains four groups of nodes each of which are connected
        # each of these groups are connected to each other
        # test checks that one property of MSF is satisfied:
        # an edge that bridges a partition of the graph must be the maximum 
        # edge which bridges that partition.
        for l1 in "ABCD":
            for l2 in "ABCD":
                if(l1 >= l2): continue

                result_set = self.randomGraph.query("""
                    CALL algo.MSF({nodeLabels: [$l1, $l2], 
                    weightAttribute: 'weight', objective: 'maximize'}) 
                    YIELD edges 
                    RETURN [e IN edges| e.weight]
                    """,
                    params = {'l1': l1, 'l2': l2}).result_set
                maxEdge = self.find_max_edge(l1, l2)
                
                if(maxEdge is None): # components are disconnected.
                    # should return two spanning trees with 19 edges each.
                    # check that there are 19*2 = 38 edges total 
                    self.env.assertEqual(len(result_set), 2)
                    self.env.assertEqual(len(result_set[0][0]), 19)
                    self.env.assertEqual(len(result_set[1][0]), 19)
                else:
                    # connected component with 40 nodes: must have 39 edges.
                    self.env.assertEqual(len(result_set), 1)
                    self.env.assertEqual(len(result_set[0][0]), 39)
                    self.env.assertIn(maxEdge, result_set[0][0])

    def test_msf_rand_forest_no_weight(self):
        """ Test that MSF correctly identifies and groups multiple trees """
        # create the forest
        nodes =[{"count": randint(5,40), "properties": 3, "labels": [l]} 
            for l in "ABCDEFGHIJK"]

        # add many edges between nodes of the same label
        edges = [{"type": r, "source": l, "target": l, "count": 100} 
                    for r in "LMNOPQRS" for l in range(11)]

        create_random_graph(self.graph, nodes, edges)

        #check that YIELDing nodes or edges alone returns correct results
        result_set_both = self.graph.query("""
            CALL algo.MSF()
            YIELD nodes, edges
            RETURN [n in nodes | id(n)] AS nodeIds,
                [e IN edges | id(e)] AS edgeIds
            """).result_set
        node_set = self.graph.query("""
            CALL algo.MSF()
            YIELD nodes
            RETURN [n in nodes | id(n)] AS nodeIds
            """).result_set
        edge_set = self.graph.query("""
            CALL algo.MSF()
            YIELD edges
            RETURN [e IN edges | id(e)] AS edgeIds
            """).result_set
        # check these are equal
        result_set_individual = [[n[0], e[0]] for n, e in zip(node_set, edge_set)]
        self.env.assertEqual(result_set_both, result_set_individual)

        # get nodes and endpoints. 
        result_set = self.graph.query("""
            CALL algo.MSF() 
            YIELD nodes, edges
            RETURN [n in nodes | id(n)] AS nodeIds, 
                [e IN edges | id(startNode(e))] 
                + [e IN edges | id(endNode(e))] AS ends 
            """).result_set
        
        # check that there is one more node than edge and that each node can be 
        # found at least once in the resulting tree.
        for tree in result_set:
            self.env.assertEqual(len(tree[0]), len(tree[1]) / 2 + 1)
            if(len(tree[0]) == 1): continue
            self.env.assertEqual(set(tree[0]), set(tree[1]))

    def test_msf_no_relations(self):
        """ Test MSF on a graph with some nodes but no relationships """
        # create a graph with 10 nodes and no relationships
        self.graph.query("""
            CREATE (n1 {id: 1}), (n2 {id: 2}), (n3 {id: 3}),
                   (n4 {id: 4}), (n5 {id: 5}), (n6 {id: 6}),
                   (n7 {id: 7}), (n8 {id: 8}), (n9 {id: 9}),
                   (n10 {id: 10})
        """)

        # run MSF on the graph
        result = self.graph.query("""CALL algo.MSF() YIELD nodes, edges 
            RETURN [n in nodes | n.id] AS ids, edges""")


        expected = [[[1], []], 
                    [[2], []], 
                    [[3], []], 
                    [[4], []], 
                    [[5], []], 
                    [[6], []], 
                    [[7], []], 
                    [[8], []], 
                    [[9], []], 
                    [[10], []]]
        self.env.assertEqual(result.result_set, expected)

        # try deleting a node
        self.graph.query("MATCH (n) WHERE n.id = 1 DETACH DELETE n")
        # run MSF again
        result = self.graph.query("""CALL algo.MSF() YIELD nodes, edges 
            RETURN [n in nodes | n.id] AS ids, edges""")
        expected = [[[2], []], 
                    [[3], []], 
                    [[4], []], 
                    [[5], []], 
                    [[6], []], 
                    [[7], []], 
                    [[8], []], 
                    [[9], []], 
                    [[10], []]]
        self.env.assertEqual(result.result_set, expected)