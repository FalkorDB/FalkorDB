from common import *

GRAPH_ID = "MSF"

class testMSF(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

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

    def test_msf_on_empty_graph(self):
        """Test MSF algorithm behavior on an empty graph"""
        
        # run MSF on empty graph
        result = self.graph.query("CALL algo.MSF(null)")
        # if we reach here, the algorithm didn't throw an exception
        
        # check if it returned empty results (acceptable behavior)
        self.env.assertEqual(len(result.result_set), 0)

    def test_msf_on_unlabeled_graph(self):
        """Test MSF algorithm on unlabeled nodes with multiple connected components"""
        print("test_msf_on_unlabeled_graph")
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

    def test_msf_on_multilable(self):
        """Test MSF algorithm on multilabled nodes with multiple connected components"""
        print("test_msf_on_multilable")

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
            yield edge, weight""")
        result_set = result.result_set
        self.env.assertEqual(len(result_set), 1)
        # Check the edge and weight
        edge, weight = result_set[0]
        print(edge)
        self.env.assertEqual(weight, 4)
        # self.env.assertEqual(edge.end_node.id, 2)


    def test_msf_with_multiedge(self):
        """Test WCC algorithm with different relationship type filters"""
        print("test_msf_with_multiedge")
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

    # def test_wcc_with_different_node_labels(self):
    #     """Test WCC algorithm with different node label filters"""

    #     # Create a graph with multiple labeled and unlabeled nodes
    #     # - L0 labeled nodes: n1, n2, n3
    #     # - L1 labeled nodes: n4, n5, n6
    #     # - Unlabeled nodes: n7, n8
    #     # All connected with relationship type R in a specific pattern
    #     self.graph.query("""
    #         CREATE
    #         (n1:L0 {id: 1}),
    #         (n2:L0 {id: 2}),
    #         (n3:L0 {id: 3}),
    #         (n4:L1 {id: 4}),
    #         (n5:L1 {id: 5}),
    #         (n6:L1 {id: 6}),
    #         (n7 {id: 7}),
    #         (n8 {id: 8}),
    #         (n1)-[:R]->(n2),
    #         (n2)-[:R]->(n3),
    #         (n3)-[:R]->(n4),
    #         (n4)-[:R]->(n5),
    #         (n5)-[:R]->(n6),
    #         (n6)-[:R]->(n7),
    #         (n7)-[:R]->(n8)
    #     """)


    #     # Test 1: Using only nodes with label L0
    #     result_L0 = self.graph.query("CALL algo.WCC({nodeLabels: ['L0']})")
    #     components_L0 = get_components(result_L0)

    #     # With only L0 nodes, we expect one component with nodes 1,2,3
    #     self.env.assertEqual(len(components_L0), 1)
    #     self.env.assertEqual(components_L0[0], [1, 2, 3])

    #     # Test 2: Using only nodes with label L1
    #     result_L1 = self.graph.query("CALL algo.WCC({nodeLabels: ['L1']})")
    #     components_L1 = get_components(result_L1)

    #     # With only L1 nodes, we expect one component with nodes 4,5,6
    #     self.env.assertEqual(len(components_L1), 1)
    #     self.env.assertEqual(components_L1[0], [4, 5, 6])

    #     # Test 3: Using nodes with both labels L0 and L1
    #     result_L0_L1 = self.graph.query("CALL algo.WCC({nodeLabels: ['L0', 'L1']})")
    #     components_L0_L1 = get_components(result_L0_L1)

    #     # With both L0 and L1 nodes, we expect one component with nodes 1-6
    #     self.env.assertEqual(len(components_L0_L1), 1)
    #     self.env.assertEqual(components_L0_L1[0], [1, 2, 3, 4, 5, 6])

    #     # Test 4: Using all nodes (null parameter)
    #     result_all = self.graph.query("CALL algo.WCC(null)")
    #     components_all = get_components(result_all)

    #     # With all nodes, we expect one component with all nodes 1-8
    #     self.env.assertEqual(len(components_all), 1)
    #     self.env.assertEqual(components_all[0], [1, 2, 3, 4, 5, 6, 7, 8])

    # def test_wcc_with_combined_labels_and_relationships(self):
    #     """Test WCC algorithm with combinations of node labels and relationship types"""

    #     # Create a complex graph with:
    #     # - Multiple node labels: L1, L2, and unlabeled nodes
    #     # - Multiple relationship types: R1, R2
    #     # The graph structure forms several potential connected components depending on filters
    #     self.graph.query("""
    #         CREATE
    #         (n1:L1 {id: 1}),
    #         (n2:L1 {id: 2}),
    #         (n3:L1 {id: 3}),
    #         (n4:L2 {id: 4}),
    #         (n5:L2 {id: 5}),
    #         (n6:L2 {id: 6}),
    #         (n7 {id: 7}),
    #         (n8 {id: 8}),
    #         (n9:L1:L2 {id: 9}),  // Node with both labels

    #         // R1 relationships
    #         (n1)-[:R1]->(n2),
    #         (n2)-[:R1]->(n3),
    #         (n4)-[:R1]->(n5),

    #         // R2 relationships
    #         (n3)-[:R2]->(n4),  // Bridge between L1 and L2 clusters
    #         (n5)-[:R2]->(n6),
    #         (n6)-[:R2]->(n7),
    #         (n7)-[:R2]->(n8),

    #         // Connections to the dual-labeled node
    #         (n3)-[:R1]->(n9),
    #         (n9)-[:R2]->(n6)
    #     """)

    #     # Test Cases:

    #     # Test 1: Only label L1 and only relationship type R1
    #     result = self.graph.query("CALL algo.WCC({nodeLabels: ['L1'], relationshipTypes: ['R1']})")
    #     components = get_components(result)
    #     # Expected: One component with nodes 1,2,3,9
    #     self.env.assertEqual(len(components), 1)
    #     self.env.assertEqual(components, [[1, 2, 3, 9]])

    #     # Test 2: Only label L2 and only relationship type R1
    #     result = self.graph.query("CALL algo.WCC({nodeLabels: ['L2'], relationshipTypes: ['R1']})")
    #     components = get_components(result)
    #     # Expected: Components should be [4,5], [6] and [9] as separate components
    #     # First, check the total number of components
    #     self.env.assertEqual(len(components), 3)  # Should have 3 components: [4,5], [6], and [9]
    #     # Verify each expected component exists
    #     self.env.assertTrue([4, 5] in components)
    #     self.env.assertTrue([6] in components)
    #     self.env.assertTrue([9] in components)

    #     # Test 3: Labels L1 and L2, only relationship type R1
    #     result = self.graph.query("CALL algo.WCC({nodeLabels: ['L1', 'L2'], relationshipTypes: ['R1']})")
    #     components = get_components(result)
    #     # Expected: Two separate components: [1,2,3,9] and [4,5]
    #     # First check number of components
    #     self.env.assertEqual(len(components), 3)

    #     # Three separate components
    #     self.env.assertEqual(components, [[6], [4, 5], [1, 2, 3, 9]])

    #     # Test 4: All labels, only relationship type R2
    #     result = self.graph.query("CALL algo.WCC({relationshipTypes: ['R2']})")
    #     components = get_components(result)
    #     # Expected components: [1], [2], [3,4], [5,6,7,8,9]
    #     self.env.assertEqual(len(components), 4)
    #     # Verify the exact components
    #     self.env.assertEqual(components, [[1], [2], [3, 4], [5, 6, 7, 8, 9]])

    #     # Test 5: Only label L1, both relationship types
    #     result = self.graph.query("CALL algo.WCC({nodeLabels: ['L1'], relationshipTypes: ['R1', 'R2']})")
    #     components = get_components(result)
    #     # Expected: One component with nodes 1,2,3,9
    #     self.env.assertEqual(components, [[1, 2, 3, 9]])

    #     # Test 6: All node labels, all relationship types
    #     result = self.graph.query("CALL algo.WCC(null)")
    #     components = get_components(result)
    #     # Expected: One component with all nodes 1-9
    #     self.env.assertEqual(len(components), 1)
    #     self.env.assertEqual(components[0], [1, 2, 3, 4, 5, 6, 7, 8, 9])

# TODO: make sure we can run WCC on a random generated graph
# test to the best of our abilities e.g. when specifying a label make sure all ndoes
# under that labels show up

