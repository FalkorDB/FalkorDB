from common import *
from random_graph import create_random_schema, create_random_graph

GRAPH_ID = "WCC"

# Helper function to extract components from query results
def get_components(result):
    components = {}
    for record in result.result_set:
        node = record[0]
        component_id = record[1]
        node_id = node.properties['id']

        if component_id not in components:
            components[component_id] = []
        components[component_id].append(node_id)

    # Sort node IDs within each component
    for component_id in components:
        components[component_id].sort()

    # Convert to sorted lists for easier comparison
    component_sets = sorted(components.values(), key=lambda x: (len(x), x))
    return component_sets

class testWCC(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def test_invalid_invocation(self):
        invalid_queries = [
                """CALL algo.WCC({nodeLabels: 'Person'})""",         # non-array nodeLabels parameter
                """CALL algo.WCC({relationshipTypes: 'KNOWS'})""",   # non-array relationshipTypes parameter
                """CALL algo.WCC({invalidParam: 'value'})""",        # unexpected extra parameters
                """CALL algo.WCC('invalid')""",                      # invalid configuration type
                """CALL algo.WCC({nodeLabels: [1, 2, 3]})""",        # integer values in nodeLabels array
                """CALL algo.WCC({relationshipTypes: [1, 2, 3]})""", # integer values in relationshipTypes array
                """CALL algo.WCC(null) YIELD node, invalidField""",  # non-existent yield field

                """CALL algo.WCC({nodeLabels: ['Person'],
                               relationshipTypes: ['KNOWS'],
                               invalidParam: 'value'})""",           # unexpected extra parameters
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except:
                pass

    def test_wcc_on_empty_graph(self):
        """Test WCC algorithm behavior on an empty graph"""
        
        # run WCC on empty graph
        result = self.graph.query("CALL algo.WCC(null)")
        # if we reach here, the algorithm didn't throw an exception
        
        # check if it returned empty results (acceptable behavior)
        self.env.assertEqual(len(result.result_set), 0)

    def test_wcc_on_unlabeled_graph(self):
        """Test WCC algorithm on unlabeled nodes with multiple connected components"""

        # Create an unlabeled graph with multiple connected components
        # - Component 1: nodes 1-2-3 forming a path
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
            (n4)-[:R]->(n5)
        """)

        # Run WCC algorithm
        result = self.graph.query("CALL algo.WCC(null) YIELD node, componentId")
        components = get_components(result)

        # We should have exactly 3 different components
        self.env.assertEqual(len(components), 3)

        # Validate each component has the correct nodes
        # Expected components after sorting:
        # - Component with 1 node (isolated node 6)
        # - Component with 2 nodes (4-5)
        # - Component with 3 nodes (1-2-3)
        self.env.assertEqual(len(components[0]), 1)  # Isolated node
        self.env.assertEqual(len(components[1]), 2)  # Two connected nodes
        self.env.assertEqual(len(components[2]), 3)  # Three connected nodes

        # Check specific node memberships
        # The isolated node component should be just node 6
        self.env.assertEqual(components[0], [6])

        # The component with 2 nodes should contain nodes 4 and 5
        self.env.assertEqual(components[1], [4, 5])

        # The component with 3 nodes should contain nodes 1, 2, and 3
        self.env.assertEqual(components[2], [1, 2, 3])

    def test_wcc_with_different_relationship_types(self):
        """Test WCC algorithm with different relationship type filters"""

        # Create an unlabeled graph with two relationship types
        # Relationship type A:
        # - n1-A->n2-A->n3 (forms one component)
        # - n4-A->n5 (forms another component)
        # - n6 (isolated)
        # Relationship type B:
        # - n3-B->n4 (connects the first two components from type A)
        # - n5-B->n6 (connects n6 to the second component)
        # - n7 (new isolated node)
        self.graph.query("""
            CREATE
            (n1 {id: 1}),
            (n2 {id: 2}),
            (n3 {id: 3}),
            (n4 {id: 4}),
            (n5 {id: 5}),
            (n6 {id: 6}),
            (n7 {id: 7}),
            (n1)-[:A]->(n2),
            (n2)-[:A]->(n3),
            (n4)-[:A]->(n5),
            (n3)-[:B]->(n4),
            (n5)-[:B]->(n6)
        """)

        # Test 1: Using only relationship type A
        result_A = self.graph.query("CALL algo.WCC({relationshipTypes: ['A']})")
        components_A = get_components(result_A)

        # With only A relationships, we expect:
        # - Component 1: nodes 1,2,3
        # - Component 2: nodes 4,5
        # - Component 3: node 6 (isolated)
        # - Component 4: node 7 (isolated)
        self.env.assertEqual(len(components_A), 4)
        self.env.assertEqual(len(components_A[0]), 1)  # Isolated node
        self.env.assertEqual(len(components_A[1]), 1)  # Isolated node
        self.env.assertEqual(len(components_A[2]), 2)  # Two connected nodes
        self.env.assertEqual(len(components_A[3]), 3)  # Three connected nodes

        # Check specific node memberships
        # Find the single-node components (they could be either 6 or 7)
        isolated_nodes = [components_A[0][0], components_A[1][0]]
        self.env.assertContains(6, isolated_nodes)
        self.env.assertContains(7, isolated_nodes)

        # The component with 2 nodes should be 4,5
        self.env.assertEqual(components_A[2], [4, 5])

        # The component with 3 nodes should be 1,2,3
        self.env.assertEqual(components_A[3], [1, 2, 3])

        # Test 2: Using only relationship type B
        result_B = self.graph.query("CALL algo.WCC({relationshipTypes: ['B']})")
        components_B = get_components(result_B)

        # With only B relationships, we expect:
        # - Component 1: node 1 (isolated)
        # - Component 2: node 2 (isolated)
        # - Component 3: nodes 3,4 (connected)
        # - Component 4: nodes 5,6 (connected)
        # - Component 5: node 7 (isolated)
        self.env.assertEqual(len(components_B), 5)

        # We should have 3 single-node components
        single_node_count = sum(1 for c in components_B if len(c) == 1)
        self.env.assertEqual(single_node_count, 3)

        # And 2 two-node components
        two_node_count = sum(1 for c in components_B if len(c) == 2)
        self.env.assertEqual(two_node_count, 2)

        # Find the two-node components
        two_node_components = [c for c in components_B if len(c) == 2]
        # One should be [3,4] and the other [5,6]
        self.env.assertContains([3, 4], two_node_components)
        self.env.assertContains([5, 6], two_node_components)

        # Test 3: Using both relationship types A and B
        result_AB = self.graph.query("CALL algo.WCC({relationshipTypes: ['A', 'B']})")
        components_AB = get_components(result_AB)

        # With both A and B relationships, we expect:
        # - Component 1: nodes 1,2,3,4,5,6 (all connected)
        # - Component 2: node 7 (isolated)
        self.env.assertEqual(len(components_AB), 2)

        # One component should have 6 nodes (1-6)
        major_component = [c for c in components_AB if len(c) == 6][0]
        self.env.assertEqual(sorted(major_component), [1, 2, 3, 4, 5, 6])

        # The other component should have just node 7
        isolated_component = [c for c in components_AB if len(c) == 1][0]
        self.env.assertEqual(isolated_component, [7])

    def test_wcc_with_different_node_labels(self):
        """Test WCC algorithm with different node label filters"""

        # Create a graph with multiple labeled and unlabeled nodes
        # - L0 labeled nodes: n1, n2, n3
        # - L1 labeled nodes: n4, n5, n6
        # - Unlabeled nodes: n7, n8
        # All connected with relationship type R in a specific pattern
        self.graph.query("""
            CREATE
            (n1:L0 {id: 1}),
            (n2:L0 {id: 2}),
            (n3:L0 {id: 3}),
            (n4:L1 {id: 4}),
            (n5:L1 {id: 5}),
            (n6:L1 {id: 6}),
            (n7 {id: 7}),
            (n8 {id: 8}),
            (n1)-[:R]->(n2),
            (n2)-[:R]->(n3),
            (n3)-[:R]->(n4),
            (n4)-[:R]->(n5),
            (n5)-[:R]->(n6),
            (n6)-[:R]->(n7),
            (n7)-[:R]->(n8)
        """)


        # Test 1: Using only nodes with label L0
        result_L0 = self.graph.query("CALL algo.WCC({nodeLabels: ['L0']})")
        components_L0 = get_components(result_L0)

        # With only L0 nodes, we expect one component with nodes 1,2,3
        self.env.assertEqual(len(components_L0), 1)
        self.env.assertEqual(components_L0[0], [1, 2, 3])

        # Test 2: Using only nodes with label L1
        result_L1 = self.graph.query("CALL algo.WCC({nodeLabels: ['L1']})")
        components_L1 = get_components(result_L1)

        # With only L1 nodes, we expect one component with nodes 4,5,6
        self.env.assertEqual(len(components_L1), 1)
        self.env.assertEqual(components_L1[0], [4, 5, 6])

        # Test 3: Using nodes with both labels L0 and L1
        result_L0_L1 = self.graph.query("CALL algo.WCC({nodeLabels: ['L0', 'L1']})")
        components_L0_L1 = get_components(result_L0_L1)

        # With both L0 and L1 nodes, we expect one component with nodes 1-6
        self.env.assertEqual(len(components_L0_L1), 1)
        self.env.assertEqual(components_L0_L1[0], [1, 2, 3, 4, 5, 6])

        # Test 4: Using all nodes (null parameter)
        result_all = self.graph.query("CALL algo.WCC(null)")
        components_all = get_components(result_all)

        # With all nodes, we expect one component with all nodes 1-8
        self.env.assertEqual(len(components_all), 1)
        self.env.assertEqual(components_all[0], [1, 2, 3, 4, 5, 6, 7, 8])

    def test_wcc_with_combined_labels_and_relationships(self):
        """Test WCC algorithm with combinations of node labels and relationship types"""

        # Create a complex graph with:
        # - Multiple node labels: L1, L2, and unlabeled nodes
        # - Multiple relationship types: R1, R2
        # The graph structure forms several potential connected components depending on filters
        self.graph.query("""
            CREATE
            (n1:L1 {id: 1}),
            (n2:L1 {id: 2}),
            (n3:L1 {id: 3}),
            (n4:L2 {id: 4}),
            (n5:L2 {id: 5}),
            (n6:L2 {id: 6}),
            (n7 {id: 7}),
            (n8 {id: 8}),
            (n9:L1:L2 {id: 9}),  // Node with both labels

            // R1 relationships
            (n1)-[:R1]->(n2),
            (n2)-[:R1]->(n3),
            (n4)-[:R1]->(n5),

            // R2 relationships
            (n3)-[:R2]->(n4),  // Bridge between L1 and L2 clusters
            (n5)-[:R2]->(n6),
            (n6)-[:R2]->(n7),
            (n7)-[:R2]->(n8),

            // Connections to the dual-labeled node
            (n3)-[:R1]->(n9),
            (n9)-[:R2]->(n6)
        """)

        # Test Cases:

        # Test 1: Only label L1 and only relationship type R1
        result = self.graph.query("CALL algo.WCC({nodeLabels: ['L1'], relationshipTypes: ['R1']})")
        components = get_components(result)
        # Expected: One component with nodes 1,2,3,9
        self.env.assertEqual(len(components), 1)
        self.env.assertEqual(components, [[1, 2, 3, 9]])

        # Test 2: Only label L2 and only relationship type R1
        result = self.graph.query("CALL algo.WCC({nodeLabels: ['L2'], relationshipTypes: ['R1']})")
        components = get_components(result)
        # Expected: Components should be [4,5], [6] and [9] as separate components
        # First, check the total number of components
        self.env.assertEqual(len(components), 3)  # Should have 3 components: [4,5], [6], and [9]
        # Verify each expected component exists
        self.env.assertTrue([4, 5] in components)
        self.env.assertTrue([6] in components)
        self.env.assertTrue([9] in components)

        # Test 3: Labels L1 and L2, only relationship type R1
        result = self.graph.query("CALL algo.WCC({nodeLabels: ['L1', 'L2'], relationshipTypes: ['R1']})")
        components = get_components(result)
        # Expected: Two separate components: [1,2,3,9] and [4,5]
        # First check number of components
        self.env.assertEqual(len(components), 3)

        # Three separate components
        self.env.assertEqual(components, [[6], [4, 5], [1, 2, 3, 9]])

        # Test 4: All labels, only relationship type R2
        result = self.graph.query("CALL algo.WCC({relationshipTypes: ['R2']})")
        components = get_components(result)
        # Expected components: [1], [2], [3,4], [5,6,7,8,9]
        self.env.assertEqual(len(components), 4)
        # Verify the exact components
        self.env.assertEqual(components, [[1], [2], [3, 4], [5, 6, 7, 8, 9]])

        # Test 5: Only label L1, both relationship types
        result = self.graph.query("CALL algo.WCC({nodeLabels: ['L1'], relationshipTypes: ['R1', 'R2']})")
        components = get_components(result)
        # Expected: One component with nodes 1,2,3,9
        self.env.assertEqual(components, [[1, 2, 3, 9]])

        # Test 6: All node labels, all relationship types
        result = self.graph.query("CALL algo.WCC(null)")
        components = get_components(result)
        # Expected: One component with all nodes 1-9
        self.env.assertEqual(len(components), 1)
        self.env.assertEqual(components[0], [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_wcc_with_random_graph(self):
        """Test WCC algorithm on a random generated graph"""

        # Create a random graph

        # Test Cases:
        # create a random graph
        nodes, edges = create_random_schema()
        create_random_graph(self.graph, nodes, edges)

        # Test 1: Validate all nodes are reported
        result = self.graph.query("CALL algo.WCC({})").result_set
        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        self.env.assertEqual(len(result), node_count)

        # Test 2: Validate all labeled nodes are reported
        q = """CALL db.labels() YIELD label
               CALL algo.WCC({nodeLabels: [label]}) YIELD node
               RETURN label, count(node)"""
        components = self.graph.query(q).result_set

        for component in components:
            lbl = component[0]
            component_size = component[1]
            lbl_node_count = self.graph.query(f"MATCH (n:{lbl}) RETURN count(n)").result_set[0][0]

            self.env.assertEqual(lbl_node_count, component_size)

