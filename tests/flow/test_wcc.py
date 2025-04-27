from common import *

GRAPH_ID = "WCC"


class testWCC(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test_invalid_invocation(self):
        invalid_queries = [
                """CALL algo.wcc({nodeLabels: 'Person'})""",         # non-array nodeLabels parameter
                """CALL algo.wcc({relationshipTypes: 'KNOWS'})""",   # non-array relationshipTypes parameter
                """CALL algo.wcc({invalidParam: 'value'})""",        # unexpected extra parameters
                """CALL algo.wcc('invalid')""",                      # invalid configuration type
                """CALL algo.wcc({nodeLabels: [1, 2, 3]})""",        # integer values in nodeLabels array
                """CALL algo.wcc({relationshipTypes: [1, 2, 3]})""", # integer values in relationshipTypes array
                """CALL algo.wcc(null) YIELD node, invalidField""",  # non-existent yield field

                """CALL algo.wcc({nodeLabels: ['Person'],
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
        # first, ensure the graph is empty
        self.graph.query("MATCH (n) DELETE n")
        
        # verify graph is empty
        result = self.graph.query("MATCH (n) RETURN count(n) as count")
        self.env.assertEquals(result.result_set[0][0], 0)
        
        # run WCC on empty graph
        result = self.graph.query("CALL algo.wcc(null)")
        # if we reach here, the algorithm didn't throw an exception
        
        # check if it returned empty results (acceptable behavior)
        self.env.assertEquals(len(result.result_set), 0)

    def test_wcc_on_unlabeled_graph(self):
        """Test WCC algorithm on unlabeled nodes with multiple connected components"""

        # clear the graph
        self.graph.query("MATCH (n) DELETE n")

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
        result = self.graph.query("CALL algo.wcc(null) YIELD node, componentId")

        # Extract node ids and component ids
        components = {}
        for record in result.result_set:
            node = record[0]  # node is at index 0
            component_id = record[1]  # componentId is at index 1
            node_id = node.properties['id']

            if component_id not in components:
                components[component_id] = []
            components[component_id].append(node_id)

        # We should have exactly 3 different components
        self.env.assertEquals(len(components), 3)

        # Validate each component has the correct nodes
        component_sets = []
        for component_id, node_ids in components.items():
            node_ids.sort()  # Sort for consistent comparison
            component_sets.append(node_ids)

        # Sort components by size and then content for deterministic comparison
        component_sets.sort(key=lambda x: (len(x), x))

        # Expected components after sorting:
        # - Component with 1 node (isolated node 6)
        # - Component with 2 nodes (4-5)
        # - Component with 3 nodes (1-2-3)
        self.env.assertEquals(len(component_sets[0]), 1)  # Isolated node
        self.env.assertEquals(len(component_sets[1]), 2)  # Two connected nodes
        self.env.assertEquals(len(component_sets[2]), 3)  # Three connected nodes

        # Check specific node memberships
        # The isolated node component should be just node 6
        self.env.assertEquals(component_sets[0], [6])

        # The component with 2 nodes should contain nodes 4 and 5
        self.env.assertEquals(component_sets[1], [4, 5])

        # The component with 3 nodes should contain nodes 1, 2, and 3
        self.env.assertEquals(component_sets[2], [1, 2, 3])

    def test_wcc_with_different_relationship_types(self):
        """Test WCC algorithm with different relationship type filters"""
        # Clear the graph
        self.graph.query("MATCH (n) DELETE n")

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

            # Convert to sorted lists for easier comparison
            component_sets = []
            for component_id, node_ids in components.items():
                node_ids.sort()
                component_sets.append(node_ids)

            # Sort by size then content
            component_sets.sort(key=lambda x: (len(x), x))
            return component_sets

        # Test 1: Using only relationship type A
        result_A = self.graph.query("CALL algo.wcc({relationshipTypes: ['A']})")
        components_A = get_components(result_A)

        # With only A relationships, we expect:
        # - Component 1: nodes 1,2,3
        # - Component 2: nodes 4,5
        # - Component 3: node 6 (isolated)
        # - Component 4: node 7 (isolated)
        self.env.assertEquals(len(components_A), 4)
        self.env.assertEquals(len(components_A[0]), 1)  # Isolated node
        self.env.assertEquals(len(components_A[1]), 1)  # Isolated node
        self.env.assertEquals(len(components_A[2]), 2)  # Two connected nodes
        self.env.assertEquals(len(components_A[3]), 3)  # Three connected nodes

        # Check specific node memberships
        # Find the single-node components (they could be either 6 or 7)
        isolated_nodes = [components_A[0][0], components_A[1][0]]
        self.env.assertTrue(6 in isolated_nodes)
        self.env.assertTrue(7 in isolated_nodes)

        # The component with 2 nodes should be 4,5
        self.env.assertEquals(components_A[2], [4, 5])

        # The component with 3 nodes should be 1,2,3
        self.env.assertEquals(components_A[3], [1, 2, 3])

        # Test 2: Using only relationship type B
        result_B = self.graph.query("CALL algo.wcc({relationshipTypes: ['B']})")
        components_B = get_components(result_B)

        # With only B relationships, we expect:
        # - Component 1: node 1 (isolated)
        # - Component 2: node 2 (isolated)
        # - Component 3: nodes 3,4 (connected)
        # - Component 4: nodes 5,6 (connected)
        # - Component 5: node 7 (isolated)
        self.env.assertEquals(len(components_B), 5)

        # We should have 3 single-node components
        single_node_count = sum(1 for c in components_B if len(c) == 1)
        self.env.assertEquals(single_node_count, 3)

        # And 2 two-node components
        two_node_count = sum(1 for c in components_B if len(c) == 2)
        self.env.assertEquals(two_node_count, 2)

        # Find the two-node components
        two_node_components = [c for c in components_B if len(c) == 2]
        # One should be [3,4] and the other [5,6]
        self.env.assertTrue([3, 4] in two_node_components)
        self.env.assertTrue([5, 6] in two_node_components)

        # Test 3: Using both relationship types A and B
        result_AB = self.graph.query("CALL algo.wcc({relationshipTypes: ['A', 'B']})")
        components_AB = get_components(result_AB)

        # With both A and B relationships, we expect:
        # - Component 1: nodes 1,2,3,4,5,6 (all connected)
        # - Component 2: node 7 (isolated)
        self.env.assertEquals(len(components_AB), 2)

        # One component should have 6 nodes (1-6)
        major_component = [c for c in components_AB if len(c) == 6][0]
        self.env.assertEquals(sorted(major_component), [1, 2, 3, 4, 5, 6])

        # The other component should have just node 7
        isolated_component = [c for c in components_AB if len(c) == 1][0]
        self.env.assertEquals(isolated_component, [7])

    def test_wcc_with_different_node_labels(self):
        """Test WCC algorithm with different node label filters"""
        # Clear the graph
        self.graph.query("MATCH (n) DETACH DELETE n")

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

            # Convert to sorted lists for easier comparison
            component_sets = []
            for component_id, node_ids in components.items():
                node_ids.sort()
                component_sets.append(node_ids)

            # Sort by size then content
            component_sets.sort(key=lambda x: (len(x), x))
            return component_sets

        # Test 1: Using only nodes with label L0
        result_L0 = self.graph.query("CALL algo.wcc({nodeLabels: ['L0']})")
        components_L0 = get_components(result_L0)

        # With only L0 nodes, we expect one component with nodes 1,2,3
        self.env.assertEquals(len(components_L0), 1)
        self.env.assertEquals(components_L0[0], [1, 2, 3])

        # Test 2: Using only nodes with label L1
        result_L1 = self.graph.query("CALL algo.wcc({nodeLabels: ['L1']})")
        components_L1 = get_components(result_L1)

        # With only L1 nodes, we expect one component with nodes 4,5,6
        self.env.assertEquals(len(components_L1), 1)
        self.env.assertEquals(components_L1[0], [4, 5, 6])

        # Test 3: Using nodes with both labels L0 and L1
        result_L0_L1 = self.graph.query("CALL algo.wcc({nodeLabels: ['L0', 'L1']})")
        components_L0_L1 = get_components(result_L0_L1)

        # With both L0 and L1 nodes, we expect one component with nodes 1-6
        self.env.assertEquals(len(components_L0_L1), 1)
        self.env.assertEquals(components_L0_L1[0], [1, 2, 3, 4, 5, 6])

        # Test 4: Using all nodes (null parameter)
        result_all = self.graph.query("CALL algo.wcc(null)")
        components_all = get_components(result_all)

        # With all nodes, we expect one component with all nodes 1-8
        self.env.assertEquals(len(components_all), 1)
        self.env.assertEquals(components_all[0], [1, 2, 3, 4, 5, 6, 7, 8])

