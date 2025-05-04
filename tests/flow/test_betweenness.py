from common import *

GRAPH_ID = "betweenness"

class testBetweenness(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def betweenness_centrality(self, nodeLabels=None, relationshipTypes=None, seed=10, samplingSize=16):
        config = { 'samplingSeed': seed, 'samplingSize': samplingSize }

        if nodeLabels is not None:
            config['nodeLabels'] = nodeLabels

        if relationshipTypes is not None:
            config['relationshipTypes'] = relationshipTypes

        return self.graph.query("""CALL algo.betweenness($config)
                                   YIELD node, score
                                   RETURN node.name, score
                                   ORDER BY score DESC""", {'config': config})

    def test_invalid_invocation(self):
        invalid_queries = [
                # non-array nodeLabels parameter
                """CALL algo.betweenness({nodeLabels: 'Person'})""",         

                # non-array relationshipTypes parameter
                """CALL algo.betweenness({relationshipTypes: 'KNOWS'})""",   

                # unexpected extra parameters
                """CALL algo.betweenness({invalidParam: 'value'})""",        

                # invalid configuration type
                """CALL algo.betweenness('invalid')""",

                # integer values in nodeLabels array
                """CALL algo.betweenness({nodeLabels: [1, 2, 3]})""",

                # integer values in relationshipTypes array
                """CALL algo.betweenness({relationshipTypes: [1, 2, 3]})""",

                # non-existent yield field
                """CALL algo.betweenness(null) YIELD node, invalidField""",

                # samplingSize is expected to a positive integer
                """CALL algo.betweenness({samplingSize: 'a'})""",

                # samplingSize is expected to a positive integer
                """CALL algo.betweenness({samplingSize: -21})""",

                # samplingSeed is expected to an integer
                """CALL algo.betweenness({samplingSeed: [1]})""",

                # unexpected extra parameters
                """CALL algo.betweenness({nodeLabels: ['Person'],
                               relationshipTypes: ['KNOWS'],
                               invalidParam: 'value'})""",
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except:
                pass

    def test_betweenness_on_empty_graph(self):
        """Test Betweenness centrality algorithm behavior on an empty graph"""
        # verify graph is empty
        result = self.graph.query("MATCH (n) RETURN count(n) as count")
        self.env.assertEquals(result.result_set[0][0], 0)
        
        # run betweenness centrality on empty graph
        result = self.betweenness_centrality()

        # if we reach here, the algorithm didn't throw an exception
        
        # check if it returned empty results (acceptable behavior)
        self.env.assertEquals(len(result.result_set), 0)

    def test_betweenness_centrality(self):
        """Test betweenness centrality algorithm on a simple directed graph structure."""

        # Create a small graph where we can reason about the betweenness centrality
        # Graph structure (directed):
        # A --> B --> C
        #       |     |
        #       v     v
        #       D --> E
        #
        # Node B should have the highest betweenness as it's on many shortest paths
        # Node C should have medium betweenness
        # Nodes A, D, E should have the lowest betweenness

        # Create the entire graph with a single Cypher query
        create_query = """
        CREATE
        (a:Person {name: 'A'}),
        (b:Person {name: 'B'}),
        (c:Person {name: 'C'}),
        (d:Person {name: 'D'}),
        (e:Person {name: 'E'}),
        (a)-[:CONNECTS]->(b),
        (b)-[:CONNECTS]->(c),
        (b)-[:CONNECTS]->(d),
        (c)-[:CONNECTS]->(e),
        (d)-[:CONNECTS]->(e)
        """
        self.graph.query(create_query)

        # Run betweenness centrality algorithm
        result = self.betweenness_centrality()

        # Extract results directly from the result set
        centrality_scores = {}
        for record in result.result_set:
            name = record[0]
            score = record[1]
            centrality_scores[name] = score

        # Check that expected relationships hold for directed graph
        # B should have highest betweenness centrality
        self.env.assertGreater(centrality_scores['B'], centrality_scores['A'])
        self.env.assertGreater(centrality_scores['B'], centrality_scores['C'])
        self.env.assertGreater(centrality_scores['B'], centrality_scores['D'])
        self.env.assertGreater(centrality_scores['B'], centrality_scores['E'])

        # C and D should have medium betweenness centrality
        self.env.assertGreater(centrality_scores['C'], centrality_scores['A'])
        self.env.assertGreater(centrality_scores['D'], centrality_scores['A'])

        # A should have zero betweenness centrality (no shortest paths pass through it)
        self.env.assertEqual(centrality_scores['A'], 0)

        # In directed graph, C and D aren't perfectly symmetrical
        # C is on the path to E, while D is also on path to E
        self.env.assertEqual(centrality_scores['C'], centrality_scores['D'])  # Still equal in this specific topology

        # E should have zero betweenness (no paths go through it)
        self.env.assertEqual(centrality_scores['E'], 0)

    def test_betweenness_centrality_node_labels(self):
        """Test betweenness centrality algorithm with nodeLabels parameter."""
        # Create a graph with multiple node labels
        # Graph structure (directed):
        #
        #    Person      Person      Person
        #     A  ------->  B  ------->  C
        #                  |            |
        #                  v            v
        #                  D  ------->  E
        #               Location     Company
        #
        # We'll compute betweenness only on Person nodes
        # Only B and C should have scores, other nodes should be excluded

        create_query = """
        CREATE
        (a:Person {name: 'A'}),
        (b:Person {name: 'B'}),
        (c:Person {name: 'C'}),
        (d:Location {name: 'D'}),
        (e:Company {name: 'E'}),
        (a)-[:CONNECTS]->(b),
        (b)-[:CONNECTS]->(c),
        (b)-[:CONNECTS]->(d),
        (c)-[:CONNECTS]->(e),
        (d)-[:CONNECTS]->(e)
        """
        self.graph.query(create_query)

        # Run betweenness centrality algorithm with nodeLabels parameter
        result = self.betweenness_centrality(nodeLabels = ['Person'])

        # Extract results directly from the result set
        centrality_scores = {}
        for record in result.result_set:
            name = record[0]
            score = record[1]
            centrality_scores[name] = score

        # Verify only Person nodes have scores
        self.env.assertEqual(len(centrality_scores), 3)

        # Verify Person nodes A, B, C are included
        self.env.assertIn('A', centrality_scores)
        self.env.assertIn('B', centrality_scores)
        self.env.assertIn('C', centrality_scores)

        # Verify non-Person nodes are excluded
        self.env.assertNotIn('D', centrality_scores)
        self.env.assertNotIn('E', centrality_scores)

        # B should have highest centrality among Person nodes
        self.env.assertGreater(centrality_scores['B'], centrality_scores['A'])
        self.env.assertGreater(centrality_scores['B'], centrality_scores['C'])

        # A should have zero betweenness (no paths through it)
        self.env.assertEqual(centrality_scores['A'], 0)

        # C should have zero betweenness (no paths through it)
        self.env.assertEqual(centrality_scores['C'], 0)

    def test_betweenness_centrality_relationship_types(self):
        """Test betweenness centrality algorithm with relationshipTypes parameter."""
        # Create a graph with multiple relationship types
        # Graph structure (directed):
        #
        #     A  --FRIEND--> B --FRIEND--> C
        #                    |             |
        #                WORKS_AT       LOCATED_IN
        #                    v             v
        #                    D --NEAR----> E
        #
        # We'll compute betweenness using only FRIEND relationships
        # This should affect which paths are considered for centrality

        create_query = """
        CREATE
        (a:Person {name: 'A'}),
        (b:Person {name: 'B'}),
        (c:Person {name: 'C'}),
        (d:Person {name: 'D'}),
        (e:Person {name: 'E'}),
        (a)-[:FRIEND]->(b),
        (b)-[:FRIEND]->(c),
        (b)-[:WORKS_AT]->(d),
        (c)-[:LOCATED_IN]->(e),
        (d)-[:NEAR]->(e)
        """
        self.graph.query(create_query)

        # Run betweenness centrality algorithm with relationshipTypes parameter
        result = self.betweenness_centrality(relationshipTypes = ['FRIEND'])

        # Extract results directly from the result set
        centrality_scores = {}
        for record in result.result_set:
            name = record[0]
            score = record[1]
            centrality_scores[name] = score

        # Verify all nodes are included in results (relationship filter doesn't exclude nodes)
        self.env.assertEqual(len(centrality_scores), 5)

        # B should have a non-zero betweenness score (it's on the A->C path via FRIEND edges)
        self.env.assertGreater(centrality_scores['B'], 0)

        # All other nodes should have zero betweenness (no FRIEND paths go through them)
        self.env.assertEqual(centrality_scores['A'], 0)
        self.env.assertEqual(centrality_scores['C'], 0)
        self.env.assertEqual(centrality_scores['D'], 0)
        self.env.assertEqual(centrality_scores['E'], 0)

        # Run algorithm again with different relationship types
        result = self.betweenness_centrality(relationshipTypes = ['WORKS_AT', 'NEAR'])

        # Extract results
        centrality_scores_2 = {}
        for record in result.result_set:
            name = record[0]
            score = record[1]
            centrality_scores_2[name] = score

        # D should now have non-zero betweenness (it's on the B->E path)
        self.env.assertGreater(centrality_scores_2['D'], 0)

        # B should have a different score than in the first test
        self.env.assertNotEqual(centrality_scores['B'], centrality_scores_2['B'])

    def test_betweenness_centrality_combined_parameters(self):
        """Test betweenness centrality with combinations of nodeLabels and relationshipTypes parameters."""
        # Create a complex graph with multiple node labels and relationship types
        # Graph structure:
        #
        #     A:Person --FRIEND--> B:Person --FRIEND--> C:Person
        #                   |                   |
        #               MANAGES             COLLEAGUE
        #                   |                   |
        #                   v                   v
        #              D:Employee --WORKS_WITH-> E:Employee
        #                   |                   |
        #                BASED_IN           VISITS
        #                   |                   |
        #                   v                   v
        #              F:Location --NEAR--> G:Location

        create_query = """
        CREATE
        (a:Person {name: 'A'}),
        (b:Person {name: 'B'}),
        (c:Person {name: 'C'}),
        (d:Employee {name: 'D'}),
        (e:Employee {name: 'E'}),
        (f:Location {name: 'F'}),
        (g:Location {name: 'G'}),
        (a)-[:FRIEND]->(b),
        (b)-[:FRIEND]->(c),
        (b)-[:MANAGES]->(d),
        (c)-[:COLLEAGUE]->(e),
        (d)-[:WORKS_WITH]->(e),
        (d)-[:BASED_IN]->(f),
        (e)-[:VISITS]->(g),
        (f)-[:NEAR]->(g)
        """
        self.graph.query(create_query)

        # Test 1: Filter only by Person nodes
        result1 = self.betweenness_centrality(nodeLabels = ['Person'])

        centrality_scores1 = {}
        for record in result1.result_set:
            name = record[0]
            score = record[1]
            centrality_scores1[name] = score

        # Should only contain Person nodes
        self.env.assertEqual(len(centrality_scores1), 3)
        self.env.assertIn('A', centrality_scores1)
        self.env.assertIn('B', centrality_scores1)
        self.env.assertIn('C', centrality_scores1)
        self.env.assertNotIn('D', centrality_scores1)

        # Test 2: Filter only by FRIEND relationships
        result2 = self.betweenness_centrality(relationshipTypes = ['FRIEND'])

        centrality_scores2 = {}
        for record in result2.result_set:
            name = record[0]
            score = record[1]
            centrality_scores2[name] = score

        # B should have highest betweenness as it's on the FRIEND path from A to C
        max_node_2 = max(centrality_scores2, key=centrality_scores2.get)
        self.env.assertEqual(max_node_2, 'B')

        # All other nodes should have 0 betweenness for FRIEND relationships
        for node in centrality_scores2:
            if node != 'B':
                self.env.assertEqual(centrality_scores2[node], 0)

        # Test 3: Combine Person nodes with FRIEND relationships
        result3 = self.betweenness_centrality(nodeLabels=['Person'],
                                              relationshipTypes=['FRIEND'])

        centrality_scores3 = {}
        for record in result3.result_set:
            name = record[0]
            score = record[1]
            centrality_scores3[name] = score

        # Should only contain Person nodes
        self.env.assertEqual(len(centrality_scores3), 3)

        # Results should match Test 2 for the Person nodes
        for node in ['A', 'B', 'C']:
            self.env.assertEqual(centrality_scores2[node], centrality_scores3[node])

        # Test 4: Employee nodes with WORKS_WITH relationships
        result4 = self.betweenness_centrality(nodeLabels=['Employee'],
                                              relationshipTypes=['WORKS_WITH'])

        centrality_scores4 = {}
        for record in result4.result_set:
            name = record[0]
            score = record[1]
            centrality_scores4[name] = score

        # Should only contain Employee nodes
        self.env.assertEqual(len(centrality_scores4), 2)
        self.env.assertIn('D', centrality_scores4)
        self.env.assertIn('E', centrality_scores4)

        # Test 5: Multiple node labels
        result5 = self.betweenness_centrality(nodeLabels=['Person', 'Employee'])

        centrality_scores5 = {}
        for record in result5.result_set:
            name = record[0]
            score = record[1]
            centrality_scores5[name] = score

        # Should contain Person and Employee nodes
        self.env.assertEqual(len(centrality_scores5), 5)
        self.env.assertIn('A', centrality_scores5)
        self.env.assertIn('B', centrality_scores5)
        self.env.assertIn('C', centrality_scores5)
        self.env.assertIn('D', centrality_scores5)
        self.env.assertIn('E', centrality_scores5)

        # Test 6: Multiple relationship types
        result6 = self.betweenness_centrality(relationshipTypes=['FRIEND', 'COLLEAGUE', 'WORKS_WITH'],
                                              seed=231231)

        centrality_scores6 = {}
        for record in result6.result_set:
            name = record[0]
            score = record[1]
            centrality_scores6[name] = score

        # D should have zero betweenness (no paths through it)
        self.env.assertEqual(centrality_scores6['D'], 0)

        # Compare B's score between tests - it should be different with more relationship types
        self.env.assertNotEqual(centrality_scores2['B'], centrality_scores6['B'])

        # Test 7: Combine all parameters including sampling
        result7 = self.betweenness_centrality(nodeLabels=['Person', 'Employee'],
                                              relationshipTypes=['FRIEND', 'WORKS_WITH'],
                                              samplingSize=3, seed=231231)

        centrality_scores7 = {}
        for record in result7.result_set:
            name = record[0]
            score = record[1]
            centrality_scores7[name] = score

        # Should only contain Person and Employee nodes
        self.env.assertEqual(len(centrality_scores7), 5)

        # Run the same test with a different seed
        result8 = self.betweenness_centrality(nodeLabels=['Person', 'Employee'],
                                              relationshipTypes=['FRIEND', 'WORKS_WITH'],
                                              samplingSize=3, seed=100)

        centrality_scores8 = {}
        for record in result8.result_set:
            name = record[0]
            score = record[1]
            centrality_scores8[name] = score

        # Different seeds should give different results
        self.env.assertNotEqual(centrality_scores7['B'], centrality_scores8['B'])

