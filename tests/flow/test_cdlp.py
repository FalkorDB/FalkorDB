from common import *

GRAPH_ID = "CDLP"

class testCDLP(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def CDLP(self, nodeLabels=None, relationshipTypes=None, maxIterations=10):
        config = { 'maxIterations': maxIterations }

        if nodeLabels is not None:
            config['nodeLabels'] = nodeLabels

        if relationshipTypes is not None:
            config['relationshipTypes'] = relationshipTypes

        return self.graph.query("""CALL algo.labelPropagation($config)
                                   YIELD node, communityId
                                   RETURN node.name, communityId
                                   ORDER BY communityId DESC""", {'config': config})

    def test_invalid_invocation(self):
        invalid_queries = [
                # non-array nodeLabels parameter
                """CALL algo.labelPropagation({nodeLabels: 'Person'})""",

                # non-array relationshipTypes parameter
                """CALL algo.labelPropagation({relationshipTypes: 'KNOWS'})""",

                # unexpected extra parameters
                """CALL algo.labelPropagation({invalidParam: 'value'})""",

                # invalid configuration type
                """CALL algo.labelPropagation('invalid')""",

                # integer values in nodeLabels array
                """CALL algo.labelPropagation({nodeLabels: [1, 2, 3]})""",

                # integer values in relationshipTypes array
                """CALL algo.labelPropagation({relationshipTypes: [1, 2, 3]})""",

                # non-existent yield field
                """CALL algo.labelPropagation(null) YIELD node, invalidField""",

                # maxIterations is expected to a positive integer
                """CALL algo.labelPropagation({maxIterations: 'a'})""",

                # maxIterations is expected to a positive integer
                """CALL algo.labelPropagation({maxIterations: -21})""",

                # maxIterations is expected to an integer
                """CALL algo.labelPropagation({maxIterations: [1]})""",

                # unexpected extra parameters
                """CALL algo.labelPropagation({nodeLabels: ['Person'],
                               relationshipTypes: ['KNOWS'],
                               invalidParam: 'value'})""",
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except:
                pass

    def test_CDLP_on_empty_graph(self):
        """Test CDLP algorithm behavior on an empty graph"""
        # verify graph is empty
        result = self.graph.query("MATCH (n) RETURN count(n) as count")
        self.env.assertEqual(result.result_set[0][0], 0)
        
        # run CDLP on empty graph
        result = self.CDLP()

        # if we reach here, the algorithm didn't throw an exception
        
        # check if it returned empty results (acceptable behavior)
        self.env.assertEqual(len(result.result_set), 0)

    def test_label_propagation_basic(self):
        """Test basic functionality of the label propagation algorithm."""
        # Create a simple graph with three completely separate communities
        # Each community has 3 nodes (odd number) with full internal connectivity
        # Graph structure:
        #
        # Community 1:    Community 2:    Community 3:
        # A -- B          D -- E          G -- H
        # |    |          |    |          |    |
        # --- C ---       --- F ---       --- I ---
        #
        # We expect exactly three distinct communities
        create_query = """
        CREATE
        /* Community 1 */
        (a:Person {name: 'A'}),
        (b:Person {name: 'B'}),
        (c:Person {name: 'C'}),
        /* Community 2 */
        (d:Person {name: 'D'}),
        (e:Person {name: 'E'}),
        (f:Person {name: 'F'}),
        /* Community 3 */
        (g:Person {name: 'G'}),
        (h:Person {name: 'H'}),
        (i:Person {name: 'I'}),

        /* Community 1 - fully connected */
        (a)-[:KNOWS]->(b),
        (a)-[:KNOWS]->(c),
        (b)-[:KNOWS]->(a),
        (b)-[:KNOWS]->(c),
        (c)-[:KNOWS]->(a),
        (c)-[:KNOWS]->(b),

        /* Community 2 - fully connected */
        (d)-[:KNOWS]->(e),
        (d)-[:KNOWS]->(f),
        (e)-[:KNOWS]->(d),
        (e)-[:KNOWS]->(f),
        (f)-[:KNOWS]->(d),
        (f)-[:KNOWS]->(e),

        /* Community 3 - fully connected */
        (g)-[:KNOWS]->(h),
        (g)-[:KNOWS]->(i),
        (h)-[:KNOWS]->(g),
        (h)-[:KNOWS]->(i),
        (i)-[:KNOWS]->(g),
        (i)-[:KNOWS]->(h)
        """
        self.graph.query(create_query)

        # Run label propagation algorithm without configuration
        result = self.CDLP()

        # Extract results
        communities = {}
        for record in result.result_set:
            name = record[0]
            community_id = record[1]
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(name)

        # We expect exactly 3 distinct communities
        self.env.assertEqual(len(communities), 3)

        # Find which community each node belongs to
        community_lookup = {}
        for community_id, members in communities.items():
            for name in members:
                community_lookup[name] = community_id

        # Nodes in the same expected community should have the same community ID
        # Community 1
        self.env.assertEqual(community_lookup['A'], community_lookup['B'])
        self.env.assertEqual(community_lookup['A'], community_lookup['C'])

        # Community 2
        self.env.assertEqual(community_lookup['D'], community_lookup['E'])
        self.env.assertEqual(community_lookup['D'], community_lookup['F'])

        # Community 3
        self.env.assertEqual(community_lookup['G'], community_lookup['H'])
        self.env.assertEqual(community_lookup['G'], community_lookup['I'])

        # Verify that the expected communities are different from each other
        self.env.assertNotEqual(community_lookup['A'], community_lookup['D'])
        self.env.assertNotEqual(community_lookup['A'], community_lookup['G'])
        self.env.assertNotEqual(community_lookup['D'], community_lookup['G'])

    def test_label_propagation_node_label_filtering(self):
        """Test that the label propagation algorithm correctly filters by node labels."""
        # Create a graph with separate communities for each node label
        # Each community has only one type of node label and is fully connected internally
        # Graph structure:
        #
        # Person Communities:           User Communities:
        # A(Person) -- B(Person)        E(User) -- F(User)
        #     |            |                |           |
        #     --- C(Person) ---             --- G(User) ---
        #
        # D(Person) -- H(Person)        I(User) -- J(User)
        #     |            |                |           |
        #     --- K(Person) ---             --- L(User) ---
        #
        # When filtering for Person nodes only, we expect two distinct communities: ABC and DHK
        # When filtering for User nodes only, we expect two distinct communities: EFG and IJL
        create_query = """
        CREATE
        /* Person Community 1 */
        (a:Person {name: 'A'}),
        (b:Person {name: 'B'}),
        (c:Person {name: 'C'}),

        /* Person Community 2 */
        (d:Person {name: 'D'}),
        (h:Person {name: 'H'}),
        (k:Person {name: 'K'}),

        /* User Community 1 */
        (e:User {name: 'E'}),
        (f:User {name: 'F'}),
        (g:User {name: 'G'}),

        /* User Community 2 */
        (i:User {name: 'I'}),
        (j:User {name: 'J'}),
        (l:User {name: 'L'}),

        /* Person Community 1 - fully connected */
        (a)-[:KNOWS]->(b),
        (a)-[:KNOWS]->(c),
        (b)-[:KNOWS]->(a),
        (b)-[:KNOWS]->(c),
        (c)-[:KNOWS]->(a),
        (c)-[:KNOWS]->(b),

        /* Person Community 2 - fully connected */
        (d)-[:KNOWS]->(h),
        (d)-[:KNOWS]->(k),
        (h)-[:KNOWS]->(d),
        (h)-[:KNOWS]->(k),
        (k)-[:KNOWS]->(d),
        (k)-[:KNOWS]->(h),

        /* User Community 1 - fully connected */
        (e)-[:KNOWS]->(f),
        (e)-[:KNOWS]->(g),
        (f)-[:KNOWS]->(e),
        (f)-[:KNOWS]->(g),
        (g)-[:KNOWS]->(e),
        (g)-[:KNOWS]->(f),

        /* User Community 2 - fully connected */
        (i)-[:KNOWS]->(j),
        (i)-[:KNOWS]->(l),
        (j)-[:KNOWS]->(i),
        (j)-[:KNOWS]->(l),
        (l)-[:KNOWS]->(i),
        (l)-[:KNOWS]->(j)
        """
        self.graph.query(create_query)

        # Run label propagation algorithm with Person node label filter
        person_result = self.CDLP(nodeLabels=['Person'])

        # Extract Person results
        person_communities = {}
        for record in person_result.result_set:
            name = record[0]
            community_id = record[1]
            if community_id not in person_communities:
                person_communities[community_id] = []
            person_communities[community_id].append(name)

        # We should only have Person nodes in the results
        all_person_nodes = []
        for members in person_communities.values():
            all_person_nodes.extend(members)
        self.env.assertEqual(sorted(all_person_nodes), ['A', 'B', 'C', 'D', 'H', 'K'])

        # We expect exactly 2 distinct Person communities
        self.env.assertEqual(len(person_communities), 2)

        # Find Person community IDs
        person_lookup = {}
        for community_id, members in person_communities.items():
            for name in members:
                person_lookup[name] = community_id

        # Verify that Person nodes in the same expected community are together
        self.env.assertEqual(person_lookup['A'], person_lookup['B'])
        self.env.assertEqual(person_lookup['A'], person_lookup['C'])
        self.env.assertEqual(person_lookup['D'], person_lookup['H'])
        self.env.assertEqual(person_lookup['D'], person_lookup['K'])

        # Verify that Person nodes in different communities have different IDs
        self.env.assertNotEqual(person_lookup['A'], person_lookup['D'])

        # Run label propagation algorithm with User node label filter
        user_result = self.CDLP(nodeLabels=['User'])

        # Extract User results
        user_communities = {}
        for record in user_result.result_set:
            name = record[0]
            community_id = record[1]
            if community_id not in user_communities:
                user_communities[community_id] = []
            user_communities[community_id].append(name)

        # We should only have User nodes in the results
        all_user_nodes = []
        for members in user_communities.values():
            all_user_nodes.extend(members)
        self.env.assertEqual(sorted(all_user_nodes), ['E', 'F', 'G', 'I', 'J', 'L'])

        # We expect exactly 2 distinct User communities
        self.env.assertEqual(len(user_communities), 2)

        # Find User community IDs
        user_lookup = {}
        for community_id, members in user_communities.items():
            for name in members:
                user_lookup[name] = community_id

        # Verify that User nodes in the same expected community are together
        self.env.assertEqual(user_lookup['E'], user_lookup['F'])
        self.env.assertEqual(user_lookup['E'], user_lookup['G'])
        self.env.assertEqual(user_lookup['I'], user_lookup['J'])
        self.env.assertEqual(user_lookup['I'], user_lookup['L'])

        # Verify that User nodes in different communities have different IDs
        self.env.assertNotEqual(user_lookup['E'], user_lookup['I'])

    def test_label_propagation_relationship_filtering(self):
        """Test that the label propagation algorithm correctly filters by relationship types."""
        # Create a graph with communities connected by different relationship types
        # Each community is fully connected internally with one specific relationship type
        # Graph structure:
        #
        # KNOWS Communities:              WORKS_WITH Communities:
        # A -- B (KNOWS)                  E -- F (WORKS_WITH)
        #  \  /                            \  /
        #   C                               G
        #
        # D -- H (KNOWS)                  I -- J (WORKS_WITH)
        #  \  /                            \  /
        #   K                               L
        #
        # When filtering for KNOWS relationships only, we expect two communities: ABC and DHK
        # When filtering for WORKS_WITH relationships only, we expect two communities: EFG and IJL
        create_query = """
        CREATE
        /* All nodes */
        (a:Person {name: 'A'}),
        (b:Person {name: 'B'}),
        (c:Person {name: 'C'}),
        (d:Person {name: 'D'}),
        (e:Person {name: 'E'}),
        (f:Person {name: 'F'}),
        (g:Person {name: 'G'}),
        (h:Person {name: 'H'}),
        (i:Person {name: 'I'}),
        (j:Person {name: 'J'}),
        (k:Person {name: 'K'}),
        (l:Person {name: 'L'}),

        /* KNOWS Community 1 */
        (a)-[:KNOWS]->(b),
        (a)-[:KNOWS]->(c),
        (b)-[:KNOWS]->(a),
        (b)-[:KNOWS]->(c),
        (c)-[:KNOWS]->(a),
        (c)-[:KNOWS]->(b),

        /* KNOWS Community 2 */
        (d)-[:KNOWS]->(h),
        (d)-[:KNOWS]->(k),
        (h)-[:KNOWS]->(d),
        (h)-[:KNOWS]->(k),
        (k)-[:KNOWS]->(d),
        (k)-[:KNOWS]->(h),

        /* WORKS_WITH Community 1 */
        (e)-[:WORKS_WITH]->(f),
        (e)-[:WORKS_WITH]->(g),
        (f)-[:WORKS_WITH]->(e),
        (f)-[:WORKS_WITH]->(g),
        (g)-[:WORKS_WITH]->(e),
        (g)-[:WORKS_WITH]->(f),

        /* WORKS_WITH Community 2 */
        (i)-[:WORKS_WITH]->(j),
        (i)-[:WORKS_WITH]->(l),
        (j)-[:WORKS_WITH]->(i),
        (j)-[:WORKS_WITH]->(l),
        (l)-[:WORKS_WITH]->(i),
        (l)-[:WORKS_WITH]->(j)
        """
        self.graph.query(create_query)

        # Run label propagation algorithm with KNOWS relationship filter
        knows_result = self.CDLP(relationshipTypes=['KNOWS'])

        # Extract KNOWS results
        knows_communities = {}
        for record in knows_result.result_set:
            name = record[0]
            community_id = record[1]
            if community_id not in knows_communities:
                knows_communities[community_id] = []
            knows_communities[community_id].append(name)

        # Find KNOWS-connected nodes (expect nodes A-C and D-K to be in their own communities)
        knows_connected_nodes = ['A', 'B', 'C', 'D', 'H', 'K']
        knows_connected_communities = {}

        for community_id, members in knows_communities.items():
            connected_members = [m for m in members if m in knows_connected_nodes]
            if connected_members:
                if community_id not in knows_connected_communities:
                    knows_connected_communities[community_id] = []
                knows_connected_communities[community_id].extend(connected_members)

        # We expect exactly 2 KNOWS communities (among the KNOWS-connected nodes)
        knows_community_count = sum(1 for members in knows_connected_communities.values()
                                  if any(node in knows_connected_nodes for node in members))
        self.env.assertEqual(knows_community_count, 2)

        # Find KNOWS community IDs for connected nodes
        knows_lookup = {}
        for community_id, members in knows_connected_communities.items():
            for name in members:
                if name in knows_connected_nodes:
                    knows_lookup[name] = community_id

        # Verify that nodes in the same KNOWS community are together
        if 'A' in knows_lookup and 'B' in knows_lookup and 'C' in knows_lookup:
            self.env.assertEqual(knows_lookup['A'], knows_lookup['B'])
            self.env.assertEqual(knows_lookup['A'], knows_lookup['C'])

        if 'D' in knows_lookup and 'H' in knows_lookup and 'K' in knows_lookup:
            self.env.assertEqual(knows_lookup['D'], knows_lookup['H'])
            self.env.assertEqual(knows_lookup['D'], knows_lookup['K'])

        # Verify that nodes in different KNOWS communities have different IDs
        if 'A' in knows_lookup and 'D' in knows_lookup:
            self.env.assertNotEqual(knows_lookup['A'], knows_lookup['D'])

        # Run label propagation algorithm with WORKS_WITH relationship filter
        works_result = self.CDLP(relationshipTypes=['WORKS_WITH'])

        # Extract WORKS_WITH results
        works_communities = {}
        for record in works_result.result_set:
            name = record[0]
            community_id = record[1]
            if community_id not in works_communities:
                works_communities[community_id] = []
            works_communities[community_id].append(name)

        # Find WORKS_WITH-connected nodes (expect nodes E-G and I-L to be in their own communities)
        works_connected_nodes = ['E', 'F', 'G', 'I', 'J', 'L']
        works_connected_communities = {}

        for community_id, members in works_communities.items():
            connected_members = [m for m in members if m in works_connected_nodes]
            if connected_members:
                if community_id not in works_connected_communities:
                    works_connected_communities[community_id] = []
                works_connected_communities[community_id].extend(connected_members)

        # We expect exactly 2 WORKS_WITH communities (among the WORKS_WITH-connected nodes)
        works_community_count = sum(1 for members in works_connected_communities.values()
                                  if any(node in works_connected_nodes for node in members))
        self.env.assertEqual(works_community_count, 2)

        # Find WORKS_WITH community IDs for connected nodes
        works_lookup = {}
        for community_id, members in works_connected_communities.items():
            for name in members:
                if name in works_connected_nodes:
                    works_lookup[name] = community_id

        # Verify that nodes in the same WORKS_WITH community are together
        if 'E' in works_lookup and 'F' in works_lookup and 'G' in works_lookup:
            self.env.assertEqual(works_lookup['E'], works_lookup['F'])
            self.env.assertEqual(works_lookup['E'], works_lookup['G'])

        if 'I' in works_lookup and 'J' in works_lookup and 'L' in works_lookup:
            self.env.assertEqual(works_lookup['I'], works_lookup['J'])
            self.env.assertEqual(works_lookup['I'], works_lookup['L'])

        # Verify that nodes in different WORKS_WITH communities have different IDs
        if 'E' in works_lookup and 'I' in works_lookup:
            self.env.assertNotEqual(works_lookup['E'], works_lookup['I'])

        # Run label propagation algorithm with both relationship types
        both_result = self.CDLP(relationshipTypes=['KNOWS', 'WORKS_WITH'])

        # Extract combined results
        both_communities = {}
        for record in both_result.result_set:
            name = record[0]
            community_id = record[1]
            if community_id not in both_communities:
                both_communities[community_id] = []
            both_communities[community_id].append(name)

        # We expect nodes in each of the 4 communities to remain in their separate communities
        # when using both relationship types
        combined_community_count = len(both_communities)
        self.env.assertEqual(combined_community_count, 4)

