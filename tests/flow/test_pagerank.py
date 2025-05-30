from common import *

GRAPH_ID = "pagerank"


class testPagerank(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def test_invalid_invocation(self):
        invalid_queries = [

            # array relationship type
            """CALL algo.pageRank('Special', {'SPECIAL_CONNECTS'}) YIELD node""",   

            # array label type
            """CALL algo.pageRank({'Special'}, 'SPECIAL_CONNECTS') YIELD node""",  

            # unexpected extra parameters
            """CALL algo.pageRank('Special', 'SPECIAL_CONNECTS','extra') YIELD node""",         

            # invalid configuration type
            """CALL algo.pageRank('invalid') YIELD node""",

            # non-existent yield field
            """CALL algo.pageRank(null, null) YIELD node, invalidField""",
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except:
                pass
    
    def test_pagerank_null_arguments(self):
        """Test PageRank algorithm when NULL is passed for both node label and relationship type arguments"""

        # Create a simple directed graph using a Cypher query
        # Graph visualization:
        #
        #    A -------> B -------> C
        #    |          ^          |
        #    |          |          |
        #    v          |          v
        #    D <------- E <------- F
        #
        # In this graph:
        # - All nodes have 1 incoming edge except B which has 2 (from A and E)
        # - The graph forms a cycle, ensuring PageRank will converge
        create_query = """
        CREATE
            (a:Node {name: 'A'}),
            (b:Node {name: 'B'}),
            (c:Node {name: 'C'}),
            (d:Node {name: 'D'}),
            (e:Node {name: 'E'}),
            (f:Node {name: 'F'}),
            (a)-[:CONNECTS]->(b),
            (b)-[:CONNECTS]->(c),
            (c)-[:CONNECTS]->(f),
            (f)-[:CONNECTS]->(e),
            (e)-[:CONNECTS]->(d),
            (d)-[:CONNECTS]->(a),
            (e)-[:CONNECTS]->(b)
        """
        self.graph.query(create_query)

        # Run PageRank on all nodes and relationships
        result = self.graph.query("CALL algo.pageRank(NULL, NULL) YIELD node, score RETURN node.name, score")

        # Validate result structure
        self.env.assertEqual(len(result.result_set), 6)  # Should return 6 nodes

        # Create a dictionary of node labels to scores
        scores = {}
        for record in result.result_set:
            node_name = record[0]
            score = record[1]

            # Extract the node label from properties
            scores[node_name] = score

        # Verify all nodes have a score
        self.env.assertEqual(len(scores), 6)

        # In this graph, nodes with more incoming edges should have higher PageRank scores
        # Node B has 2 incoming edges, others have 1
        # Check that B has the highest score
        b_score = scores.get("B", 0)
        for label, score in scores.items():
            if label != "B":
                self.env.assertGreaterEqual(b_score, score)

        # Verify all scores sum to approximately 1.0 (allowing for floating point errors)
        score_sum = sum(scores.values())
        self.env.assertAlmostEqual(score_sum, 1.0, delta=0.0001)

        # Verify all scores are positive
        for label, score in scores.items():
            self.env.assertGreater(score, 0)

    def test_pagerank_specific_labels(self):
        """Test PageRank with specific node label and relationship type"""

        # Create a graph with both regular and special nodes
        # Graph visualization:
        #
        # Regular nodes:    A -------> B
        #
        # Special nodes:    S1 -------> S2
        #                   (SPECIAL_CONNECTS)
        create_query = """
        CREATE
            (a:Node {name: 'A'}),
            (b:Node {name: 'B'}),
            (a)-[:CONNECTS]->(b),
            (s1:Special {name: 'S1'}),
            (s2:Special {name: 'S2'}),
            (s1)-[:SPECIAL_CONNECTS]->(s2)
        """
        self.graph.query(create_query)

        # Run PageRank only on Special nodes and SPECIAL_CONNECTS relationships
        result = self.graph.query("""CALL algo.pageRank('Special', 'SPECIAL_CONNECTS')
                                     YIELD node, score
                                     RETURN node.name, score""")

        # Should only return the 2 Special nodes
        self.env.assertEqual(len(result.result_set), 2)

        # Verify node labels
        labels = set()
        for record in result.result_set:
            node_name = record[0]
            labels.add(node_name)

        self.env.assertEqual(labels, {"S1", "S2"})

        # Since S2 has an incoming edge and S1 doesn't, S2 should have a higher score
        scores = {}
        for record in result.result_set:
            node_name = record[0]
            score = record[1]
            scores[node_name] = score

        self.env.assertGreater(scores["S2"], scores["S1"])

    def test_pagerank_empty_graph(self):
        """Test PageRank on an empty graph"""
        # Run PageRank on the empty graph
        result = self.graph.query("CALL algo.pageRank(NULL, NULL) YIELD node, score")

        # Should return an empty result set
        self.env.assertEqual(len(result.result_set), 0)

