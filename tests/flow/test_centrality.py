from common import *

GRAPH_ID = "centrality"


class testCentrality(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def centrality(
        self,
        nodeLabels=None,
        relationshipTypes=None,
        weightAttribute=None,
        defaultWeight=None,
    ):
        config = {}

        if nodeLabels is not None:
            config["nodeLabels"] = nodeLabels

        if relationshipTypes is not None:
            config["relationshipTypes"] = relationshipTypes

        if weightAttribute is not None:
            config["weightAttribute"] = weightAttribute

        if defaultWeight is not None:
            config["defaultWeight"] = defaultWeight

        return self.graph.query(
            """CALL algo.Centrality($config)
                                   YIELD node, score
                                   RETURN node.name, score
                                   ORDER BY score DESC""",
            {"config": config},
        )

    def test01_invalid_invocation(self):
        invalid_queries = [
            # non-array nodeLabels parameter
            """CALL algo.Centrality({nodeLabels: 'Person'})""",
            # non-array relationshipTypes parameter
            """CALL algo.Centrality({relationshipTypes: 'KNOWS'})""",
            # unexpected extra parameter
            """CALL algo.Centrality({invalidParam: 'value'})""",
            # integer values in nodeLabels array
            """CALL algo.Centrality({nodeLabels: [1, 2, 3]})""",
            # integer values in relationshipTypes array
            """CALL algo.Centrality({relationshipTypes: [1, 2, 3]})""",
            # non-map config argument
            """CALL algo.Centrality('invalid')""",
            # non-existent yield field
            """CALL algo.Centrality(null) YIELD node, invalidField""",
            # non-integer defaultWeight
            """CALL algo.Centrality({defaultWeight: 'notanint'})""",
            # non-string weightAttribute
            """CALL algo.Centrality({weightAttribute: 123})""",
            # too many unknown fields
            """CALL algo.Centrality({nodeLabels: ['Person'],
                           relationshipTypes: ['KNOWS'],
                           invalidParam: 'value'})""",
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except:
                pass

    def test02_centrality_empty_graph(self):
        """Test harmonic centrality on an empty graph returns no results"""
        result = self.graph.query("MATCH (n) RETURN count(n) as count")
        self.env.assertEqual(result.result_set[0][0], 0)

        result = self.centrality()
        self.env.assertEqual(len(result.result_set), 0)

    def test03_centrality_basic(self):
        """Test harmonic centrality on a directed path graph.

        Graph (directed):  A --> B --> C --> D

        Harmonic closeness centrality:
          A: reaches B(d=1), C(d=2), D(d=3)  → score ≈ 1 + 0.5 + 0.33 = 1.83
          B: reaches C(d=1), D(d=2)           → score ≈ 1 + 0.5     = 1.5
          C: reaches D(d=1)                   → score ≈ 1
          D: no outgoing edges                → score = 0.0 (exact)

        Expected ordering: A > B > C > D  (with D == 0)
        """
        self.graph.query("""
            CREATE
            (a:Node {name: 'A'}),
            (b:Node {name: 'B'}),
            (c:Node {name: 'C'}),
            (d:Node {name: 'D'}),
            (a)-[:EDGE]->(b),
            (b)-[:EDGE]->(c),
            (c)-[:EDGE]->(d)
        """)

        result = self.centrality()
        scores = {row[0]: row[1] for row in result.result_set}

        # ordering assertions (HLL gives accurate estimates for small cardinalities)
        self.env.assertGreater(scores["A"], scores["B"])
        self.env.assertGreater(scores["B"], scores["C"])
        self.env.assertGreater(scores["C"], scores["D"])

        # D has no outgoing edges — score is exactly 0
        self.env.assertEqual(scores["D"], 0.0)

    def test04_centrality_star(self):
        """Test harmonic centrality on a directed star graph.

        Graph (directed): Center --> L1, L2, L3, L4, L5

        Center reaches all leaves at distance 1 → score ≈ 5.
        All leaves have no outgoing edges → score = 0.0 (exact).
        """
        self.graph.query("""
            CREATE
            (c:Node {name: 'Center'}),
            (l1:Node {name: 'L1'}),
            (l2:Node {name: 'L2'}),
            (l3:Node {name: 'L3'}),
            (l4:Node {name: 'L4'}),
            (l5:Node {name: 'L5'}),
            (c)-[:EDGE]->(l1),
            (c)-[:EDGE]->(l2),
            (c)-[:EDGE]->(l3),
            (c)-[:EDGE]->(l4),
            (c)-[:EDGE]->(l5)
        """)

        result = self.centrality()
        scores = {row[0]: row[1] for row in result.result_set}

        # center has the highest score
        for leaf in ["L1", "L2", "L3", "L4", "L5"]:
            self.env.assertGreater(scores["Center"], scores[leaf])

        # all leaves have no outgoing edges — score is exactly 0
        for leaf in ["L1", "L2", "L3", "L4", "L5"]:
            self.env.assertEqual(scores[leaf], 0.0)

    def test05_centrality_node_labels(self):
        """Test harmonic centrality with nodeLabels filter.

        Graph (directed):
          A:Person --> B:Person --> C:Person
                       |
                       v
                       D:Location --> E:Company

        Running with nodeLabels=['Person'] should return only A, B, C.
        B should have highest score among Person nodes (it reaches C via KNOWS).
        """
        self.graph.query("""
            CREATE
            (a:Person {name: 'A'}),
            (b:Person {name: 'B'}),
            (c:Person {name: 'C'}),
            (d:Location {name: 'D'}),
            (e:Company {name: 'E'}),
            (a)-[:KNOWS]->(b),
            (b)-[:KNOWS]->(c),
            (b)-[:WORKS_AT]->(d),
            (d)-[:LOCATED_IN]->(e)
        """)

        result = self.centrality(nodeLabels=["Person"])
        scores = {row[0]: row[1] for row in result.result_set}

        # only Person nodes returned
        self.env.assertEqual(len(scores), 3)
        self.env.assertContains("A", scores)
        self.env.assertContains("B", scores)
        self.env.assertContains("C", scores)
        self.env.assertNotContains("D", scores)
        self.env.assertNotContains("E", scores)

        # A reaches B and C through Person subgraph → highest score
        self.env.assertGreater(scores["A"], scores["B"])
        self.env.assertGreater(scores["B"], scores["C"])

        # C has no outgoing Person edges
        self.env.assertEqual(scores["C"], 0.0)

    def test06_centrality_relationship_types(self):
        """Test harmonic centrality with relationshipTypes filter.

        Graph (directed):
          A --FRIEND--> B --FRIEND--> C
                        |
                     WORKS_AT
                        |
                        v
                        D --NEAR--> E

        Running with relationshipTypes=['FRIEND'] should only traverse FRIEND edges.
        A should have the highest score as it can reach B and C via FRIEND edges.
        D and E have no FRIEND outgoing edges, so score == 0.
        """
        self.graph.query("""
            CREATE
            (a:Node {name: 'A'}),
            (b:Node {name: 'B'}),
            (c:Node {name: 'C'}),
            (d:Node {name: 'D'}),
            (e:Node {name: 'E'}),
            (a)-[:FRIEND]->(b),
            (b)-[:FRIEND]->(c),
            (b)-[:WORKS_AT]->(d),
            (d)-[:NEAR]->(e)
        """)

        result = self.centrality(relationshipTypes=["FRIEND"])
        scores = {row[0]: row[1] for row in result.result_set}

        # all nodes returned
        self.env.assertEqual(len(scores), 5)

        # A reaches B and C via FRIEND, B reaches only C
        self.env.assertGreater(scores["A"], scores["B"])
        self.env.assertGreater(scores["B"], scores["C"])

        # C, D, E have no outgoing FRIEND edges
        self.env.assertEqual(scores["C"], 0.0)
        self.env.assertEqual(scores["D"], 0.0)
        self.env.assertEqual(scores["E"], 0.0)

        # compare to full graph traversal — A's score should differ
        result_full = self.centrality()
        scores_full = {row[0]: row[1] for row in result_full.result_set}

        self.env.assertNotEqual(scores["A"], scores_full["A"])

    def test07_centrality_combined_parameters(self):
        """Test harmonic centrality with both nodeLabels and relationshipTypes.

        Graph (directed):
          A:Person --FRIEND--> B:Person --FRIEND--> C:Person
                               |
                            MANAGES
                               |
                               v
                          D:Employee --WORKS_WITH--> E:Employee

        With nodeLabels=['Person'] and relationshipTypes=['FRIEND']:
          Only Person nodes are included, only FRIEND edges are traversed.
          A: reaches B(d=1), C(d=2) → highest score
          B: reaches C(d=1)
          C: no outgoing FRIEND edges → score = 0
        """
        self.graph.query("""
            CREATE
            (a:Person {name: 'A'}),
            (b:Person {name: 'B'}),
            (c:Person {name: 'C'}),
            (d:Employee {name: 'D'}),
            (e:Employee {name: 'E'}),
            (a)-[:FRIEND]->(b),
            (b)-[:FRIEND]->(c),
            (b)-[:MANAGES]->(d),
            (d)-[:WORKS_WITH]->(e)
        """)

        result = self.centrality(nodeLabels=["Person"], relationshipTypes=["FRIEND"])
        scores = {row[0]: row[1] for row in result.result_set}

        # only Person nodes returned
        self.env.assertEqual(len(scores), 3)
        self.env.assertContains("A", scores)
        self.env.assertContains("B", scores)
        self.env.assertContains("C", scores)
        self.env.assertNotContains("D", scores)
        self.env.assertNotContains("E", scores)

        # A reaches B and C, B reaches only C
        self.env.assertGreater(scores["A"], scores["B"])
        self.env.assertGreater(scores["B"], scores["C"])

        # C has no outgoing FRIEND edges within Person subgraph
        self.env.assertEqual(scores["C"], 0.0)

    def test08_centrality_all_zero_weights(self):
        """Test that all scores are 0 when every node has weight attribute = 0.

        When all node weights are 0, every HLL sketch is initialized with
        0 hashes (no seeds). No propagation can ever start, so all scores
        must be exactly 0.0 regardless of graph topology.
        """
        # a "weird" graph with several edges but all nodes have w=0
        self.graph.query("""
            CREATE
            (a:Node {name: 'A', w: 0}),
            (b:Node {name: 'B', w: 0}),
            (c:Node {name: 'C', w: 0}),
            (d:Node {name: 'D', w: 0}),
            (a)-[:EDGE]->(b),
            (b)-[:EDGE]->(c),
            (c)-[:EDGE]->(a),
            (a)-[:EDGE]->(d),
            (d)-[:EDGE]->(b)
        """)

        result = self.centrality(weightAttribute="w", defaultWeight=0)
        scores = {row[0]: row[1] for row in result.result_set}

        # all nodes are returned (defaultWeight=0 keeps them all)
        self.env.assertEqual(len(scores), 4)

        # with all HLL sketches empty, no information propagates -> all zeros
        for name in ["A", "B", "C", "D"]:
            self.env.assertEqual(scores[name], 0.0)

    def test09_centrality_weighted(self):
        """Test harmonic centrality with weightAttribute.

        Graph (directed): S0, S1, ..., S999 each -EDGE-> A -EDGE-> B (B.score=1)

        The `score` attribute seeds each node's HLL sketch (weight = # hashes).

        Case 1 - defaultWeight=0:
          All 1002 nodes included. Only B seeds with 1 hash (B.score=1).
          A reaches B at d=1          -> score(A) ~= 1.
          Each Si reaches B at d=2    -> score(Si) ~= 0.5.
          B has no outgoing edges     -> score(B) = 0 (exact).

        Case 2 - no defaultWeight:
          Only B is included (only B has the `score` attribute).
          B has no outgoing edges     -> score(B) = 0; only B is returned.
        """
        # build hub graph; only B carries the score attribute
        parts = ["CREATE (b:Hub {name: 'B', score: 1}), (a:Hub {name: 'A'})"]
        for i in range(1000):
            parts.append(f"(s{i}:Hub {{name: 'S{i}'}})")
        create_q = parts[0] + ", " + ", ".join(parts[1:])
        create_q += ", (a)-[:EDGE]->(b)"
        for i in range(1000):
            create_q += f", (s{i})-[:EDGE]->(a)"
        self.graph.query(create_q)

        # ---- Case 1: defaultWeight=0, all nodes in subgraph ----
        result1 = self.centrality(weightAttribute="score", defaultWeight=0)
        scores1 = {row[0]: row[1] for row in result1.result_set}

        self.env.assertEqual(len(scores1), 1002)

        # B: no outgoing edges -> exact 0
        self.env.assertEqual(scores1["B"], 0.0)

        # A: reaches B at d=1 -> score ~= 1
        self.env.assertGreater(scores1["A"], 0.4)
        self.env.assertLess(scores1["A"], 1.6)

        # source nodes: reach B at d=2 via A -> score ~= 0.5
        for i in range(1000):
            self.env.assertGreater(scores1[f"S{i}"], 0.1)
            self.env.assertLess(scores1[f"S{i}"], 0.9)

        # A should score higher than any individual source node
        self.env.assertGreater(scores1["A"], scores1["S0"])

        # ---- Case 2: no defaultWeight, only B participates ----
        result2 = self.centrality(weightAttribute="score")
        scores2 = {row[0]: row[1] for row in result2.result_set}

        # only B has the score attribute -> only B is returned
        self.env.assertEqual(len(scores2), 1)
        self.env.assertContains("B", scores2)

        # B has no outgoing edges in the single-node subgraph -> score = 0
        self.env.assertEqual(scores2["B"], 0.0)
