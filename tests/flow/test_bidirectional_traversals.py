from common import *

class testBidirectionalTraversals(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.populate_acyclic_graph()
        self.populate_cyclic_graph()

    def populate_acyclic_graph(self):
        self.acyclic_graph = self.db.select_graph("G")
        # Construct a graph with the form:
        # (v1)-[:E]->(v2)-[:E]->(v3)
        node_props = ['v1', 'v2', 'v3']

        nodes = []
        for idx, v in enumerate(node_props):
            node = Node(alias=f"n_{idx}", labels="L", properties={"val": v})
            nodes.append(node)

        e0 = Edge(nodes[0], "E", nodes[1], properties={"val": 0})
        e1 = Edge(nodes[1], "E", nodes[2], properties={"val": 1})

        nodes_str = [str(n) for n in nodes]
        query = f"CREATE {','.join(nodes_str)}, {e0}, {e1}"
        self.acyclic_graph.query(query)

    def populate_cyclic_graph(self):
        self.graph_with_cycle = self.db.select_graph("H")
        # Construct a graph with the form:
        # (v1)-[:E]->(v2)-[:E]->(v3), (v2)-[:E]->(v1)
        node_props = ['v1', 'v2', 'v3']

        nodes = []
        for idx, v in enumerate(node_props):
            nodes.append(Node(alias=f"n_{idx}", labels="L", properties={"val": v}))

        e0 = Edge(nodes[0], "E", nodes[1])
        e1 = Edge(nodes[1], "E", nodes[2])
        e3 = Edge(nodes[1], "E", nodes[0]) # Introduce a cycle between v2 and v1.

        nodes_str = [str(n) for n in nodes]
        query = f"CREATE {','.join(nodes_str)}, {e0}, {e1}, {e3}"
        self.graph_with_cycle.query(query)

    # Test traversals that don't specify an edge direction.
    def test01_bidirectional_traversals(self):
        query = """MATCH (a)-[:E]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.acyclic_graph.query(query)
        # Each relation should appear twice with the source and destination swapped in the second result.
        expected_result = [['v1', 'v2'],
                           ['v2', 'v1'],
                           ['v2', 'v3'],
                           ['v3', 'v2']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Test undirected traversals with a referenced edge.
        query = """MATCH (a)-[e:E]-(b) RETURN ID(e), a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.acyclic_graph.query(query)
        expected_result = [[0, 'v1', 'v2'],
                           [0, 'v2', 'v1'],
                           [1, 'v2', 'v3'],
                           [1, 'v3', 'v2']]
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Test 0-hop undirected traversals.
    def test02_bidirectional_zero_hop_traversals(self):
        query = """MATCH (a)-[*0]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.acyclic_graph.query(query)
        expected_result = [['v1', 'v1'],
                           ['v2', 'v2'],
                           ['v3', 'v3']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # TODO doesn't work - returns each node with itself as source and destination in adition to expected results.
        # Test combinations of directed and undirected traversals.
        #  query = """MATCH (a)-[:E]->()-[]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        #  actual_result = self.acyclic_graph.query(query)
        #  expected_result = [['v1', 'v3']]
        #  self.env.assertEqual(actual_result.result_set, expected_result)

        # TODO doesn't work for the same reason.
        # Test fixed-length multi-hop undirected traversals.
        #  query = """MATCH (a)-[:E*2]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        #  actual_result = self.acyclic_graph.query(query)
        #  expected_result = [[0, 'v1', 'v3'],
                           #  [0, 'v3', 'v1']]
        #  self.env.assertEqual(actual_result.result_set, expected_result)

    # Test variable-length traversals that don't specify an edge direction.
    def test03_bidirectional_variable_length_traversals(self):
        query = """MATCH (a)-[*]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.acyclic_graph.query(query)
        # Each combination of distinct node source and destination should appear once.
        expected_result = [['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v2', 'v1'],
                           ['v2', 'v3'],
                           ['v3', 'v1'],
                           ['v3', 'v2']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Should generate the same results as the previous query.
        query = """MATCH (a)-[*1..2]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.acyclic_graph.query(query)
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Test collecting self and all direct neighbors.
    def test04_bidirectional_variable_bounded_length_traversals(self):
        query = """MATCH (a)-[*0..1]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.acyclic_graph.query(query)
        # Each combination of distinct node source and destination should appear once.
        expected_result = [['v1', 'v1'],
                           ['v1', 'v2'],
                           ['v2', 'v1'],
                           ['v2', 'v2'],
                           ['v2', 'v3'],
                           ['v3', 'v2'],
                           ['v3', 'v3']]
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Test bidirectional query on nonexistent edge.
    def test05_bidirectional_variable_length_traversals_over_nonexistent_type(self):
        query = """MATCH (a)-[:NONEXISTENT*]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.acyclic_graph.query(query)
        expected_result = []
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Test bidirectional query on real edge or nonexistent edge.
    def test06_bidirectional_variable_length_traversals_over_partial_existing_types(self):
        query = """MATCH (a)-[:NONEXISTENT|:E*]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.acyclic_graph.query(query)
        # Each combination of distinct node source and destination should appear once.
        expected_result = [['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v2', 'v1'],
                           ['v2', 'v3'],
                           ['v3', 'v1'],
                           ['v3', 'v2']]
        self.env.assertEqual(actual_result.result_set, expected_result)

    # TODO returns 16 rows; 18 rows expected.
    # The missing two rows are both `['v2', 'v3']
    # Test bidirectional query on two real edge types.
    #  def test07_bidirectional_variable_length_traversals_over_multiple_existing_types(self):
        #  # Generate new dest->src edges between every current src->dest pair.
        #  query = """MATCH (a {val: 'v1'})-[e]->(b {val: 'v2'}) CREATE (a)-[:CLONE]->(b)"""
        #  actual_result = self.acyclic_graph.query(query)
        #  self.env.assertEqual(actual_result.relationships_created, 1)

        #  query = """MATCH (a)-[:E|:CLONE*]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        #  actual_result = self.acyclic_graph.query(query)
        #  expected_result = [['v1', 'v1'],
                           #  ['v1', 'v1'],
                           #  ['v1', 'v2'],
                           #  ['v1', 'v2'],
                           #  ['v1', 'v3'],
                           #  ['v1', 'v3'],
                           #  ['v2', 'v1'],
                           #  ['v2', 'v1'],
                           #  ['v2', 'v2'],
                           #  ['v2', 'v2'],
                           #  ['v2', 'v3'],
                           #  ['v2', 'v3'],
                           #  ['v2', 'v3'],
                           #  ['v3', 'v1'],
                           #  ['v3', 'v1'],
                           #  ['v3', 'v2'],
                           #  ['v3', 'v2'],
                           #  ['v3', 'v2']]
        #  self.env.assertEqual(actual_result.result_set, expected_result)

    # Test bidirectional query on two real edge types.
    def test08_bidirectional_variable_bounded_length_traversals_over_multiple_existing_types(self):
        # Generate one new edge between v1 and v2.
        query = """MATCH (a {val: 'v1'})-[e]->(b {val: 'v2'}) CREATE (a)-[:CLONE]->(b)"""
        actual_result = self.acyclic_graph.query(query)
        self.env.assertEqual(actual_result.relationships_created, 1)

        query = """MATCH (a)-[:E|:CLONE*1..2]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.acyclic_graph.query(query)
        expected_result = [['v1', 'v1'],
                           ['v1', 'v1'],
                           ['v1', 'v2'],
                           ['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v1', 'v3'],
                           ['v2', 'v1'],
                           ['v2', 'v1'],
                           ['v2', 'v2'],
                           ['v2', 'v2'],
                           ['v2', 'v3'],
                           ['v3', 'v1'],
                           ['v3', 'v1'],
                           ['v3', 'v2']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Delete cloned edge.
        query = """MATCH ()-[e:CLONE]->() DELETE e"""
        actual_result = self.acyclic_graph.query(query)
        self.env.assertEqual(actual_result.relationships_deleted, 1)

    # Test traversals that don't specify an edge direction in a graph with a cycle.
    def test09_bidirectional_traversals_with_cycle(self):
        # Test undirected traversals with a referenced edge.
        # TODO The variant query in which the edge is not referenced does not work:
        #  query = """MATCH (a)-[:E]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        query = """MATCH (a)-[e:E]-(b) RETURN ID(e) AS id, a.val, b.val ORDER BY id, a.val, b.val"""
        actual_result = self.graph_with_cycle.query(query)
        # Each relation should appear twice with the source and destination swapped in the second result.
        expected_result = [[0, 'v1', 'v2'],
                           [0, 'v2', 'v1'],
                           [1, 'v2', 'v3'],
                           [1, 'v3', 'v2'],
                           [2, 'v1', 'v2'],
                           [2, 'v2', 'v1']]
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Test variable-length traversals that don't specify an edge direction.
    def test10_bidirectional_variable_length_traversals_with_cycle(self):
        # TODO returns 16 rows; 18 rows expected.
        # The missing two rows are both `['v2', 'v3']
        #  query = """MATCH (a)-[*]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""

        query = """MATCH (a)-[*1..2]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph_with_cycle.query(query)
        # Each src/dest pair (including when the source and dest are the same) is returned twice
        # except for (v2)-[]->(v3), which correctly only occurs once as the missing traversal pattern takes 3 hops.
        expected_result = [['v1', 'v1'],
                           ['v1', 'v1'],
                           ['v1', 'v2'],
                           ['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v1', 'v3'],
                           ['v2', 'v1'],
                           ['v2', 'v1'],
                           ['v2', 'v2'],
                           ['v2', 'v2'],
                           ['v2', 'v3'],
                           ['v3', 'v1'],
                           ['v3', 'v1'],
                           ['v3', 'v2']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Collect self and all direct neighbors with the pattern (v1)-[]-(v2) repeated.
        query = """MATCH (a)-[*0..1]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph_with_cycle.query(query)
        expected_result = [['v1', 'v1'],
                           ['v1', 'v2'],
                           ['v1', 'v2'],
                           ['v2', 'v1'],
                           ['v2', 'v1'],
                           ['v2', 'v2'],
                           ['v2', 'v3'],
                           ['v3', 'v2'],
                           ['v3', 'v3']]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test11_bidirectional_multiple_edge_type(self):
        # Construct a simple graph:
        # (a)-[E1]->(b), (c)-[E2]->(d)

        g = self.db.select_graph("multi_edge_type")

        a  = Node(alias='a', properties={'val': 'a'})
        b  = Node(alias='b', properties={'val': 'b'})
        c  = Node(alias='c', properties={'val': 'c'})
        d  = Node(alias='d', properties={'val': 'd'})
        ab = Edge(a, "E1", b)
        cd = Edge(c, "E2", d)

        g.query(f"CREATE {a}, {b}, {c}, {d}, {ab}, {cd}")

        query = """MATCH (a)-[:E1|:E2]-(z) RETURN a.val, z.val ORDER BY a.val, z.val"""
        actual_result = g.query(query)

        expected_result = [['a', 'b'],
                           ['b', 'a'],
                           ['c', 'd'],
                           ['d', 'c']]

        self.env.assertEqual(actual_result.result_set, expected_result)

    # Test bidirectional traversals resolved by an ExpandInto op.
    def test12_bidirectional_expand_into(self):
        query = """MATCH (a), (b) WITH a, b MATCH (a)-[e:E]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.acyclic_graph.query(query)
        # Each relation should appear twice with the source and destination swapped in the second result.
        expected_result = [['v1', 'v2'],
                           ['v2', 'v1'],
                           ['v2', 'v3'],
                           ['v3', 'v2']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Verify result against the equivalent conditional traversal.
        query = """MATCH (a)-[:E]-(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        traverse_result = self.acyclic_graph.query(query)
        self.env.assertEqual(actual_result.result_set, traverse_result.result_set)

        # Test undirected traversals with a referenced edge.
        query = """MATCH (a), (b) WITH a, b MATCH (a)-[e:E]-(b) RETURN e.val, a.val, b.val ORDER BY e.val, a.val, b.val"""
        actual_result = self.acyclic_graph.query(query)
        expected_result = [[0, 'v1', 'v2'],
                           [0, 'v2', 'v1'],
                           [1, 'v2', 'v3'],
                           [1, 'v3', 'v2']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Verify result against the equivalent conditional traversal.
        query = """MATCH (a)-[e:E]-(b) RETURN e.val, a.val, b.val ORDER BY e.val, a.val, b.val"""
        traverse_result = self.acyclic_graph.query(query)
        self.env.assertEqual(actual_result.result_set, traverse_result.result_set)

    def test13_multiple_bidirectional_edges(self):
        # Traverse over 2 bidirectional edges.
        # Cypher's relationship-uniqueness forbids binding the same edge to
        # both patterns, so v1 can only reach v3 (and vice versa) — a result
        # like ['v1', 'v1'] would require walking (e0) back and forth.
        query = """MATCH (a)-[]-()-[]-(c) RETURN a.val, c.val ORDER BY a.val, c.val"""

        actual_result = self.acyclic_graph.query(query)
        expected_result = [['v1', 'v3'],
                           ['v3', 'v1']]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test14_relationship_uniqueness(self):
        # A single edge cannot satisfy a two-hop pattern: matching would have
        # to reuse it for both hops, which Cypher's relationship-uniqueness
        # forbids (issue #2441). Anonymous edges are no exception.
        g = self.db.select_graph("single_edge")
        g.query("CREATE (:L {val: 'v1'})-[:E]->(:L {val: 'v2'})")

        for pattern in ["(a)-[:E]-(b)-[:E]-(c)",
                        "(a)-[e1:E]-(b)-[e2:E]-(c)",
                        "(a)-[e1:E]-(b)-[:E]-(c)",
                        "(a)-[:E]-(b)-[e2:E]-(c)",
                        "(a)--(b)--(c)",
                        "(a)-[]-(b)-[]-(c)"]:
            actual_result = g.query(f"MATCH {pattern} RETURN count(*)")
            self.env.assertEqual(actual_result.result_set, [[0]])

        # a named path over the same pattern is bound by the same rule
        actual_result = g.query("MATCH p = (a)-[:E]-(b)-[:E]-(c) RETURN count(*)")
        self.env.assertEqual(actual_result.result_set, [[0]])

        # a single hop still matches, in both directions
        actual_result = g.query("MATCH (a)-[:E]-(b) RETURN a.val, b.val ORDER BY a.val")
        self.env.assertEqual(actual_result.result_set, [['v1', 'v2'], ['v2', 'v1']])

        # a fresh MATCH clause starts a new uniqueness scope
        actual_result = g.query("MATCH (a)-[:E]->(b) MATCH (c)-[:E]->(d) RETURN count(*)")
        self.env.assertEqual(actual_result.result_set, [[1]])

        # distinct edges satisfy the pattern
        actual_result = self.acyclic_graph.query(
                "MATCH (a)-[:E]-(b)-[:E]-(c) RETURN a.val, b.val, c.val ORDER BY a.val")
        self.env.assertEqual(actual_result.result_set, [['v1', 'v2', 'v3'],
                                                        ['v3', 'v2', 'v1']])

    def test15_relationship_uniqueness_disjoint_types(self):
        # Hops restricted to disjoint relationship types can never bind the
        # same edge, so uniqueness must not filter anything out here.
        g = self.db.select_graph("two_types")
        g.query("""CREATE (a:L {val: 'v1'})-[:E]->(b:L {val: 'v2'}),
                          (b)-[:F]->(:L {val: 'v3'})""")

        actual_result = g.query(
                "MATCH (a)-[:E]->(b)-[:F]->(c) RETURN a.val, b.val, c.val")
        self.env.assertEqual(actual_result.result_set, [['v1', 'v2', 'v3']])

        # a self-loop is a single edge: it cannot fill both hops
        loop = self.db.select_graph("self_loop")
        loop.query("CREATE (n:L {val: 'v1'})-[:E]->(n)")
        actual_result = loop.query("MATCH (a)-[:E]->(b)-[:E]->(c) RETURN count(*)")
        self.env.assertEqual(actual_result.result_set, [[0]])

