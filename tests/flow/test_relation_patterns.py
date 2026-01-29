from common import *

GRAPH_ID = "relation_patterns"


class testRelationPattern(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        # Construct a graph with the form:
        # (v1)-[:e]->(v2)-[:e]->(v3)
        node_props = ['v1', 'v2', 'v3']

        nodes = []
        for idx, v in enumerate(node_props):
            node = Node(alias=f"n_{idx}", labels="L", properties={"val": v})
            nodes.append(node)

        e0 = Edge(nodes[0], "e", nodes[1])
        e1 = Edge(nodes[1], "e", nodes[2])

        self.graph.query(f"CREATE {nodes[0]}, {nodes[1]}, {nodes[2]}, {e0}, {e1}")

    # Test patterns that traverse 1 edge.
    def test01_one_hop_traversals(self):
        # Conditional traversal with label
        query = """MATCH (a)-[:e]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        result_a = self.graph.query(query)

        # Conditional traversal without label
        query = """MATCH (a)-[]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        result_b = self.graph.query(query)

        # Fixed-length 1-hop traversal with label
        query = """MATCH (a)-[:e*1]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        result_c = self.graph.query(query)

        # Fixed-length 1-hop traversal without label
        query = """MATCH (a)-[*1]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        result_d = self.graph.query(query)

        # default minimum length is 1
        # the following query is equivalent to:
        # MATCH (a)-[]->(b) RETURN a.val, b.val ORDER BY a.val, b.val
        query = """MATCH (a)-[*..1]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        result_e = self.graph.query(query)

        self.env.assertEquals(result_b.result_set, result_a.result_set)
        self.env.assertEquals(result_c.result_set, result_a.result_set)
        self.env.assertEquals(result_d.result_set, result_a.result_set)
        self.env.assertEquals(result_e.result_set, result_a.result_set)

    # Test patterns that traverse 2 edges.
    def test02_two_hop_traversals(self):
        # Conditional two-hop traversal without referenced intermediate node
        query = """MATCH (a)-[:e]->()-[:e]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Fixed-length two-hop traversal (same expected result)
        query = """MATCH (a)-[:e*2]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Variable-length traversal with a minimum bound of 2 (same expected result)
        query = """MATCH (a)-[*2..]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Conditional two-hop traversal with referenced intermediate node
        query = """MATCH (a)-[:e]->(b)-[:e]->(c) RETURN a.val, b.val, c.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2', 'v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Test variable-length patterns
    def test03_var_len_traversals(self):
        # Variable-length traversal with label
        query = """MATCH (a)-[:e*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v2', 'v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Variable-length traversal without label (same expected result)
        query = """MATCH (a)-[*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Variable-length traversal with bounds 1..2 (same expected result)
        query = """MATCH (a)-[:e*1..2]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Variable-length traversal with bounds 0..1
        # This will return every node and itself, as well as all
        # single-hop edges.
        query = """MATCH (a)-[:e*0..1]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v1'],
                           ['v1', 'v2'],
                           ['v2', 'v2'],
                           ['v2', 'v3'],
                           ['v3', 'v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Test variable-length patterns with alternately labeled source
    # and destination nodes, which can cause different execution sequences.
    def test04_variable_length_labeled_nodes(self):
        # Source and edge labeled variable-length traversal
        query = """MATCH (a:L)-[:e*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v2', 'v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Destination and edge labeled variable-length traversal (same expected result)
        query = """MATCH (a)-[:e*]->(b:L) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Source labeled variable-length traversal (same expected result)
        query = """MATCH (a:L)-[*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Destination labeled variable-length traversal (same expected result)
        query = """MATCH (a)-[*]->(b:L) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Test traversals over explicit relationship types
    def test05_relation_types(self):
        # Add two nodes and two edges of a new type.
        # The new form of the graph will be:
        # (v1)-[:e]->(v2)-[:e]->(v3)-[:q]->(v4)-[:q]->(v5)
        query = """MATCH (n {val: 'v3'}) CREATE (n)-[:q]->(:L {val: 'v4'})-[:q]->(:L {val: 'v5'})"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.nodes_created, 2)
        self.env.assertEquals(actual_result.relationships_created, 2)

        # Verify the graph structure
        query = """MATCH (a)-[e]->(b) RETURN a.val, b.val, TYPE(e) ORDER BY TYPE(e), a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2', 'e'],
                           ['v2', 'v3', 'e'],
                           ['v3', 'v4', 'q'],
                           ['v4', 'v5', 'q']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Verify conditional traversals with explicit relation types
        query = """MATCH (a)-[:e]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v2', 'v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (a)-[:q]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v3', 'v4'],
                           ['v4', 'v5']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Verify conditional traversals with multiple explicit relation types
        query = """MATCH (a)-[e:e|:q]->(b) RETURN a.val, b.val, TYPE(e) ORDER BY TYPE(e), a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2', 'e'],
                           ['v2', 'v3', 'e'],
                           ['v3', 'v4', 'q'],
                           ['v4', 'v5', 'q']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Verify variable-length traversals with explicit relation types
        query = """MATCH (a)-[:e*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v2', 'v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (a)-[:q*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v3', 'v4'],
                           ['v3', 'v5'],
                           ['v4', 'v5']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Verify variable-length traversals with multiple explicit relation types
        query = """MATCH (a)-[:e|:q*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v1', 'v4'],
                           ['v1', 'v5'],
                           ['v2', 'v3'],
                           ['v2', 'v4'],
                           ['v2', 'v5'],
                           ['v3', 'v4'],
                           ['v3', 'v5'],
                           ['v4', 'v5']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Test traversals over transposed edge matrices.
    def test06_transposed_traversals(self):
        # The intermediate node 'b' will be used to form the scan operation because it is filtered.
        # As such, one of the traversals must be transposed.
        query = """MATCH (a)-[e]->(b {val:'v3'})-[]->(c:L) RETURN COUNT(e)"""
        plan = str(self.graph.explain(query))

        # Verify that the execution plan contains two traversals following opposing edge directions.
        self.env.assertIn("<-", plan)
        self.env.assertIn("->", plan)

        # Verify results.
        actual_result = self.graph.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test07_transposed_multi_hop(self):
        redis_con = self.env.getConnection()
        g = self.db.select_graph("tran_multi_hop")

        # (a)-[R]->(b)-[R]->(c)<-[R]-(d)<-[R]-(e)
        a  = Node(alias='a', properties={"val": 'a'})
        b  = Node(alias='b', properties={"val": 'b'})
        c  = Node(alias='c', properties={"val": 'c'})
        d  = Node(alias='d', properties={"val": 'd'})
        e  = Node(alias='e', properties={"val": 'e'})
        ab = Edge(a, "R", b)
        bc = Edge(b, "R", c)
        ed = Edge(e, "R", d)
        dc = Edge(d, "R", c)

        g.query(f"CREATE {a}, {b}, {c}, {d}, {e}, {ab}, {bc}, {ed}, {dc}")

        q = """MATCH (a)-[*2]->(b)<-[*2]-(c) RETURN a.val, b.val, c.val ORDER BY a.val, b.val, c.val"""
        actual_result = g.query(q)
        expected_result = [['a', 'c', 'a'], ['a', 'c', 'e'], ['e', 'c', 'a'], ['e', 'c', 'e']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test08_transposed_varlen_traversal(self):
        # Verify that variable-length traversals with nested transpose operations perform correctly.
        query = """MATCH (a {val: 'v1'})-[*]-(b {val: 'v2'})-[:e]->(:L {val: 'v3'}) RETURN a.val ORDER BY a.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test09_transposed_elem_order(self):
        redis_con = self.env.getConnection()
        g = self.db.select_graph("transpose_patterns")

        # Create a new graph of the form:
        # (A)<-[1]-(B)-[2]->(C)
        g.query("CREATE (a:A)<-[:E {val:'ba'}]-(b:B)-[:E {val:'bc'}]->(c:C)")

        queries = ["MATCH (a:A)<-[e1]-(b:B)-[e2]->(c:C) RETURN e1.val, e2.val",
                   "MATCH (a:A) WITH a MATCH (a)<-[e1]-(b:B)-[e2]->(c:C) RETURN e1.val, e2.val",
                   "MATCH (b:B) WITH b MATCH (a:A)<-[e1]-(b)-[e2]->(c:C) RETURN e1.val, e2.val",
                   "MATCH (c:C) WITH c MATCH (a:A)<-[e1]-(b:B)-[e2]->(c) RETURN e1.val, e2.val",
                   ]
        expected_result = [['ba', 'bc']]
        for query in queries:
            actual_result = g.query(query)
            self.env.assertEquals(actual_result.result_set, expected_result)

    def test10_triple_edge_type(self):
        # Construct a simple graph:
        # (A)-[X]->(B)
        # (A)-[Y]->(C)
        # (A)-[Z]->(D)
        g = self.db.select_graph("triple_edge_type")
        q = "CREATE(a:A), (b:B), (c:C), (d:D), (a)-[:X]->(b), (a)-[:Y]->(c), (a)-[:Z]->(d)"
        g.query(q)

        labels = ['X', 'Y', 'Z']
        expected_result = [[['B']], [['C']], [['D']]]

        q = "MATCH (a)-[:{L0}|:{L1}|:{L2}]->(b) RETURN labels(b) AS label ORDER BY label"
        import itertools
        for perm in itertools.permutations(labels):
            res = g.query(q.format(L0=perm[0], L1=perm[1], L2=perm[2]))
            self.env.assertEquals(res.result_set, expected_result)

    def test11_shared_node_detection(self):
        # Construct a simple graph
        # (s)<-[:A]-(x)
        # (x)<-[:B]-(x)
        # (x)-[:B]->(t)
        # (t)<-[:B]-(x)
        g = self.db.select_graph("shared_node")
        q = "MERGE (s)<-[:A]-(x)<-[:B]-(x)-[:B]->(t)<-[:B]-(x)"
        result = g.query(q)

        self.env.assertEquals(result.nodes_created, 3)
        self.env.assertEquals(result.relationships_created, 4)

        result = g.query(q)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.relationships_created, 0)

    # test error reporting for invalid min, max variable length edge length
    def test12_lt_zero_hop_traversals(self):
        # Construct an empty graph
        g = self.db.select_graph("lt_zero_hop_traversals")

        queries = [
            "MATCH p=()-[*..0]->() RETURN nodes(p) AS nodes",
            "MATCH p=()-[*1..0]->() RETURN nodes(p) AS nodes",
            "MATCH p=()-[*2..1]->() RETURN nodes(p) AS nodes",
            "MATCH p=()-[e*2..1]->() RETURN nodes(p) AS nodes",
            "MATCH p=()-[e:R*20..10]->() RETURN nodes(p) AS nodes",
            "MATCH p=()-[]->()-[*1..0]->() RETURN nodes(p) AS nodes",
        ]
        for query in queries:
            self._assert_exception(g, query,
                "Variable length path, maximum number of hops must be greater or equal to minimum number of hops.")

    def test13_return_var_len_edge_array(self):
        # Construct a simple graph:
        # (A)-[R]->(b)
        # (b)-[R]->(c)
        g = self.db.select_graph("return_var_len_edge_array")
        q = "CREATE (a)-[:R]->(b)-[:R]->(c)"
        g.query(q)

        query_to_expected_result = [
            ("MATCH (a)-[r*2..2]->(b) RETURN size(nodes(r))" , [[3]]),
            ("MATCH (a)-[r:R*2..2]->(b) RETURN size(nodes(r))" , [[3]]),
            ("MATCH (a)-[r*1..2]->(b) RETURN size(nodes(r)) AS x ORDER BY x" , [[2], [2], [3]]),
            ("MATCH (a)-[r*0..2]->(b) RETURN size(nodes(r)) AS x ORDER BY x" , [[1], [1], [1], [2], [2], [3]]),
            ("MATCH (a)-[r*0..1]->(b) RETURN size(nodes(r)) AS x ORDER BY x" , [[1], [1], [1], [2], [2]]),
            ("MATCH (a)-[r*0..0]->(b) RETURN size(nodes(r)) AS x ORDER BY x" , [[1], [1], [1]]),
        ]
        for query, expected_result in query_to_expected_result:
            actual_result = g.query(query)
            self.env.assertEquals(actual_result.result_set, expected_result)

        # for patterns of length equals to one, the expected result is of type edge
        q = "MATCH (a)-[r*1..1]->(b) RETURN r"
        actual_result = g.query(q)

        e01 = actual_result.result_set[0][0]      
        self.env.assertEquals(e01.src_node, 0)
        self.env.assertEquals(e01.dest_node, 1)
        self.env.assertEquals(e01.relation, 'R')

        e12 = actual_result.result_set[1][0]
        self.env.assertEquals(e12.src_node, 1)
        self.env.assertEquals(e12.dest_node, 2)
        self.env.assertEquals(e12.relation, 'R')

    # Test that the same relationship cannot be bound to multiple variables
    # in a single MATCH pattern (relationship uniqueness within a pattern)
    # Regression test for: https://github.com/FalkorDB/FalkorDB/issues/1469
    def test14_relationship_uniqueness_in_pattern(self):
        # Create a minimal graph: 2 nodes connected by a single relationship
        g = self.db.select_graph("relationship_uniqueness")
        g.query("CREATE ()-[:R]->()")

        # Sanity check: verify the graph has exactly one relationship
        sanity_result = g.query("MATCH ()-[r]->() RETURN count(r) AS rel_count")
        self.env.assertEquals(sanity_result.result_set, [[1]])

        # Bug query: pattern requires two distinct relationships
        # Since only one relationship exists, this should return 0 rows
        # (the same relationship cannot be used for both 'a' and 'b')
        bug_result = g.query("MATCH ()<-[a]-()-[b]->() RETURN count(*) AS rows")
        self.env.assertEquals(bug_result.result_set, [[0]])

        # Verify with different pattern variations
        # Pattern: (x)<-[a]-(y)-[b]->(z) - requires two distinct edges from y
        result = g.query("MATCH (x)<-[a]-(y)-[b]->(z) RETURN count(*)")
        self.env.assertEquals(result.result_set, [[0]])

        # Now add a second relationship to verify the fix allows valid patterns
        g.query("MATCH (n) WHERE id(n) = 1 CREATE (n)-[:R]->()")

        # After adding a second edge, we should get results
        result = g.query("MATCH ()<-[a]-()-[b]->() RETURN count(*)")
        # Node 1 has both an incoming edge (from node 0) and an outgoing edge (to node 2)
        self.env.assertEquals(result.result_set, [[1]])

        # Verify the two edge IDs are different
        result = g.query("MATCH ()<-[a]-()-[b]->() RETURN id(a), id(b)")
        self.env.assertEquals(len(result.result_set), 1)
        edge_a_id = result.result_set[0][0]
        edge_b_id = result.result_set[0][1]
        self.env.assertNotEqual(edge_a_id, edge_b_id)

        # Test with bidirectional pattern - same edge should not be used twice
        g2 = self.db.select_graph("relationship_uniqueness_bidir")
        g2.query("CREATE (a)-[:R]->(b)")

        # Bidirectional pattern with single edge - should return 0
        result = g2.query("MATCH ()-[a]-()-[b]-() RETURN count(*)")
        self.env.assertEquals(result.result_set, [[0]])

        # Add another edge
        g2.query("MATCH (n) WHERE id(n) = 0 CREATE (n)-[:S]->(m)")

        # Now there are two edges, so patterns requiring 2 edges should work
        result = g2.query("MATCH ()-[a]-()-[b]-() WHERE id(a) <> id(b) RETURN count(*)")
        # This should have results now
        self.env.assertTrue(result.result_set[0][0] >= 1)
