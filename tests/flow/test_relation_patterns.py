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

        self.env.assertEqual(result_b.result_set, result_a.result_set)
        self.env.assertEqual(result_c.result_set, result_a.result_set)
        self.env.assertEqual(result_d.result_set, result_a.result_set)
        self.env.assertEqual(result_e.result_set, result_a.result_set)

    # Test patterns that traverse 2 edges.
    def test02_two_hop_traversals(self):
        # Conditional two-hop traversal without referenced intermediate node
        query = """MATCH (a)-[:e]->()-[:e]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v3']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Fixed-length two-hop traversal (same expected result)
        query = """MATCH (a)-[:e*2]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Variable-length traversal with a minimum bound of 2 (same expected result)
        query = """MATCH (a)-[*2..]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Conditional two-hop traversal with referenced intermediate node
        query = """MATCH (a)-[:e]->(b)-[:e]->(c) RETURN a.val, b.val, c.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2', 'v3']]
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Test variable-length patterns
    def test03_var_len_traversals(self):
        # Variable-length traversal with label
        query = """MATCH (a)-[:e*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v2', 'v3']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Variable-length traversal without label (same expected result)
        query = """MATCH (a)-[*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Variable-length traversal with bounds 1..2 (same expected result)
        query = """MATCH (a)-[:e*1..2]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, expected_result)

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
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Test variable-length patterns with alternately labeled source
    # and destination nodes, which can cause different execution sequences.
    def test04_variable_length_labeled_nodes(self):
        # Source and edge labeled variable-length traversal
        query = """MATCH (a:L)-[:e*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v2', 'v3']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Destination and edge labeled variable-length traversal (same expected result)
        query = """MATCH (a)-[:e*]->(b:L) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Source labeled variable-length traversal (same expected result)
        query = """MATCH (a:L)-[*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Destination labeled variable-length traversal (same expected result)
        query = """MATCH (a)-[*]->(b:L) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Test traversals over explicit relationship types
    def test05_relation_types(self):
        # Add two nodes and two edges of a new type.
        # The new form of the graph will be:
        # (v1)-[:e]->(v2)-[:e]->(v3)-[:q]->(v4)-[:q]->(v5)
        query = """MATCH (n {val: 'v3'}) CREATE (n)-[:q]->(:L {val: 'v4'})-[:q]->(:L {val: 'v5'})"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.nodes_created, 2)
        self.env.assertEqual(actual_result.relationships_created, 2)

        # Verify the graph structure
        query = """MATCH (a)-[e]->(b) RETURN a.val, b.val, TYPE(e) ORDER BY TYPE(e), a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2', 'e'],
                           ['v2', 'v3', 'e'],
                           ['v3', 'v4', 'q'],
                           ['v4', 'v5', 'q']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Verify conditional traversals with explicit relation types
        query = """MATCH (a)-[:e]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v2', 'v3']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        query = """MATCH (a)-[:q]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v3', 'v4'],
                           ['v4', 'v5']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Verify conditional traversals with multiple explicit relation types
        query = """MATCH (a)-[e:e|:q]->(b) RETURN a.val, b.val, TYPE(e) ORDER BY TYPE(e), a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2', 'e'],
                           ['v2', 'v3', 'e'],
                           ['v3', 'v4', 'q'],
                           ['v4', 'v5', 'q']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Verify variable-length traversals with explicit relation types
        query = """MATCH (a)-[:e*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v2', 'v3']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        query = """MATCH (a)-[:q*]->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v3', 'v4'],
                           ['v3', 'v5'],
                           ['v4', 'v5']]
        self.env.assertEqual(actual_result.result_set, expected_result)

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
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Test traversals over transposed edge matrices.
    def test06_transposed_traversals(self):
        # The intermediate node 'b' will be used to form the scan operation because it is filtered.
        # As such, one of the traversals must be transposed.
        query = """MATCH (a)-[e]->(b {val:'v3'})-[]->(c:L) RETURN COUNT(e)"""
        plan = str(self.graph.explain(query))

        # Verify that the execution plan contains two traversals following opposing edge directions.
        self.env.assertContains("<-", plan)
        self.env.assertContains("->", plan)

        # Verify results.
        actual_result = self.graph.query(query)
        expected_result = [[1]]
        self.env.assertEqual(actual_result.result_set, expected_result)

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
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test08_transposed_varlen_traversal(self):
        # Verify that variable-length traversals with nested transpose operations perform correctly.
        query = """MATCH (a {val: 'v1'})-[*]-(b {val: 'v2'})-[:e]->(:L {val: 'v3'}) RETURN a.val ORDER BY a.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1']]
        self.env.assertEqual(actual_result.result_set, expected_result)

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
            self.env.assertEqual(actual_result.result_set, expected_result)

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
            self.env.assertEqual(res.result_set, expected_result)

    def test11_shared_node_detection(self):
        # Construct a simple graph
        # (s)<-[:A]-(x)
        # (x)<-[:B]-(x)
        # (x)-[:B]->(t)
        # (t)<-[:B]-(x)
        g = self.db.select_graph("shared_node")
        q = "MERGE (s)<-[:A]-(x)<-[:B]-(x)-[:B]->(t)<-[:B]-(x)"
        result = g.query(q)

        self.env.assertEqual(result.nodes_created, 3)
        self.env.assertEqual(result.relationships_created, 4)

        result = g.query(q)
        self.env.assertEqual(result.nodes_created, 0)
        self.env.assertEqual(result.relationships_created, 0)

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
            self.env.assertEqual(actual_result.result_set, expected_result)

        # for patterns of length equals to one, the expected result is of type edge
        q = "MATCH (a)-[r*1..1]->(b) RETURN r"
        actual_result = g.query(q)

        e01 = actual_result.result_set[0][0]      
        self.env.assertEqual(e01.src_node, 0)
        self.env.assertEqual(e01.dest_node, 1)
        self.env.assertEqual(e01.relation, 'R')

        e12 = actual_result.result_set[1][0]
        self.env.assertEqual(e12.src_node, 1)
        self.env.assertEqual(e12.dest_node, 2)
        self.env.assertEqual(e12.relation, 'R')

    # Relationship uniqueness and multigraph cardinality must not depend on
    # whether the relationship (or the middle node) is named.
    def test14_relationship_uniqueness_ignores_naming(self):
        # 3-cycle: 3 edges * 2 endpoints * 1 remaining incident edge = 6
        g = self.db.select_graph("rel_uniqueness_cycle")
        g.query("CREATE (a:N), (b:N), (c:N), (a)-[:R]->(b), (b)-[:R]->(c), (c)-[:R]->(a)")
        for pattern in ["(x)-[r1]-()-[r2]-(z)",
                        "(x)-[r1]-(m)-[r2]-(z)",
                        "(x)-[]-()-[]-(z)",
                        "(x)-[]-(m)-[]-(z)",
                        "(x)-[:R]-()-[:R]-(z)"]:
            res = g.query(f"MATCH {pattern} RETURN count(*)")
            self.env.assertEqual(res.result_set, [[6]])

        # converging pattern: Alice->Bob<-Charlie, 2 distinct edges
        g = self.db.select_graph("rel_uniqueness_converge")
        g.query("CREATE (:P {n:'A'})-[:F]->(:P {n:'B'})<-[:F]-(:P {n:'C'})")
        for pattern in ["(p1:P)-[:F]->(p2:P)<-[:F]-(p3:P)",
                        "(p1:P)-[e1:F]->(p2:P)<-[e2:F]-(p3:P)"]:
            res = g.query(f"MATCH {pattern} RETURN count(p1)")
            self.env.assertEqual(res.result_set, [[2]])

        # multigraph: parallel edges keep their multiplicity
        g = self.db.select_graph("rel_uniqueness_multigraph")
        g.query("CREATE (a:V {id:1}), (b:V {id:2}), (c:V {id:3}), "
                "(a)-[:R {eid:1}]->(b), (a)-[:R {eid:2}]->(b), "
                "(b)-[:R {eid:3}]->(c), (b)-[:R {eid:4}]->(c), "
                "(c)-[:R {eid:5}]->(a), (a)-[:R {eid:6}]->(a)")
        expected = [[2, 2], [3, 4]]
        for pattern in ["(a:V {id:1})-[:R*2]->(b:V)",
                        "(a:V {id:1})-[r1:R]->(v:V)-[r2:R]->(b:V)",
                        "(a:V {id:1})-[:R]->(v:V)-[:R]->(b:V)",
                        "(a:V {id:1})-[:R]->()-[:R]->(b:V)"]:
            res = g.query(f"MATCH {pattern} RETURN b.id AS dst, count(*) AS n ORDER BY dst")
            self.env.assertEqual(res.result_set, expected)
        for pattern in ["(a:V)-[:R]->(b:V)", "(a:V)-[r:R]->(b:V)"]:
            res = g.query(f"MATCH {pattern} RETURN count(*)")
            self.env.assertEqual(res.result_set, [[6]])

    # A fused anonymous chain counts paths, not (src, dst) pairs.
    def test15_fused_chain_counts_paths(self):
        # diamond: two paths a -> c through distinct anonymous middle nodes
        g = self.db.select_graph("fused_chain_diamond")
        g.query("CREATE (a:A), (b1:B), (b2:B), (c:C), "
                "(a)-[:R]->(b1), (a)-[:R]->(b2), (b1)-[:R]->(c), (b2)-[:R]->(c)")
        for pattern in ["(a:A)-[:R]->()-[:R]->(c:C)", "(a:A)-[:R]->(m)-[:R]->(c:C)"]:
            res = g.query(f"MATCH {pattern} RETURN count(*)")
            self.env.assertEqual(res.result_set, [[2]])
        # the anonymous form runs as one fused traverse
        plan = str(g.explain("MATCH (a:A)-[:R]->()-[:R]->(c:C) RETURN count(*)"))
        self.env.assertContains("fused hop", plan)

        # self-loops: two on `a` and an edge a -> b. Distinct edge pairs:
        # (s1,s2), (s2,s1) to a, and (s1,e), (s2,e) to b.
        g = self.db.select_graph("fused_chain_self_loops")
        g.query("CREATE (a:A), (b:B), (a)-[:R]->(a), (a)-[:R]->(a), (a)-[:R]->(b)")
        for pattern in ["(a:A)-[:R]->()-[:R]->(c)",
                        "(a:A)-[r1:R]->(m)-[r2:R]->(c)",
                        "(a:A)-[]->()-[]->(c)"]:
            res = g.query(f"MATCH {pattern} RETURN count(*)")
            self.env.assertEqual(res.result_set, [[4]])

    # A pair joined by several edges of different types is several matches.
    def test16_multi_type_pair(self):
        g = self.db.select_graph("multi_type_pair")
        g.query("CREATE (a:A)-[:R]->(b:B), (a)-[:S]->(b), (a)-[:R]->(b)")
        for pattern, expected in [("(a:A)-[:R|S]->(b)", 3),
                                  ("(a:A)-[]->(b)", 3),
                                  ("(a:A)-[:R]->(b)", 2),
                                  # 3 edges, each paired with another one
                                  ("(a:A)-[]->()<-[]-(c)", 6)]:
            res = g.query(f"MATCH {pattern} RETURN count(*)")
            self.env.assertEqual(res.result_set, [[expected]])

    # Uniqueness among same-type anonymous edges holds on every path: batched
    # (compared by node), per-row, and with parallel edges (compared by id).
    def test17_converging_edges_are_distinct(self):
        g = self.db.select_graph("converging_distinct")
        g.query("CREATE (a:A)-[:R]->(m:M)<-[:R]-(c:A), (a)-[:R]->(a), (m)-[:R]->(m)")
        for query, expected in [
            ("MATCH (a)-[:R]->(m)<-[:R]-(c) RETURN count(*)", 6),
            ("MATCH (a)-[r1:R]->(m)<-[r2:R]-(c) RETURN count(*)", 6),
            ("MATCH (a)-[:R]->(m)-[:R]->(c) RETURN count(*)", 3),
            ("MATCH (a)-[:R]->(m)<-[:R]-(c)-[:R]->(d) RETURN count(*)", 2),
            ("MATCH (a)<-[:R]-(m)-[:R]->(c) RETURN count(*)", 2),
        ]:
            self.env.assertEqual(g.query(query).result_set, [[expected]])
        # a parallel edge makes the same node pair two different edges
        g.query("MATCH (a:A)-[:R]->(m:M) CREATE (a)-[:R]->(m)")
        res = g.query("MATCH (a)-[:R]->(m)<-[:R]-(c) RETURN count(*)")
        self.env.assertEqual(res.result_set, [[20]])

    # A pair joined by edges of different types holds several edges although
    # no single type has a parallel one; every traverse path counts them all.
    def test18_cross_type_pair_without_parallel_edges(self):
        g = self.db.select_graph("cross_type_pair")
        g.query("CREATE (a:A)-[:R]->(b:B), (a)-[:S]->(b)")
        for query in [
            "MATCH (a:A)-[:R|S]->(b:B) RETURN count(*)",
            "MATCH (a:A)-[]->(b:B) RETURN count(*)",
            "MATCH (a:A)-[:R|S]-(b:B) RETURN count(*)",
            "MATCH (a:A)-[]-(b:B) RETURN count(*)",
            "MATCH (a:A)-[r:R|S]->(b:B) RETURN count(*)",
            "OPTIONAL MATCH (a:A)-[:R|S]-(b:B) RETURN count(*)",
            "MATCH (a:A), (b:B) WITH a, b MATCH (a)-[:R|S]->(b) RETURN count(*)",
            "MATCH (a:A), (b:B) WITH a, b MATCH (a)-[]->(b) RETURN count(*)",
        ]:
            self.env.assertEqual(g.query(query).result_set, [[2]], depth=1)

    # One input entry fanning out into more rows than a batch (or a u16 row
    # index) holds is split, and stops at the record cap.
    def test19_fused_multiplicity_is_batched(self):
        g = self.db.select_graph("fused_multiplicity")
        # the big entry sits between two that pass, whichever way the scan runs
        g.query("CREATE (:A {v: 2})-[:R]->(:M)-[:S]->(:C {v: 2}), "
                "(a:A {v: 1})-[:R]->(m:M)-[:S]->(c:C {v: 2}), "
                "(:A {v: 2})-[:R]->(:M)-[:S]->(:C {v: 2})")
        g.query("UNWIND range(1, 256) AS i MATCH (a:A {v: 1}), (m:M)<-[:R]-(a), (c:C)<-[:S]-(m) "
                "CREATE (a)-[:R]->(m), (m)-[:S]->(c)")
        base = "MATCH (a:A)-[:R]->()-[:S]->(c:C) "
        self.env.assertEqual(g.query(base + "RETURN count(*)").result_set, [[257 * 257 + 2]])
        # a filter over rows of one oversized entry must keep each row's own values
        res = g.query(base + "WHERE a.v + c.v >= 3 RETURN a.v, count(*) ORDER BY a.v")
        self.env.assertEqual(res.result_set, [[1, 257 * 257], [2, 2]])
        res = g.query(base + "WHERE a.v + c.v = 99 RETURN count(*)")
        self.env.assertEqual(res.result_set, [[0]])
        res = g.query(base + "RETURN c.v LIMIT 5")
        self.env.assertEqual(res.result_set, [[2]] * 5)

    # A traverse drops the rows whose edges the pattern already used, so a LIMIT
    # above it cannot cap the traverse beneath it: the first row of the lower
    # traverse is a->b, which the converging pattern has no match for.
    def test20_limit_does_not_cap_a_row_dropping_traverse(self):
        g = self.db.select_graph("limit_through_traverse")
        g.query("CREATE (a:P)-[:R]->(b:P), (c:P)-[:R]->(m:P), (d:P)-[:R]->(m), (m)-[:S]->(z:P)")
        for query in [
            "MATCH (x)-[:R]->(y)<-[:R]-(z) RETURN 1 LIMIT 1",
            "MATCH (x)-[r1:R]->(y)<-[r2:R]-(z) RETURN 1 LIMIT 1",
        ]:
            self.env.assertEqual(g.query(query).result_set, [[1]], depth=1)

    # The parallel-edge counts and pending-delta structures a traversal reuses
    # must follow writes: the fused count, the unfused named count and the
    # edge total agree after every kind of change.
    def test18_counts_follow_writes(self):
        g = self.db.select_graph("counts_follow_writes")
        g.query("UNWIND range(0, 9) AS i CREATE (:W {i: i})")
        g.query("MATCH (a:W), (b:W) WHERE b.i = (a.i + 1) % 10 CREATE (a)-[:R]->(b)")

        def check():
            fused = g.query("MATCH (a:W)-[:R]->()-[:R]->(c) RETURN count(*)").result_set[0][0]
            named = g.query("MATCH (a:W)-[r1:R]->(m)-[r2:R]->(c) RETURN count(r1) + 0 * count(r2)").result_set[0][0]
            single = g.query("MATCH (a:W)-[:R]->(b) RETURN count(*)").result_set[0][0]
            total = g.query("MATCH ()-[r:R]->() RETURN count(r)").result_set[0][0]
            self.env.assertEqual(fused, named)
            self.env.assertEqual(single, total)
            return fused, total

        self.env.assertEqual(check(), (10, 10))
        # a second edge on a committed pair promotes it to a multi-edge pair
        g.query("MATCH (a:W {i: 0}), (b:W {i: 1}) CREATE (a)-[:R]->(b)")
        # out(0) and in(1) are now 2: 8 single-edge middles + 2 + 2
        self.env.assertEqual(check(), (12, 11))
        # more parallel edges, and an edge into a node that has none
        g.query("MATCH (a:W {i: 0}), (b:W {i: 1}) CREATE (a)-[:R]->(b), (a)-[:R]->(b)")
        g.query("MATCH (a:W {i: 5}), (b:W {i: 5}) CREATE (a)-[:R]->(b)")
        fused, total = check()
        self.env.assertEqual(total, 14)
        # deleting one of the parallel edges and a single-edge pair
        g.query("MATCH (a:W {i: 0})-[r:R]->(b:W {i: 1}) WITH r LIMIT 1 DELETE r")
        g.query("MATCH (a:W {i: 7})-[r:R]->(b:W {i: 8}) DELETE r")
        fused, total = check()
        self.env.assertEqual(total, 12)

    # A LIMIT must still return rows when the first sources have no match and
    # when a later traverse drops what an earlier one produced (a label check, a
    # uniqueness check): no cap may be pushed below an operator that can drop.
    def test20_limit_under_dropping_traverses(self):
        g = self.db.select_graph("limit_dropping")
        # 300 isolated sources first, then chains whose only D is far away
        g.query("UNWIND range(1, 300) AS i CREATE (:S {id: i})")
        g.query("UNWIND range(1, 300) AS i CREATE (:S {id: 1000 + i})-[:E]->(:X {id: i})")
        g.query("MATCH (x:X {id: 300}) CREATE (x)-[:E]->(:D {id: 99})")
        for q in ["MATCH (s:S)-[]-()-[]-(d:D) RETURN d.id LIMIT 1",
                  "MATCH (s:S)-[:E]->()-[:E]->(d:D) RETURN d.id LIMIT 1",
                  "MATCH (s:S)-[:E]->(x)-[:E]->(d:D) RETURN d.id LIMIT 1"]:
            self.env.assertEqual(g.query(q).result_set, [[99]])
        # converging edges: the first rows reuse one edge and are dropped
        g2 = self.db.select_graph("limit_dropping_converge")
        g2.query("CREATE (a:A)-[:R]->(m:M)<-[:R]-(c:A)")
        res = g2.query("MATCH (x)-[:R]->(y)<-[:R]-(z) RETURN 1 LIMIT 1")
        self.env.assertEqual(len(res.result_set), 1)
        # a fused chain under a small LIMIT returns exactly that many rows
        g3 = self.db.select_graph("limit_fused")
        g3.query("UNWIND range(1, 500) AS i CREATE (:N {id: i})")
        g3.query("MATCH (a:N), (b:N) WHERE b.id = a.id % 500 + 1 CREATE (a)-[:R]->(b)")
        for lim in (1, 10, 100):
            res = g3.query(f"MATCH (a:N)-[:R]->()-[:R]->(c) RETURN c.id LIMIT {lim}")
            self.env.assertEqual(len(res.result_set), lim)

    # An anonymous edge whose type no other edge of the pattern can bind checks
    # no uniqueness. Edge id 0 here is the U edge; the two T edges, compared by
    # node, bind no real id, and the U traverse used to reject its edge as
    # "already used" against their placeholder 0.
    def test21_disjoint_type_edge_is_not_compared_to_the_others(self):
        g = self.db.select_graph("disjoint_type_edge")
        g.query("CREATE (a:A {id: 1}), (b:B {id: 2}), (c:C {id: 3})")
        g.query("MATCH (b:B), (c:C) CREATE (b)-[:U]->(c)")
        g.query("MATCH (a:A), (b:B) CREATE (b)-[:T]->(a), (a)-[:T]->(b), (a)-[:T]->(b), (a)-[:T]->(b)")
        res = g.query("MATCH (n0:B)<-[:T]-(n1)<-[:T]-(n2)-[:U]->(n3) RETURN n3.id, count(*)")
        self.env.assertEqual(res.result_set, [[3, 3]])
        res = g.query("MATCH (n0:B)<-[r:T]-(n1)<-[s:T]-(n2)-[:U]->(n3) RETURN count(*)")
        self.env.assertEqual(res.result_set, [[3]])

    # A traverse the planner transposed (it starts at the pattern's `to`) runs
    # as one product over the backward adjacency: parallel edges, several types
    # (one of them unknown), labels at both ends, and writes after the load.
    def test22_backward_batched_traverse(self):
        g = self.db.select_graph("backward_batched")
        g.query("UNWIND range(0, 5) AS i CREATE (:L {id: i})")
        g.query("MATCH (a:L), (b:L) WHERE b.id = (a.id + 1) % 6 CREATE (a)-[:R]->(b), (a)-[:R]->(b)")
        g.query("MATCH (a:L {id: 0}), (b:L {id: 3}) CREATE (a)-[:S]->(b)")
        g.query("MATCH (a:L {id: 2}) CREATE (a)-[:R]->(a)")
        # brute force the expected counts from the edge list itself
        edges = [(i, (i + 1) % 6, "R") for i in range(6)] * 2 + [(0, 3, "S"), (2, 2, "R")]

        def paths(types):
            n = 0
            for i, (s1, d1, t1) in enumerate(edges):
                for j, (s2, d2, t2) in enumerate(edges):
                    if i != j and d1 == d2 and t1 in types and t2 in types:
                        n += 1
            return n
        for pat, types in [("(a:L)-[:R]->(m)<-[:R]-(b)", {"R"}),
                           ("(a:L)-[:R|S]->(m)<-[:R|S]-(b)", {"R", "S"}),
                           ("(a:L)-[:R|NOPE]->(m)<-[:R|NOPE]-(b)", {"R"}),
                           ("(a:L)-[]->(m)<-[]-(b)", {"R", "S"})]:
            for where in ["", " WHERE a.id < 3", " WHERE b.id = 2"]:
                res = g.query(f"MATCH {pat}{where} RETURN count(*)")
                want = 0
                for i, (s1, d1, t1) in enumerate(edges):
                    for j, (s2, d2, t2) in enumerate(edges):
                        if i == j or d1 != d2 or t1 not in types or t2 not in types:
                            continue
                        if where == " WHERE a.id < 3" and not s1 < 3:
                            continue
                        if where == " WHERE b.id = 2" and s2 != 2:
                            continue
                        want += 1
                self.env.assertEqual(res.result_set, [[want]])
        # an edge added after the first run is seen by the cached structures
        g.query("MATCH (a:L {id: 1}), (b:L {id: 3}) CREATE (a)-[:R]->(b), (a)-[:R]->(b)")
        edges += [(1, 3, "R")] * 2
        res = g.query("MATCH (a:L)-[:R]->(m)<-[:R]-(b) RETURN count(*)")
        self.env.assertEqual(res.result_set, [[paths({"R"})]])

    # Undirected anonymous hops over disjoint types are not deduplicated by
    # (source, destination): each pair of edges is its own path.
    def test23_undirected_disjoint_types_count_every_edge_pair(self):
        g = self.db.select_graph("undirected_disjoint")
        g.query("CREATE (n0:P {id: 0}), (n1:P {id: 1}), (n2:P {id: 2}), "
                "(n2)-[:T]->(n0), (n2)-[:T]->(n0), (n0)-[:R]->(n0), (n0)-[:R]->(n1)")
        res = g.query("MATCH (a:P {id: 2})-[:T]-()-[:R]-(b) RETURN a.id, b.id ORDER BY b.id")
        # via n0: 2 T edges x (the R self-loop, and the R to n1)
        self.env.assertEqual(sorted(res.result_set), sorted([[2, 0]] * 2 + [[2, 1]] * 2))
        res = g.query("MATCH (a:P {id: 2})-[:T]-(m)-[:R]-(b) RETURN count(*)")
        self.env.assertEqual(res.result_set, [[4]])
