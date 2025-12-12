from common import *
import re

nodes = {}
GRAPH_ID = "optional_match"


class testOptionalFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        global nodes
        # Construct a graph with the form:
        # (v1)-[:E1]->(v2)-[:E2]->(v3), (v4)
        node_props = ['v1', 'v2', 'v3', 'v4']

        for idx, v in enumerate(node_props):
            node = Node(alias=f"n_{idx}", labels="L", properties={"v": v})
            nodes[v] = node

        e0 = Edge(nodes['v1'], "E1", nodes['v2'])
        e1 = Edge(nodes['v2'], "E2", nodes['v3'])

        nodes_str = [str(node) for node in nodes.values()]
        self.graph.query(f"CREATE {', '.join(nodes_str)}, {e0}, {e1}")

    # Optional MATCH clause that does not interact with the mandatory MATCH.
    def test01_disjoint_optional(self):
        query = """MATCH (a {v: 'v1'}) OPTIONAL MATCH (b) RETURN a.v, b.v ORDER BY a.v, b.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v1'],
                           ['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v1', 'v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Optional MATCH clause that extends the mandatory MATCH pattern and has matches for all results.
    def test02_optional_traverse(self):
        query = """MATCH (a) WHERE a.v IN ['v1', 'v2'] OPTIONAL MATCH (a)-[]->(b) RETURN a.v, b.v ORDER BY a.v, b.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v2', 'v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Optional MATCH clause that extends the mandatory MATCH pattern and has null results.
    def test03_optional_traverse_with_nulls(self):
        query = """MATCH (a) OPTIONAL MATCH (a)-[]->(b) RETURN a.v, b.v ORDER BY a.v, b.v"""
        actual_result = self.graph.query(query)
        # (v3) and (v4) have no outgoing edges.
        expected_result = [['v1', 'v2'],
                           ['v2', 'v3'],
                           ['v3', None],
                           ['v4', None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Optional MATCH clause that extends the mandatory MATCH pattern and has a WHERE clause.
    def test04_optional_traverse_with_predicate(self):
        query = """MATCH (a) OPTIONAL MATCH (a)-[]->(b) WHERE b.v = 'v2' RETURN a.v, b.v ORDER BY a.v, b.v"""
        actual_result = self.graph.query(query)
        # only (v1) has an outgoing edge to (v2).
        expected_result = [['v1', 'v2'],
                           ['v2', None],
                           ['v3', None],
                           ['v4', None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Optional MATCH clause with endpoints resolved by the mandatory MATCH pattern.
    def test05_optional_expand_into(self):
        query = """MATCH (a)-[]->(b) OPTIONAL MATCH (a)-[e]->(b) RETURN a.v, b.v, TYPE(e) ORDER BY a.v, b.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2', 'E1'],
                           ['v2', 'v3', 'E2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # The OPTIONAL MATCH exactly repeats the MATCH, producing identical results.
        query_without_optional = """MATCH (a)-[e]->(b) RETURN a.v, b.v, TYPE(e) ORDER BY a.v, b.v"""
        result_without_optional = self.graph.query(query_without_optional)
        self.env.assertEquals(actual_result.result_set, result_without_optional.result_set)

    # Optional MATCH clause with endpoints resolved by the mandatory MATCH pattern and new filters introduced.
    def test06_optional_expand_into_with_reltype(self):
        query = """MATCH (a)-[]->(b) OPTIONAL MATCH (a)-[e:E2]->(b) RETURN a.v, b.v, TYPE(e) ORDER BY a.v, b.v"""
        actual_result = self.graph.query(query)
        # Only (v2)-[E2]->(v3) fulfills the constraint of the OPTIONAL MATCH clause.
        expected_result = [['v1', 'v2', None],
                           ['v2', 'v3', 'E2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Optional MATCH clause with endpoints resolved by the mandatory MATCH pattern, but no mandatory traversal.
    def test07_optional_expand_into_cartesian_product(self):
        query = """MATCH (a {v: 'v1'}), (b) OPTIONAL MATCH (a)-[e]->(b) RETURN a.v, b.v, TYPE(e) ORDER BY a.v, b.v"""
        actual_result = self.graph.query(query)
        # All nodes are represented, but (v1)-[E1]->(v2) is the only matching connection.
        expected_result = [['v1', 'v1', None],
                           ['v1', 'v2', 'E1'],
                           ['v1', 'v3', None],
                           ['v1', 'v4', None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # TODO ExpandInto doesn't evaluate bidirectionally properly
    # Optional MATCH clause with endpoints resolved by the mandatory MATCH pattern and a bidirectional optional pattern.
    #  def test08_optional_expand_into_bidirectional(self):
        #  query = """MATCH (a), (b {v: 'v2'}) OPTIONAL MATCH (a)-[e]-(b) RETURN a.v, b.v, TYPE(e) ORDER BY a.v, b.v"""
        #  actual_result = self.graph.query(query)
        #  # All nodes are represented, but only edges with (v2) as an endpoint match.
        #  expected_result = [['v1', 'v2', 'E1'],
                           #  ['v2', 'v2', None],
                           #  ['v3', 'v2', 'E2'],
                           #  ['v3', 'v2', None]]
        #  self.env.assertEquals(actual_result.result_set, expected_result)

    # Optional MATCH clause with variable-length traversal and some results match.
    def test09_optional_variable_length(self):
        query = """MATCH (a) OPTIONAL MATCH (a)-[*]->(b) RETURN a.v, b.v ORDER BY a.v, b.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v1', 'v3'],
                           ['v2', 'v3'],
                           ['v3', None],
                           ['v4', None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Optional MATCH clause with variable-length traversal and all results match.
    def test10_optional_variable_length_all_matches(self):
        query = """MATCH (a {v: 'v1'}) OPTIONAL MATCH (a)-[*]->(b) RETURN a.v, b.v ORDER BY a.v, b.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v1', 'v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Optional MATCH clause with a variable-length traversal that has no matches.
    def test11_optional_variable_length_no_matches(self):
        query = """MATCH (a {v: 'v3'}) OPTIONAL MATCH (a)-[*]->(b) RETURN a.v, b.v ORDER BY a.v, b.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v3', None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Multiple interdependent optional MATCH clauses.
    def test12_multiple_optional_traversals(self):
        query = """MATCH (a)
                   OPTIONAL MATCH (a)-[]->(b)
                   OPTIONAL MATCH (b)-[]->(c)
                   RETURN a.v, b.v, c.v
                   ORDER BY a.v, b.v, c.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2', 'v3'],
                           ['v2', 'v3', None],
                           ['v3', None, None],
                           ['v4', None, None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Multiple interdependent optional MATCH clauses with both directed and bidirectional traversals.
    def test13_multiple_optional_multi_directional_traversals(self):
        query = """MATCH (a) OPTIONAL MATCH (a)-[]-(b) OPTIONAL MATCH (b)-[]->(c) RETURN a.v, b.v, c.v ORDER BY a.v, b.v, c.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2', 'v3'],
                           ['v2', 'v1', 'v2'],
                           ['v2', 'v3', None],
                           ['v3', 'v2', 'v3'],
                           ['v4', None, None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Multiple interdependent optional MATCH clauses with exclusively bidirectional traversals.
    def test14_multiple_optional_bidirectional_traversals(self):
        query = """MATCH (a) OPTIONAL MATCH (a)-[]-(b) OPTIONAL MATCH (b)-[]-(c) RETURN a.v, b.v, c.v ORDER BY a.v, b.v, c.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2', 'v1'],
                           ['v1', 'v2', 'v3'],
                           ['v2', 'v1', 'v2'],
                           ['v2', 'v3', 'v2'],
                           ['v3', 'v2', 'v1'],
                           ['v3', 'v2', 'v3'],
                           ['v4', None, None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Build a named path in an optional clause.
    def test15_optional_named_path(self):
        query = """MATCH (a)
                   OPTIONAL MATCH p = (a)-[]->(b)
                   RETURN length(p)
                   ORDER BY length(p)"""
        actual_result = self.graph.query(query)
        # 2 nodes have outgoing edges and 2 do not, so expected 2 paths of length 1 and 2 null results.
        expected_result = [[1],
                           [1],
                           [None],
                           [None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Return a result set with null values in the first record and non-null values in subsequent records.
    def test16_optional_null_first_result(self):
        query = """MATCH (a) OPTIONAL MATCH (a)-[e]->(b) RETURN a, b, TYPE(e) ORDER BY EXISTS(b), a.v, b.v"""
        actual_result = self.graph.query(query)
        expected_result = [[nodes['v3'], None, None],
                           [nodes['v4'], None, None],
                           [nodes['v1'], nodes['v2'], 'E1'],
                           [nodes['v2'], nodes['v3'], 'E2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test17_optional_label_introductions(self):
        query = """MATCH (a)
                   OPTIONAL MATCH (a:L)-[]->(b:L)
                   RETURN a.v, b.v
                   ORDER BY a.v, b.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v2', 'v3'],
                           ['v3', None],
                           ['v4', None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Make sure highly connected nodes aren't lost
    def test18_optional_over_intermidate(self):
        query = """MATCH (a)-[]->(b)-[]->(c) OPTIONAL MATCH (b)-[]->(c) RETURN a"""
        plan = str(self.graph.explain(query))
        # Expecting to find "Expand Into" operation as both 'b' and 'c'
        # are bounded, which means 'b' is treated as an intermidate node
        # that needs to be tracked.
        self.env.assertIn("Expand Into", plan)

    # Validate that filters are created properly when OPTIONAL MATCH is the first clause.
    def test19_leading_optional_match(self):
        query = """MATCH (n) WHERE n.v = 'v1' RETURN n.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Validate that path filters on OPTIONAL MATCH clauses are constructed properly.
    def test20_optional_path_filter(self):
        query = """MATCH (n {v: 'v1'}) OPTIONAL MATCH (m:L)-[]->() WHERE (n)--() RETURN n.v, m.v ORDER BY n.v, m.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v1'],
                           ['v1', 'v2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (n) OPTIONAL MATCH (m {v:'v1'})--() WHERE (n)--() RETURN n.v, m.v ORDER BY n.v, m.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v1'],
                           ['v2', 'v1'],
                           ['v3', 'v1'],
                           ['v4', None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """OPTIONAL MATCH (n {v: 'v1'}) OPTIONAL MATCH (m {v: 'v2'}) WHERE (n)--(m) RETURN n.v, m.v"""

    # Test placement of filters that don't rely on variable references.
    def test21_optional_filters_without_references(self):
        query = """OPTIONAL MATCH (a {v: 'v1'}), (b {v: 'v2'}) WHERE false RETURN a, b"""
        actual_result = self.graph.query(query)
        expected_result = [[None, None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """OPTIONAL MATCH (a {v: 'v1'}), (b {v: 'v2'}) WHERE true RETURN a.v, b.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # validate that the correct plan is populated and executed when OPTIONAL
    # does not introduce any new variables
    def test22_optional_repeats_reference(self):
        query = """MATCH (n1) OPTIONAL MATCH (n1) WHERE n1.nonexistent > 0 RETURN n1.v ORDER BY n1.v"""
        plan = str(self.graph.explain(query))
        # the first child of the Apply op should be a scan and the
        # second should be the OPTIONAL subtree
        self.env.assertTrue(re.search('Apply\s+All Node Scan | (n1)\s+Optional\s+Filter\s+Argument', plan))

        actual_result = self.graph.query(query)
        expected_result = [['v1'],
                           ['v2'],
                           ['v3'],
                           ['v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test23_optional_after_apply(self):
        self.graph.delete()
        query = """WITH [0, 0] AS n0 OPTIONAL MATCH () MERGE ()"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.nodes_created, 1)

    def test24_optional_and_cartesian_product(self):
        self.graph.delete()
        self.graph.query("CREATE ()-[:A]->()")
        query = """OPTIONAL MATCH (), ({x:0, x:1}) RETURN 0"""
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test25_optional_no_matchings(self):
        # due to delayed init within the Apply op this used to crash the server
        # discovered by Celine Wuest

        # run on an empty graph
        self.graph.delete()

        q = """WITH 0 AS n0
               OPTIONAL MATCH ()
               WITH 0 AS n1
               LIMIT 0
               CREATE ()"""

        self.graph.query(q)

    # validate Optional Conditional Traverse operation is used
    def test26_optional_batch_traversal(self):
        query = """MATCH (a)
                   OPTIONAL MATCH (a)-[]->(b)
                   OPTIONAL MATCH (b)-[]->(c)
                   RETURN a, b, c"""

        # Expecting to find "Optional Conditional Traverse" operations
        plan = str(self.graph.explain(query))
        self.env.assertIn("Optional Conditional Traverse | (a)->(b)", plan)
        self.env.assertIn("Optional Conditional Traverse | (b)->(c)", plan)

    # Test multiple named paths in a single OPTIONAL MATCH clause (regression test for issue #1353)
    def test27_multiple_named_paths_in_optional_match(self):
        # This query should not crash - it has multiple comma-separated patterns in OPTIONAL MATCH
        # The batch optimization should skip this pattern as it produces a CartesianProduct
        query = """OPTIONAL MATCH (n0)--(n1)
                   OPTIONAL MATCH p0 = (n1), p1 = (n0)
                   RETURN *"""
        
        # Query should complete without crashing
        try:
            actual_result = self.graph.query(query)
            # Verify the query completes and returns results
            # With our test graph: 4 nodes (v1, v2, v3, v4), edges: v1->v2, v2->v3
            # First OPTIONAL MATCH produces pairs: (v1,v2), (v2,v1), (v2,v3), (v3,v2)
            # Second OPTIONAL MATCH creates cartesian product of named paths
            # We just verify it doesn't crash and returns some results
            self.env.assertTrue(actual_result.result_set is not None)
        except Exception as e:
            self.env.fail(f"Query crashed with error: {e}")

    # Test simpler case: multiple named paths without traversal
    def test28_named_paths_without_traversal(self):
        # Multiple named node patterns in a single OPTIONAL MATCH
        query = """MATCH (a {v: 'v1'})
                   OPTIONAL MATCH p0 = (a), p1 = (b {v: 'v2'})
                   RETURN a.v, b.v, length(p0), length(p1)
                   ORDER BY a.v, b.v"""
        
        actual_result = self.graph.query(query)
        # p0 and p1 are single-node paths (length 0)
        expected_result = [['v1', 'v2', 0, 0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

