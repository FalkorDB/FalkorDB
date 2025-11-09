from common import *

GRAPH_ID = "value_comparison"
values = ["str1", "str2", False, True, 5, 10.5]


class testValueComparison(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        nodes = []
        for idx, v in enumerate(values):
            nodes.append(Node(alias=f"n{idx}", labels="value", properties={"val": v}))

        # Add an additional node with no properties
        nodes.append(Node(labels="value"))

        nodes_str = [str(n) for n in nodes]
        self.graph.query(f"CREATE {','.join(nodes_str)}")

    # Verify the ordering of values that can and cannot be directly compared
    def test_orderability(self):
        query = """MATCH (v:value) RETURN v.val ORDER BY v.val"""
        actual_result = self.graph.query(query)
        expected = [['str1'],
                    ['str2'],
                    [False],
                    [True],
                    [5],
                    [10.5],
                    [None]]
        self.env.assertEquals(actual_result.result_set, expected)

        # Expect the results to appear in reverse when using descending order
        query = """MATCH (v:value) RETURN v.val ORDER BY v.val DESC"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected[::-1])

    # From the Cypher specification:
    # "In a mixed set, any numeric value is always considered to be higher than any string value"
    def test_mixed_type_min(self):
        query = """MATCH (v:value) RETURN MIN(v.val)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 'str1')

    def test_mixed_type_max(self):
        query = """MATCH (v:value) RETURN MAX(v.val)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 10.5)

    # Verify that disjoint types pass <> filters
    def test_disjoint_comparisons(self):
        # Compare all node pairs under a Cartesian product
        query = """MATCH (v:value), (w:value) WHERE ID(v) <> ID(w) AND v.val = w.val RETURN v"""
        actual_result = self.graph.query(query)
        # No nodes have the same property, so there should be 0 equal results
        expected_result_count = 0
        self.env.assertEquals(
            len(actual_result.result_set), expected_result_count)

        query = """MATCH (v:value), (w:value) WHERE ID(v) <> ID(w) AND v.val <> w.val RETURN v"""
        actual_result = self.graph.query(query)
        # Every comparison should produce an inequal result
        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0] # The node with value set as "null" should not be returned or be part of evaluation.
        expected_result_count = (node_count - 1) * (node_count - 2)
        self.env.assertEquals(
            len(actual_result.result_set), expected_result_count)

    # Verify that comparisons between very small and very large values are ordered properly.
    def test_large_comparisons(self):
        query = """UNWIND [933, 1099511628237] AS val RETURN val ORDER BY val"""
        actual_result = self.graph.query(query)
        expected = [[933], [1099511628237]]
        self.env.assertEquals(actual_result.result_set, expected)

    # Verify that AND conditions on true, false, and NULL values evaluate appropriately
    def test_AND_truth_tables(self):
        # Test two non-NULL values
        query = """RETURN true AND true, true AND false, false AND true, false AND false"""
        actual_result = self.graph.query(query)
        expected_val = [True, False, False, False] # Truth table for AND
        self.env.assertEquals(actual_result.result_set[0], expected_val)

        # false AND null == false 
        query = """RETURN false AND NULL"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], False)

        # true AND null == null 
        query = """RETURN true AND NULL"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # Test two NULL values
        query = """RETURN NULL AND NULL"""
        actual_result = self.graph.query(query)
        # AND comparisons with two NULL values evaluate to NULL
        self.env.assertEquals(actual_result.result_set[0][0], None)

    # Verify that OR conditions on true, false, and NULL values evaluate appropriately
    def test_OR_truth_tables(self):
        # Test two non-NULL values
        query = """RETURN true OR true, true OR false, false OR true, false OR false"""
        actual_result = self.graph.query(query)
        expected_val = [True, True, True, False] # Truth table for OR
        self.env.assertEquals(actual_result.result_set[0], expected_val)

        # false OR null == null
        query = """RETURN false OR NULL"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # true OR null == true
        query = """RETURN true OR NULL"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], True)

        # null OR null == null
        query = """RETURN NULL OR NULL"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

    # Verify that XOR conditions on true, false, and NULL values evaluate appropriately
    def test_XOR_truth_tables(self):
        # Test two non-NULL values
        query = """RETURN true XOR true, true XOR false, false XOR true, false XOR false"""
        actual_result = self.graph.query(query)
        expected_val = [False, True, True, False] # Truth table for XOR
        self.env.assertEquals(actual_result.result_set[0], expected_val)

        # Test one NULL value
        query = """RETURN true XOR null, false XOR null"""
        actual_result = self.graph.query(query)
            # XOR comparisons with one NULL value always evaluate to null
        expected_val = [None, None]
        self.env.assertEquals(actual_result.result_set[0], expected_val)

        # Test two NULL values
        query = """RETURN NULL XOR NULL"""
        actual_result = self.graph.query(query)
        # XOR comparisons with two NULL values evaluate to NULL
        self.env.assertEquals(actual_result.result_set[0][0], None)

    # Verify that NOT conditions on true, false, and NULL values evaluate appropriately
    def test_NOT_truth_tables(self):
        # Test non-NULL values
        query = """RETURN NOT true, NOT false"""
        actual_result = self.graph.query(query)
        expected_val = [False, True] # Truth table (single-valued) for NOT
        self.env.assertEquals(actual_result.result_set[0], expected_val)

        # NOT null == null
        query = """RETURN NOT NULL"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

    def test_coalesce(self):
        query = """MATCH (n) RETURN COALESCE(n.a, n.b, n.c)"""
        actual_result = self.graph.query(query)
        # Test default value - everything is null.
        self.env.assertEquals([[None], [None], [None], [None], [None], [None], [None]], actual_result.result_set)
        query = """MATCH (n) SET n.c = 'c' RETURN COALESCE(n.a, n.b, n.c)"""
        actual_result = self.graph.query(query)
        # Test value search, last expressions is not null.
        self.env.assertEquals([['c'], ['c'], ['c'], ['c'], ['c'], ['c'], ['c']], actual_result.result_set)
        query = """MATCH (n) SET n.b = 2 RETURN COALESCE(n.a, n.b, n.c)"""
        actual_result = self.graph.query(query)
        # Test value search, second expressions is not null.
        self.env.assertEquals([[2], [2], [2], [2], [2], [2], [2]], actual_result.result_set)
        query = """MATCH (n) SET n.a = 1.1 RETURN COALESCE(n.a, n.b, n.c)"""
        actual_result = self.graph.query(query)
        # Test value search, first expressions is not null.
        self.env.assertEquals([[1.1], [1.1], [1.1], [1.1], [1.1], [1.1], [1.1]], actual_result.result_set)

    # Verify string concatenation in comparison expressions
    def test_string_concat_comparison(self):
        # Test additional string concatenation comparison cases
        query = """RETURN 'abc' <= ('def' + 'ghi')"""
        actual_result = self.graph.query(query)
        # 'abc' <= 'defghi' should be true
        self.env.assertEquals(actual_result.result_set[0][0], True)

        query = """RETURN 'xyz' > ('abc' + 'def')"""
        actual_result = self.graph.query(query)
        # 'xyz' > 'abcdef' should be true
        self.env.assertEquals(actual_result.result_set[0][0], True)

        query = """RETURN 'test' = ('te' + 'st')"""
        actual_result = self.graph.query(query)
        # 'test' = 'test' should be true
        self.env.assertEquals(actual_result.result_set[0][0], True)

