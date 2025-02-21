from common import *
import re

values = ["str1", "str2", False, True, 5, 10.5]
GRAPH_ID = "with_caluse"

class testWithClause(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()
 
    def populate_graph(self):
        # Populate a graph with two labels, each containing the same property values but different keys.
        # Each node pair is connected by an edge from label_a to label_b
        nodes = []
        edges = []
        for idx, v in enumerate(values):
            src  = Node(alias=f"s{idx}", labels="label_a", properties={"a_val": v, "a_idx": idx})
            dest = Node(alias=f"t{idx}", labels="label_b", properties={"b_val": v, "b_idx": idx})
            e    = Edge(src, 'connects', dest, properties={"edgeval": idx})
            nodes.append(src)
            nodes.append(dest)
            edges.append(e)

        nodes_str = [str(n) for n in nodes]
        edges_str = [str(e) for e in edges]
        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")
    
    # Verify that graph entities specified in a WITH clause are returned properly
    def test01_with_scalar_read_queries(self):
        query = """MATCH (a:label_a) WITH a.a_val AS val RETURN val ORDER BY val"""
        actual_result = self.graph.query(query)
        expected = [['str1'],
                    ['str2'],
                    [False],
                    [True],
                    [5],
                    [10.5]]
        self.env.assertEqual(actual_result.result_set, expected)


        query = """MATCH (a:label_a) WITH a.a_val AS val ORDER BY val SKIP 1 LIMIT 1 RETURN val"""
        actual_result = self.graph.query(query)
        expected = [['str2']]
        self.env.assertEqual(actual_result.result_set, expected)


        query = """MERGE (a:label_a {a_val: 5}) WITH a.a_val AS val ORDER BY val RETURN val"""
        actual_result = self.graph.query(query)
        expected = [[5]]
        self.env.assertEqual(actual_result.properties_set, 0)

        self.env.assertEqual(actual_result.result_set, expected)


        # Merge on existing edge
        query = """MERGE ()-[e:connects {edgeval: 5}]->() WITH e.edgeval AS val RETURN val"""
        actual_result = self.graph.query(query)
        expected = [[5]]
        self.env.assertEqual(actual_result.properties_set, 0)

        self.env.assertEqual(actual_result.result_set, expected)

    def test02_with_arithmetic_op_read_queries(self):
        # Iterate over nodes
        query = """MATCH (a) WITH ID(a) AS id RETURN id ORDER BY id"""
        actual_result = self.graph.query(query)
        expected = [[0],
                    [1],
                    [2],
                    [3],
                    [4],
                    [5],
                    [6],
                    [7],
                    [8],
                    [9],
                    [10],
                    [11]]
        self.env.assertEqual(actual_result.result_set, expected)

        # Iterate over edges
        query = """MATCH ()-[e]->() WITH ID(e) AS id RETURN id ORDER BY id"""
        actual_result = self.graph.query(query)

        expected = [[0],
                    [1],
                    [2],
                    [3],
                    [4],
                    [5]]
        self.env.assertEqual(actual_result.result_set, expected)

        # Perform chained arithmetic ops
        query = """MATCH (a)-[]->(b) WHERE a.a_val > 0 AND b.b_val > 0 WITH a.a_val * 2 + b.b_val AS val ORDER BY val RETURN val"""
        actual_result = self.graph.query(query)

        expected = [[15],
                    [31.5]]
        self.env.assertEqual(actual_result.result_set, expected)

    def test03_with_aggregate_op_read_queries(self):
        query = """MATCH (a)-[e]->() WITH COUNT(a.a_val) AS count_res, SUM(ID(e)) AS sum_res RETURN count_res, sum_res"""
        actual_result = self.graph.query(query)
        expected = [[6, 15]]
        self.env.assertEqual(actual_result.result_set, expected)

    # TODO UNWIND support needs to be extended for combinations like UNWIND...MATCH
    def test04_with_unwind_expressions(self):
        query = """UNWIND [1, 2, 3] AS x WITH x AS y RETURN y"""
        actual_result = self.graph.query(query)

        expected = [[1],
                    [2],
                    [3]]
        self.env.assertEqual(actual_result.result_set, expected)

        query = """UNWIND [1, 2, 3] AS x WITH x * 2 AS y RETURN y"""
        actual_result = self.graph.query(query)

        expected = [[2],
                    [4],
                    [6]]
        self.env.assertEqual(actual_result.result_set, expected)

        query = """UNWIND [1, 2, 3] AS x WITH x * 2 AS y WITH y * 2 AS z RETURN z"""
        actual_result = self.graph.query(query)

        expected = [[4],
                    [8],
                    [12]]
        self.env.assertEqual(actual_result.result_set, expected)

        query = """UNWIND [1, 2, 3] AS x WITH x * 2 AS y WITH y * 2 AS z WITH MIN(z) as min RETURN min"""
        actual_result = self.graph.query(query)

        expected = [[4]]
        self.env.assertEqual(actual_result.result_set, expected)

    def test05_with_create_expressions(self):
        query = """CREATE (c:c_label {c_val: 25}) WITH c AS c RETURN c.c_val AS val"""
        actual_result = self.graph.query(query)
        expected = [[25]]
        self.env.assertEqual(actual_result.result_set, expected)
        self.env.assertEqual(actual_result.labels_added, 1)
        self.env.assertEqual(actual_result.nodes_created, 1)
        self.env.assertEqual(actual_result.properties_set, 1)

        # TODO Update CREATE to accept variable arguments from UNWIND, WITH, etc
        query = """UNWIND [5] AS a WITH a AS b CREATE (:unwind_label {prop: 'some_constant'})"""
        actual_result = self.graph.query(query)

        self.env.assertEqual(actual_result.labels_added, 1)
        self.env.assertEqual(actual_result.nodes_created, 1)
        self.env.assertEqual(actual_result.properties_set, 1)
        query = """MATCH (a:unwind_label) RETURN a.prop"""
        actual_result = self.graph.query(query)

        # Verify that the entity was created properly
        expected = [['some_constant']]
        self.env.assertEqual(actual_result.result_set, expected)

    def test06_update_expressions(self):
        query = """MATCH (c:c_label) SET c.c_val = 50 WITH c.c_val AS val RETURN val"""
        actual_result = self.graph.query(query)
        expected = [[50]]
        self.env.assertEqual(actual_result.result_set, expected)
        self.env.assertEqual(actual_result.properties_set, 1)

    # Verify that projected nodes, edges, and scalars can be returned properly
    def test07_projected_graph_entities(self):
        query = """MATCH (a)-[e]->(b) WITH a, e, b.b_val AS b_val ORDER BY a.a_val LIMIT 2 RETURN *"""
        actual_result = self.graph.query(query)

        # Validate the header strings of the 3 columns.
        # NOTE - currently, RETURN * populates values in alphabetical order, but that is subject to later change.
        self.env.assertEqual(actual_result.header[0][1], 'a')
        self.env.assertEqual(actual_result.header[1][1], 'b_val')
        self.env.assertEqual(actual_result.header[2][1], 'e')

        # Verify that 2 rows and 3 columns are returned.
        self.env.assertEqual(len(actual_result.result_set), 2)
        self.env.assertEqual(len(actual_result.result_set[0]), 3)

    def test08_filter_projected(self):
        # If x is odd then ceil(x/2) > floor(x/2)
        # otherwise, x is even and ceil(x/2) == floor(x/2).
        # ceil(5/2) = 3, floor(5/2) = 2
        # ceil(6/2) = 3, floor(6/2) = 3.
        query = """unwind(range(0, 10)) as x with x as x where ceil(x/2.0) > floor(x/2.0) return count(x)"""
        actual_result = self.graph.query(query)
        # Expecting count of 5: [1,3,5,7,9].
        self.env.assertEqual(actual_result.result_set[0], [5])

    # Verify that filters can properly be placed in the scope up to WITH and the expressions projected by it.
    def test09_filter_placement(self):
        # Place a filter on a projected expression.
        query = """UNWIND [1,2,3] AS a WITH a WHERE a = 2 RETURN a"""
        actual_result = self.graph.query(query)
        expected = [[2]]
        self.env.assertEqual(actual_result.result_set, expected)

        # Place a filter on a projected expression that is aliased by the WITH clause.
        query = """UNWIND [1,2,3] AS a WITH a AS b WHERE a = 2 RETURN b"""
        actual_result = self.graph.query(query)
        expected = [[2]]
        self.env.assertEqual(actual_result.result_set, expected)

    def test10_filter_placement_validate_scopes(self):
        # Verify that filters cannot be placed in earlier scopes.
        query = """UNWIND ['scope1'] AS a
                   WITH a AS b
                   UNWIND ['scope2'] AS a
                   WITH a
                   WHERE a = 'scope1'
                   RETURN a"""
        actual_result = self.graph.query(query)
        expected = [] # No results should be returned
        self.env.assertEqual(actual_result.result_set, expected)

        query = """UNWIND ['scope1'] AS a WITH a AS b UNWIND ['scope2'] AS a WITH a WHERE a = 'scope2' RETURN a"""
        actual_result = self.graph.query(query)
        expected = [['scope2']]
        self.env.assertEqual(actual_result.result_set, expected)

        # Verify that WITH filters are properly placed in scope without violating Apply restrictions.
        query = """MATCH (a) OPTIONAL MATCH (b) WITH a, b WHERE b.fakeprop = true RETURN a, b"""
        actual_result = self.graph.query(query)
        expected = [] # No results should be returned
        self.env.assertEqual(actual_result.result_set, expected)
        # Verify that the Filter op appears directly above the Apply operation in the ExecutionPlan.
        plan = str(self.graph.explain(query))
        self.env.assertTrue(re.search('Filter\s+Apply', plan))

        # Verify that filters on projected aliases do not get placed before the projection op.
        query = """UNWIND [1] AS a WITH a AS b, 'projected' AS a WHERE a = 1 RETURN a"""
        plan = str(self.graph.explain(query))
        actual_result = self.graph.query(query)
        expected = [] # No results should be returned
        self.env.assertEqual(actual_result.result_set, expected)
        self.env.assertTrue(re.search('Filter\s+Project', plan))

        query = """UNWIND [1] AS a WITH a AS b, 'projected' AS a WHERE a = 'projected' RETURN a"""
        plan = str(self.graph.explain(query))
        actual_result = self.graph.query(query)
        expected = [['projected']] # The projected string should be returned
        self.env.assertTrue(re.search('Filter\s+Project', plan))

    def test11_valid_order_by_aliases(self):
        # Verify that ORDER BY aliases match previously defined references
        query = """UNWIND [1,2,3] AS a WITH a ORDER BY a RETURN a"""
        actual_result = self.graph.query(query)
        expected_result = [[1],
                           [2],
                           [3]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        invalid_queries = ["UNWIND [1,2,3] AS a WITH a ORDER BY nonexistent RETURN a",
                           "UNWIND [1,2,3] AS a WITH a ORDER BY a, nonexistent RETURN a",
                           "UNWIND [1,2,3] AS a WITH a AS b UNWIND [1,2,3] AS c WITH c ORDER BY a RETURN c"]
        for query in invalid_queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                # Expecting an error.
                self.env.assertIn("not defined", str(e))

    def test12_cartesian_product_reset_single_response(self):
        # Verify that WITH projections that with no children are reset
        # properly by CartesianProduct ops
        query = """WITH 1 AS x MATCH (a:label_a), (b:label_b) RETURN a.v, b.v"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(len(actual_result.result_set), 36)
