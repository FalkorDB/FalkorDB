from common import *
from index_utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

GRAPH_ID = "bound_variables"

class testBoundVariables(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        # Construct a graph with the form:
        # (v1)-[:E]->(v2)-[:E]->(v3)
        node_props = ['v1', 'v2', 'v3']

        nodes = []
        for idx, v in enumerate(node_props):
            node = Node(alias=f"n_{idx}", labels="L", properties={"val": v})
            nodes.append(node)
        nodes_str = [str(n) for n in nodes]

        e0 = Edge(nodes[0], "E", nodes[1])
        e1 = Edge(nodes[1], "E", nodes[2])

        self.graph.query(f"CREATE {','.join(nodes_str)}, {e0}, {e1}")

    def test01_with_projected_entity(self):
        query = """MATCH (a:L {val: 'v1'}) WITH a MATCH (a)-[e]->(b) RETURN b.val"""
        actual_result = self.graph.query(query)

        # Verify that this query does not generate a Cartesian product.
        execution_plan = str(self.graph.explain(query))
        self.env.assertNotContains('Cartesian Product', execution_plan)

        # Verify results.
        expected_result = [['v2']]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test02_match_create_bound_variable(self):
        # Extend the graph such that the new form is:
        # (v1)-[:E]->(v2)-[:E]->(v3)-[:e]->(v4)
        query = """MATCH (a:L {val: 'v3'}) CREATE (a)-[:E]->(b:L {val: 'v4'}) RETURN b.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v4']]
        self.env.assertEqual(actual_result.result_set, expected_result)
        self.env.assertEqual(actual_result.relationships_created, 1)
        self.env.assertEqual(actual_result.nodes_created, 1)

    def test03_procedure_match_bound_variable(self):
        # Create a full-text index.
        create_node_fulltext_index(self.graph, "L", "val", sync=True)

        # Project the result of scanning this index into a MATCH pattern.
        query = """CALL db.idx.fulltext.queryNodes('L', 'v1') YIELD node MATCH (node)-[]->(b) RETURN b.val"""
        # Verify that execution begins at the procedure call and proceeds into the traversals.
        execution_plan = str(self.graph.explain(query))
        # Verify that Node By Fulltext Index Scan appears later in the plan than
        # its parent, Conditional Traverse.
        traverse_idx = execution_plan.index("Conditional Traverse")
        call_idx = execution_plan.index("Node By Fulltext Index Scan")
        self.env.assertTrue(call_idx > traverse_idx)

        # Verify the results
        actual_result = self.graph.query(query)
        expected_result = [['v2']]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test04_projected_scanned_entity(self):
        query = """MATCH (a:L {val: 'v1'}) WITH a MATCH (a), (b {val: 'v2'}) RETURN a.val, b.val"""
        actual_result = self.graph.query(query)

        # Verify that this query generates exactly 2 scan ops.
        execution_plan = str(self.graph.explain(query))
        self.env.assertEqual(2, execution_plan.count('Scan'))

        # Verify results.
        expected_result = [['v1', 'v2']]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test05_unwind_reference_entities(self):
        query = """MATCH ()-[a]->() UNWIND a as x RETURN id(x)"""
        actual_result = self.graph.query(query)

        # Verify results.
        expected_result = [[0], [1], [2]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test06_override_bound_with_label(self):
        """Tests that we override a bound alias with a new scan if it has a
        label"""

        # clear the db
        self.graph.delete()

        # create one node with label `N`
        res = self.graph.query("CREATE (:N)")
        self.env.assertEqual(res.nodes_created, 1)

        res = self.graph.query("MATCH(n:N) WITH n MATCH (n:X) RETURN n")

        # make sure no nodes were returned
        self.env.assertEqual(len(res.result_set), 0)

    def test07_rebind_non_entity_variable_conflicts(self):
        self.graph.delete()
        self.graph.query("CREATE (:A)-[:R]->(:B)")

        def expect_err(query, expected_msg):
            try:
                self.graph.query(query)
                assert False, f"Expected error '{expected_msg}' but query succeeded: {query}"
            except ResponseError as e:
                self.env.assertContains(expected_msg, str(e))

        # Path variable bound to non-entity (map, integer, string, boolean, arithmetic, function, case)
        expect_err("WITH {x: 1} AS p MATCH p = ()-[]-() RETURN p", "Variable `p` already declared")
        expect_err("WITH 1 AS p MATCH p = ()-[]-() RETURN p", "Variable `p` already declared")
        expect_err("WITH 'str' AS p MATCH p = ()-[]-() RETURN p", "Variable `p` already declared")
        expect_err("WITH true AS p MATCH p = ()-[]-() RETURN p", "Variable `p` already declared")
        expect_err("WITH 1 + 2 AS p MATCH p = ()-[]-() RETURN p", "Variable `p` already declared")
        expect_err("WITH size([]) AS p MATCH p = ()-[]-() RETURN p", "Variable `p` already declared")
        expect_err("WITH CASE WHEN true THEN 1 ELSE 2 END AS p MATCH p = ()-[]-() RETURN p", "Variable `p` already declared")

        # Relationship variable bound to non-entity (map, integer, list, boolean, arithmetic, case)
        expect_err("WITH {x: 1} AS r MATCH ()-[r]-() RETURN r", "The alias 'r' was specified for both a node and a relationship.")
        expect_err("WITH 100 AS r MATCH ()-[r]-() RETURN r", "The alias 'r' was specified for both a node and a relationship.")
        expect_err("WITH [1, 2] AS r MATCH ()-[r]-() RETURN r", "The alias 'r' was specified for both a node and a relationship.")
        expect_err("WITH true AS r MATCH ()-[r]-() RETURN r", "The alias 'r' was specified for both a node and a relationship.")
        expect_err("WITH 1 + 2 AS r MATCH ()-[r]-() RETURN r", "The alias 'r' was specified for both a node and a relationship.")
        expect_err("WITH size([]) AS r MATCH ()-[r]-() RETURN r", "The alias 'r' was specified for both a node and a relationship.")
        expect_err("WITH CASE WHEN true THEN 1 ELSE 2 END AS r MATCH ()-[r]-() RETURN r", "The alias 'r' was specified for both a node and a relationship.")

        # Node variable bound to non-entity (map, integer, list, boolean, arithmetic, case)
        expect_err("WITH {x: 1} AS n MATCH (n) RETURN n", "The alias 'n' was specified for both a node and a relationship.")
        expect_err("WITH 42 AS n MATCH (n) RETURN n", "The alias 'n' was specified for both a node and a relationship.")
        expect_err("WITH [1, 2] AS n MATCH (n) RETURN n", "The alias 'n' was specified for both a node and a relationship.")
        expect_err("WITH true AS n MATCH (n) RETURN n", "The alias 'n' was specified for both a node and a relationship.")
        expect_err("WITH 1 + 2 AS n MATCH (n) RETURN n", "The alias 'n' was specified for both a node and a relationship.")
        expect_err("WITH size([]) AS n MATCH (n) RETURN n", "The alias 'n' was specified for both a node and a relationship.")
        expect_err("WITH CASE WHEN true THEN 1 ELSE 2 END AS n MATCH (n) RETURN n", "The alias 'n' was specified for both a node and a relationship.")

        # Legitimate entity reuse succeeds
        res = self.graph.query("MATCH (a:A)-[r:R]->(b:B) WITH a, r, b MATCH (a)-[r]->(b) RETURN a, r, b")
        self.env.assertEqual(len(res.result_set), 1)

        # Variable-length relationship projection carries exemption for per-edge property access
        plan = str(self.graph.explain("MATCH ()-[r:R*1..2]->() WITH r AS s WHERE s.prop = 1 RETURN count(*)"))
        self.env.assertContains("Filter", plan)

        # Null-bound entity inputs continue through binding
        res = self.graph.query("WITH NULL AS a MATCH (a) RETURN a")
        self.env.assertEqual(res.result_set, [[None]])

        res = self.graph.query("WITH NULL AS a MATCH (a)-[e]->(b) RETURN a, e, b")
        self.env.assertEqual(len(res.result_set), 0)

        # Projected NULL values remain valid for DELETE
        self.graph.query("WITH NULL AS n DELETE n")

        # Arithmetic inference returns integer for integer addition, compatible with range()
        res = self.graph.query("WITH 1 + 1 AS x RETURN range(0, x)")
        self.env.assertEqual(res.result_set, [[[0, 1, 2]]])

        # Function argument validation accepts source unions (e.g. startNode returning Union(Node, Null) into id())
        res = self.graph.query("MATCH ()-[r]->() WITH startNode(r) AS n RETURN id(n)")
        self.env.assertEqual(len(res.result_set), 1)
