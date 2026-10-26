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

    def test07_named_path_over_bound_node(self):
        """`p = (n)` over an already-bound `n` is the one-node path: never
        null, and the rows that bound `n` are kept (#2311, #2727, #2737)"""

        g = self.db.select_graph("bound_node_path")
        g.query("CREATE (:A {id: 1, k0: 'x'})-[:R]->(:B {id: 2})")

        # OPTIONAL MATCH: the path is bound, not null-padded (#2311, #2737)
        for label in [":A", ""]:
            res = g.query(f"""MATCH (n{label} {{id: 1}})
                              OPTIONAL MATCH p = (n)
                              RETURN n.id, length(p), nodes(p)[0] = n""")
            self.env.assertEqual(res.result_set, [[1, 0, True]], depth=1)

        # MATCH: the rows produced before the pattern reach it (#2727)
        res = g.query("""MATCH (n0:A) WITH n0 AS alias4
                         MATCH (alias4), p1 = (alias4)
                         RETURN alias4.k0, length(p1)""")
        self.env.assertEqual(res.result_set, [['x', 0]])

        res = g.query("MATCH (n:A) WITH * MATCH p = (n) RETURN count(p)")
        self.env.assertEqual(res.result_set, [[1]])

        res = g.query("""MATCH (n:A) CALL db.labels() YIELD label
                         MATCH (n), p = (n)
                         RETURN label, length(p) ORDER BY label""")
        self.env.assertEqual(res.result_set, [['A', 0], ['B', 0]])

        # inline filters on the bound node still apply
        res = g.query("""MATCH (n) WITH n MATCH p = (n {id: 2})
                         RETURN n.id, length(p)""")
        self.env.assertEqual(res.result_set, [[2, 0]])

        # a path over an edge whose every element is already bound
        res = g.query("""MATCH (a)-[r]->(b) WITH a, r, b
                         MATCH p = (a)-[r]->(b)
                         RETURN a.id, length(p)""")
        self.env.assertEqual(res.result_set, [[1, 1]])

        # the issue's full query: `alias4` survives the pipeline (#2727)
        res = g.query("""OPTIONAL MATCH p0 = (n0 {k0: 'x'})
                         WITH * WHERE n0 IS NOT NULL
                         CALL algo.BFS(n0, 2, 'R') YIELD nodes AS alias0
                         WITH alias0, n0.k0 AS alias1, n0 AS alias4, p0
                         ORDER BY alias1
                         MATCH (alias4), p1 = (alias4)
                         RETURN alias4.id, length(p1), size(alias0), alias1""")
        self.env.assertEqual(res.result_set, [[1, 0, 1, 'x']])

        g.delete()
