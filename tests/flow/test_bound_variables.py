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

    # An incoming hop from a node bound by a previous traverse must find the
    # sources through the per-type transposes, not a transpose of the whole
    # adjacency matrix (#3573). Covers the few-type and many-type paths.
    def test_incoming_hop_from_bound_node(self):
        for n_types in (3, 20):
            g = self.db.select_graph(f"incoming_hop_{n_types}")
            try:
                g.query("UNWIND range(0, 59) AS i CREATE (:n {id: 'test' + toString(i)})")
                for k in range(n_types):
                    g.query(f"MATCH (a:n), (b:n) WHERE id(b) = (id(a) * 7 + {k * 5 + 1}) % 60 "
                            f"AND id(a) % {n_types} = {k} CREATE (a)-[:R{k}]->(b)")
                g.query("MATCH (a:n {id: 'test0'}), (b:n {id: 'test7'}) CREATE (a)-[:R0]->(b), (b)-[:R1]->(a)")
                for i in range(0, 60, 7):
                    for tail in ("", "WHERE x.id <> 'none'"):
                        q = (f"MATCH (:n {{id: 'test{i}'}})-[]->(m)<-[]-(x:n) {tail} "
                             "RETURN id(m), id(x) ORDER BY id(m), id(x)")
                        got = g.query(q).result_set
                        # Independent oracle: hop backwards from m over the
                        # typed relationships only.
                        want = set()
                        for k in range(n_types):
                            want |= {tuple(r) for r in g.query(
                                f"MATCH (:n {{id: 'test{i}'}})-[]->(m)<-[:R{k}]-(x:n) "
                                "RETURN id(m), id(x)").result_set}
                        self.env.assertEqual(sorted(set(map(tuple, got))), sorted(want))
            finally:
                g.delete()
