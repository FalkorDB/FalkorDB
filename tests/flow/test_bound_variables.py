from common import *
from index_utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

redis_graph = None
GRAPH_ID = "G"


class testBoundVariables(FlowTestsBase):
    def __init__(self):
        self.env = Env(decodeResponses=True)
        global redis_graph
        redis_con = self.env.getConnection()
        redis_graph = Graph(redis_con, GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        global redis_graph

        # Construct a graph with the form:
        # (v1)-[:E]->(v2)-[:E]->(v3)
        node_props = ['v1', 'v2', 'v3']

        nodes = []
        for idx, v in enumerate(node_props):
            node = Node(label="L", properties={"val": v})
            nodes.append(node)
            redis_graph.add_node(node)

        edge = Edge(nodes[0], "E", nodes[1])
        redis_graph.add_edge(edge)

        edge = Edge(nodes[1], "E", nodes[2])
        redis_graph.add_edge(edge)

        redis_graph.commit()

    def test01_with_projected_entity(self):
        query = """MATCH (a:L {val: 'v1'}) WITH a MATCH (a)-[e]->(b) RETURN b.val"""
        actual_result = redis_graph.query(query)

        # Verify that this query does not generate a Cartesian product.
        execution_plan = redis_graph.execution_plan(query)
        self.env.assertNotIn('Cartesian Product', execution_plan)

        # Verify results.
        expected_result = [['v2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test02_match_create_bound_variable(self):
        # Extend the graph such that the new form is:
        # (v1)-[:E]->(v2)-[:E]->(v3)-[:e]->(v4)
        query = """MATCH (a:L {val: 'v3'}) CREATE (a)-[:E]->(b:L {val: 'v4'}) RETURN b.val"""
        actual_result = redis_graph.query(query)
        expected_result = [['v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)
        self.env.assertEquals(actual_result.relationships_created, 1)
        self.env.assertEquals(actual_result.nodes_created, 1)

    def test03_procedure_match_bound_variable(self):
        # Create a full-text index.
        create_node_fulltext_index(redis_graph, "L", "val", sync=True)

        # Project the result of scanning this index into a MATCH pattern.
        query = """CALL db.idx.fulltext.queryNodes('L', 'v1') YIELD node MATCH (node)-[]->(b) RETURN b.val"""
        # Verify that execution begins at the procedure call and proceeds into the traversals.
        execution_plan = redis_graph.execution_plan(query)
        # For the moment, we'll just verify that ProcedureCall appears later in the plan than
        # its parent, Conditional Traverse.
        traverse_idx = execution_plan.index("Conditional Traverse")
        call_idx = execution_plan.index("ProcedureCall")
        self.env.assertTrue(call_idx > traverse_idx)

        # Verify the results
        actual_result = redis_graph.query(query)
        expected_result = [['v2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test04_projected_scanned_entity(self):
        query = """MATCH (a:L {val: 'v1'}) WITH a MATCH (a), (b {val: 'v2'}) RETURN a.val, b.val"""
        actual_result = redis_graph.query(query)

        # Verify that this query generates exactly 2 scan ops.
        execution_plan = redis_graph.execution_plan(query)
        self.env.assertEquals(2, execution_plan.count('Scan'))

        # Verify results.
        expected_result = [['v1', 'v2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test05_unwind_reference_entities(self):
        query = """MATCH ()-[a]->() UNWIND a as x RETURN id(x)"""
        actual_result = redis_graph.query(query)

        # Verify results.
        expected_result = [[0], [1], [2]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test06_override_bound_with_label(self):
        """Tests that we override a bound alias with a new scan if it has a
        label"""

        # clear the db
        self.env.flush()
        redis_graph = Graph(self.env.getConnection(), GRAPH_ID)

        # create one node with label `N`
        res = redis_graph.query("CREATE (:N)")
        self.env.assertEquals(res.nodes_created, 1)

        res = redis_graph.query("MATCH(n:N) WITH n MATCH (n:X) RETURN n")

        # make sure no nodes were returned
        self.env.assertEquals(len(res.result_set), 0)
