import re
from common import *
from index_utils import *
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from demo import QueryInfo

GRAPH_ID = "path_filters"


class testPathFilter(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def setUp(self):
        self.conn.delete(GRAPH_ID)

    def test00_simple_path_filter(self):
        node0 = Node(alias="n0", node_id=0, labels="L")
        node1 = Node(alias="n1", node_id=1, labels="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {edge01}")

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) RETURN n"
        result_set = self.graph.query(query)
        expected_results = [[node0]]
        query_info = QueryInfo(query = query, description="Tests simple path filter", expected_result = expected_results)
        self._assert_resultset_equals_expected(result_set, query_info)

    def test01_negated_simple_path_filter(self):
        node0 = Node(alias="n0", node_id=0, labels="L")
        node1 = Node(alias="n1", node_id=1, labels="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {edge01}")

        query = "MATCH (n:L) WHERE NOT (n)-[:R]->(:L) RETURN n"
        result_set = self.graph.query(query)
        expected_results = [[node1]]
        query_info = QueryInfo(query = query, description="Tests simple negated path filter", expected_result = expected_results)
        self._assert_resultset_equals_expected(result_set, query_info)

    def test02_test_path_filter_or_property_filter(self):
        node0  = Node(alias="n0", node_id=0, labels="L")
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {edge01}")

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) OR n.x=1 RETURN n"
        result_set = self.graph.query(query)
        expected_results = [[node0],[node1]]
        query_info = QueryInfo(query = query, description="Tests OR condition with simple filter and path filter", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test03_path_filter_or_negated_path_filter(self):
        node0  = Node(alias="n0", node_id=0, labels="L")
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {edge01}")

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) OR NOT (n)-[:R]->(:L) RETURN n"
        result_set = self.graph.query(query)
        expected_results = [[node0],[node1]]
        query_info = QueryInfo(query = query, description="Tests OR condition with path and negated path filters", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test04_test_level_1_nesting_logical_operators_over_path_and_property_filters(self):
        node0  = Node(alias="n0", node_id=0, labels="L")
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {edge01}")

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) OR (n.x=1 AND NOT (n)-[:R]->(:L)) RETURN n"
        result_set = self.graph.query(query)
        expected_results = [[node0],[node1]]
        query_info = QueryInfo(query = query, description="Tests AND condition with simple filter and negated path filter", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test05_test_level_2_nesting_logical_operators_over_path_and_property_filters(self):
        node0  = Node(alias="n0", node_id=0, labels="L")
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {edge01}")

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) OR (n.x=1 AND (n.x = 2 OR NOT (n)-[:R]->(:L))) RETURN n"
        result_set = self.graph.query(query)
        expected_results = [[node0],[node1]]
        query_info = QueryInfo(query = query, description="Tests AND condition with simple filter and nested OR", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test06_test_level_2_nesting_logical_operators_over_path_filters(self):
        node0  = Node(alias="n0", node_id=0, labels="L")
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x':1})
        node2  = Node(alias="n2", node_id=2, labels="L2")
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        edge12 = Edge(src_node=node1, dest_node=node2, relation="R2")
        self.graph.query(f"CREATE {node0}, {node1}, {node2}, {edge01}, {edge12}")

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) OR (n.x=1 AND ((n)-[:R2]->(:L2) OR (n)-[:R]->(:L))) RETURN n"
        result_set = self.graph.query(query)
        expected_results = [[node0],[node1]]
        query_info = QueryInfo(query = query, description="Tests AND condition with simple filter and nested OR", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test07_test_edge_filters(self):
        node0  = Node(alias="n0", node_id=0, labels="L", properties={'x': 'a'})
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x': 'b'})
        node2  = Node(alias="n2", node_id=2, labels="L", properties={'x': 'c'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R", properties={'x': 1})
        edge12 = Edge(src_node=node1, dest_node=node2, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {node2}, {edge01}, {edge12}")

        query = "MATCH (n:L) WHERE (n)-[:R {x:1}]->() RETURN n.x"
        result_set = self.graph.query(query)
        expected_results = [['a']]
        query_info = QueryInfo(query = query, description="Tests pattern filter edge conditions", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test08_indexed_child_stream_resolution(self):
        node0  = Node(alias="n0", node_id=0, labels="L", properties={'x': 'a'})
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x': 'b'})
        node2  = Node(alias="n2", node_id=2, labels="L", properties={'x': 'c'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        edge12 = Edge(src_node=node1, dest_node=node2, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {node2}, {edge01}, {edge12}")

        # Create index.
        result_set = create_node_range_index(self.graph, 'L', 'x', sync=True)
        self.env.assertEquals(result_set.indices_created, 1)

        # Issue a query in which the bound variable stream of the SemiApply op is an Index Scan.
        query = "MATCH (n:L) WHERE (:L)<-[]-(n)<-[]-(:L {x: 'a'}) AND n.x = 'b' RETURN n.x"
        result_set = self.graph.query(query)
        expected_results = [['b']]
        self.env.assertEquals(result_set.result_set, expected_results)

    def test09_no_invalid_expand_into(self):
        node0  = Node(alias="n0", node_id=0, labels="L", properties={'x': 'a'})
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x': 'b'})
        node2  = Node(alias="n2", node_id=2, labels="L", properties={'x': 'c'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        edge12 = Edge(src_node=node1, dest_node=node2, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {node2}, {edge01}, {edge12}")

        # Issue a query in which the match stream and the bound stream must both perform traversal.
        query = "MATCH (n:L)-[]->(:L) WHERE ({x: 'a'})-[]->(n) RETURN n.x"
        plan = str(self.graph.explain(query))
        # Verify that the execution plan has no Expand Into and two traversals.
        self.env.assertNotIn("Expand Into", plan)
        self.env.assertEquals(2, plan.count("Conditional Traverse"))

        result_set = self.graph.query(query)
        expected_results = [['b']]
        self.env.assertEquals(result_set.result_set, expected_results)

    def test10_verify_apply_results(self):
        # Build a graph with 3 nodes and 3 edges, 2 of which have the same source.
        node0  = Node(alias="n0", node_id=0, labels="L", properties={'x': 'a'})
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x': 'b'})
        node2  = Node(alias="n2", node_id=2, labels="L", properties={'x': 'c'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        edge02 = Edge(src_node=node0, dest_node=node2, relation="R")
        edge12 = Edge(src_node=node1, dest_node=node2, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {node2}, {edge01}, {edge02}, {edge12}")

        query = "MATCH (n:L) WHERE (n)-[]->() RETURN n.x ORDER BY n.x"
        result_set = self.graph.query(query)
        # Each source node should be returned exactly once.
        expected_results = [['a'], ['b']]
        self.env.assertEquals(result_set.result_set, expected_results)

    def test11_unbound_path_filters(self):
        # Build a graph with 2 nodes connected by 1 edge.
        node0  = Node(alias="n0", node_id=0, labels="L", properties={'x': 'a'})
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x': 'b'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {edge01}")

        # Emit a query that uses an AntiSemiApply op to return values.
        query = "MATCH (n:L) WHERE NOT (:L)-[]->() RETURN n.x ORDER BY n.x"
        result_set = self.graph.query(query)
        # The WHERE filter evaluates to false, no results should be returned.
        expected_result = []
        self.env.assertEquals(result_set.result_set, expected_result)

        # Emit a query that uses a SemiApply op to return values.
        query = "MATCH (n:L) WHERE (:L)-[]->() RETURN n.x ORDER BY n.x"
        result_set = self.graph.query(query)
        # The WHERE filter evaluates to true, all results should be returned.
        expected_result = [['a'],
                           ['b']]
        self.env.assertEquals(result_set.result_set, expected_result)

    def test12_label_introduced_in_path_filter(self):
        # Build a graph with 2 nodes connected by 1 edge.
        node0  = Node(alias="n0", node_id=0, labels="L", properties={'x': 'a'})
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x': 'b'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {edge01}")

        # Write a WHERE filter that introduces label data.
        query = "MATCH (a1)-[]->(a2) WHERE (a1:L)-[]->(a2:L) return a1.x, a2.x"
        result_set = self.graph.query(query)
        expected_result = [['a', 'b']]
        self.env.assertEquals(result_set.result_set, expected_result)

    def test13_path_filter_in_different_scope(self):
        # Create a graph of the form:
        # (c)-[]->(a)-[]->(b)
        node0  = Node(alias="n0", node_id=0, labels="L", properties={'x': 'a'})
        node1  = Node(alias="n1", node_id=1, labels="L", properties={'x': 'b'})
        node2  = Node(alias="n2", node_id=2, labels="L", properties={'x': 'c'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        edge12 = Edge(src_node=node1, dest_node=node2, relation="R")
        self.graph.query(f"CREATE {node0}, {node1}, {node2}, {edge01}, {edge12}")

        # Match nodes with an outgoing edge that optionally have an incoming edge.
        query = "MATCH (a) OPTIONAL MATCH (a)<-[]-() WITH a WHERE (a)-[]->() return a.x ORDER BY a.x"
        result_set = self.graph.query(query)
        expected_result = [['a'],
                           ['b']]
        self.env.assertEquals(result_set.result_set, expected_result)

    def test14_path_and_predicate_filters(self):
        # Build a graph with 2 nodes connected by 1 edge.
        self.graph.query("CREATE (:L {x:'a'})-[:R]->(:L {x:'b'})")

        # Write a WHERE clause that evaluates a predicate on a node and a path filter.
        query = "MATCH (a:L) WHERE (a)-[]->() AND a.x = 'a' return a.x"
        plan_1 = str(self.graph.explain(query))
        # The predicate filter should be evaluated between the Apply and Scan ops.
        self.env.assertTrue(re.search('Semi Apply\s+Filter\s+Node By Label Scan', plan_1))
        result_set = self.graph.query(query)
        expected_result = [['a']]
        self.env.assertEquals(result_set.result_set, expected_result)

        # Swap the order of the WHERE clause filters.
        query = "MATCH (a:L) WHERE a.x = 'a' AND (a)-[]->() return a.x"
        plan_2 = str(self.graph.explain(query))
        # The plan should be identical to the one constructed previously.
        self.env.assertEqual(plan_1, plan_2)

    def test15_named_path_filter_position(self):
        # make sure the named path filter are positioned correctly
        # named paths are a bit different than ordinary aliases e.g. 'n'
        # a named path 'p' is constructed from its individual components
        # at projection time. this is why filters applied to named paths
        # need to be positioned right below the projection operation
        # forming the named path

        # create graph
        self.graph.query("CREATE (:A {v:1}), (:A {v:2})")

        q = """MATCH p=(n)
               WITH p
               WHERE all(x in nodes(p) WHERE x.v=1)
               RETURN count(1)"""

        plan = self.graph.explain(q)

        op_result = (plan.structured_plan)
        self.env.assertEqual(op_result.name, 'Results')
        self.env.assertEqual(len(op_result.children), 1)

        op_aggregate = op_result.children[0]
        self.env.assertEqual(op_aggregate.name, 'Aggregate')
        self.env.assertEqual(len(op_aggregate.children), 1)

        op_filter = op_aggregate.children[0]
        self.env.assertEqual(op_filter.name, 'Filter')
        self.env.assertEqual(len(op_filter.children), 1)

        op_project = op_filter.children[0]
        self.env.assertEqual(op_project.name, 'Project')
        self.env.assertEqual(len(op_project.children), 1)

        res = self.graph.query(q).result_set
        self.env.assertEqual(res[0][0], 1)

    def test16_xor_with_path_filter(self):
        # regression test for issue #1466
        # XOR combined with a path filter must return an error, not crash
        try:
            self.graph.query(
                "MATCH (n) WHERE NOT((()--()) XOR TRUE) RETURN n")
            self.env.assertTrue(False, "Expected an error")
        except ResponseError as e:
            self.env.assertContains("XOR", str(e))
