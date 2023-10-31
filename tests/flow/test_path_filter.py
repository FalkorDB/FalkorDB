import re
from common import *
from index_utils import *
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from demo import QueryInfo

GRAPH_ID = "G"
redis_con = None
redis_graph = None


class testPathFilter(FlowTestsBase):
    def __init__(self):
        self.env = Env(decodeResponses=True)
        global redis_con
        redis_con = self.env.getConnection()

    def setUp(self):
        global redis_graph
        redis_graph = Graph(redis_con, GRAPH_ID)
        self.env.flush()

    def test00_simple_path_filter(self):
        node0 = Node(node_id=0, label="L")
        node1 = Node(node_id=1, label="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_edge(edge01)
        redis_graph.flush()

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) RETURN n"
        result_set = redis_graph.query(query)
        expected_results = [[node0]]
        query_info = QueryInfo(query = query, description="Tests simple path filter", expected_result = expected_results)
        self._assert_resultset_equals_expected(result_set, query_info)

    def test01_negated_simple_path_filter(self):
        node0 = Node(node_id=0, label="L")
        node1 = Node(node_id=1, label="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_edge(edge01)
        redis_graph.flush()

        query = "MATCH (n:L) WHERE NOT (n)-[:R]->(:L) RETURN n"
        result_set = redis_graph.query(query)
        expected_results = [[node1]]
        query_info = QueryInfo(query = query, description="Tests simple negated path filter", expected_result = expected_results)
        self._assert_resultset_equals_expected(result_set, query_info)

    def test02_test_path_filter_or_property_filter(self):
        node0 = Node(node_id=0, label="L")
        node1 = Node(node_id=1, label="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_edge(edge01)
        redis_graph.flush()

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) OR n.x=1 RETURN n"
        result_set = redis_graph.query(query)
        expected_results = [[node0],[node1]]
        query_info = QueryInfo(query = query, description="Tests OR condition with simple filter and path filter", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test03_path_filter_or_negated_path_filter(self):
        node0 = Node(node_id=0, label="L")
        node1 = Node(node_id=1, label="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_edge(edge01)
        redis_graph.flush()

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) OR NOT (n)-[:R]->(:L) RETURN n"
        result_set = redis_graph.query(query)
        expected_results = [[node0],[node1]]
        query_info = QueryInfo(query = query, description="Tests OR condition with path and negated path filters", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test04_test_level_1_nesting_logical_operators_over_path_and_property_filters(self):
        node0 = Node(node_id=0, label="L")
        node1 = Node(node_id=1, label="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_edge(edge01)
        redis_graph.flush()

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) OR (n.x=1 AND NOT (n)-[:R]->(:L)) RETURN n"
        result_set = redis_graph.query(query)
        expected_results = [[node0],[node1]]
        query_info = QueryInfo(query = query, description="Tests AND condition with simple filter and negated path filter", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test05_test_level_2_nesting_logical_operators_over_path_and_property_filters(self):
        node0 = Node(node_id=0, label="L")
        node1 = Node(node_id=1, label="L", properties={'x':1})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_edge(edge01)
        redis_graph.flush()

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) OR (n.x=1 AND (n.x = 2 OR NOT (n)-[:R]->(:L))) RETURN n"
        result_set = redis_graph.query(query)
        expected_results = [[node0],[node1]]
        query_info = QueryInfo(query = query, description="Tests AND condition with simple filter and nested OR", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test06_test_level_2_nesting_logical_operators_over_path_filters(self):
        node0 = Node(node_id=0, label="L")
        node1 = Node(node_id=1, label="L", properties={'x':1})
        node2 = Node(node_id=2, label="L2")
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        edge12 = Edge(src_node=node1, dest_node=node2, relation="R2")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_node(node2)
        redis_graph.add_edge(edge01)
        redis_graph.add_edge(edge12)
        redis_graph.flush()

        query = "MATCH (n:L) WHERE (n)-[:R]->(:L) OR (n.x=1 AND ((n)-[:R2]->(:L2) OR (n)-[:R]->(:L))) RETURN n"
        result_set = redis_graph.query(query)
        expected_results = [[node0],[node1]]
        query_info = QueryInfo(query = query, description="Tests AND condition with simple filter and nested OR", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test07_test_edge_filters(self):
        node0 = Node(node_id=0, label="L", properties={'x': 'a'})
        node1 = Node(node_id=1, label="L", properties={'x': 'b'})
        node2 = Node(node_id=2, label="L", properties={'x': 'c'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R", properties={'x': 1})
        edge12 = Edge(src_node=node1, dest_node=node2, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_node(node2)
        redis_graph.add_edge(edge01)
        redis_graph.add_edge(edge12)
        redis_graph.flush()

        query = "MATCH (n:L) WHERE (n)-[:R {x:1}]->() RETURN n.x"
        result_set = redis_graph.query(query)
        expected_results = [['a']]
        query_info = QueryInfo(query = query, description="Tests pattern filter edge conditions", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(result_set, query_info)

    def test08_indexed_child_stream_resolution(self):
        node0 = Node(node_id=0, label="L", properties={'x': 'a'})
        node1 = Node(node_id=1, label="L", properties={'x': 'b'})
        node2 = Node(node_id=2, label="L", properties={'x': 'c'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        edge12 = Edge(src_node=node1, dest_node=node2, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_node(node2)
        redis_graph.add_edge(edge01)
        redis_graph.add_edge(edge12)
        redis_graph.flush()

        # Create index.
        result_set = create_node_exact_match_index(redis_graph, 'L', 'x', sync=True)
        self.env.assertEquals(result_set.indices_created, 1)

        # Issue a query in which the bound variable stream of the SemiApply op is an Index Scan.
        query = "MATCH (n:L) WHERE (:L)<-[]-(n)<-[]-(:L {x: 'a'}) AND n.x = 'b' RETURN n.x"
        result_set = redis_graph.query(query)
        expected_results = [['b']]
        self.env.assertEquals(result_set.result_set, expected_results)

    def test09_no_invalid_expand_into(self):
        node0 = Node(node_id=0, label="L", properties={'x': 'a'})
        node1 = Node(node_id=1, label="L", properties={'x': 'b'})
        node2 = Node(node_id=2, label="L", properties={'x': 'c'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        edge12 = Edge(src_node=node1, dest_node=node2, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_node(node2)
        redis_graph.add_edge(edge01)
        redis_graph.add_edge(edge12)
        redis_graph.flush()

        # Issue a query in which the match stream and the bound stream must both perform traversal.
        query = "MATCH (n:L)-[]->(:L) WHERE ({x: 'a'})-[]->(n) RETURN n.x"
        plan = redis_graph.execution_plan(query)
        # Verify that the execution plan has no Expand Into and two traversals.
        self.env.assertNotIn("Expand Into", plan)
        self.env.assertEquals(2, plan.count("Conditional Traverse"))

        result_set = redis_graph.query(query)
        expected_results = [['b']]
        self.env.assertEquals(result_set.result_set, expected_results)

    def test10_verify_apply_results(self):
        # Build a graph with 3 nodes and 3 edges, 2 of which have the same source.
        node0 = Node(node_id=0, label="L", properties={'x': 'a'})
        node1 = Node(node_id=1, label="L", properties={'x': 'b'})
        node2 = Node(node_id=2, label="L", properties={'x': 'c'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        edge02 = Edge(src_node=node0, dest_node=node2, relation="R")
        edge12 = Edge(src_node=node1, dest_node=node2, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_node(node2)
        redis_graph.add_edge(edge01)
        redis_graph.add_edge(edge02)
        redis_graph.add_edge(edge12)
        redis_graph.flush()

        query = "MATCH (n:L) WHERE (n)-[]->() RETURN n.x ORDER BY n.x"
        result_set = redis_graph.query(query)
        # Each source node should be returned exactly once.
        expected_results = [['a'], ['b']]
        self.env.assertEquals(result_set.result_set, expected_results)

    def test11_unbound_path_filters(self):
        # Build a graph with 2 nodes connected by 1 edge.
        node0 = Node(node_id=0, label="L", properties={'x': 'a'})
        node1 = Node(node_id=1, label="L", properties={'x': 'b'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_edge(edge01)
        redis_graph.flush()

        # Emit a query that uses an AntiSemiApply op to return values.
        query = "MATCH (n:L) WHERE NOT (:L)-[]->() RETURN n.x ORDER BY n.x"
        result_set = redis_graph.query(query)
        # The WHERE filter evaluates to false, no results should be returned.
        expected_result = []
        self.env.assertEquals(result_set.result_set, expected_result)

        # Emit a query that uses a SemiApply op to return values.
        query = "MATCH (n:L) WHERE (:L)-[]->() RETURN n.x ORDER BY n.x"
        result_set = redis_graph.query(query)
        # The WHERE filter evaluates to true, all results should be returned.
        expected_result = [['a'],
                           ['b']]
        self.env.assertEquals(result_set.result_set, expected_result)

    def test12_label_introduced_in_path_filter(self):
        # Build a graph with 2 nodes connected by 1 edge.
        node0 = Node(node_id=0, label="L", properties={'x': 'a'})
        node1 = Node(node_id=1, label="L", properties={'x': 'b'})
        edge01 = Edge(src_node=node0, dest_node=node1, relation="R")
        redis_graph.add_node(node0)
        redis_graph.add_node(node1)
        redis_graph.add_edge(edge01)
        redis_graph.flush()

        # Write a WHERE filter that introduces label data.
        query = "MATCH (a1)-[]->(a2) WHERE (a1:L)-[]->(a2:L) return a1.x, a2.x"
        result_set = redis_graph.query(query)
        expected_result = [['a', 'b']]
        self.env.assertEquals(result_set.result_set, expected_result)

    def test13_path_filter_in_different_scope(self):
        # create a graph of the form:
        # (a)-[]->(b)-[]->(c)
        a = Node(node_id=0, label="L", properties={'x': 'a'})
        b = Node(node_id=1, label="L", properties={'x': 'b'})
        c = Node(node_id=2, label="L", properties={'x': 'c'})
        ab = Edge(src_node=a, dest_node=b, relation="R")
        bc = Edge(src_node=b, dest_node=c, relation="R")
        redis_graph.add_node(a)
        redis_graph.add_node(b)
        redis_graph.add_node(c)
        redis_graph.add_edge(ab)
        redis_graph.add_edge(bc)
        redis_graph.flush()

        # match nodes with an outgoing edge that optionally have an incoming edge
        query = """MATCH (n)
                   OPTIONAL MATCH (n)<-[]-()
                   WHERE exists((n)-[]->())
                   WITH n
                   RETURN n.x
                   ORDER BY n.x"""
        result_set = redis_graph.query(query)
        expected_result = [['a'],
                           ['b']]
        self.env.assertEquals(result_set.result_set, expected_result)

    def test14_path_and_predicate_filters(self):
        # Build a graph with 2 nodes connected by 1 edge.
        redis_graph.query("CREATE (:L {x:'a'})-[:R]->(:L {x:'b'})")

        # Write a WHERE clause that evaluates a predicate on a node and a path filter.
        query = "MATCH (a:L) WHERE (a)-[]->() AND a.x = 'a' return a.x"
        plan_1 = redis_graph.execution_plan(query)
        # The predicate filter should be evaluated between the Apply and Scan ops.
        self.env.assertTrue(re.search('Semi Apply\s+Filter\s+Node By Label Scan', plan_1))
        result_set = redis_graph.query(query)
        expected_result = [['a']]
        self.env.assertEquals(result_set.result_set, expected_result)

        # Swap the order of the WHERE clause filters.
        query = "MATCH (a:L) WHERE a.x = 'a' AND (a)-[]->() return a.x"
        plan_2 = redis_graph.execution_plan(query)
        # The plan should be identical to the one constructed previously.
        self.env.assertEqual(plan_1, plan_2)

