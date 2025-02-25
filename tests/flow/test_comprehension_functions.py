from common import *
from execution_plan_util import locate_operation

GRAPH_ID = "list_comprehension"

def _check_pattern_comprehension_plan(plan: ExecutionPlan):
    apply = locate_operation(plan.structured_plan, "Apply")
    return apply and                                          \
        len(apply.children) == 2 and                          \
        apply.children[1].name == "Aggregate" and             \
        locate_operation(apply.children[1], "Argument")

class testComprehensionFunctions(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        # Construct a graph with the form:
        # (v1)-[e1]->(v2)-[e2]->(v3)
        node_props = ['v1', 'v2', 'v3']

        nodes = []
        for idx, v in enumerate(node_props):
            node = Node(alias=f"n_{idx}", labels="L", properties={"val": v})
            nodes.append(node)
        nodes_str = [str(node) for node in nodes]

        e0 = Edge(nodes[0], "E", nodes[1], properties={"edge_val": ['v1', 'v2']})
        e1 = Edge(nodes[1], "E", nodes[2], properties={"edge_val": ['v2', 'v3']})

        self.graph.query(f"CREATE {', '.join(nodes_str)}, {e0}, {e1}")

    # Test list comprehension queries with scalar inputs and a single result row
    def test01_list_comprehension_single_return(self):
        expected_result = [[[2, 6]]]

        # Test logically identical queries that generate the same input array with different methods.
        query = """WITH [1,2,3] AS arr RETURN [elem IN arr WHERE elem % 2 = 1 | elem * 2]"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """RETURN [elem IN [1,2,3] WHERE elem % 2 = 1 | elem * 2]"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """RETURN [elem IN range(1,3) WHERE elem % 2 = 1 | elem * 2]"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test02_list_comprehension_no_filter_no_map(self):
        expected_result = [[[1, 2, 3]]]
        query = """WITH [1,2,3] AS arr RETURN [elem IN arr]"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)
        query = """RETURN [elem IN [1,2,3]]"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test03_list_comprehension_map_no_filter(self):
        query = """WITH [1,2,3] AS arr RETURN [elem IN arr | elem * 2]"""
        actual_result = self.graph.query(query)
        expected_result = [[[2, 4, 6]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test04_list_comprehension_filter_no_map(self):
        query = """WITH [1,2,3] AS arr RETURN [elem IN arr WHERE elem % 2 = 1]"""
        actual_result = self.graph.query(query)
        expected_result = [[[1, 3]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test05_list_comprehension_on_allocated_values(self):
        query = """WITH [toUpper('str1'), toUpper('str2'), toUpper('str3')] AS arr RETURN [elem IN arr]"""
        actual_result = self.graph.query(query)
        expected_result = [[['STR1', 'STR2', 'STR3']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """WITH [toUpper('str1'), toUpper('str2'), toUpper('str3')] AS arr RETURN [elem IN arr WHERE toLower(elem) = 'str2']"""
        actual_result = self.graph.query(query)
        expected_result = [[['STR2']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """WITH [toUpper('str1'), toUpper('str2'), toUpper('str3')] AS arr RETURN [elem IN arr WHERE toLower(elem) = 'str2' | elem + 'low']"""
        actual_result = self.graph.query(query)
        expected_result = [[['STR2low']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test06_list_comprehension_on_graph_entities(self):
        query = """MATCH p=()-[*]->() WITH nodes(p) AS nodes RETURN [elem IN nodes]"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), 3)

        query = """MATCH p=()-[*]->() WITH nodes(p) AS nodes WITH [elem IN nodes | elem.val] AS vals RETURN vals ORDER BY vals"""
        actual_result = self.graph.query(query)
        expected_result = [[['v1', 'v2']],
                           [['v1', 'v2', 'v3']],
                           [['v2', 'v3']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[*]->() WITH nodes(p) AS nodes RETURN [elem IN nodes WHERE elem.val = 'v2' | elem.val]"""
        actual_result = self.graph.query(query)
        expected_result = [[['v2']],
                           [['v2']],
                           [['v2']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[*]->() WITH nodes(p) AS nodes RETURN [elem IN nodes WHERE elem.val = 'v2' | elem.val + 'a']"""
        actual_result = self.graph.query(query)
        expected_result = [[['v2a']],
                           [['v2a']],
                           [['v2a']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test07_list_comprehension_in_where_predicate(self):
        # List comprehension with predicate in WHERE predicate on MATCH clause - evaluates to true
        query = """MATCH (n) WHERE n.val IN [x in ['v1', 'v3']] RETURN n.val ORDER BY n.val"""
        actual_result = self.graph.query(query)
        expected_result = [['v1'],
                           ['v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # List comprehension with predicate in WHERE predicate - evaluates to true
        query = """WITH 1 AS a WHERE a IN [x in [1, 2]] RETURN a"""
        actual_result = self.graph.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # List comprehension with predicate in WHERE predicate - evaluates to false
        query = """WITH 1 AS a WHERE a IN [x in [2,3]] RETURN a"""
        actual_result = self.graph.query(query)
        expected_result = []
        self.env.assertEquals(actual_result.result_set, expected_result)

        # List comprehension with predicate and eval in WHERE predicate - evaluates to false
        query = """WITH 1 AS a WHERE size([i in [2,3] WHERE i > 5]) > 0 RETURN a"""
        actual_result = self.graph.query(query)
        expected_result = []
        self.env.assertEquals(actual_result.result_set, expected_result)

        # List comprehension without predicate or eval in WHERE predicate - evaluates to true
        query = """WITH 1 AS a WHERE size([i in [2,3]]) > 0 RETURN a"""
        actual_result = self.graph.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test08_list_comprehension_on_property_array(self):
        query = """MATCH (n)-[e]->() WITH n, e ORDER BY n.val RETURN [elem IN e.edge_val WHERE elem = n.val]"""
        actual_result = self.graph.query(query)
        expected_result = [[['v1']],
                           [['v2']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test09_nested_list_comprehension(self):
        query = """RETURN [elem IN [nested_val IN range(0, 6) WHERE nested_val % 2 = 0] WHERE elem * 2 >= 4 | elem * 2]"""
        actual_result = self.graph.query(query)
        expected_result = [[[4, 8, 12]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test10_any_all_comprehension_acceptance(self):
        # Reject ANY and ALL comprehensions that don't include a WHERE predicate.
        try:
            self.graph.query("RETURN any(x IN [1,2])")
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting a type error.
            self.env.assertIn("requires a WHERE predicate", str(e))

        try:
            self.graph.query("RETURN all(x IN [1,2])")
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting a type error.
            self.env.assertIn("requires a WHERE predicate", str(e))

    def test11_any_all_truth_table(self):
        # Test inputs and predicates where ANY and ALL are both false.
        query = """RETURN any(x IN [0,1] WHERE x = 2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[False]])

        query = """RETURN all(x IN [0,1] WHERE x = 2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[False]])

        # Test inputs and predicates where ANY is true and ALL is false.
        query = """RETURN any(x IN [0,1] WHERE x = 1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[True]])

        query = """RETURN all(x IN [0,1] WHERE x = 1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[False]])

        # Test inputs and predicates where ANY and ALL are both true.
        query = """RETURN any(x IN [0,1] WHERE x = 0 OR x = 1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[True]])

        query = """RETURN all(x IN [0,1] WHERE x = 0 OR x = 1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[True]])

        # Test inputs and predicates where ANY and ALL are both NULL.
        query = """RETURN any(x IN NULL WHERE x = 1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[None]])

        query = """RETURN all(x IN NULL WHERE x = 1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[None]])

    def test12_any_all_on_property_arrays(self):
        # The first array evaluates to ['v1', 'v2'] and the second evaluates to ['v2', 'v3']
        query = """MATCH ()-[e]->() WITH e ORDER BY e.edge_val RETURN ANY(elem IN e.edge_val WHERE elem = 'v2' OR elem = 'v3')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[True], [True]])

        query = """MATCH ()-[e]->() WITH e ORDER BY e.edge_val RETURN ALL(elem IN e.edge_val WHERE elem = 'v2' OR elem = 'v3')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[False], [True]])

    def test13_any_all_path_filtering(self):
        # Use ANY and ALL to introspect on named variable-length paths.
        # All paths should be returned using both ANY and ALL filters.
        expected_result = [['v1'], ['v1'], ['v2']]
        query = """MATCH p=()-[*]->() WHERE any(node IN nodes(p) WHERE node.val STARTS WITH 'v') WITH head(nodes(p)) AS n RETURN n.val ORDER BY n.val"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[*]->() WHERE all(node IN nodes(p) WHERE node.val STARTS WITH 'v') WITH head(nodes(p)) AS n RETURN n.val ORDER BY n.val"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Run a query in which 2 paths pass an ANY filter and 1 path passes an ALL filter.
        query = """MATCH p=()-[*0..1]->() WHERE any(node IN nodes(p) WHERE node.val = 'v1') RETURN length(p) ORDER BY length(p)"""
        actual_result = self.graph.query(query)
        expected_result = [[0], [1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[*0..1]->() WHERE all(node IN nodes(p) WHERE node.val = 'v1') RETURN length(p) ORDER BY length(p)"""
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test14_simple_pattern_comprehension(self):
        # Match all nodes and collect their destination's property in an array
        query = """MATCH (a) RETURN a.val AS v, [(a)-[]->(b) | b.val] ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        actual_result = self.graph.query(query)
        expected_result = [['v1', ['v2']],
                           ['v2', ['v3']],
                           ['v3', []]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test logically equivalent rewrites of the pattern comprehension
        query = """MATCH (a) RETURN a.val AS v, [(a)-[:E]->(b:L) | b.val] ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (a) RETURN a.val AS v, [(a:L)-[:E]->(b:L) | b.val] ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (a) RETURN a.val AS v, [(b)<-[:E]-(a) | b.val] ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (a) RETURN a.val AS v, [(b)<-[:E]-(a) WHERE NOT FALSE | b.val] ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test15_variable_length_pattern_comprehension(self):
        # Match all nodes and collect their destination's property over all hops in an array
        query = """MATCH (a) RETURN a.val AS v, [(a)-[*0..]->(b) | b.val] ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        actual_result = self.graph.query(query)
        expected_result = [['v1', ['v1', 'v2', 'v3']],
                           ['v2', ['v2', 'v3']],
                           ['v3', ['v3']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test logically equivalent rewrites of the pattern comprehension
        query = """MATCH (a) RETURN a.val AS v, [(a)-[:E*0..]->(b) | b.val] ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (a) RETURN a.val AS v, [(a)-[:E*0..]->(b:L) | b.val] ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test16_nested_pattern_comprehension(self):
        # Perform pattern comprehension inside a function call
        query = """MATCH (a) RETURN a.val AS v, size([p=(a)-[*0..]->() | p]) ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        actual_result = self.graph.query(query)
        expected_result = [['v1', 3],
                           ['v2', 2],
                           ['v3', 1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test17_pattern_comprehension_in_aggregation(self):
        # Perform pattern comprehension as an aggregation key
        query = """UNWIND range(1, 3) AS x MATCH (a) RETURN COUNT(a) AS v, [p=(a)-[]->(b) | b.val] AS w ORDER BY v, w"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        actual_result = self.graph.query(query)
        expected_result = [[3, []],
                           [3, ['v2']],
                           [3, ['v3']]]

        self.env.assertEquals(actual_result.result_set, expected_result)

        # Perform pattern comprehension in an aggregation value
        query = """UNWIND range(1, 3) AS x MATCH (a) RETURN a.val AS v, collect([p=(a)-[*0..]->(b) | b.val]) ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        actual_result = self.graph.query(query)
        expected_result = [['v1', [['v1', 'v2', 'v3'], ['v1', 'v2', 'v3'], ['v1', 'v2', 'v3']]],
                           ['v2', [['v2', 'v3'], ['v2', 'v3'], ['v2', 'v3']]],
                           ['v3', [['v3'], ['v3'], ['v3']]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test18_pattern_comprehension_with_filters(self):
        # Match all nodes and collect their destination's property in an array
        query = """MATCH (a) RETURN a.val AS v, [(a)-[]->(b {val: 'v2'}) | b.val] ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        apply = locate_operation(plan.structured_plan, "Apply")
        filter = locate_operation(apply.children[1], "Filter")
        self.env.assertIsNotNone(filter)
        actual_result = self.graph.query(query)
        expected_result = [['v1', ['v2']],
                           ['v2', []],
                           ['v3', []]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (a) RETURN a.val AS v, [(a)-[]->(b) WHERE b.val CONTAINS '3' | b.val] ORDER BY v"""
        plan = self.graph.explain(query)
        self.env.assertTrue(_check_pattern_comprehension_plan(plan))
        apply = locate_operation(plan.structured_plan, "Apply")
        filter = locate_operation(apply.children[1], "Filter")
        self.env.assertIsNotNone(filter)
        actual_result = self.graph.query(query)
        expected_result = [['v1', []],
                           ['v2', ['v3']],
                           ['v3', []]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test19_variable_redefinition(self):
        # Use a list comprehension's variable in two different contexts
        # The shared variable is x
        query = """MATCH x=(a {val: 'v3'})
                   RETURN [x IN nodes(x) | [elem IN range(1, 2) | size(({val: 'v2'})-[]->(x))]]"""
        actual_result = self.graph.query(query)
        expected_result = [[[[1, 1]]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Clear data
        self.graph.delete()

        # Create 10 nodes with incremental v attribute
        self.graph.query("UNWIND range(1, 10) AS x CREATE (:N{v:x})")

        # Validate n used correctly in each scope
        # Iterate each node create an array with v values from 1 to v
        query = """MATCH p=(n) RETURN [n in nodes(p) | [n in range(1, n.v) | n]]"""
        actual_result = self.graph.query(query)
        expected_result = [
            [[[1]]],
            [[[1, 2]]],
            [[[1, 2, 3]]],
            [[[1, 2, 3, 4]]],
            [[[1, 2, 3, 4, 5]]],
            [[[1, 2, 3, 4, 5, 6]]],
            [[[1, 2, 3, 4, 5, 6, 7]]],
            [[[1, 2, 3, 4, 5, 6, 7, 8]]],
            [[[1, 2, 3, 4, 5, 6, 7, 8, 9]]],
            [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]] ]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test20_pattern_comprehension_in_switch_case(self):
        query = "RETURN CASE WHEN [()-[]-() | 1] THEN [()-[]-() | 0] END AS v3"
        actual_result = self.graph.query(query)
        expected_result = [[[]]]
        self.env.assertEquals(actual_result.result_set, expected_result)
        
        # Create a single relationship
        query = "CREATE ()-[:R]->()"
        self.graph.query(query)

        # Lookup for undirected relationship. For each releationship there will be an entry with the value 1.
        # The result of the case will be to lookup again and for each relationship add the value 0 to an array.
        # Since there is one relationship and the pattern matching is undirected we will get two matches.
        query = "RETURN CASE WHEN [()-[]-() | 1] THEN [()-[]-() | 0] END AS v3"
        actual_result = self.graph.query(query)
        expected_result = [[[0, 0]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Similar but with more complex plan
        query = "RETURN CASE WHEN [()-[]-() | 1] THEN [()-[]-() | 0] WHEN [()-[]-()-[]-() | 1] THEN [()-[]-()-[]-() | 0] END AS v3"
        actual_result = self.graph.query(query)
        expected_result = [[[0, 0]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Clear data
        self.graph.delete()

        # Create:
        # Node a with 2 outgoing relationships to b and c
        # Node d without relationships
        query = "CREATE (a {val: 1})-[:R1]->(b {val: 2}), (a)-[:R2]->(c {val: 3}), (d {val: 4})"
        self.graph.query(query)

        # Lookup for undirected relationship. For each releationship there will be an entry with the value 1.
        # The result of the case will be to lookup again and for each relationship add the LHS node's val field value to an array.
        # Since there is two relationships:
        # a-->b will yield 2 matches
        # a-->c will yield 2 matches
        # The result should be an array with [1, 1, 2, 3]

        query = "RETURN CASE WHEN [()-[]-() | 1] THEN [(a)-[]-() | a.val] END AS v3"
        actual_result = self.graph.query(query)
        expected_result = [[[1, 1, 2, 3]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # For each node, lookup for its undirected relationship. For each releationship there will be an entry with the value 1.
        # The result of the case will be to lookup again and for each relationship add the matched node's val field value to an array.
        # Since there is two relationships:
        # a, a-->b, a-->c will yield [1, 1]
        # b, a--b will yield [2]
        # c, a--c will yield [3]
        # d, will yield [] as there are no relationship matches
        # The result should be an array with [1, 1, 2, 3]
        query = "MATCH (n) WITH CASE WHEN [(n)-[]-() | 1] THEN [(n)-[]-() | n.val] END AS v3 RETURN v3"
        actual_result = self.graph.query(query)
        expected_result = [[[1, 1]], [[2]], [[3]], [[]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

