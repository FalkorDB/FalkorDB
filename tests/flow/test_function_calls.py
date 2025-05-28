from cmath import isinf, isnan
from common import *
import json
import math

people = ["Roi", "Alon", "Ailon", "Boaz"]

class testFunctionCallsFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("G")
        self.populate_graph()

    def populate_graph(self):
        nodes = {}
        # Create entities
        for idx, p in enumerate(people):
            if idx % 2 == 0:
                labels = ["person"]
            else:
                labels = ["person", "student"]
            node = Node(alias=f"n_{idx}", labels=labels, properties={"name": p, "val": idx})
            nodes[node.alias] = node
        nodes_str = [str(n) for n in nodes.values()]

        # Fully connected graph
        edges = []
        for src in nodes:
            for dest in nodes:
                if src != dest:
                    edge = Edge(nodes[src], "know", nodes[dest])
                    edges.append(edge)
                    edge = Edge(nodes[src], "works_with", nodes[dest])
                    edges.append(edge)
        edges_str = [str(e) for e in edges]

        # create graph entities
        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")

        # Duplicate know edges
        query = """MATCH (a)-[:know]->(b) CREATE (a)-[:know]->(b)"""
        self.graph.query(query)

    def expect_error(self, query, expected_err_msg):
        try:
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn(expected_err_msg, str(e))

    def expect_type_error(self, query):
        self.expect_error(query, "Type mismatch")
    
    def get_res_and_assertEquals(self, query, expected_result):
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Validate capturing of errors prior to query execution.
    def test01_compile_time_errors(self):
        query = """RETURN toUpper(5)"""
        self.expect_type_error(query)

        query = """RETURN 'a' * 2"""
        self.expect_type_error(query)

        query = """RETURN max(1 + min(2))"""
        self.expect_error(query, "Can't use aggregate functions inside of aggregate functions")

    def test02_boolean_comparisons(self):
        query = """RETURN true = 5"""
        actual_result = self.graph.query(query)
        expected_result = [[False]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """RETURN true <> 'str'"""
        actual_result = self.graph.query(query)
        expected_result = [[True]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """RETURN 'anything' <> NULL"""
        actual_result = self.graph.query(query)
        expected_result = [[None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """RETURN 'anything' = NULL"""
        actual_result = self.graph.query(query)
        expected_result = [[None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """RETURN 10 >= 1.5"""
        actual_result = self.graph.query(query)
        expected_result = [[True]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """RETURN -1 < 1"""
        actual_result = self.graph.query(query)
        expected_result = [[True]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test03_boolean_errors(self):
        query = """RETURN 'str' < 5.5"""
        expected_result = [[None]]
        self.get_res_and_assertEquals(query, expected_result)

        query = """RETURN true > 5"""
        self.get_res_and_assertEquals(query, expected_result)

        query = """MATCH (a) RETURN a < 'anything' LIMIT 1"""
        self.get_res_and_assertEquals(query, expected_result)

    def test04_entity_functions(self):
        query = "RETURN ID(5)"
        self.expect_type_error(query)

        query = "MATCH (a) RETURN ID(a) ORDER BY ID(a) LIMIT 3"
        actual_result = self.graph.query(query)
        expected_result = [[0], [1], [2]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = "MATCH (a)-[e]->() RETURN ID(e) ORDER BY ID(e) LIMIT 3"
        actual_result = self.graph.query(query)
        expected_result = [[0], [1], [2]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = "RETURN EXISTS(null)"
        actual_result = self.graph.query(query)
        expected_result = [[False]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = "RETURN EXISTS('anything')"
        actual_result = self.graph.query(query)
        expected_result = [[True]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test07_nonmap_errors(self):
        query = """MATCH (a) WITH a.name AS scalar RETURN scalar.name"""
        self.expect_type_error(query)

    def test08_apply_all_function(self):
        query = "MATCH () RETURN COUNT(*)"
        actual_result = self.graph.query(query)
        expected_result = [[4]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = "UNWIND [1, 2] AS a RETURN COUNT(*)"
        actual_result = self.graph.query(query)
        expected_result = [[2]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # COLLECT should associate false and 'false' to different groups.
        query = "UNWIND [false,'false',0,'0'] AS a RETURN a, count(a) order by a"
        actual_result = self.graph.query(query)
        expected_result = [['0', 1], ["false", 1], [False, 1], [0, 1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test09_static_aggregation(self):
        query = "RETURN count(*)"
        actual_result = self.graph.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = "RETURN max(2)"
        actual_result = self.graph.query(query)
        expected_result = [[2]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = "RETURN max([2])"
        actual_result = self.graph.query(query)
        expected_result = [[[2]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = "RETURN min(3)"
        actual_result = self.graph.query(query)
        expected_result = [[3]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = "RETURN min([3])"
        actual_result = self.graph.query(query)
        expected_result = [[[3]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test10_modulo_inputs(self):
        # Validate modulo with integer inputs.
        query = "RETURN 5 % 2"
        actual_result = self.graph.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate modulo with a floating-point dividend.
        query = "RETURN 5.5 % 2"
        actual_result = self.graph.query(query)
        expected_result = [[1.5]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate modulo with a floating-point divisor.
        query = "RETURN 5 % 2.5"
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate modulo with both a floating-point dividen and a floating-point divisor.
        query = "RETURN 5.5 % 2.5"
        actual_result = self.graph.query(query)
        expected_result = [[0.5]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate modulo with negative integer inputs.
        query = "RETURN -5 % -2"
        actual_result = self.graph.query(query)
        expected_result = [[-1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate modulo with negative floating-point inputs.
        query = "RETURN -5.5 % -2.5"
        actual_result = self.graph.query(query)
        expected_result = [[-0.5]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate modulo on edge case -LONG_MIN%-1.
        # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=30484
        query = "RETURN toInteger(1.2289948315394e+19) % -1"
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate integer dividend modulo by 0
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 3 % 0"
        # (error) Division by zero
        query = "RETURN 3 % 0"
        try:
            actual_result = self.graph.query(query)
        except redis.ResponseError as e:
            self.env.assertContains("Division by zero", str(e))

        # Validate floating-point dividend modulo by 0
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 1.0 % 0"
        # 1) 1) "1.0 % 0"
        # 2) 1) 1) "-nan"
        query = "RETURN 1.0 % 0"
        actual_result = self.graph.query(query)
        self.env.assertTrue(math.isnan(actual_result.result_set[0][0]))

        # Validate integer dividend modulo by 0.0
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 1 % 0.0"
        # 1) 1) "1 % 0.0"
        # 2) 1) 1) "-nan"
        query = "RETURN 1 % 0.0"
        actual_result = self.graph.query(query)
        self.env.assertTrue(math.isnan(actual_result.result_set[0][0]))

        # Validate floating-point dividend modulo by 0.0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 1.0 % 0.0"
        # 1) 1) "1.0 % 0.0"
        # 2) 1) 1) "-nan"       
        query = "RETURN 1.0 % 0.0"
        actual_result = self.graph.query(query)
        self.env.assertTrue(math.isnan(actual_result.result_set[0][0]))

        # Validate floating-point 0.0 modulo by floating-point different from 0
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 0.0 % 5.1"
        # 1) 1) "0.0 % 5.1"
        # 2) 1) 1) "0"
        query = "RETURN 0.0 % 5.1"
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate floating-point 0.0 modulo by integer different from 0
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 0.0 % 7"
        # 1) 1) "0.0 % 7"
        # 2) 1) 1) "0"
        query = "RETURN 0.0 % 7"
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate integer 0 modulo by floating-point different from 0
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 0 % 7.5"
        # 1) 1) "0 % 7.5"
        # 2) 1) 1) "0"
        query = "RETURN 0 % 7.5"
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate integer 0 modulo by integer different from 0
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 0 % 7"
        # 1) 1) "0 % 7"
        # 2) 1) 1) (integer) 0
        query = "RETURN 0 % 7"
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate floating-point 0.0 modulo by infinite
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 0.0 % ( 1 / 0.0 )"
        # 1) 1) "0.0 % ( 1 / 0.0 )"
        # 2) 1) 1) "0"
        query = "RETURN 0.0 % ( 1 / 0.0 )"
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate integer 0 modulo by infinite
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 0 % ( 1 / 0.0 )"
        # 1) 1) "0 % ( 1 / 0.0 )"
        # 2) 1) 1) "0"
        query = "RETURN 0 % ( 1 / 0.0 )"
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate integer modulo by infinite
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 3 % ( 1 / 0.0 )"
        # 1) 1) "3 % ( 1 / 0.0 )"
        # 2) 1) 1) "3"
        query = "RETURN 3 % ( 1 / 0.0 )"
        actual_result = self.graph.query(query)
        expected_result = [[3]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate floating-point modulo by infinite
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 2.5 % ( 1 / 0.0 )"
        # 1) 1) "2.5 % ( 1 / 0.0 )"
        # 2) 1) 1) "2.5"
        query = "RETURN 2.5 % ( 1 / 0.0 )"
        actual_result = self.graph.query(query)
        expected_result = [[2.5]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate infinite modulo by integer 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN ( 1 / 0.0 ) % 7"
        # 1) 1) "( 1 / 0.0 ) % 7"
        # 2) 1) 1) "-nan"
        query = "RETURN ( 1 / 0.0 ) % 7"
        actual_result = self.graph.query(query)
        self.env.assertTrue(math.isnan(actual_result.result_set[0][0]))

        # Validate floating-point 0.0 modulo by floating-point 0.0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN  0.0 % 0.0"
        # 1) 1) "0.0 % 0.0"
        # 2) 1) 1) "-nan"
        query = "RETURN 0.0 % 0.0"
        actual_result = self.graph.query(query)
        self.env.assertTrue(math.isnan(actual_result.result_set[0][0]))

        # Validate integer 0 modulo by floating 0.0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN  0 % 0.0"
        # 1) 1) "0 % 0.0"
        # 2) 1) 1) "-nan"
        query = "RETURN 0 % 0.0"
        actual_result = self.graph.query(query)
        self.env.assertTrue(math.isnan(actual_result.result_set[0][0]))

        # Validate floating-point 0.0 modulo by integer 0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN  0.0 % 0"
        # 1) 1) "0 % 0.0"
        # 2) 1) 1) "-nan"
        query = "RETURN  0.0 % 0"
        actual_result = self.graph.query(query)
        self.env.assertTrue(math.isnan(actual_result.result_set[0][0]))

        # Validate integer 0 modulo by 0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN  0 % 0"
        # (error) Division by zero
        query = "RETURN 0 % 0"
        try:
            actual_result = self.graph.query(query)
        except redis.ResponseError as e:
            self.env.assertContains("Division by zero", str(e))

    # Aggregate functions should handle null inputs appropriately.
    def test11_null_aggregate_function_inputs(self):
        # SUM should sum all non-null inputs.
        query = """UNWIND [1, NULL, 3] AS a RETURN sum(a)"""
        actual_result = self.graph.query(query)
        expected_result = [[4]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # SUM should return 0 given a fully NULL input.
        query = """WITH NULL AS a RETURN sum(a)"""
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # COUNT should count all non-null inputs.
        query = """UNWIND [1, NULL, 3] AS a RETURN count(a)"""
        actual_result = self.graph.query(query)
        expected_result = [[2]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # COUNT should return 0 given a fully NULL input.
        query = """WITH NULL AS a RETURN count(a)"""
        actual_result = self.graph.query(query)
        expected_result = [[0]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # COLLECT should ignore null inputs.
        query = """UNWIND [1, NULL, 3] AS a RETURN collect(a)"""
        actual_result = self.graph.query(query)
        expected_result = [[[1, 3]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # COLLECT should return an empty array on all null inputs.
        query = """WITH NULL AS a RETURN collect(a)"""
        actual_result = self.graph.query(query)
        expected_result = [[[]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Verify that nested functions that perform heap allocations return properly.
    def test12_nested_heap_functions(self):
        query = """MATCH p = (n) WITH head(nodes(p)) AS node RETURN node.name ORDER BY node.name"""
        actual_result = self.graph.query(query)
        expected_result = [['Ailon'],
                           ['Alon'],
                           ['Boaz'],
                           ['Roi']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # CASE...WHEN statements should properly handle NULL, false, and true evaluations.
    def test13_case_when_inputs(self):
        # Simple case form: single value evaluation.
        query = """UNWIND [NULL, true, false] AS v RETURN v, CASE v WHEN true THEN v END"""
        actual_result = self.graph.query(query)
        expected_result = [[None, None],
                           [True, True],
                           [False, None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """UNWIND [NULL, true, false] AS v RETURN v, CASE v WHEN true THEN v WHEN false THEN v END"""
        actual_result = self.graph.query(query)
        expected_result = [[None, None],
                           [True, True],
                           [False, False]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Generic case form: evaluation for each case.
        query = """UNWIND [NULL, true, false] AS v RETURN v, CASE WHEN v THEN v END"""
        actual_result = self.graph.query(query)
        # Only the true value should return non-NULL.
        expected_result = [[None, None],
                           [True, True],
                           [False, None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """UNWIND [NULL, true, false] AS v RETURN v, CASE WHEN v IS NOT NULL THEN v END"""
        actual_result = self.graph.query(query)
        # The true and false values should both return non-NULL.
        expected_result = [[None, None],
                           [True, True],
                           [False, False]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # CASE...WHEN statements should manage allocated values properly.
    def test14_case_when_memory_management(self):
        # Simple case form: single value evaluation.
        query = """WITH 'A' AS a WITH CASE a WHEN 'A' THEN toString(a) END AS key RETURN toLower(key)"""
        actual_result = self.graph.query(query)
        expected_result = [['a']]
        self.env.assertEquals(actual_result.result_set, expected_result)
        # Generic case form: evaluation for each case.
        query = """WITH 'A' AS a WITH CASE WHEN true THEN toString(a) END AS key RETURN toLower(key)"""
        actual_result = self.graph.query(query)
        expected_result = [['a']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test15_aggregate_error_handling(self):
        functions = ["avg",
                     "collect",
                     "count",
                     "max",
                     "min",
                     "sum",
                     "percentileDisc",
                     "percentileCont",
                     "stDev"]
        # Test all functions for invalid argument counts.
        for function in functions:
            query = """UNWIND range(0, 10) AS val RETURN %s(val, val, val)""" % (function)
            self.expect_error(query, "Received 3 arguments")

        # Test numeric functions for invalid input types.
        numeric_functions = ["avg",
                             "sum",
                             "stDev"]
        for function in numeric_functions:
            query = """UNWIND ['a', 'b', 'c'] AS val RETURN %s(val)""" % (function)
            self.expect_type_error(query)

        # Test invalid numeric input for percentile function.
        query = """UNWIND range(0, 10) AS val RETURN percentileDisc(val, -1)"""
        self.expect_error(query, "must be a number in the range 0.0 to 1.0")

    # startNode and endNode calls should return the appropriate nodes.
    def test16_edge_endpoints(self):
        query = """MATCH (a)-[e]->(b) RETURN a.name, startNode(e).name, b.name, endNode(e).name"""
        actual_result = self.graph.query(query)
        for row in actual_result.result_set:
            self.env.assertEquals(row[0], row[1])
            self.env.assertEquals(row[2], row[3])

    def test17_to_json(self):
        # Test JSON literal values in an array.
        query = """RETURN toJSON([1, 0.000000000000001, 'str', true, NULL])"""
        actual_result = self.graph.query(query)
        parsed = json.loads(actual_result.result_set[0][0])
        self.env.assertEquals(parsed, [1, 0.000000000000001, "str", True, None])

        # Test JSON an empty array value.
        query = """WITH [] AS arr RETURN toJSON(arr)"""
        actual_result = self.graph.query(query)
        parsed = json.loads(actual_result.result_set[0][0])
        self.env.assertEquals(parsed, [])

        # Test JSON an empty map value.
        query = """WITH {} AS map RETURN toJSON(map)"""
        actual_result = self.graph.query(query)
        parsed = json.loads(actual_result.result_set[0][0])
        self.env.assertEquals(parsed, {})

        # Test converting a map projection.
        query = """MATCH (n {val: 1}) RETURN toJSON(n {.val, .name})"""
        actual_result = self.graph.query(query)
        parsed = json.loads(actual_result.result_set[0][0])
        self.env.assertEquals(parsed, {"name": "Alon", "val": 1})

        # Test converting a full node.
        query = """MATCH (n {val: 1}) RETURN toJSON(n)"""
        actual_result = self.graph.query(query)
        parsed = json.loads(actual_result.result_set[0][0])
        self.env.assertEquals(parsed, {"type": "node", "id": 1, "labels": ["person", "student"], "properties": {"name": "Alon", "val": 1}})

        # Test converting a full edge.
        query = """MATCH ({val: 0})-[e:works_with]->({val: 1}) RETURN toJSON(e)"""
        actual_result = self.graph.query(query)
        start = {"id": 0, "labels": ["person"], "properties": {"name": "Roi", "val": 0}}
        end = {"id": 1, "labels": ["person", "student"], "properties": {"name": "Alon", "val": 1}}
        parsed = json.loads(actual_result.result_set[0][0])
        self.env.assertEquals(parsed, {"type": "relationship", "id": 1, "relationship": "works_with", "properties": {}, "start": start, "end": end})

        # Test converting a path.
        query = """MATCH path=({val: 0})-[e:works_with]->({val: 1}) RETURN toJSON(path)"""
        actual_result = self.graph.query(query)
        expected = [{'type': 'node', 'id': 0, 'labels': ['person'], 'properties': {'name': 'Roi', 'val': 0}}, {'type': 'relationship', 'id': 1, 'relationship': 'works_with', 'properties': {}, 'start': {'id': 0, 'labels': ['person'], 'properties': {'name': 'Roi', 'val': 0}}, 'end': {'id': 1, 'labels': ['person', 'student'], 'properties': {'name': 'Alon', 'val': 1}}}, {'type': 'node', 'id': 1, 'labels': ['person', 'student'], 'properties': {'name': 'Alon', 'val': 1}}]
        parsed = json.loads(actual_result.result_set[0][0])
        self.env.assertEquals(parsed, expected)

        # Test JSON literal values in an point.
        query = """RETURN toJSON(point({ longitude: 167.697555, latitude: 0.402313 }))"""
        actual_result = self.graph.query(query)
        parsed = json.loads(actual_result.result_set[0][0])
        self.env.assertEquals(parsed, {"crs": "wgs-84", "latitude": 0.402313, "longitude": 167.697556, "height": None})

    # Memory should be freed properly when the key values are heap-allocated.
    def test18_allocated_keys(self):
        query = """UNWIND ['str1', 'str1', 'str2', 'str1'] AS key UNWIND [1, 2, 3] as agg RETURN toUpper(key) AS key, collect(DISTINCT agg) ORDER BY key"""
        actual_result = self.graph.query(query)
        expected_result = [['STR1', [1, 2, 3]],
                           ['STR2', [1, 2, 3]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test19_has_labels(self):
        # Test existing label
        query = """MATCH (n) WHERE n:person RETURN n.name"""
        actual_result = self.graph.query(query)
        expected_result = [['Roi'], ['Alon'], ['Ailon'], ['Boaz']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test not existing label
        query = """MATCH (n) WHERE n:L RETURN n.name"""
        actual_result = self.graph.query(query)
        expected_result = []
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test multi label
        query = """MATCH (n) WHERE n:person:L RETURN n.name"""
        actual_result = self.graph.query(query)
        expected_result = []
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test multi label
        query = """MATCH (n) WHERE n:person:student RETURN n.name"""
        actual_result = self.graph.query(query)
        expected_result = [['Alon'], ['Boaz']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test or between different labels label
        query = """MATCH (n) WHERE n:person OR n:L RETURN n.name"""
        actual_result = self.graph.query(query)
        expected_result = [['Roi'], ['Alon'], ['Ailon'], ['Boaz']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test multi label using functions
        query = """MATCH (n) WHERE hasLabels(n, ['person', 'L']) RETURN n.name"""
        actual_result = self.graph.query(query)
        expected_result = []
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test multi label using functions
        query = """MATCH (n) WHERE hasLabels(n, ['person', 'student']) RETURN n.name"""
        actual_result = self.graph.query(query)
        expected_result =  [['Alon'], ['Boaz']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test has labels using functions mismatch type
        query = """MATCH (n) WHERE hasLabels(n, ['person', 1]) RETURN n.name"""
        try:
            self.graph.query(query)
        except redis.ResponseError as e:
            self.env.assertContains("Type mismatch: expected String but was Integer", str(e))

    def test20_keys(self):
        # Test retrieving keys of a nested map
        query = """RETURN keys({a: 5, b: 10})"""
        actual_result = self.graph.query(query)
        expected_result = [[['a', 'b']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test retrieving keys of a map reference
        query = """WITH {a: 5, b: 10} AS map RETURN keys(map)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test retrieving keys of a node
        query = """MATCH (n:person {name: 'Roi'}) RETURN keys(n)"""
        actual_result = self.graph.query(query)
        expected_result = [[['name', 'val']]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test retrieving keys of an (empty) edge
        query = """MATCH (:person {name: 'Roi'})-[e:works_with]->(:person {name: 'Alon'}) RETURN keys(e)"""
        actual_result = self.graph.query(query)
        expected_result = [[[]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test a null input
        query = """WITH NULL AS map RETURN keys(map)"""
        actual_result = self.graph.query(query)
        expected_result = [[None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test trying to retrieve keys of an invalid type
        query = """WITH 10 AS map RETURN keys(map)"""
        self.expect_type_error(query)

    def test21_distinct_memory_management(self):
        # validate behavior of the DISTINCT function with allocated values
        query = """MATCH (a {val: 0}) RETURN collect(DISTINCT a { .name })"""
        actual_result = self.graph.query(query)
        expected_result = [[[{'name': 'Roi'}]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test22_large_list_argument(self):
        # validate that large lists arguments are not allocated on stack
        large_list = str([1] * 1000000)
        query = f"""RETURN {large_list}"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set[0][0]), 1000000)
    
    def test23_toInteger(self):
        # expect calling toInteger to succeed
        query_to_expected_result = {
            """RETURN toInteger(1)""": [[1]],
            """RETURN toInteger(1.1)""": [[1]],
            """RETURN toInteger(1.9)""": [[1]],
            """RETURN toInteger('1')""": [[1]],
            """RETURN toInteger('1.1')""": [[1]],
            """RETURN toInteger('1.9')""": [[1]],
            """RETURN toInteger(true)""": [[1]],
            """RETURN toInteger(false)""": [[0]],
            """RETURN toInteger('1790460441484152222')""": [[1790460441484152222]],
            """RETURN toInteger('-1790460441484152222')""": [[-1790460441484152222]],
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

        # expect calling toInteger to return NULL
        queries = [
            """RETURN toInteger('z')""",
            """RETURN toInteger(NULL)""",
            """RETURN toInteger('')""",
            """RETURN toInteger('18446744073709551616')""",
            """RETURN toInteger('-18446744073709551616')""",
        ]
        for query in queries:
            actual_result = self.graph.query(query)
            self.env.assertEquals(actual_result.result_set[0][0], None)

    def test24_substring(self):
        query_to_expected_result = {
            """RETURN SUBSTRING('muchacho', 0, 4)""": [["much"]],
            """RETURN SUBSTRING('muchacho', 3, 20)""": [["hacho"]],
            """RETURN SUBSTRING(NULL, 3, 20)""": [[None]],
            """RETURN SUBSTRING('ab', 1, 999999999999999)""": [["b"]],
            # test unicode charecters
            """RETURN SUBSTRING('丁丂七丄丅丆万丈三上', 3, 4)""" : [['丄丅丆万']],
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
        
        self.expect_error("""RETURN SUBSTRING("muchacho", 3, -20)""",
            "length must be a non-negative integer")

        self.expect_error("""RETURN SUBSTRING("muchacho", -3, 3)""",
            "start must be a non-negative integer")

    def test25_left(self):
        query_to_expected_result = {
            "RETURN LEFT('muchacho', 4)" : [['much']],
            "RETURN LEFT('muchacho', 100)" : [['muchacho']],
            "RETURN LEFT(NULL, -1)" : [[None]],
            "RETURN LEFT(NULL, 100)" : [[None]],
            "RETURN LEFT(NULL, NULL)" : [[None]],
            # test unicode charecters
            "RETURN LEFT('丁丂七丄丅丆万丈三上', 4)" : [['丁丂七丄']],
            "RETURN LEFT('丁丂七丄丅丆万丈三上', 100)" : [['丁丂七丄丅丆万丈三上']],
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

        # invalid length argument
        queries = [
            """RETURN LEFT('', -100)""",
            """RETURN LEFT('a', NULL)""",
            ]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertEqual(str(e), "length must be a non-negative integer")

        # invalid input types
        queries = [
            """RETURN LEFT(NULL, 'a')""",
            """RETURN LEFT(NULL, 1.3)""",
        ]
        for query in queries:
            self.expect_type_error(query)

    def test26_right(self):
        query_to_expected_result = {
            "RETURN RIGHT('muchacho', 4)" : [['acho']],
            "RETURN RIGHT('muchacho', 100)" : [['muchacho']],
            "RETURN RIGHT(NULL, -1)" : [[None]],
            "RETURN RIGHT(NULL, 100)" : [[None]],
            "RETURN RIGHT(NULL, NULL)" : [[None]],
            # test unicode charecters
            "RETURN RIGHT('丁丂七丄丅丆万丈三上', 4)" : [['万丈三上']],
            "RETURN RIGHT('丁丂七丄丅丆万丈三上', 100)" : [['丁丂七丄丅丆万丈三上']],
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

        # invalid length argument
        queries = [
            """RETURN RIGHT('', -100)""",
            """RETURN RIGHT('a', NULL)""",
            ]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertEqual(str(e), "length must be a non-negative integer")

        # invalid input types
        queries = [
            """RETURN RIGHT(NULL, 'a')""",
            """RETURN RIGHT(NULL, 1.3)""",
        ]
        for query in queries:
            self.expect_type_error(query)

    def test27_string_concat(self):
        larg_double = 1.123456e300
        query = f"""RETURN '' + {larg_double} + {larg_double}"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "%f%f" % (larg_double, larg_double))

    def test28_sqrt(self):
        query = """RETURN sqrt(0)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 0)

        query = """RETURN sqrt(9801)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 99)

        query = """RETURN sqrt(-1)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isnan(actual_result.result_set[0][0]))

        query = """RETURN sqrt(-9)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isnan(actual_result.result_set[0][0]))

        query = """RETURN sqrt(-0.0000000001)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isnan(actual_result.result_set[0][0]))

        query = """RETURN sqrt(null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        query = """RETURN sqrt(2540.95581553)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 50.4078943770715)
    
    def test29_toBoolean(self):
        # all other toBoolean cases (boolean, strings, null, errors) are covered in TCK
        # integers
        query = """RETURN toBoolean(0)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], False)
        query = """RETURN toBoolean(1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], True)
        query = """RETURN toBoolean(-1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], True)

    def test29_toBooleanOrNull(self):
        # boolean
        query = """RETURN toBooleanOrNull(true)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], True)
        query = """RETURN toBooleanOrNull(false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], False)

        # strings
        query = """RETURN toBooleanOrNull('TruE')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], True)
        query = """RETURN toBooleanOrNull('FaLsE')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], False)
        query = """RETURN toBooleanOrNull('not a boolean')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # integers
        query = """RETURN toBooleanOrNull(0)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], False)
        query = """RETURN toBooleanOrNull(1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], True)
        query = """RETURN toBooleanOrNull(-1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], True)

        # null
        query = """RETURN toBooleanOrNull(null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # float
        query = """RETURN toBooleanOrNull(0.1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # list
        query = """RETURN toBooleanOrNull([true])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # node
        query = """CREATE (n) RETURN toBooleanOrNull(n)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # edge
        query = """CREATE ()-[r:R]->() RETURN toBooleanOrNull(r)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

    def test30_toFloatOrNull(self):
        # floats
        query = """RETURN toFloatOrNull(1.2)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1.2, 0.0001)

        # strings
        query = """RETURN toFloatOrNull('1.23')"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1.23, 0.0001)
        query = """RETURN toFloatOrNull('1.2.3')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # integers
        query = """RETURN toFloatOrNull(0.1)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.1, 0.0001)
        query = """RETURN toFloatOrNull(1)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1.0, 0.0001)
        query = """RETURN toFloatOrNull(-1)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], -1.0, 0.0001)

        # null
        query = """RETURN toFloatOrNull(null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # boolean
        query = """RETURN toFloatOrNull(true)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)
        query = """RETURN toFloatOrNull(false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # list
        query = """RETURN toFloatOrNull([1.0])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # node
        query = """CREATE (n) RETURN toFloatOrNull(n)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # edge
        query = """CREATE ()-[r:R]->() RETURN toFloatOrNull(r)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

    def test31_toIntegerOrNull(self):
        # integers
        query = """RETURN toIntegerOrNull(0)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 0)
        query = """RETURN toIntegerOrNull(1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 1)
        query = """RETURN toIntegerOrNull(-1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], -1)

        # floats
        query = """RETURN toIntegerOrNull(0.1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 0)
        query = """RETURN toIntegerOrNull(0.9)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 0)

        # strings
        query = """RETURN toIntegerOrNull('1')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 1)
        query = """RETURN toIntegerOrNull('1.2')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 1)

        # null
        query = """RETURN toIntegerOrNull(null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # boolean
        query = """RETURN toIntegerOrNull(true)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 1)
        query = """RETURN toIntegerOrNull(false)"""
        actual_result =self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 0)

        # list
        query = """RETURN toIntegerOrNull([1])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # node
        query = """CREATE (n) RETURN toIntegerOrNull(n)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # edge
        query = """CREATE ()-[r:R]->() RETURN toIntegerOrNull(r)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

    def test32_toStringOrNull(self):
        # strings
        query = """RETURN toStringOrNull('1')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "1")
        query = """RETURN toStringOrNull('1.2')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "1.2")
        query = """RETURN toStringOrNull('hello')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "hello")

        # integers
        query = """RETURN toStringOrNull(0)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "0")
        query = """RETURN toStringOrNull(1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "1")
        query = """RETURN toStringOrNull(-1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "-1")

        # floats
        query = """RETURN toStringOrNull(0.1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "0.100000")
        query = """RETURN toStringOrNull(0.9)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "0.900000")

        # boolean
        query = """RETURN toStringOrNull(true)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "true")
        query = """RETURN toStringOrNull(false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "false")
        
        # null
        query = """RETURN toStringOrNull(null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)
  
         # list
        query = """RETURN toStringOrNull([1])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # node
        query = """CREATE (n) RETURN toStringOrNull(n)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        # edge
        query = """CREATE ()-[r:R]->() RETURN toStringOrNull(r)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)
    
    def test33_toString(self):
        # strings
        query = """RETURN toString('1')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "1")
        query = """RETURN toString('1.2')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "1.2")
        query = """RETURN toString('hello')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "hello")

        # integers
        query = """RETURN toString(0)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "0")
        query = """RETURN toString(1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "1")
        query = """RETURN toString(-1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "-1")

        # floats
        query = """RETURN toString(0.1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "0.100000")
        query = """RETURN toString(0.9)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "0.900000")

        # boolean
        query = """RETURN toString(true)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "true")
        query = """RETURN toString(false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], "false")
        
        # null
        query = """RETURN toString(null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], None)

        queries = [
            """RETURN toString([1])""",                   # list
            """CREATE (n) RETURN toString(n)""",          # node
            """CREATE ()-[r:R]->() RETURN toString(r)"""  # edge
            ]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                # Expecting a type error.
                self.env.assertIn("Type mismatch", str(e))

    def test34_split(self):
        query_to_expected_result = {
            "RETURN split(null, ',')": [[None]],
            "RETURN split('hello world', null)": [[None]],
            "RETURN split('hello world', ',')": [[["hello world"]]],
            "RETURN split('hello world', '')": [[['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']]],
            "RETURN split('aa', 'a')": [[["", "", ""]]],
            "RETURN split('', ',')": [[[""]]],
            "RETURN split('', '')": [[[""]]],
            "RETURN split('aAbBcC', '')": [[['a', 'A', 'b', 'B', 'c', 'C']]],
            "RETURN split('aAbBcC', 'B')": [[['aAb', 'cC']]],
            # test unicode charecters
            "RETURN split('丁丂七丄丅丆万丈三上', '丄')": [[["丁丂七", "丅丆万丈三上"]]],
            "RETURN split('丁丂七丅', '')": [[["丁", "丂", "七", "丅"]]],
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

    def test35_min_max(self):
        query = "UNWIND [[1], [2], [2], [1]] AS x RETURN max(x), min(x)"
        actual_result = self.graph.query(query)
        expected_result = [[[2], [1]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = "UNWIND [1, 2, '1' ,'2' ,[1] ,[2] ,1 ,2, '1', '2', NULL, True] AS x RETURN max(x), min(x)"
        actual_result = self.graph.query(query)
        expected_result = [[2, [1]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test36_log(self):

        # log(0)
        query = """RETURN log(0)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isinf(-actual_result.result_set[0][0]))

        # log(-1)
        query = """RETURN log(-1)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isnan(actual_result.result_set[0][0]))

        # log(1)
        query = """RETURN log(1)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0, 0.0001)

        # log(10)
        query = """RETURN log(10)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 2.30258509299405, 0.0001)

        # log10(0)
        query = """RETURN log10(0)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isinf(-actual_result.result_set[0][0]))

        # log10(-11.3)
        query = """RETURN log10(-11.3)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isnan(actual_result.result_set[0][0]))

        # log10(100)
        query = """RETURN log10(100)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 2, 0.0001)

        # log10(110)
        query = """RETURN log10(110)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 2.04139268515822, 0.0001)

        # log(True)
        query = """RETURN log(True)"""
        self.expect_type_error(query)

        # log10(True)
        query = """RETURN log10(True)"""
        self.expect_type_error(query)

    def test37_exp(self):
        # exp(0)
        query = """RETURN exp(0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1.0, 0.0001)

        # exp(1)
        query = """RETURN exp(1)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 2.71828182845905, 0.0001)

        # exp(1.4)
        query = """RETURN exp(1.4)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 4.05519996684467, 0.0001)

        # exp(-1.2)
        query = """RETURN exp(0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1, 0.0001)

        # e()
        query = """RETURN e()"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 2.71828182845905, 0.0001)

        # exp(True)
        query = """RETURN exp(True)"""
        self.expect_type_error(query)

    def test38_properties(self):
        # null input
        query = """RETURN properties(null)"""
        query_result = self.graph.query(query)
        expected_result = [[None]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # map input
        query = """WITH {val: 5, nested: {nested_val: 'nested_str'}} AS map RETURN properties(map)"""
        query_result = self.graph.query(query)
        expected_result = [[{'val': 5, 'nested': {'nested_val': 'nested_str'}}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # node input
        query = """CREATE (p:Person {name: 'Alexa', city: 'Buga', age: 44}) RETURN properties(p)"""
        query_result = self.graph.query(query)
        expected_result = [[{'name': 'Alexa', 'city': 'Buga', 'age': 44}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # edge input
        query = """CREATE (a:X)-[r:R {name:'R1', len:5}]->(b:Y) RETURN properties(r)"""
        query_result = self.graph.query(query)
        expected_result = [[{'name': 'R1', 'len': 5}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # properies of entity properties subset
        query = """MATCH (p:Person {name: 'Alexa'}) RETURN properties(p{.name, .age})"""
        query_result = self.graph.query(query)
        expected_result = [[{'name': 'Alexa', 'age': 44}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """MATCH ()-[r:R {name:'R1', len:5}]->() RETURN properties(r{.name})"""
        query_result = self.graph.query(query)
        expected_result = [[{'name': 'R1'}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # string input
        query = """RETURN properties('a')"""
        self.expect_type_error(query)

        # integer input
        query = """RETURN properties(1)"""
        self.expect_type_error(query)

        # list input
        query = """RETURN properties([1, 2, 3])"""
        self.expect_type_error(query)

        # call without input arguments
        query = """RETURN properties()"""
        self.expect_error(query, "Received 0 arguments")

    def test39_sin(self):
        # sin(0)
        query = """RETURN sin(0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.0, 0.0001)

        # sin(1.5)
        query = """RETURN sin(1.5)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.997494986604054, 0.0001)

        # sin(1.5)
        query = """RETURN sin(-2.45)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], -0.637764702134504, 0.0001)

        # sin(null)
        query = """RETURN sin(null)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # sin(True)
        query = """RETURN sin(True)"""
        self.expect_type_error(query)

        # sin(2,3)
        query = """RETURN sin(2,3)"""
        self.expect_error(query, "Received 2 arguments to function 'sin', expected at most 1")

    def test40_cos(self):
        # cos(0)
        query = """RETURN cos(0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1.0, 0.0001)

        # cos(1.6)
        query = """RETURN cos(1.6)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], -0.0291995223012888, 0.0001)

        # cos(-3.27)
        query = """RETURN cos(-3.27)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], -0.991767098339465, 0.0001)

        # cos(null)
        query = """RETURN cos(null)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # cos(True)
        query = """RETURN cos(True)"""
        self.expect_type_error(query)

        # cos(2,3)
        query = """RETURN cos(2,3)"""
        self.expect_error(query, "Received 2 arguments to function 'cos', expected at most 1")

    def test41_tan(self):
        # tan(0)
        query = """RETURN tan(0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.0, 0.0001)

        # tan(3.8)
        query = """RETURN tan(3.8)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.773556090503126, 0.0001)

        # tan(-1.97)
        query = """RETURN tan(-1.97)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 2.37048352994376, 0.0001)

        # tan(null)
        query = """RETURN tan(null)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # tan(True)
        query = """RETURN tan(True)"""
        self.expect_type_error(query)

        # tan(2,3)
        query = """RETURN tan(2,3)"""
        self.expect_error(query, "Received 2 arguments to function 'tan', expected at most 1")

    def test42_cot(self):
        # cot(0)
        query = """RETURN cot(0)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isinf(actual_result.result_set[0][0]))

        # cot(5.77)
        query = """RETURN cot(5.77)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], -1.77447133917238, 0.0001)

        # cot(-4.655)
        query = """RETURN cot(-4.655)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], -0.0574520669374087, 0.0001)

        # cot(null)
        query = """RETURN cot(null)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # cot(True)
        query = """RETURN cot(True)"""
        self.expect_type_error(query)

        # cot(6,7.5)
        query = """RETURN cot(2,3)"""
        self.expect_error(query, "Received 2 arguments to function 'cot', expected at most 1")

    def test43_asin(self):
        # asin(0)
        query = """RETURN asin(0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.0, 0.0001)

        # asin(0.89)
        query = """RETURN asin(0.89)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1.09734516952283, 0.0001)

        # asin(-0.38)
        query = """RETURN asin(-0.38)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], -0.389796296474261, 0.0001)

        # asin(1.3)
        query = """RETURN asin(1.3)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isnan(actual_result.result_set[0][0]))

        # asin(-1.1)
        query = """RETURN asin(-1.1)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isnan(actual_result.result_set[0][0]))

        # asin(null)
        query = """RETURN asin(null)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # asin(True)
        query = """RETURN asin(True)"""
        self.expect_type_error(query)

        # asin(2,3)
        query = """RETURN asin(2,3)"""
        self.expect_error(query, "Received 2 arguments to function 'asin', expected at most 1")

    def test44_acos(self):
        # acos(0)
        query = """RETURN acos(0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1.5707963267949, 0.0001)

        # acos(0.45)
        query = """RETURN acos(0.45)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1.1040309877476, 0.0001)

        # acos(-0.39)
        query = """RETURN acos(-0.39)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1.97142791949627, 0.0001)

        # acos(1.12)
        query = """RETURN acos(1.12)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isnan(actual_result.result_set[0][0]))

        # acos(-1.21)
        query = """RETURN acos(-1.21)"""
        actual_result = self.graph.query(query)
        self.env.assertTrue(isnan(actual_result.result_set[0][0]))

        # acos(null)
        query = """RETURN acos(null)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # acos(True)
        query = """RETURN acos(True)"""
        self.expect_type_error(query)

        # acos(2,3)
        query = """RETURN acos(2,3)"""
        self.expect_error(query, "Received 2 arguments to function 'acos', expected at most 1")
    
    def test45_atan(self):
        # atan(0)
        query = """RETURN atan(0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.0, 0.0001)

        # atan(6.95)
        query = """RETURN atan(6.95)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1.42789222318575, 0.0001)

        # atan(-7.43)
        query = """RETURN atan(-7.43)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], -1.43701077132559, 0.0001)

        # atan(null)
        query = """RETURN atan(null)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # atan(True)
        query = """RETURN atan(True)"""
        self.expect_type_error(query)

        # atan(2,3)
        query = """RETURN atan(2,3)"""
        self.expect_error(query, "Received 2 arguments to function 'atan', expected at most 1")

    def test46_atan2(self):
        # atan2(0,0)
        query = """RETURN atan2(0,0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.0, 0.0001)

        # atan2(1.7, -2.3)
        query = """RETURN atan2(1.7, -2.3)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 2.50508443780184, 0.0001)

        # atan2(-3.2, 7.3)
        query = """RETURN atan2(-3.2, 7.3)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], -0.413128832278401, 0.0001)

        # atan2(1,null)
        query = """RETURN atan2(1,null)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # atan2(null,2)
        query = """RETURN atan2(null,2)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # atan2(3,True)
        query = """RETURN atan2(3,True)"""
        self.expect_type_error(query)

        # atan2(2)
        query = """RETURN atan2(2)"""
        self.expect_error(query, "Received 1 arguments to function 'atan2', expected at least 2")
        
        # atan2(2,3,4)
        query = """RETURN atan2(2,3,4)"""
        self.expect_error(query, "Received 3 arguments to function 'atan2', expected at most 2")


    def test47_degrees(self):
        # degrees(0)
        query = """RETURN degrees(0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.0, 0.0001)

        # degrees(45.67)
        query = """RETURN degrees(45.67)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 2616.69825036247, 0.0001)

        # degrees(-44.56)
        query = """RETURN degrees(-44.56)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], -2553.09993510295, 0.0001)

        # degrees(null)
        query = """RETURN degrees(null)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # degrees(True)
        query = """RETURN degrees(True)"""
        self.expect_type_error(query)

        # degrees(2,3)
        query = """RETURN degrees(2,3)"""
        self.expect_error(query, "Received 2 arguments to function 'degrees', expected at most 1")   

    def test48_radians(self):
        # radians(0)
        query = """RETURN radians(0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.0, 0.0001)

        # radians(65.78)
        query = """RETURN radians(65.78)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 1.14807758196187, 0.0001)

        # radians(-99.33)
        query = """RETURN radians(-99.33)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], -1.73363554600597, 0.0001)

        # radians(null)
        query = """RETURN radians(null)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # radians(True)
        query = """RETURN radians(True)"""
        self.expect_type_error(query)

        # radians(2,3)
        query = """RETURN radians(2,3)"""
        self.expect_error(query, "Received 2 arguments to function 'radians', expected at most 1")


    def test49_pi(self):
        # pi()
        query = """RETURN pi()"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 3.14159265358979, 0.0001)

        # pi(2)
        query = """RETURN pi(null)"""
        self.expect_error(query, "Received 1 arguments to function 'pi', expected at most 0")

    def test50_haversin(self):
        # haversin(0)
        query = """RETURN haversin(0)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.0, 0.0001)

        # haversin(6.59)
        query = """RETURN haversin(6.59)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.0233497787816533, 0.0001)

        # haversin(-78.53)
        query = """RETURN haversin(-78.53)"""
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], 0.999975910061948, 0.0001)

        # haversin(null)
        query = """RETURN haversin(null)"""
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # haversin(True)
        query = """RETURN haversin(True)"""
        self.expect_type_error(query)

        # haversin(2,3)
        query = """RETURN haversin(2,3)"""
        self.expect_error(query, "Received 2 arguments to function 'haversin', expected at most 1")           

    def test51_isempty(self):
        # null input, the expected result is null
        query = "RETURN isEmpty(null)"
        actual_result = self.graph.query(query)
        self.env.assertIsNone(actual_result.result_set[0][0])

        # inputs with expected result = True
        queries = [
            """RETURN isEmpty('')""",
            """RETURN isEmpty([])""",
            """WITH {} AS map RETURN isEmpty(map)"""
        ]
        for query in queries:
            actual_result = self.graph.query(query)
            self.env.assertEquals(actual_result.result_set, [[True]])

        # inputs with expected result = False
        queries = [
            """RETURN isEmpty('abc')""",
            """RETURN isEmpty(['a', 'b', 'c'])""",
            """RETURN isEmpty([null])""",
            """WITH {val: 1, nested: {nested_val: 'nested_str'}} AS map RETURN isEmpty(map)""",
            """WITH {x:null} AS map RETURN isEmpty(map)"""
        ]
        for query in queries:
            actual_result = self.graph.query(query)
            self.env.assertEquals(actual_result.result_set, [[False]])

        # invalid input types
        queries = [
            """RETURN isEmpty(true)""",
            """RETURN isEmpty(0)""",
            """RETURN isEmpty(1.3)""",
            """CREATE (n) RETURN isEmpty(n)""",
            """CREATE (a:X)-[r:R]->(b:Y) RETURN isEmpty(r)"""
        ]
        for query in queries:
            self.expect_type_error(query)

    def test52_Expression(self):
        query_to_expected_result = {
            "RETURN 'muchacho'": [['muchacho']],
            "RETURN 1": [[1]],
            "RETURN 1+2*3": [[7]],
            "RETURN 1 + 1 + 1 + 1 + 1 + 1": [[6]],
            "RETURN ABS(-5 + 2 * 1)": [[3]],
            "RETURN 'a' + 'b'": [['ab']],
            "RETURN 1 + 2 + 'a' + 2 + 1": [['3a21']],
            "RETURN 2 * 2 + 'a' + 3 * 3": [['4a9']],
            "RETURN 9 % 5": [[4]],
            "RETURN 9 % 5 % 3": [[1]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
        
    def test53_NullArithmetic(self):
        query_to_expected_result = {
            "RETURN null + 1": [[None]],
            "RETURN 1 + null": [[None]],
            "RETURN null - 1": [[None]],
            "RETURN 1 - null": [[None]],
            "RETURN 1 * null": [[None]],
            "RETURN null / 1": [[None]],
            "RETURN 1 / null": [[None]],
            "RETURN 5 % null": [[None]],
            "RETURN null % 5": [[None]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test54_Abs(self):
        query_to_expected_result = {
            "RETURN ABS(1)": [[1]],
            "RETURN ABS(-1)": [[1]],
            "RETURN ABS(0)": [[0]],
            "RETURN ABS(NULL)": [[None]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test55_Aggregate(self):
        query_to_expected_result = {
            "UNWIND [1, 1, 1] AS one RETURN SUM(one)": [[3]],
            "UNWIND [1, 1, 1] AS one WITH SUM(one) AS s RETURN s+2": [[5]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test56_Ciel(self):
        query_to_expected_result = {
            "RETURN CEIL(0.5)": [[1]],
            "RETURN CEIL(1)": [[1]],
            "RETURN CEIL(0.1)": [[1]],
            "RETURN CEIL(NULL)": [[None]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test57_Floor(self):
        query_to_expected_result = {
            "RETURN FLOOR(0.5)": [[0]], 
            "RETURN FLOOR(1)": [[1]], 
            "RETURN FLOOR(0.1)": [[0]], 
            "RETURN FLOOR(NULL)": [[None]] 
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test58_Round(self):
        query_to_expected_result = {
            "RETURN ROUND(0)": [[0]], 
            "RETURN ROUND(0.49)": [[0]], 
            "RETURN ROUND(0.5)": [[1]], 
            "RETURN ROUND(1)": [[1]], 
            "RETURN ROUND(NULL)": [[None]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

    def test59_Sign(self):
        query_to_expected_result = {
            "RETURN SIGN(0)": [[0]], 
            "RETURN SIGN(-1)": [[-1]], 
            "RETURN SIGN(1)": [[1]], 
            "RETURN SIGN(NULL)": [[None]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

    def test60_Pow(self):
        query_to_expected_result = {
            "RETURN pow(1,0)": [[1]], 
            "RETURN 1^0": [[1]], 
            "RETURN pow(0,1)": [[0]], 
            "RETURN 0^1": [[0]], 
            "RETURN pow(0,0)": [[1]], 
            "RETURN 0^0": [[1]], 
            "RETURN pow(2,3)": [[8]], 
            "RETURN 2^3": [[8]], 
            "RETURN pow(2,-3)": [[0.125]], 
            "RETURN 2^-3": [[0.125]], 
            "RETURN 2^(-3)": [[0.125]], 
            "RETURN pow(0.5,2)": [[0.25]], 
            "RETURN 0.5^2": [[0.25]], 
            "RETURN pow(-1,2)": [[1]], 
            "RETURN -1^2": [[1]], 
            "RETURN (-1)^2": [[1]], 
            "RETURN pow(NULL,1)": [[None]], 
            "RETURN NULL^1": [[None]], 
            "RETURN pow(1,NULL)": [[None]], 
            "RETURN 1^NULL": [[None]], 
            "RETURN pow(NULL,NULL)": [[None]], 
            "RETURN NULL^NULL": [[None]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test61_Reverse(self):
        query_to_expected_result = {
            "RETURN REVERSE('muchacho')": [["ohcahcum"]], 
            "RETURN REVERSE('')": [[""]], 
            "RETURN REVERSE(NULL)": [[None]],
            # test unicode charecters
            "RETURN reverse('丁丂七丄丅丆万丈三上')": [["上三丈万丆丅丄七丂丁"]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test62_LTrim(self):
        query_to_expected_result = {
            "RETURN lTrim('   muchacho')": [["muchacho"]], 
            "RETURN lTrim('muchacho   ')": [["muchacho   "]], 
            "RETURN lTrim('   much   acho   ')": [["much   acho   "]], 
            "RETURN lTrim('muchacho')": [["muchacho"]], 
            "RETURN lTrim(NULL)": [[None]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test63_RTrim(self):
        query_to_expected_result = {
            "RETURN rTrim('   muchacho')": [["   muchacho"]], 
            "RETURN rTrim('muchacho   ')": [["muchacho"]], 
            "RETURN rTrim('   much   acho   ')": [["   much   acho"]], 
            "RETURN rTrim('muchacho')": [["muchacho"]], 
            "RETURN rTrim(NULL)": [[None]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test64_Trim(self):
        query_to_expected_result = {
            "RETURN trim('   muchacho')": [["muchacho"]],
            "RETURN trim('muchacho   ')": [["muchacho"]],
            "RETURN trim('   much   acho   ')": [["much   acho"]],
            "RETURN trim('muchacho')": [["muchacho"]],
            "RETURN trim(NULL)": [[None]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test65_ToLower(self):
        query_to_expected_result = {
            "RETURN toLower('MuChAcHo')": [['muchacho']],
            "RETURN toLower('mUcHaChO')": [['muchacho']],
            "RETURN toLower(NULL)": [[None]],
            # test unicode charecters
            "RETURN toLower('ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω')": [["ααββγγδδεεζζηηθθιικκλλμμννξξοοππρρσσςττυυφφχχψψωω"]],
            "RETURN toLower('АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЬьЭэЮюЯя')": [["ааббввггддеежжззииййккллммннооппррссттууффххццччшшщщььээююяя"]],
            "RETURN toLower('AbCdEfGhIjKlMnOpQrStUvWxYzÄöÜß')":  [["abcdefghijklmnopqrstuvwxyzäöüß"]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

        self.expect_error('RETURN toLower(replace("�", "", "   "))', "Invalid UTF8 string")
    
    def test66_ToUpper(self):
        query_to_expected_result = {
            "RETURN toUpper('MuChAcHo')": [['MUCHACHO']],
            "RETURN toUpper('mUcHaChO')": [['MUCHACHO']],
            "RETURN toUpper(NULL)": [[None]],
            # test unicode charecters
            "RETURN toUpper('ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω')": [["ΑΑΒΒΓΓΔΔΕΕΖΖΗΗΘΘΙΙΚΚΛΛΜΜΝΝΞΞΟΟΠΠΡΡΣΣΣΤΤΥΥΦΦΧΧΨΨΩΩ"]],
            "RETURN toUpper('АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЬьЭэЮюЯя')":  [["ААББВВГГДДЕЕЖЖЗЗИИЙЙККЛЛММННООППРРССТТУУФФХХЦЦЧЧШШЩЩЬЬЭЭЮЮЯЯ"]],
            "RETURN toUpper('AbCdEfGhIjKlMnOpQrStUvWxYzÄöÜß')":  [["ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜẞ"]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

        self.expect_error('RETURN toUpper(replace("�", "", "   "))', "Invalid UTF8 string")
    
    def test67_Exists(self):
        query_to_expected_result = {
            "RETURN EXISTS(null)": [[0]],
            "RETURN EXISTS(1)": [[1]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test68_Case(self):
        query_to_expected_result = {
            "RETURN CASE 'brown' WHEN 'blue' THEN 1+0 WHEN 'brown' THEN 2-0 ELSE 3*1 END": [[2]],
            "RETURN CASE 'green' WHEN 'blue' THEN 1+0 WHEN 'brown' THEN 2-0 ELSE 3*1 END": [[3]],
            "RETURN CASE WHEN NULL THEN 1+0 WHEN true THEN 2-0 END": [[2]],
            "RETURN CASE WHEN NULL THEN 1+0 WHEN NULL THEN 2-0 ELSE 3*1 END": [[3]],
            "RETURN CASE WHEN NULL THEN 1+0 WHEN NULL THEN 2-0 END": [[None]],
            "RETURN CASE NULL WHEN NULL THEN NULL ELSE 'else' END AS result": [[None]],
            "RETURN CASE NULL WHEN 'value' THEN 'value' WHEN NULL THEN NULL ELSE 'else' END AS result": [[None]],
            "RETURN CASE NULL WHEN 'when' THEN 'then' ELSE NULL END AS result": [[None]],
            "RETURN CASE 'value' WHEN NULL THEN NULL ELSE true END AS result": [[True]],
            "RETURN CASE 'value' WHEN NULL THEN NULL WHEN 'value' THEN true ELSE false END AS result": [[True]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test69_AND(self):
        scenarios = [
            ['TRUE', 'FALSE', False],
            ['FALSE', 'TRUE', False],
            ['TRUE', 'TRUE', True],
            ['FALSE', 'FALSE', False],
            ['NULL', 'FALSE', None],
            ['FALSE', 'NULL', None],
            ['TRUE', 'NULL', None],
            ['NULL', 'TRUE', None],
            ['NULL', 'NULL', None]
        ]
        for i in range(0, len(scenarios), 3):
            lhs = scenarios[i][0]
            rhs = scenarios[i][1]
            expected = scenarios[i][2]
            actual_result = self.graph.query(f"RETURN {lhs} AND {rhs}").result_set[0][0]
            self.env.assertEquals(actual_result, expected)
    
    def test70_OR(self):
        scenarios = [
            ['TRUE', 'FALSE', True],
            ['FALSE', 'TRUE', True],
            ['TRUE', 'TRUE', True],
            ['FALSE', 'FALSE', False],
            ['NULL', 'FALSE', None],
            ['FALSE', 'NULL', None],
            ['TRUE', 'NULL', True],
            ['NULL', 'TRUE', True],
            ['NULL', 'NULL', None]
        ]
        for i in range(len(scenarios)):
            lhs = scenarios[i][0]
            rhs = scenarios[i][1]
            expected = scenarios[i][2]
            actual_result = self.graph.query(f"RETURN {lhs} OR {rhs}").result_set[0][0]
            self.env.assertEquals(actual_result, expected)
    
    def test71_XOR(self):
        scenarios = [
            ['TRUE', 'FALSE', True],
            ['FALSE', 'TRUE', True],
            ['TRUE', 'TRUE', False],
            ['FALSE', 'FALSE', False],
            ['NULL', 'FALSE', None],
            ['FALSE', 'NULL', None],
            ['TRUE', 'NULL', None],
            ['NULL', 'TRUE', None],
            ['NULL', 'NULL', None]
        ]
        for i in range(len(scenarios)):
            lhs = scenarios[i][0]
            rhs = scenarios[i][1]
            expected = scenarios[i][2]
            actual_result = self.graph.query(f"RETURN {lhs} XOR {rhs}").result_set[0][0]
            self.env.assertEquals(actual_result, expected)

    def test72_NOT(self):
        scenarios = [
            ['TRUE', False],
            ['FALSE', True],
            ['NULL', None]
        ]
        for i in range(len(scenarios)):
            b = scenarios[i][0]
            expected = scenarios[i][1]
            actual_result = self.graph.query(f"RETURN NOT {b}").result_set[0][0]
            self.env.assertEquals(actual_result, expected)
    
    def test73_LT(self):
        scenarios = [
            ['1','1', False],
            ['1', '2', True],
            ['2', '1', False],
            ['2', 'NULL', None],
            ['2', '2', False],
            ['1', 'NULL', None],
            ['NULL', '2', None],
            ['NULL', '1', None],
            ['NULL', 'NULL', None]
        ]
        for i in range(len(scenarios)):
            lhs = scenarios[i][0]
            rhs = scenarios[i][1]
            expected = scenarios[i][2]
            actual_result = self.graph.query(f"RETURN {lhs} < {rhs}").result_set[0][0]
            self.env.assertEquals(actual_result, expected)
    
    def test74_LE(self):
        scenarios = [
            ['1','1', True],
            ['1', '2', True],
            ['2', '1', False],
            ['2', 'NULL', None],
            ['2', '2', True],
            ['1', 'NULL', None],
            ['NULL', '2', None],
            ['NULL', '1', None],
            ['NULL', 'NULL', None]
        ]
        for i in range(len(scenarios)):
            lhs = scenarios[i][0]
            rhs = scenarios[i][1]
            expected = scenarios[i][2]
            actual_result = self.graph.query(f"RETURN {lhs} <= {rhs}").result_set[0][0]
            self.env.assertEquals(actual_result, expected)

    def test75_EQ(self):
        scenarios = [
            ['1','1', True],
            ['1', '2', False],
            ['2', '1', False],
            ['2', 'NULL', None],
            ['2', '2', True],
            ['1', 'NULL', None],
            ['NULL', '2', None],
            ['NULL', '1', None],
            ['NULL', 'NULL', None]
        ]
        for i in range(len(scenarios)):
            lhs = scenarios[i][0]
            rhs = scenarios[i][1]
            expected = scenarios[i][2]
            actual_result = self.graph.query(f"RETURN {lhs} = {rhs}").result_set[0][0]
            self.env.assertEquals(actual_result, expected)
    
    def test76_NE(self):
        scenarios = [
            ['1','1', False],
            ['1', '2', True],
            ['2', '1', True],
            ['2', 'NULL', None],
            ['2', '2', False],
            ['1', 'NULL', None],
            ['NULL', '2', None],
            ['NULL', '1', None],
            ['NULL', 'NULL', None]
        ]
        for i in range(len(scenarios)):
            lhs = scenarios[i][0]
            rhs = scenarios[i][1]
            expected = scenarios[i][2]
            actual_result = self.graph.query(f"RETURN {lhs} <> {rhs}").result_set[0][0]
            self.env.assertEquals(actual_result, expected)
    
    def test77_List(self):
        arr = [1, 2.3, '4', True, False, None]
        query = "RETURN [1,2.3,'4',TRUE,FALSE, NULL]"
        actual_result = self.graph.query(query).result_set[0][0]
        if not type(actual_result) is list:
            assert(False)                   # Fail if the record returned is not a list.
        for i in range(len(arr)):
            self.env.assertEquals(actual_result[i], arr[i])

    def test78_ListSlice(self):
        arr = [0,1,2,3,4,5,6,7,8,9,10]
        query_to_expected_result = {
            "RETURN [0,1,2,3,4,5,6,7,8,9,10][3]": [[arr[3]]],
            "RETURN [0,1,2,3,4,5,6,7,8,9,10][-3]": [[arr[-3]]],
            "RETURN [0,1,2,3,4,5,6,7,8,9,10][0..3]": [[arr[0:3]]],
            "RETURN [0,1,2,3,4,5,6,7,8,9,10][0..-5]": [[arr[0:-5]]],
            "RETURN [0,1,2,3,4,5,6,7,8,9,10][-5..]": [[arr[-5:]]],
            "RETURN [0,1,2,3,4,5,6,7,8,9,10][..4]": [[arr[:4]]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test79_Range(self):
        query_to_expected_result = {
            "RETURN range(0,10)": [[[i for i in range(11)]]],
            "RETURN range(2,18,3)": [[[i for i in range(2, 18, 3)]]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test80_IN(self):
        query_to_expected_result = {
            "RETURN 3 IN [1,2,3]": [[True]],
            "RETURN 4 IN [1,2,3]": [[False]],
            "RETURN [1,2] IN [1,2,3]": [[False]],
            "RETURN [1,2] IN [[1,2],3]": [[True]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

    def test81_ISNULL(self):
        arr = ["NULL", "1", "1.2", "TRUE", "FALSE", "'string'", "[1,2,3]"]
        for ind, s in enumerate(arr):
            query1 = f'RETURN {s} IS NOT NULL'
            expected1 = [[True]] if ind!=0 else [[False]]
            self.get_res_and_assertEquals(query1, expected1)
            query2 = f'RETURN {s} IS NULL'
            expected2 = [[False]] if ind!=0 else [[True]]
            self.get_res_and_assertEquals(query2, expected2)
    
    def test82_Coalesce(self):
        query_to_expected_result = {
            "RETURN coalesce(1)": [[1]],
            "RETURN coalesce(NULL, 1)": [[1]],
            "RETURN coalesce(NULL, NULL, 500, NULL)": [[500]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)
    
    def test83_Replace(self):
        query_to_expected_result = {
            "RETURN replace('abcabc', 'a', '00')": [["00bc00bc"]],
            "RETURN replace('abcabc', 'bc', '0')": [["a0a0"]],
            "RETURN replace('abcabc', 'abc', '')": [[""]],
            "RETURN replace('abcabc', 'ab', '')": [["cc"]],
            "RETURN replace('abcabc', '', '0')": [["0a0b0c0a0b0c0"]],
            # test unicode charecters
            # changing half unicode charecter will not change the original string
            "RETURN replace('丁丂七丄丅丆万丈三上', '\xe4', 'X')": [["丁丂七丄丅丆万丈三上"]],
            "RETURN replace('丁丂七丄丅丆万丈三上', '丄', 'X')": [["丁丂七X丅丆万丈三上"]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

    def test84_RandomUUID(self):
        query = "RETURN randomUUID()"
        actual_result = self.graph.query(query).result_set[0][0]
        self.env.assertEquals(actual_result[8], '-')
        self.env.assertEquals(actual_result[13], '-')
        self.env.assertEquals(actual_result[14], '4')
        self.env.assertEquals(actual_result[18], '-')
        if not actual_result[19] in ['8', '9', 'a', 'b']:
            assert(False)
        self.env.assertEquals(actual_result[23], '-')

    def test85_division_inputs(self):
        # Validate integer dividend division by 0
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 3 / 0"
        # (error) Division by zero
        query = "RETURN 3 / 0"
        try:
            actual_result = self.graph.query(query)
        except redis.ResponseError as e:
            self.env.assertContains("Division by zero", str(e))
        
        # Validate floating-point dividend division by 0
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 1.0 / 0"
        # 1) 1) "1.0 / 0"
        # 2) 1) 1) "inf"
        query = "RETURN 1.0 / 0"
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0],float('inf'))

        # Validate negative floating-point dividend division by 0
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN -1.0 / 0"
        # 1) 1) "-1.0 / 0"
        # 2) 1) 1) "-inf"
        query = "RETURN -1.0 / 0"
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0],float('-inf'))

        # Validate integer dividend division by 0.0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 1 / 0.0"
        # 1) 1) "1 / 0.0"
        # 2) 1) 1) "inf"
        query = "RETURN 1 / 0.0"
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0],float('inf'))

        # Validate negative integer dividend division by 0.0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN -1 / 0.0"
        # 1) 1) "-1 / 0.0"
        # 2) 1) 1) "-inf"
        query = "RETURN -1 / 0.0"
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0],float('-inf'))

        # Validate floating-point dividend division by 0.0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 1.0 / 0.0"
        # 1) 1) "1.0 / 0.0"
        # 2) 1) 1) "inf"
        query = "RETURN 1.0 / 0.0"
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0],float('inf'))

        # Validate negative floating-point dividend division by 0.0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN -1.0 / 0.0"
        # 1) 1) "-1.0 / 0.0"
        # 2) 1) 1) "-inf"
        query = "RETURN -1.0 / 0.0"
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0],float('-inf'))

        # Validate floating-point 0.0 divided by floating-point 0.0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 0.0 / 0.0"
        # 1) 1) "0.0 / 0.0"
        # 2) 1) 1) "-nan"
        query = "RETURN 0.0 / 0.0"
        actual_result = self.graph.query(query)
        self.env.assertTrue(math.isnan(actual_result.result_set[0][0]))

        # Validate integer 0 divided by floating 0.0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 0 / 0.0"
        # 1) 1) "0 / 0.0"
        # 2) 1) 1) "-nan"
        query = "RETURN 0 / 0.0"
        actual_result = self.graph.query(query)
        self.env.assertTrue(math.isnan(actual_result.result_set[0][0]))

        # Validate floating-point 0.0 divided by integer 0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 0.0 / 0"
        # 1) 1) "0.0 / 0"
        # 2) 1) 1) "-nan"
        query = "RETURN 0.0 / 0"
        actual_result = self.graph.query(query)
        self.env.assertTrue(math.isnan(actual_result.result_set[0][0]))

        # Validate integer 0 divided by 0 
        # redis-cli output example:
        # 127.0.0.1:6379> self.GRAPH.QUERY g "RETURN 0 / 0"
        # (error) Division by zero
        query = "RETURN 0 / 0"
        try:
            actual_result = self.graph.query(query)
        except redis.ResponseError as e:
            self.env.assertContains("Division by zero", str(e))

    def test86_type_mismatch_message(self):
        # A list of queries and errors which are expected to occur with the
        # specified query.
        queries_with_errors = {
            "RETURN tail(1)": "Type mismatch: expected List or Null but was Integer",
            "CREATE (n) RETURN hasLabels(n, 1)": "Type mismatch: expected List but was Integer",
            "CREATE ()-[r:R]->() RETURN hasLabels(r, ['abc', 'def'])": "Type mismatch: expected Node or Null but was Edge",
            "RETURN toBoolean(1.2)": "Type mismatch: expected String, Boolean, Integer, or Null but was Float",
            "RETURN isEmpty(1)": "Type mismatch: expected Map, List, String, or Null but was Integer",
            "CREATE ()-[r:R]->() RETURN toString(r)": "Type mismatch: expected Datetime, Duration, String, Boolean, Integer, Float, Null, or Point but was Edge",
        }
        for query, error in queries_with_errors.items():
            self.expect_error(query, error)

    def test87_typeof(self):
        query_to_expected_result = {
            "RETURN typeOf(NULL)" : [['Null']],
            "RETURN typeOf([1,2])" : [['List']],
            "RETURN typeOf({a: 1})" : [['Map']],
            "RETURN typeOf(point({latitude:1,longitude:2}))" : [['Point']],
            "RETURN typeOf(1), typeOf('1'), typeOf(true)" : [['Integer', 'String', 'Boolean']],
            "MATCH path=({val: 0})-[e:works_with]->({val: 1}) RETURN typeOf(path)" : [['Path']],
            "CREATE (a)-[b:B]->(c) RETURN typeOf(a), typeOf(b), typeOf(c)" : [['Node', 'Edge', 'Node']],
            "CREATE (a:A {x:1, y:'1', z:true}) RETURN typeOf(a.x), typeOf(a.y), typeOf(a.z)" : [['Integer', 'String', 'Boolean']],
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

    def test88_in_out_degree(self):
        # clear graph
        self.graph.delete()

        # given the graph (a)
        # in/out degree of 'a' is 0
        self.graph.query("CREATE (a:A)")
        result_set = self.graph.query("MATCH (a:A) RETURN inDegree(a), outDegree(a)").result_set
        in_degree = result_set[0][0]
        out_degree = result_set[0][1]
        self.env.assertEqual(in_degree, 0)
        self.env.assertEqual(out_degree, 0)

        # given the graph: (a)-[:E]->(b)
        # in degree of 'a' is 0
        # out degree of 'a' is 1
        # in degree of 'b' is 1
        # out degree of 'b' is 0
        self.graph.query("MATCH (a:A) CREATE (a)-[:E]->(b:B)")
        result_set = self.graph.query("MATCH (a:A), (b:B) RETURN inDegree(a), outDegree(a), inDegree(b), outDegree(b)").result_set
        a_in_degree = result_set[0][0]
        a_out_degree = result_set[0][1]
        b_in_degree = result_set[0][2]
        b_out_degree = result_set[0][3]
        self.env.assertEqual(a_in_degree, 0)
        self.env.assertEqual(a_out_degree, 1)
        self.env.assertEqual(b_in_degree, 1)
        self.env.assertEqual(b_out_degree, 0)

        # given the graph (a)-[:E]->(b), (a)-[:E]->(b)
        # in degree of 'a' is 0
        # out degree of 'a' is 2
        # in degree of 'b' is 2
        # out degree of 'b' is 0
        self.graph.query("MATCH (a:A), (b:B) CREATE (a)-[:E]->(b)")
        result_set = self.graph.query("MATCH (a:A), (b:B) RETURN inDegree(a), outDegree(a), inDegree(b), outDegree(b)").result_set
        a_in_degree = result_set[0][0]
        a_out_degree = result_set[0][1]
        b_in_degree = result_set[0][2]
        b_out_degree = result_set[0][3]
        self.env.assertEqual(a_in_degree, 0)
        self.env.assertEqual(a_out_degree, 2)
        self.env.assertEqual(b_in_degree, 2)
        self.env.assertEqual(b_out_degree, 0)

        # given the graph (a)-[:E]->(b), (a)-[:E]->(b) (a)-[:E0]->(b), (a)-[:E1]->(b)
        # in degree of 'a' is 0
        # out degree of 'a' is 4
        # in degree of 'b' is 4
        # out degree of 'b' is 0
        self.graph.query("MATCH (a:A), (b:B) CREATE (a)-[:E0]->(b), (a)-[:E1]->(b)")
        result_set = self.graph.query("MATCH (a:A), (b:B) RETURN inDegree(a), outDegree(a), inDegree(b), outDegree(b)").result_set
        a_in_degree = result_set[0][0]
        a_out_degree = result_set[0][1]
        b_in_degree = result_set[0][2]
        b_out_degree = result_set[0][3]
        self.env.assertEqual(a_in_degree, 0)
        self.env.assertEqual(a_out_degree, 4)
        self.env.assertEqual(b_in_degree, 4)
        self.env.assertEqual(b_out_degree, 0)

        # given the graph (a)-[:E]->(b), (a)-[:E]->(b) (a)-[:E0]->(b), (a)-[:E1]->(b)
        # in degree of 'a' for relation 'E' is 0
        # out degree of 'a' for relation 'E' is 2
        # in degree of 'b' for relation 'E' is 2
        # out degree of 'b' for relation 'E' is 0
        result_set = self.graph.query("MATCH (a:A), (b:B) RETURN inDegree(a, 'E'), outDegree(a, 'E'), inDegree(b, 'E'), outDegree(b, 'E')").result_set
        a_in_degree = result_set[0][0]
        a_out_degree = result_set[0][1]
        b_in_degree = result_set[0][2]
        b_out_degree = result_set[0][3]
        self.env.assertEqual(a_in_degree, 0)
        self.env.assertEqual(a_out_degree, 2)
        self.env.assertEqual(b_in_degree, 2)
        self.env.assertEqual(b_out_degree, 0)

        # given the graph (a)-[:E]->(b), (a)-[:E]->(b) (a)-[:E0]->(b), (a)-[:E1]->(b)
        # in degree of 'a' for relationships 'E0' and 'E1' is 0
        # out degree of 'a' for relationships 'E0' and 'E1' is 2
        # in degree of 'b' for relationships 'E0' and 'E1' is 2
        # out degree of 'b' for relationships 'E0' and 'E1' is 0
        result_set = self.graph.query("MATCH (a:A), (b:B) RETURN inDegree(a, 'E0', 'E1'), outDegree(a, 'E0', 'E1'), inDegree(b, 'E0', 'E1'), outDegree(b, 'E0', 'E1')").result_set
        a_in_degree = result_set[0][0]
        a_out_degree = result_set[0][1]
        b_in_degree = result_set[0][2]
        b_out_degree = result_set[0][3]
        self.env.assertEqual(a_in_degree, 0)
        self.env.assertEqual(a_out_degree, 2)
        self.env.assertEqual(b_in_degree, 2)
        self.env.assertEqual(b_out_degree, 0)

        # given the graph (a)-[:E]->(b), (a)-[:E]->(b) (a)-[:E0]->(b), (a)-[:E1]->(b)
        # in degree of 'a' for relationships 'E', 'E0' and 'E1' is 0
        # out degree of 'a' for relationships 'E', 'E0' and 'E1' is 4
        # in degree of 'b' for relationships 'E', 'E0' and 'E1' is 4
        # out degree of 'b' for relationships 'E', 'E0' and 'E1' is 0
        result_set = self.graph.query("MATCH (a:A), (b:B) RETURN inDegree(a, 'E', 'E0', 'E1'), outDegree(a, 'E', 'E0', 'E1'), inDegree(b, 'E', 'E0', 'E1'), outDegree(b, 'E', 'E0', 'E1')").result_set
        a_in_degree = result_set[0][0]
        a_out_degree = result_set[0][1]
        b_in_degree = result_set[0][2]
        b_out_degree = result_set[0][3]
        self.env.assertEqual(a_in_degree, 0)
        self.env.assertEqual(a_out_degree, 4)
        self.env.assertEqual(b_in_degree, 4)
        self.env.assertEqual(b_out_degree, 0)

        # in/out degree a none existing relationship type is 0
        result_set = self.graph.query("MATCH (a:A) RETURN inDegree(a, 'none_existing'), outDegree(a, 'none_existing')").result_set
        in_degree = result_set[0][0]
        out_degree = result_set[0][1]
        self.env.assertEqual(in_degree, 0)
        self.env.assertEqual(out_degree, 0)

        # clear graph
        self.graph.delete()

        # given the graph (a)-[:E]->(a)
        # in/out degree of 'a' is 1
        self.graph.query("CREATE (a)-[:E]->(a)")
        result_set = self.graph.query("MATCH (a) RETURN inDegree(a), outDegree(a)").result_set
        in_degree = result_set[0][0]
        out_degree = result_set[0][1]
        self.env.assertEqual(in_degree, 1)
        self.env.assertEqual(out_degree, 1)

        # given the graph (a)-[:R]->(b)
        # out degree of 'a' is 1
        # in degree of 'b' is 1
        self.graph.query("CREATE (a:A)-[:R]->(b:B)")
        queries = [
            """MATCH (a:A) RETURN outdegree(a, 'R')""",
            """MATCH (a:A) RETURN outdegree(a, ['R'])""",
            """MATCH (a:A) RETURN outdegree(a, 'R', 'R')""",
            """MATCH (b:B) RETURN indegree(b, 'R')""",
            """MATCH (b:B) RETURN indegree(b, ['R', 'R'])""",
            """MATCH (b:B) RETURN indegree(b, 'R', 'R')""",
        ]
        for query in queries:
            actual_result = self.graph.query(query)
            self.env.assertEquals(actual_result.result_set, [[1]])

        # test type mismatch
        queries = [
            """MATCH (a:A) RETURN outdegree(a, a)""",           # node
            """MATCH (a:A) RETURN outdegree(a, [1])""",         # integer
            """MATCH (a:A) RETURN outdegree(a, [1.4])""",       # float
            """MATCH (a:A) RETURN outdegree(a, 'R', 1)""",      # integer after string
            """MATCH (a:A) RETURN outdegree(a, ['R', 1])""",    # integer element in list
            """MATCH (a:A) RETURN outdegree(a, 'R', ['R'])""",  # wrong signature: string and list
            ]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                # Expecting a type error.
                self.env.assertIn("Type mismatch", str(e))

        # test wrong argument number
        queries = [
            """MATCH (a:A) RETURN outdegree()""",
            """MATCH (a:A) RETURN outdegree(a, ['R'], 'a')""",
            """MATCH (a:A) RETURN outdegree(a, ['R'], ['R'])""",
            """MATCH (b:B) RETURN indegree()""",
            """MATCH (b:B) RETURN indegree(b, ['R'], 'a')""",
            """MATCH (b:B) RETURN indegree(b, ['R'], ['R'])""",
            ]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                # Expecting a type error.
                self.env.assertIn("Received", str(e))


    def test89_JOIN(self):
        # NULL input should return NULL
        expected_result = [None]
        query = """WITH NULL as list RETURN string.join(null, '')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # 2nd arg should be string
        try:
            self.graph.query("RETURN string.join(['HELL','OW'], 2)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected String but was Integer", str(e))

        # Test without input argument
        try:
            query = """RETURN string.join()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'string.join', expected at least 1", str(e))

        # Test with 3 input argument
        try:
            query = """RETURN string.join(['HELL','OW'], ' ', '')"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 3 arguments to function 'string.join', expected at most 2", str(e))

        # list args should be string
        try:
            self.graph.query("RETURN string.join(['HELL', 2], ' ')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected String but was Integer", str(e))

        # list args should be string
        try:
            self.graph.query("RETURN string.join(['HELL', 'OW', 2, 'now'], ' ')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected String but was Integer", str(e))

        # list args should be string
        try:
            self.graph.query("RETURN string.join([3, 'OW', 'now'], ' ')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected String but was Integer", str(e))

        ### Test valid inputs ###
        expected_result = ['HELLOW']
        query = """RETURN string.join(['HELL','OW'])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['HELL OW']
        query = """RETURN string.join(['HELL','OW'], ' ')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['HELL']
        query = """RETURN string.join(['HELL'], ' ')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['HELL OW NOW']
        query = """RETURN string.join(['HELL','OW', 'NOW'], ' ')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # join overflow
        q = """UNWIND RANGE(0, 10000000) as x
               WITH collect('looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong string') as list
               RETURN string.join(list, 'loooooooooooooooooooooooooooooooooooooooooonggggggggggggggggggggggggggggggggggggggggggggggggggggg delimiterrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
               """
        try:
            result = self.graph.query(q)
            self.env.assertFalse(True)
        except ResponseError as e:
            self.env.assertContains("String overflow", str(e))

        # join empty list
        queries = ["RETURN string.join([])",
                   "RETURN string.join([], '|')"]

        for q in queries:
            actual_result = self.graph.query(q).result_set[0][0]
            # expecting an empty string
            self.env.assertEquals(actual_result, "")

    def test90_size(self):
        query_to_expected_result = {
            "RETURN size(NULL)" : [[None]],
            "RETURN size('abcd')" : [[4]],
            "RETURN size('丁丂七丄丅丆万丈三上')" : [[10]]
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

    def test91_MATCHREGEX(self):
        # NULL input should return empty list
        expected_result = [[]]
        query = """WITH NULL as string RETURN string.matchRegEx(null, "bla")"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # NULL input should return empty list
        expected_result = [[]]
        query = """WITH NULL as string RETURN string.matchRegEx("bla", null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test invalid regex
        try:
            query = """RETURN string.matchRegEx('aa', '?')"""
            actual_result = self.graph.query(query)
        except ResponseError as e:
            self.env.assertContains("Invalid regex", str(e))

        # 1st arg should be string
        try:
            self.graph.query("RETURN string.matchRegEx(2, 'bla')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected String or Null but was Integer", str(e))

        # 2nd arg should be string
        try:
            self.graph.query("RETURN string.matchRegEx('bla', 2)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected String or Null but was Integer", str(e))

        # Test without input argument
        try:
            query = """RETURN string.matchRegEx()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'string.matchRegEx', expected at least 2", str(e))

        # Test with 3 input argument
        try:
            query = """RETURN string.matchRegEx('bla', 'dsds', '')"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 3 arguments to function 'string.matchRegEx', expected at most 2", str(e))

        ### Test valid inputs ###
        expected_result = [[['<header h1>txt1</header>', 'h1', 'txt1']]]
        query = """RETURN string.matchRegEx('blabla <header h1>txt1</header>', '<header (\\w+)>(\\w+)</header>')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[['<header h1>txt1</header>', 'h1', 'txt1'], ['<header h2>txt2</header>', 'h2', 'txt2']]]
        query = """RETURN string.matchRegEx('blabla <header h1>txt1</header> blabla <header h2>txt2</header>', '<header (\\w+)>(\\w+)</header>')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[['?']]]
        query = """RETURN string.matchRegEx('?', '\\\\?')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[['a'], ['a']]]
        query = """RETURN string.matchRegEx('aba', 'a')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[]]
        query = """RETURN string.matchRegEx('', 'a')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[]]
        query = """RETURN string.matchRegEx('bla', '(bla)(bal)')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[['bla9', 'bla']]]
        query = """RETURN string.matchRegEx('bla9', '(bla)[(bal)9]')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[['bla9', 'bla']]]
        query = """RETURN string.matchRegEx('bla9', '(bla)[(bal)9]')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[['😉']]]
        query = """RETURN string.matchRegEx('😉', '😉')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # proof for Avi that I need to change the parser
        #expected_result = [[[]]]
        #query = """RETURN string.matchRegEx('aa', '(?:\\?)')"""
        #actual_result = self.graph.query(query)
        #self.env.assertEquals(actual_result.result_set[0], expected_result)

    def test92_REPLACEREGEX(self):
        # NULL input should return NULL
        expected_result = [None]
        query = """WITH NULL as string RETURN string.replaceRegEx(null, "bla")"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # NULL input should return NULL
        expected_result = [None]
        query = """WITH NULL as string RETURN string.replaceRegEx("bla", null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # NULL input should return NULL
        expected_result = [None]
        query = """WITH NULL as string RETURN string.replaceRegEx("bla", "bla", null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test invalid regex
        try:
            query = """RETURN string.replaceRegEx('aa', '?')"""
            actual_result = self.graph.query(query)
        except ResponseError as e:
            self.env.assertContains("Invalid regex", str(e))

        # 1st arg should be string
        try:
            self.graph.query("RETURN string.replaceRegEx(2, 'bla')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected String or Null but was Integer", str(e))

        # 2nd arg should be string
        try:
            self.graph.query("RETURN string.replaceRegEx('bla', 2)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected String or Null but was Integer", str(e))

        # 3rd arg should be string
        try:
            self.graph.query("RETURN string.replaceRegEx('bla', 'bla', 2)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected String or Null but was Integer", str(e))

        # Test without input argument
        try:
            query = """RETURN string.replaceRegEx()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'string.replaceRegEx', expected at least 2", str(e))

        # Test with 4 input argument
        try:
            query = """RETURN string.replaceRegEx('bla', 'dsds', 'fdsf', '')"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 4 arguments to function 'string.replaceRegEx', expected at most 3", str(e))

        ### Test valid inputs ###
        expected_result = ['blabla hellow']
        query = """RETURN string.replaceRegEx('blabla <header h1>txt1</header>', '<header (\\w+)>(\\w+)</header>', 'hellow')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['blabla hellow blabla hellow']
        query = """RETURN string.replaceRegEx('blabla <header h1>txt1</header> blabla <header h2>txt2</header>', '<header (\\w+)>(\\w+)</header>', 'hellow')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['m']
        query = """RETURN string.replaceRegEx('?', '\\\\?', 'm')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['ac']
        query = """RETURN string.replaceRegEx('abc', '[b]')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['a55c']
        query = """RETURN string.replaceRegEx('abc', '[b]', '55')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['ac']
        query = """RETURN string.replaceRegEx('abc', '[b]', '')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['ac']
        query = """RETURN string.replaceRegEx('abcb', '[b]', '')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['']
        query = """RETURN string.replaceRegEx('', '[b]', 'bla')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['']
        query = """RETURN string.replaceRegEx('', '[b]', 'bla')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['blablala']
        query = """RETURN string.replaceRegEx('bbla', '[b]', 'bla')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = ['bl😀a']
        query = """RETURN string.replaceRegEx('bl😉a', '😉', '😀')"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

    def test93_overflow(self):
        # Test integer overflow caused by string to long conversion
        queries_with_errors = {
            "RETURN 10000000000000000000000" : "Integer overflow '10000000000000000000000'",
            "RETURN -10000000000000000000000" : "Integer overflow '-10000000000000000000000'",
            "RETURN 9223372036854775808" : "Integer overflow '9223372036854775808'",
            "RETURN -9223372036854775809" : "Integer overflow '-9223372036854775809'",
        }
        for query, error in queries_with_errors.items():
            self.expect_error(query, error)

        # Test valid queries
        query_to_expected_result = {
            "RETURN 10^-324" : [[0]],
            "RETURN pow(10,-324)" : [[0]],
            "RETURN 10^+324" : [[float('inf')]],
            "RETURN pow(10,324)" : [[float('inf')]],
            "RETURN 0.5 + pow(10,-324)" : [[0.5]],
            "RETURN pow(100,200), 5" : [[float('inf'), 5]],
            "RETURN 9223372036854775807" : [[9223372036854775807]],
            "RETURN -9223372036854775808" : [[-9223372036854775808]],
        }
        for query, expected_result in query_to_expected_result.items():
            self.get_res_and_assertEquals(query, expected_result)

    def test94_vector(self):
        # Test invalid inputs
        err_msg = "vectorf32 expects an array of numbers"
        queries_with_errors = [
                "RETURN vecf32([1.2, 'a'])",
                "RETURN vecf32([1.2, NULL])",
                "RETURN vecf32([1.2, true])",
                "RETURN vecf32([1.2, []])",
                "RETURN vecf32([1.2, {a:2}])",
                "RETURN vecf32([1.2, point({latitude:1,longitude:2})])",
                "RETURN vecf32([1.2, vecf32([1])])",
        ]

        for q in queries_with_errors:
            self.expect_error(q, err_msg)

    def test95_prev(self):
        res = self.graph.query("UNWIND range(1, 5) AS x RETURN prev(x)")
        self.env.assertEquals(res.result_set, [[None], [1], [2], [3], [4]])

        res = self.graph.query("UNWIND range(1, 5) AS x RETURN prev(prev(x))")
        self.env.assertEquals(res.result_set, [[None], [None], [1], [2], [3]])

        res = self.graph.query("UNWIND range(1, 5) AS x RETURN prev(tostring(x) + tostring(x))")
        self.env.assertEquals(res.result_set, [[None], ['11'], ['22'], ['33'], ['44']])
