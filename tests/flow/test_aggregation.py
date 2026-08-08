from common import *
from math import floor, ceil, sqrt

GRAPH_ID = "aggregations"

class testAggregations():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def get_res_and_assertEquals(self, query, expected_result):
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, expected_result)
    
    def get_res_and_assertAlmostEquals(self, query, expected_result):
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(actual_result.result_set[0][0], expected_result[0][0], 0.0001)

    def assert_rejected(self, graph, query, expected_message):
        try:
            result = graph.query(query)
        except ResponseError as e:
            if expected_message not in str(e):
                raise AssertionError(
                    f"{query!r} was rejected with {str(e)!r}, "
                    f"which does not mention {expected_message!r}") from e
            return
        raise AssertionError(
            f"{query!r} should have been rejected, returned {result.result_set!r}")

    # test aggregation default values
    # default values should be returned when the aggregation operation
    # was not given any data to process
    # and the aggregation doesn't specify any keys
    def test01_empty_aggregation(self):
        # default aggregation values
        expected_result = [0,    # count
                           None, # min
                           None, # max
                           0,    # sum
                           None, # avg
                           0,    # stDev
                           0,    # stDevP
                           [],   # collect,
                           None, # percentileDisc
                           None  # percentileCont
                           ]

        query = """MATCH (n) WHERE n.v = 'noneExisting'
                   RETURN count(n), min(n.v), max(n.v), sum(n.v), avg(n.v),
                   stDev(n.v), stDevP(n.v), collect(n),
                   percentileDisc(n.v, 0.5), percentileCont(n.v, 0.5)"""
        result = self.graph.query(query)
        self.env.assertEqual(result.result_set[0], expected_result)

        # issue a similar query only perform aggregations within a WITH clause
        query = """MATCH (n) WHERE n.v = 'noneExisting'
                   WITH count(n) as A, min(n.v) as B, max(n.v) as C, sum(n.v) as D,
                   avg(n.v) as E, stDev(n.v) as F,  stDevP(n.v) as G,
                   collect(n) as H, percentileDisc(n.v, 0.5) as I,
                   percentileCont(n.v, 0.5) as J
                   RETURN *"""

        result = self.graph.query(query)
        self.env.assertEqual(result.result_set[0], expected_result)
    
    def test02_countTest(self):
        query = "UNWIND [NULL, NULL, NULL, NULL, NULL] AS x RETURN count(1)"
        expected = 5
        actual_result = self.graph.query(query).result_set[0][0]
        self.env.assertEqual(actual_result, expected)
    
    def test03_partialCountTest(self):
        query = "UNWIND [NULL, 1, NULL, 1, NULL, 1, NULL, 1, NULL, 1] AS x RETURN count(x)"
        expected = 5
        actual_result = self.graph.query(query).result_set[0][0]
        self.env.assertEqual(actual_result, expected)
    
    def test04_percentileCont(self):
        expected_values = []
        percentile_doubles = [0, 0.1, 0.33, 0.5, 1]
        arr = [2, 4, 6, 8, 10]
        count = 5
        for i in range(5):
            x = percentile_doubles[i] * (count - 1)
            lower_idx = floor(x)
            upper_idx = ceil(x)
            if lower_idx == upper_idx or lower_idx == (count - 1):
                expected_values.append([[arr[lower_idx]]])
                continue
            lower = arr[lower_idx]
            upper = arr[upper_idx]

            expected_values.append([[lower * (upper_idx - x) + (upper * (x - lower_idx))]])

        for i in range(5):
            query = f'UNWIND [2, 4, 6, 8, 10] AS x RETURN percentileCont(x, {percentile_doubles[i]})'
            self.get_res_and_assertAlmostEquals(query, expected_values[i])
    
    def test05_percentileDisc(self):
        percentile_doubles = [0, 0.1, 0.33, 0.5, 1]
        expected = [0, 0, 1, 2, 4]
        expectedResults = [0] * 5
        for i in range(1, 6):
            expectedResults[i-1] = i * 2
        for i in range(5):
            query = f'UNWIND [2, 4, 6, 8, 10] AS x RETURN percentileDisc(x, {percentile_doubles[i]})'
            self.get_res_and_assertAlmostEquals(query, [[expectedResults[expected[i]]]])
        
        query = f'UNWIND [0.5, 0, 1] AS x RETURN percentileDisc(x, 0)'
        self.get_res_and_assertAlmostEquals(query, [[0]])
    
    def test06_StDev(self):
        # Edge case - less than 2 arguments.
        self.get_res_and_assertEquals("RETURN stDev(5.1)", [[0]])
        # 10 first integers.
        query = f'UNWIND [1, 2, 3, 4, 5, 6 , 7, 8, 9, 10] AS x RETURN stDev(x)'
        sum = 0
        for i in range(1, 11):
            sum += i
        mean = sum / 10
        tmp_var = 0
        for i in range(1, 11):
            tmp_var += pow(i-mean, 2)
        sample_var = tmp_var / 9
        sample_res = sqrt(sample_var)
        self.get_res_and_assertAlmostEquals(query, [[sample_res]])

    def test07_AverageDoubleOverflow(self):
        double_max = '1.7976931348623157e+308'
        query = f'UNWIND [{double_max}, {double_max} / 2] AS x RETURN avg(x)'
        query2 = f'RETURN ({double_max} / 2 + {double_max} / 4)'
        res1 = self.graph.query(query).result_set[0][0]
        res2 = self.graph.query(query2).result_set[0][0]
        self.env.assertEqual(res1, res2)
    
    def test08_AggregateLongOverflow(self):
        long_max = 2147483647
        query = f'UNWIND [{long_max}, {long_max / 2}] AS x RETURN avg(x)'
        expected = [[long_max / 2 + long_max / 4]]
        self.get_res_and_assertAlmostEquals(query, expected)
    
    def test09_AggregateWithNullFilter(self):
        query = 'CREATE (:L {p:0.0/0.0})'
        self.graph.query(query)

        query = 'MATCH (n:L) WHERE (null <> false) XOR true RETURN COUNT(n)'
        expected = [[0]]
        self.get_res_and_assertAlmostEquals(query, expected)

    def test10_AggregateCollection(self):
        """
            Collect a bunch of different items
        """

        # Collecting null values
        q = "RETURN collect(NULL) AS collection"
        expected = [[[]]]  # Empty collection
        self.get_res_and_assertEquals(q, expected)

        # Collecting integers
        q = "RETURN collect(1) AS collection"
        expected = [[[1]]]
        self.get_res_and_assertEquals(q, expected)

        # Collecting multiple integers
        q = "UNWIND [1, 2, 3] AS x RETURN collect(x)"
        expected = [[[1, 2, 3]]]
        self.get_res_and_assertEquals(q, expected)

        # Collecting floating point numbers
        q = "UNWIND [1.1, 2.2, 3.3] AS x RETURN collect(x)"
        expected = [[[1.1, 2.2, 3.3]]]
        self.get_res_and_assertEquals(q, expected)

        # Collecting strings
        q = "UNWIND ['a', 'b', 'c'] AS x RETURN collect(x)"
        expected = [[['a', 'b', 'c']]]
        self.get_res_and_assertEquals(q, expected)

        # Collecting booleans
        q = "UNWIND [true, false] AS x RETURN collect(x)"
        expected = [[[True, False]]]
        self.get_res_and_assertEquals(q, expected)

        # Collecting mixed data types
        q = "UNWIND [1, 'a', 3.14, true, NULL] AS x RETURN collect(x)"
        expected = [[[1, 'a', 3.14, True]]]  # Null collection should be empty
        self.get_res_and_assertEquals(q, expected)

        # Collecting an empty set
        q = "MATCH (n:NoneExisting) WHERE false RETURN collect(n)"
        expected = [[[]]]
        self.get_res_and_assertEquals(q, expected)

        # Create a few nodes
        res = self.graph.query("""CREATE (a:Person {name: 'Alice'}),
                                         (b:Person {name: 'Bob'})
                                  RETURN a, b""")
        nodes = [res.result_set[0][0], res.result_set[0][1]]

        # Collecting nodes properties
        q = """MATCH (p:Person)
               WITH p
               ORDER BY p.name
               RETURN collect(p.name) AS names"""

        expected = ['Alice', 'Bob']
        res = self.graph.query(q).result_set
        self.env.assertEqual(res[0][0], expected)

        # Collecting nodes
        q = """MATCH (p:Person)
               WITH p
               ORDER BY p.name
               RETURN collect(p) AS nodes"""
        expected = nodes
        res = self.graph.query(q).result_set
        self.env.assertEqual(res[0][0], expected)

        # Collecting relationships
        res = self.graph.query("""UNWIND range(0,1) AS x
                                  CREATE (:A)-[r:RELATES_TO]->(:B)
                                  RETURN r
                                  ORDER BY ID(r)""")
        edges = [row[0] for row in res.result_set]

        q = """MATCH (:A)-[r:RELATES_TO]->(:B)
               WITH r
               ORDER BY ID(r)
               RETURN collect(r) AS edges"""

        expected = edges
        res = self.graph.query(q).result_set[0][0]
        self.env.assertEqual(res, expected)

        # Collecting maps (dictionaries)
        q = "RETURN collect({key:'value', num:42}) AS collection"
        expected = [[[{'key': 'value', 'num': 42}]]]
        self.get_res_and_assertEquals(q, expected)

        # Collecting lists
        q = "RETURN collect([1, 2, 3]) AS collection"
        expected = [[[ [1, 2, 3] ]]]
        self.get_res_and_assertEquals(q, expected)

        # Collecting large dataset
        q = """UNWIND range(1, 10000) AS x
               RETURN collect({x:x, str: toString(x)}), collect([x, -x])"""
        expected = [
                [ {'x':x, 'str': str(x)} for x in range(1, 10001) ],
                [ [x, -x]                for x in range(1, 10001) ]
        ]
        res = self.graph.query(q).result_set[0]
        self.env.assertEqual(res, expected)

        # Collecting passed values
        q = """
        WITH [{a: 'Hello', b: [1, {x: 'y'}, 'GoodBye!']}, 4] AS collection
        MATCH (p:Person)
        WITH p, collection[ID(p) % 2] AS elem0, collect(collection[0]) AS collection
        ORDER BY ID(p) ASC
        WITH elem0, collection[0] AS collection
        RETURN elem0, collection
        """

        res = self.graph.query(q).result_set
        expected = [
            [4, {'a': 'Hello', 'b': [1, {'x': 'y'}, 'GoodBye!']}],
            [{'a': 'Hello', 'b': [1, {'x': 'y'}, 'GoodBye!']},
             {'a': 'Hello', 'b': [1, {'x': 'y'}, 'GoodBye!']}]
        ]

        self.env.assertEqual(res, expected)

    # a grouping key read outside the aggregation must produce its grouped
    # value, whether it is written as the name the clause projects it as or in
    # the original form the clause groups by
    def test11_GroupingKeyReadOutsideAggregation(self):
        g = self.db.select_graph("grouping_keys")
        g.query("CREATE (:P {a: 10, b: 20}), (:P {a: 10, b: 30})")

        for query, expected in [
                # projected under its own name
                ("MATCH (n:P) WITH n.a AS a RETURN a, a + count(*)",     [[10, 12]]),
                # projected as a property, read as a property
                ("MATCH (n:P) RETURN n.a, n.a + count(*)",               [[10, 12]]),
                # the whole node is the grouping key
                ("MATCH (n:P) RETURN n.a, count(*), n.a * 2",            [[10, 2, 20]]),
                # a key read from inside a map, alongside an aggregation
                ("MATCH (n:P) RETURN n.a AS a, {same: n.a = 10, kids: collect(n.b)} AS m",
                 [[10, {'same': True, 'kids': [20, 30]}]]),
                # a key captured by a construct that binds its own variables;
                # these are evaluated inline, so the capture has to resolve too
                ("MATCH (n:P) RETURN n.a AS a, count(*) + reduce(s = 0, x IN [1] | s + n.a)",
                 [[10, 12]]),
                ("MATCH (n:P) RETURN n.a AS a, count(*) + size([x IN [1] WHERE x < n.a | x])",
                 [[10, 3]]),
                ("MATCH (n:P) RETURN n.a AS a, count(*), any(x IN [1] WHERE x < n.a)",
                 [[10, 2, True]]),
                # the variables those constructs bind are local, so they are
                # neither rewritten nor demanded as grouping keys
                ("MATCH (n:P) RETURN count(*) + reduce(s = 0, x IN [1,2] | s + x)", [[5]]),
                ("MATCH (n:P) RETURN count(*), [x IN [1,2] | x * 2]",              [[2, [2, 4]]]),
                # no key involved at all
                ("MATCH (n:P) RETURN 1 + count(*)",                      [[3]]),
                ("MATCH (n:P) RETURN collect(n.a)[0]",                   [[10]]),
                ("MATCH (n:P) RETURN max(n.a) - min(n.b)",               [[-10]]),
        ]:
            self.env.assertEqual(g.query(query).result_set, expected)

        g.delete()

    # reading a variable outside an aggregation without projecting it as a
    # grouping key is an implicit grouping expression, which openCypher
    # CIP2021-07-07 rejects; it used to silently evaluate to null and collapse
    # the enclosing expression
    def test12_ImplicitGroupingKeyIsRejected(self):
        g = self.db.select_graph("grouping_keys")
        g.query("CREATE (:P {a: 10, b: 20}), (:P {a: 10, b: 30})")

        for query, read in [
                ("MATCH (n:P) RETURN n.a + count(*)",                     "n.a"),
                # n.a is a grouping key here, n.b is not
                ("MATCH (n:P) RETURN n.a, n.b + count(*)",                "n.b"),
                ("MATCH (n:P) WITH 0 AS i, n RETURN collect(n.a)[i]",     "i"),
                ("UNWIND [1, 2] AS n WITH 2 AS paths, n "
                 "RETURN CASE WHEN paths > 0 AND count(n) > 0 THEN 'PASS' ELSE 'FAIL' END",
                 "paths"),
                # the same read, one WITH earlier
                ("MATCH (n:P) WITH n.b AS b, n WITH b + count(n) AS r RETURN r", "b"),
                # a construct that binds its own variables does not excuse a key
                # it captures without projecting
                ("MATCH (n:P) RETURN n.a AS a, count(*) + reduce(s = 0, x IN [1] | s + n.b)",
                 "n.b"),
        ]:
            self.assert_rejected(
                g, query, f"'{read}' is read outside an aggregation function")

        g.delete()

    # adding a column to the projection must not change what the other columns
    # evaluate to
    def test13_ProjectingAKeyDoesNotChangeOtherColumns(self):
        g = self.db.select_graph("grouping_keys")
        g.query("CREATE (a:N {id: 1, v: 0})-[:R]->(b:N {id: 2}), (b)-[:R]->(a)")

        head = ("MATCH p = (n:N)-[:R*2..2]->(n) SET n.v = 1 "
                "WITH count(p) AS paths MATCH (n:N {v: 1}) RETURN ")
        tail = "CASE WHEN paths > 0 AND count(n) > 0 THEN 'PASS' ELSE 'FAIL' END AS v"

        with_diagnostics = g.query(head + "paths, count(n) AS updated, " + tail).result_set
        self.env.assertEqual(with_diagnostics, [[2, 2, 'PASS']])

        # without `paths` projected the query is now rejected rather than
        # quietly disagreeing with the row above
        self.assert_rejected(
            g, head + tail, "'paths' is read outside an aggregation function")

        g.delete()
