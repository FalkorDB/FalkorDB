from common import *

GRAPH_ID = "map_test"


class testMap(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        # Construct a graph with the form:
        # (v1)-[:E]->(v2)-[:E]->(v3)
        q = """CREATE (:L {val:1})-[:E]->(:L {val:2})-[:E]->(:L {val:3})"""
        self.graph.query(q)

    # Validate basic map lookup operations
    def test01_basic_map_accesses(self):
        # Return a full map
        query = """WITH {val: 5} AS map RETURN map"""
        query_result = self.graph.query(query)
        expected_result = [[{'val': 5}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Return a map value with dot notation
        query = """WITH {val: 5} AS map RETURN map.val"""
        query_result = self.graph.query(query)
        expected_result = [[5]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Return a map value with bracket notation
        query = """WITH {val: 5} AS map RETURN map['val']"""
        query_result = self.graph.query(query)
        expected_result = [[5]]
        self.env.assertEquals(query_result.result_set, expected_result)

    # Validate map projection behavior
    def test02_map_projections(self):
        query = """MATCH (a) RETURN a {.val} ORDER BY a.val"""
        query_result = self.graph.query(query)
        expected_result = [[{'val': 1}],
                           [{'val': 2}],
                           [{'val': 3}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """WITH 'lit' AS literal MATCH (a) RETURN a {.val, literal} ORDER BY a.val"""
        query_result = self.graph.query(query)
        expected_result = [[{'val': 1, 'literal': 'lit'}],
                           [{'val': 2, 'literal': 'lit'}],
                           [{'val': 3, 'literal': 'lit'}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """MATCH (n) RETURN n {}"""
        query_result = self.graph.query(query)
        expected_result = [[{}],
                           [{}],
                           [{}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """MATCH (n) RETURN n {.*}  ORDER BY n.val"""
        query_result = self.graph.query(query)
        expected_result =  [[{'val': 1}],
                            [{'val': 2}],
                            [{'val': 3}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        queries = [
            """MATCH (n) RETURN n {.*, .h}  ORDER BY n.val""",
            """MATCH (n) RETURN n {.h, .*}  ORDER BY n.val""",
            """MATCH (n) RETURN n {.*, .h, .*}  ORDER BY n.val""",
            """MATCH (n) RETURN n {.h, .*, .h, .val}  ORDER BY n.val""",
        ]
        expected_result =  [[{'val': 1, 'h': None}],
                            [{'val': 2, 'h': None}],
                            [{'val': 3, 'h': None}]]
        for query in queries:
            query_result = self.graph.query(query)
            self.env.assertEquals(query_result.result_set, expected_result)

        query = """MATCH (n:unexisting) RETURN n {}"""
        query_result = self.graph.query(query)
        expected_result = []
        self.env.assertEquals(query_result.result_set, expected_result)

        # map projection unexisting property
        query = """CREATE (a:A {name: 'abc', y: 3, z: 43}) RETURN a{.h}"""
        query_result = self.graph.query(query)
        expected_result = [[{'h': None}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # map projection all properties and unexisting property
        queries = [
            """CREATE (a:A {name: 'abc', y: 3, z: 43}) RETURN a{.*, .h}""",
            """CREATE (a:A {name: 'abc', y: 3, z: 43}) RETURN a{.h, .*}""",
            """CREATE (a:A {name: 'abc', y: 3, z: 43}) RETURN a{.*, .z, .h, .*}""",
        ]
        expected_result = [[{'name':'abc', 'y': 3, 'z': 43, 'h': None}]]
        for query in queries:
            query_result = self.graph.query(query)
            self.env.assertEquals(query_result.result_set, expected_result)

        # map projection attributes
        query = """CREATE (a:A {name: 'abc', y: 3, z: 43}) RETURN a{vname: a.name, vy: a.y, vz: a.z}"""
        expected_result = [[{'vname': 'abc', 'vy': 3, 'vz': 43}]]
        query_result = self.graph.query(query)
        self.env.assertEquals(query_result.result_set, expected_result)

        # map projection on a null entity should evaluate to null
        # (Neo4j semantics) e.g. via OPTIONAL MATCH that did not match
        queries = [
            """OPTIONAL MATCH (n:unexisting) RETURN n {.foo}""",
            """OPTIONAL MATCH (n:unexisting) RETURN n {.foo, bar: 5}""",
            """OPTIONAL MATCH (n:unexisting) RETURN n {.*}""",
            """OPTIONAL MATCH (n:unexisting) RETURN n {.*, .foo}""",
        ]
        expected_result = [[None]]
        for query in queries:
            query_result = self.graph.query(query)
            self.env.assertEquals(query_result.result_set, expected_result)

        # collect() on a map projection of a null entity should skip nulls,
        # producing an empty list (matches Neo4j behavior)
        query = """OPTIONAL MATCH (n:unexisting) RETURN collect(DISTINCT n {.foo, .bar}) AS items"""
        query_result = self.graph.query(query)
        self.env.assertEquals(query_result.result_set, [[[]]])

        # nested projections
        queries = [
            """MATCH (a), (b) RETURN a{.*, b {.*} }""",
            """MATCH (a) RETURN a{.*, c {.*} }""",
        ]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                self.env.assertContains("Invalid input '{': expected ':', ',' or '}'", str(e))

    # Validate behaviors of nested maps
    def test03_nested_maps(self):
        # Return a map with nesting
        query = """WITH {val: 5, nested: {nested_val: 'nested_str'}} AS map RETURN map"""
        query_result = self.graph.query(query)
        expected_result = [[{'val': 5, 'nested': {'nested_val': 'nested_str'}}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Return just the nested value
        query = """WITH {val: 5, nested: {nested_val: 'nested_str'}} AS map RETURN map.nested"""
        query_result = self.graph.query(query)
        expected_result = [[{'nested_val': 'nested_str'}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Perform operations on map values
        query = """WITH {val: 5, nested: {nested_val: 'nested_str'}} AS map RETURN map.val + '_' + map.nested.nested_val"""
        query_result = self.graph.query(query)
        expected_result = [['5_nested_str']]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Deeply nested map
        query = """RETURN {a: {b: {c: {d: {e: {f: {g: {h: {i: {j: {}}}}}}}}}}}"""
        query_result = self.graph.query(query)
        expected_result = [[{'a': {'b': {'c': {'d': {'e': {'f': {'g': {'h': {'i': {'j': {}}}}}}}}}}}]]
        self.env.assertEquals(query_result.result_set, expected_result)

    # Validate map sorting logic (first by keys, then by values)
    def test04_map_sorting(self):
        query = """UNWIND[{b: 1}, {a: 2}] AS map RETURN map ORDER BY map"""
        query_result = self.graph.query(query)
        expected_result = [[{'a': 2}],
                           [{'b': 1}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """UNWIND[{a: 2}, {a: 1}] AS map RETURN map ORDER BY map"""
        query_result = self.graph.query(query)
        expected_result = [[{'a': 1}],
                           [{'a': 2}]]
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """UNWIND[{a: 2}, {x: 1, k: 2}] AS map RETURN map ORDER BY map"""
        query_result = self.graph.query(query)
        expected_result = [[{'a': 2}],
                           [{'x': 1, 'k': 2}]]
        self.env.assertEquals(query_result.result_set, expected_result)

    # Validate map comparison logic (first by keys, then by values)
    def test05_map_comparison(self):
        query = """WITH {b: 2} AS map_1, {a: 1} AS map_2 RETURN map_1 > map_2, map_1 < map_2, map_1 = map_2, map_1 <> map_2"""
        query_result = self.graph.query(query)
        expected_result = [[True, False, False, True]]
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """WITH {a: 2} AS map_1, {a: 1} AS map_2 RETURN map_1 > map_2, map_1 < map_2, map_1 = map_2, map_1 <> map_2"""
        query_result = self.graph.query(query)
        expected_result = [[True, False, False, True]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Map equality is not predicated on key order.
        query = """WITH {a: 1, b: 2} AS map_1, {b: 2, a: 1} AS map_2 RETURN map_1 > map_2, map_1 < map_2, map_1 = map_2, map_1 <> map_2"""
        query_result = self.graph.query(query)
        expected_result = [[False, False, True, False]]
        self.env.assertEquals(query_result.result_set, expected_result)

    # Validate that maps are handled correctly by the DISTINCT operator.
    def test05_map_distinct(self):
        # Map uniqueness is not predicated on key order.
        query = """UNWIND[{b: 2, a: 1}, {b: 2, a: 1}, {a: 1, b: 2}] AS map RETURN DISTINCT map"""
        query_result = self.graph.query(query)
        expected_result = [[{'a': 1, 'b': 2}]]
        self.env.assertEquals(query_result.result_set, expected_result)

    # Validate that trying to access a map with a non-string key errors gracefully.
    def test06_map_invalid_key_lookup(self):
        try:
            query = """WITH {val: 5} AS map RETURN map[0]"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("Type mismatch", str(e))

    # Validate that map projections with invalid identifiers error gracefully.
    def test07_map_invalid_identifier(self):
        try:
            query = """RETURN 5 {v: 'b'}"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("Encountered unhandled type", str(e))

    # validate that function accesses of scalar-reducible maps do not access invalid memory
    def test08_map_safe_return_value(self):
        query = """RETURN {a: 5, b: 'xx'}.b"""
        query_result = self.graph.query(query)
        expected_result = [['xx']]
        self.env.assertEquals(query_result.result_set, expected_result)

        query = """RETURN {a: 5, b: toUpper('xx')}.b"""
        query_result = self.graph.query(query)
        expected_result = [['XX']]
        self.env.assertEquals(query_result.result_set, expected_result)

    def test09_merge_map(self):
        query = """RETURN {name: 'John', age: 30} + {age: 40, city: 'New York'}"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], {'name': 'John', 'age': 40, 'city': 'New York'})

        query = """RETURN {name: 'John', age: 30} + 1"""
        try:
            self.graph.query(query)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("Cannot merge a map with a non-map value", str(e))

    def test10_map_merge_null(self):
        # merge maps where one of the maps is null
        q = "CREATE (n:N {name:'John', age:30})"
        res = self.graph.query(q).result_set

        q = """MATCH (n:N)
               OPTIONAL MATCH (n)-[]->(m:M)
               RETURN n{.*, x: COLLECT({z:1})}, n"""

        res = self.graph.query(q).result_set
        self.env.assertEquals(len(res), 1)
        self.env.assertEquals(res[0][0], {'name': 'John', 'age': 30, 'x': [{'z':1}]})

        q = """MATCH (n:N)
               OPTIONAL MATCH (n)-[]->(m:M)
               RETURN n, n{x: COLLECT({z:1}), .*}"""

        res = self.graph.query(q).result_set
        self.env.assertEquals(len(res), 1)
        self.env.assertEquals(res[0][1], {'name': 'John', 'age': 30, 'x': [{'z':1}]})


    # Validate map comparison when values are of different types,
    # null or NaN. The result must not depend on key order.
    def test11_map_comparison_mixed_value_types(self):
        cases = [
            # (expression, expected)
            ("{x: 1.0} = {x: 'x'}", False),
            ("{x: 1.0} <> {x: 'x'}", True),
            ("{x: 1} = {x: 1.0}", True),
            ("[{x: 1.0}] = [{x: 'x'}]", False),
            ("[{x: 1.0}] <> [{x: 'x'}]", True),
            ("{x: {y: 1.0}} = {x: {y: 'x'}}", False),
            ("{x: 1.0} IN [{x: 'x'}]", False),
            ("{x: 1.0} IN [{x: 'x'}, {x: 1}]", True),
            ("CASE {x: 1.0} WHEN {x: 'x'} THEN 1 ELSE 0 END", 0),
            ("{x: vecf32([0.5])} = {x: 'x'}", False),
            ("{x: date('2024-01-01')} = {x: 1}", False),
            # a definite difference makes the maps unequal, despite a null
            ("{a: null, b: 1} = {a: 1, b: 2}", False),
            ("{a: 1, b: null} = {a: 2, b: 1}", False),
            ("{a: 'x', b: null} = {a: 1, b: 1}", False),
            # otherwise a null makes the comparison null
            ("{a: null, b: 1} = {a: 1, b: 1}", None),
            ("{a: 1, b: null} = {a: 1, b: 1}", None),
            # NaN is never equal, wherever its key sorts
            ("{a: 0.0/0.0, b: 1} = {a: 0.0/0.0, b: 1}", False),
            ("{a: 1, b: 0.0/0.0} = {a: 1, b: 0.0/0.0}", False),
            ("{a: 0.0/0.0, b: 1} <> {a: 0.0/0.0, b: 1}", True),
            ("{a: 0.0/0.0, b: null} = {a: 0.0/0.0, b: 1}", False),
        ]

        for exp, expected in cases:
            res = self.graph.query(f"RETURN {exp}").result_set
            self.env.assertEquals(res[0][0], expected, message=exp)

    # Validate every operator on maps holding NaN, null and mixed types.
    # Equality (=, <>, IN, list equality) fails on any unequal pair
    # of values, otherwise it is null if any pair is null.
    # Ordering (<, >, <=, >=) is decided by the first pair of values which
    # are not equal, using the scalar rules for that pair:
    # null or disjoint types give null and NaN gives false.
    def test12_map_operators_with_null_and_nan(self):
        NaN = "0.0/0.0"
        T, F, N = True, False, None
        cases = [
            # (m, n, =, <, >, <=, >=)
            ("{a: 1, b: 2}",          "{a: 1, b: 2}",         T, F, F, T, T),
            ("{a: 1, b: 2}",          "{a: 1, b: 3}",         F, T, F, T, F),
            ("{a: 1}",                "{a: 1.0}",             T, F, F, T, T),
            ("{a: 1}",                "{b: 1}",               F, T, F, T, F),
            ("{a: 1.0}",              "{a: 'x'}",             F, N, N, N, N),
            ("{a: null}",             "{a: 1}",               N, N, N, N, N),
            ("{a: null}",             "{a: null}",            N, N, N, N, N),
            (f"{{a: {NaN}}}",         "{a: 1}",               F, F, F, F, F),
            (f"{{a: {NaN}}}",         f"{{a: {NaN}}}",        F, F, F, F, F),
            ("{a: null, b: 1}",       "{a: 1, b: 1}",         N, N, N, N, N),
            ("{a: null, b: 1}",       "{a: 1, b: 2}",         F, N, N, N, N),
            ("{a: 1, b: null}",       "{a: 2, b: 1}",         F, T, F, T, F),
            ("{a: 1, b: null}",       "{a: 1, b: 1}",         N, N, N, N, N),
            (f"{{a: null, b: {NaN}}}", "{a: null, b: 1}",     F, N, N, N, N),
            (f"{{a: {NaN}, b: null}}", "{a: 1, b: 1}",        F, F, F, F, F),
            (f"{{a: {NaN}, b: 1}}",   "{a: 1, b: 2}",         F, F, F, F, F),
            (f"{{a: 1, b: {NaN}}}",   "{a: 2, b: 1}",         F, T, F, T, F),
            (f"{{a: 1, b: {NaN}}}",   "{a: 1, b: 1}",         F, F, F, F, F),
            ("{a: 1.0, b: null}",     "{a: 'x', b: 1}",       F, N, N, N, N),
            ("{a: null, b: 1.0}",     "{a: 1, b: 'x'}",       F, N, N, N, N),
            ("{a: {x: null}, b: 1}",  "{a: {x: 1}, b: 2}",    F, N, N, N, N),
            (f"{{a: {{x: {NaN}}}, b: 1}}", "{a: {x: 1}, b: 1}", F, F, F, F, F),
        ]

        def neg(v):
            return None if v is None else not v

        for m, n, eq, lt, gt, le, ge in cases:
            # each comparison is checked in both directions
            for x, y, xlt, xgt, xle, xge in [(m, n, lt, gt, le, ge),
                                             (n, m, gt, lt, ge, le)]:
                # IN with a null element is true if an element is equal,
                # null otherwise
                in_null = True if eq else None
                exps = [
                    (f"{x} = {y}",                eq),
                    (f"{x} <> {y}",               neg(eq)),
                    (f"{x} IN [{y}]",             eq),
                    (f"{x} IN [{y}, null]",       in_null),
                    (f"[{x}] = [{y}]",            eq),
                    (f"[{x}] <> [{y}]",           neg(eq)),
                    (f"[0, {x}] = [0, {y}]",      eq),
                    (f"{x} < {y}",                xlt),
                    (f"{x} > {y}",                xgt),
                    (f"{x} <= {y}",               xle),
                    (f"{x} >= {y}",               xge),
                ]
                q = f"RETURN {', '.join(e for e, _ in exps)}"
                res = self.graph.query(q).result_set[0]
                for (exp, expected), actual in zip(exps, res):
                    self.env.assertEquals(actual, expected, message=exp)

                # the same comparisons as WHERE filters
                for i, (exp, expected) in enumerate(exps):
                    q = f"WITH 1 AS one WHERE {exp} RETURN count(*)"
                    res = self.graph.query(q).result_set
                    self.env.assertEquals(res[0][0], int(expected is True),
                                          message=f"WHERE {exp}")

                    # a filter negates '<' by testing '>=' and NaN fails
                    # both, while NOT (NaN < 1) projects true, for scalars
                    # as well, so NOT is only checked for equality
                    if i >= 7:
                        continue
                    q = f"WITH 1 AS one WHERE NOT ({exp}) RETURN count(*)"
                    res = self.graph.query(q).result_set
                    self.env.assertEquals(res[0][0], int(expected is False),
                                          message=f"WHERE NOT {exp}")

    # Validate equality of scalars and lists holding NaN or null
    def test13_equality_with_nan_and_null(self):
        NaN = "(0.0/0.0)"
        cases = [
            # (expression, expected)
            (f"{NaN} = {NaN}", False),
            (f"{NaN} IN [1]", False),
            (f"{NaN} IN [{NaN}]", False),
            (f"1 IN [{NaN}, 1]", True),
            (f"1 IN [{NaN}, null]", None),
            (f"[{NaN}] = [1]", False),
            (f"[{NaN}] = [{NaN}]", False),
            (f"[1, {NaN}] <> [1, 1]", True),
            ("[null] = [null]", None),
            ("[1, null] = [1, 1]", None),
            ("[1, null] = [2, 1]", False),
            ("[null, 1] = [1, 2]", False),
            ("[1, 2] = [1, 2.0]", True),
            ("[1] = [1, 2]", False),
            ("CASE 1 WHEN 1.0 THEN 1 ELSE 0 END", 1),
        ]

        for exp, expected in cases:
            res = self.graph.query(f"RETURN {exp}").result_set
            self.env.assertEquals(res[0][0], expected, message=exp)
