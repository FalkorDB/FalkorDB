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
