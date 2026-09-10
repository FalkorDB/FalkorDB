from common import *

GRAPH_ID = "null_handling"

class testNullHandlingFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        # Create a single node.
        self.graph.query("CREATE (:L {v: 'v1'})")

    # Error when attempting to create a relationship with a null endpoint.
    def test01_create_null(self):
        try:
            query = """MATCH (a) OPTIONAL MATCH (a)-[nonexistent_edge]->(nonexistent_node) CREATE (nonexistent_node)-[:E]->(a)"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

        try:
            query = """MATCH (a) OPTIONAL MATCH (a)-[nonexistent_edge]->(nonexistent_node) CREATE (a)-[:E]->(nonexistent_node)"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    # Error when attempting to merge a relationship with a null endpoint.
    def test02_merge_null(self):
        try:
            query = """MATCH (a) OPTIONAL MATCH (a)-[nonexistent_edge]->(nonexistent_node) MERGE (nonexistent_node)-[:E]->(a)"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

        try:
            query = """MATCH (a) OPTIONAL MATCH (a)-[nonexistent_edge]->(nonexistent_node) MERGE (a)-[:E]->(nonexistent_node)"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    # SET should update attributes on non-null entities and ignore null entities.
    def test03_set_null(self):
        query = """MATCH (a) OPTIONAL MATCH (a)-[nonexistent_edge]->(nonexistent_node) SET a.v2 = true, nonexistent_node.v2 = true, a.v3 = nonexistent_node.v3 RETURN a.v2, nonexistent_node.v2, a.v3"""
        actual_result = self.graph.query(query)
        # The property should be set on the real node and ignored on the null entity.
        assert(actual_result.properties_set == 1)
        expected_result = [[True, None, None]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    # DELETE should ignore null entities.
    def test04_delete_null(self):
        query = """MATCH (a) OPTIONAL MATCH (a)-[nonexistent_edge]->(nonexistent_node) DELETE nonexistent_node"""
        actual_result = self.graph.query(query)
        assert(actual_result.nodes_deleted == 0)

    # Functions should handle null inputs appropriately.
    def test05_null_function_inputs(self):
        query = """MATCH (a) OPTIONAL MATCH (a)-[r]->(b) RETURN type(r), labels(b), b.v * 5"""
        actual_result = self.graph.query(query)
        expected_result = [[None, None, None]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Path functions should handle null inputs appropriately.
    def test06_null_named_path_function_inputs(self):
        query = """MATCH (a) OPTIONAL MATCH p = (a)-[r]->() RETURN p, length(p), collect(relationships(p))"""
        actual_result = self.graph.query(query)
        # The path and function calls on it should return NULL, while collect() returns an empty array.
        expected_result = [[None, None, []]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Scan and traversal operations should gracefully handle NULL inputs.
    def test07_null_graph_entity_inputs(self):
        query = """WITH NULL AS a MATCH (a) RETURN a"""
        actual_result = self.graph.query(query)
        # Expect one NULL entity to be returned.
        expected_result = [[None]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        query = """WITH NULL AS a MATCH (a)-[e]->(b) RETURN a, e, b"""
        plan = str(self.graph.explain(query))
        # Verify that we are attempting to perform a traversal but no scan.
        self.env.assertNotContains("Scan", plan)
        self.env.assertContains("Conditional Traverse", plan)
        actual_result = self.graph.query(query)
        # Expect no results.
        expected_result = []
        self.env.assertEqual(actual_result.result_set, expected_result)

        query = """WITH NULL AS e MATCH (a:L)-[e]->(b) RETURN a, e, b"""
        plan = str(self.graph.explain(query))
        # Verify that we are performing a scan and traversal.
        self.env.assertContains("Conditional Traverse", plan)
        actual_result = self.graph.query(query)
        # Expect no results.
        expected_result = []
        self.env.assertEqual(actual_result.result_set, expected_result)

    # ValueHashJoin ops should not treat null values as equal.
    def test08_null_value_hash_join(self):
        query = """MATCH (a), (b) WHERE a.fakeval = b.fakeval RETURN a, b"""
        plan = str(self.graph.explain(query))
        # Verify that we are performing a ValueHashJoin
        self.env.assertContains("Value Hash Join", plan)
        actual_result = self.graph.query(query)
        # Expect no results.
        expected_result = []
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Perform a sanity check on a ValueHashJoin that returns a result
        query = """MATCH (a), (b) WHERE a.v = b.v RETURN a.v, b.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v1']]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # A join key that merely *contains* a null is not itself null, but
        # Cypher's three-valued logic still makes `[1, null] = [1, null]` NULL
        # rather than TRUE. The ValueHashJoin must agree with the equivalent
        # Filter plan instead of matching on plain structural equality (#2775).
        graph = self.db.select_graph(GRAPH_ID + "_nested_null_join")
        # Clear first: an assertion failure below skips the delete at the end,
        # and a rerun against the same server would otherwise append a second
        # copy of the fixture and fail for the wrong reason. GRAPH.DELETE
        # errors on an absent key, so the first run has to tolerate that.
        try:
            graph.delete()
        except redis.ResponseError:
            pass
        graph.query("""CREATE (:A {i: 1, s: 'x', w: 1}), (:A {i: 2, s: 'y'}),
                              (:B {i: 1, s: 'x', w: 1}), (:B {i: 2, s: 'y'}),
                              (:B {i: 3, s: 'z'})""")

        # Predicates building a join key that holds a null at some depth --
        # a.missing / b.missing are absent, hence NULL. None may produce a row.
        null_keys = ["[a.i, a.missing] = [b.i, b.missing]",
                     "[[a.i, a.missing]] = [[b.i, b.missing]]",
                     "{k: a.missing} = {k: b.missing}",
                     "[{k: a.missing}] = [{k: b.missing}]",
                     # a vector has no comparison of its own, so it used to
                     # mask the null beside it in the same list
                     "[vecf32([1.0]), a.missing] = [vecf32([1.0]), b.missing]"]

        for predicate in null_keys:
            join = f"MATCH (a:A), (b:B) WHERE {predicate} RETURN a.i, b.i"
            # The WITH is a barrier, so here the predicate stays a Filter over
            # a Cartesian Product -- the same query, without the join rewrite.
            filtered = f"MATCH (a:A), (b:B) WITH a, b WHERE {predicate} RETURN a.i, b.i"

            # Verify the optimized plan really is the one under test.
            self.env.assertContains("Value Hash Join", str(graph.explain(join)))
            self.env.assertNotContains("Value Hash Join", str(graph.explain(filtered)))

            expected_result = graph.query(filtered).result_set
            self.env.assertEqual(expected_result, [])
            self.env.assertEqual(graph.query(join).result_set, expected_result)

        # Null-free keys must still join, across every table representation:
        # integers (the i64 fast path), strings, lists and maps.
        non_null_keys = [("a.i = b.i", [[1, 1], [2, 2]]),
                         ("a.s = b.s", [[1, 1], [2, 2]]),
                         ("[a.i, a.s] = [b.i, b.s]", [[1, 1], [2, 2]]),
                         ("{p: a.i, q: a.s} = {p: b.i, q: b.s}", [[1, 1], [2, 2]]),
                         # Only one side of this key is null, and coalesce removes it.
                         ("coalesce(a.w, 0) = coalesce(b.w, 0)", [[1, 1], [2, 2], [2, 3]])]

        for predicate, expected_result in non_null_keys:
            query = f"MATCH (a:A), (b:B) WHERE {predicate} RETURN a.i, b.i ORDER BY a.i, b.i"
            self.env.assertContains("Value Hash Join", str(graph.explain(query)))
            self.env.assertEqual(graph.query(query).result_set, expected_result)

        # A value Cypher cannot compare at all must not report equality just
        # because it sits inside a list. Two vectors are never equal to each
        # other, whatever their contents, and a null beside one still wins.
        incomparable = [("vecf32([1.0,2.0]) = vecf32([1.0,2.0])", False),
                        ("[vecf32([1.0,2.0])] = [vecf32([1.0,2.0])]", False),
                        ("[vecf32([1.0,2.0])] = [vecf32([9.0,9.0])]", False),
                        ("[vecf32([1.0]), null] = [vecf32([1.0]), null]", None),
                        # length still decides lists of unequal length
                        ("[vecf32([1.0])] = [vecf32([1.0]), 1]", False)]

        for expression, expected in incomparable:
            actual = graph.query(f"RETURN {expression}").result_set[0][0]
            self.env.assertEqual(actual, expected)

        graph.delete()
