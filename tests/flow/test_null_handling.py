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

    # A simple CASE compares its subject with `=`, so a null on either side
    # selects no branch. Every pair below is checked against `=` itself, since
    # the two must not disagree.
    def test09_null_simple_case(self):
        # `eq` is what `=` answers for the pair; None means null, which must
        # stay distinct from False
        for subject, when, eq in [("null",       "null",       None),
                                  ("null",       "1",          None),
                                  ("1",          "null",       None),
                                  ("1",          "1",          True),
                                  ("1.0",        "1",          True),
                                  ("'a'",        "'a'",        True),
                                  ("'a'",        "'b'",        False),
                                  ("[1,2]",      "[1,2]",      True),
                                  # a null anywhere inside makes `=` null
                                  ("[1,null]",   "[1,null]",   None),
                                  ("{a:null}",   "{a:null}",   None)]:
            q = f"RETURN CASE {subject} WHEN {when} THEN 'm' ELSE 'no' END AS v, {subject} = {when} AS eq"
            branch, actual_eq = self.graph.query(q).result_set[0]

            self.env.assertEqual(actual_eq, eq)
            # the branch is taken exactly when `=` is true, never when it is null
            self.env.assertEqual(branch, 'm' if eq is True else 'no')

        # with no ELSE, an unmatched subject yields null rather than a branch
        res = self.graph.query("RETURN CASE null WHEN null THEN 'm' END AS v")
        self.env.assertEqual(res.result_set, [[None]])

        # the searched form is unaffected: it tests truthiness, not equality
        res = self.graph.query("RETURN CASE WHEN null THEN 'm' ELSE 'no' END AS v")
        self.env.assertEqual(res.result_set, [['no']])

        # a null subject falls through to a later arm that does match
        res = self.graph.query(
            "UNWIND [1, null, 2] AS x "
            "RETURN CASE x WHEN null THEN 'isnull' WHEN 1 THEN 'one' ELSE 'other' END AS v")
        self.env.assertEqual(res.result_set, [['one'], ['other'], ['other']])
