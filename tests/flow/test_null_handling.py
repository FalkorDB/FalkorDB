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
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

        try:
            query = """MATCH (a) OPTIONAL MATCH (a)-[nonexistent_edge]->(nonexistent_node) CREATE (a)-[:E]->(nonexistent_node)"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    # Error when attempting to merge a relationship with a null endpoint.
    def test02_merge_null(self):
        try:
            query = """MATCH (a) OPTIONAL MATCH (a)-[nonexistent_edge]->(nonexistent_node) MERGE (nonexistent_node)-[:E]->(a)"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

        try:
            query = """MATCH (a) OPTIONAL MATCH (a)-[nonexistent_edge]->(nonexistent_node) MERGE (a)-[:E]->(nonexistent_node)"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    # SET should update attributes on non-null entities and ignore null entities.
    def test03_set_null(self):
        query = """MATCH (a) OPTIONAL MATCH (a)-[nonexistent_edge]->(nonexistent_node) SET a.v2 = true, nonexistent_node.v2 = true, a.v3 = nonexistent_node.v3 RETURN a.v2, nonexistent_node.v2, a.v3"""
        actual_result = self.graph.query(query)
        # The property should be set on the real node and ignored on the null entity.
        assert(actual_result.properties_set == 1)
        expected_result = [[True, None, None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

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
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Path functions should handle null inputs appropriately.
    def test06_null_named_path_function_inputs(self):
        query = """MATCH (a) OPTIONAL MATCH p = (a)-[r]->() RETURN p, length(p), collect(relationships(p))"""
        actual_result = self.graph.query(query)
        # The path and function calls on it should return NULL, while collect() returns an empty array.
        expected_result = [[None, None, []]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Scan and traversal operations should gracefully handle NULL inputs.
    def test07_null_graph_entity_inputs(self):
        query = """WITH NULL AS a MATCH (a) RETURN a"""
        actual_result = self.graph.query(query)
        # Expect one NULL entity to be returned.
        expected_result = [[None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """WITH NULL AS a MATCH (a)-[e]->(b) RETURN a, e, b"""
        plan = str(self.graph.explain(query))
        # Verify that we are attempting to perform a traversal but no scan.
        self.env.assertNotIn("Scan", plan)
        self.env.assertIn("Conditional Traverse", plan)
        actual_result = self.graph.query(query)
        # Expect no results.
        expected_result = []
        self.env.assertEquals(actual_result.result_set, expected_result)

        try:
            query = """WITH NULL AS e MATCH (a:L)-[e]->(b) RETURN a, e, b"""
            plan = self.graph.explain(query)
            # not expecting to reach this point
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("The bound edge 'e' can't be redeclared in a MERGE clause", str(e))

    # ValueHashJoin ops should not treat null values as equal.
    def test08_null_value_hash_join(self):
        query = """MATCH (a), (b) WHERE a.fakeval = b.fakeval RETURN a, b"""
        plan = str(self.graph.explain(query))
        # Verify that we are performing a ValueHashJoin
        self.env.assertIn("Value Hash Join", plan)
        actual_result = self.graph.query(query)
        # Expect no results.
        expected_result = []
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Perform a sanity check on a ValueHashJoin that returns a result
        query = """MATCH (a), (b) WHERE a.v = b.v RETURN a.v, b.v"""
        actual_result = self.graph.query(query)
        expected_result = [['v1', 'v1']]
        self.env.assertEquals(actual_result.result_set, expected_result)
