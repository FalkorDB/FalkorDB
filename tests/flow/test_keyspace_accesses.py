from common import *

from tests.flow.query_info import QueryInfo

GRAPH_ID = "G"
NEW_GRAPH_ID = "G2"


class testKeyspaceAccesses(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
    
    def test00_test_data_valid_after_rename(self):
        node0 = Node(node_id=0, labels="L", properties={'name':'x', 'age':1})
        self.graph.query(f"CREATE {node0}")
        self.redis_con.rename(GRAPH_ID, NEW_GRAPH_ID)

        self.graph = self.db.select_graph(NEW_GRAPH_ID)

        node1 = Node(node_id=1, labels="L", properties={'name':'x', 'age':1})
        self.graph.query(f"CREATE {node1}")

        query = "MATCH (n) return n"
        expected_results = [[node0], [node1]]
        query_info = QueryInfo(query = query, description="Tests data is valid after renaming", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(self.graph.query(query), query_info)

    # Graph queries should fail gracefully on accessing non-graph keys.
    def test01_graph_access_on_invalid_key(self):
        self.redis_con.set("integer_key", 5)
        self.graph = self.db.select_graph("integer_key")
        try:
            query = """MATCH (n) RETURN noneExistingFunc(n.age) AS cast"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            self.env.assertContains("Existing key has wrong Redis type", str(e))
            pass

        try:
            self.redis_con.execute_command("GRAPH.DELETE", "integer_key")
            assert(False)
        except redis.ResponseError as e:
            self.env.assertContains("Existing key has wrong Redis type", str(e))

    # Fail gracefully on attempting a graph deletion of an empty key.
    def test02_graph_delete_on_empty_key(self):
        self.graph = self.db.select_graph("nonexistent_key")
        try:
            self.graph.delete()
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            assert("empty key" in str(e))
            pass

    # A command that finds a non-graph value where it reads a graph, or where it would
    # store a new one, replies the same words either way, on Redis and Valkey alike.
    # The binding crate checks the key's type in `get_value` and again in `set_value`,
    # and words that error "wrong Valkey type"; clients have always seen "wrong Redis
    # type".
    def test03_wrong_type_wording_on_read_and_create(self):
        WRONG_TYPE = "Existing key has wrong Redis type"
        self.redis_con.set("string_key", "x")

        # The named key holds a string.
        reads = [
            ["GRAPH.QUERY", "string_key", "RETURN 1"],
            ["GRAPH.RO_QUERY", "string_key", "RETURN 1"],
            ["GRAPH.PROFILE", "string_key", "RETURN 1"],
            ["GRAPH.EXPLAIN", "string_key", "RETURN 1"],
            ["GRAPH.RECORD", "string_key", "RETURN 1"],
            ["GRAPH.DELETE", "string_key"],
            ["GRAPH.SLOWLOG", "string_key"],
            ["GRAPH.MEMORY", "USAGE", "string_key"],
            ["GRAPH.BULK", "string_key", "BEGIN", 0, 0, 0, 0],
            ["GRAPH.CONSTRAINT", "DROP", "string_key", "UNIQUE", "NODE", "L",
             "PROPERTIES", 1, "p"],
            ["GRAPH.EFFECT", "string_key", b"\x03\x00"],
            ["GRAPH.COPY", "string_key", "copy_dest"],
            ["GRAPH.RESTORE", "string_key", b"x"],
        ]
        # The named key is empty, so each of these creates a graph, and stores it at
        # the key truncated at its first NUL (C's key for the graph's name). That key
        # holds a string, so the write is what fails.
        create_key = "string_key\x00tail"
        creates = [
            ["GRAPH.QUERY", create_key, "RETURN 1"],
            ["GRAPH.PROFILE", create_key, "RETURN 1"],
            ["GRAPH.RECORD", create_key, "RETURN 1"],
            ["GRAPH.BULK", create_key, "BEGIN", 0, 0, 0, 0],
            ["GRAPH.CONSTRAINT", "CREATE", create_key, "UNIQUE", "NODE", "L",
             "PROPERTIES", 1, "p"],
            ["GRAPH.EFFECT", create_key, b"\x03\x00"],
        ]
        # One connection per command: GRAPH.QUERY and GRAPH.PROFILE run the query
        # before they store the new graph, so a failed store is followed by the query's
        # own reply. On a shared connection the next command would read it.
        for cmd in reads + creates:
            con = self.env.getConnection()
            try:
                con.execute_command(*cmd)
                self.env.assertTrue(False, message=f"{cmd} succeeded")
            except redis.ResponseError as e:
                self.env.assertEqual(str(e), WRONG_TYPE, message=f"{cmd}")
            finally:
                con.close()

        # Nothing overwrote the string or left a graph behind.
        self.env.assertEqual(self.redis_con.get("string_key"), "x")
        self.env.assertEqual(self.redis_con.exists(create_key, "copy_dest"), 0)
        self.env.assertFalse("string_key" in self.redis_con.execute_command("GRAPH.LIST"))
        self.redis_con.delete("string_key")
