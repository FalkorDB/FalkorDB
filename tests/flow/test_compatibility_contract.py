import re
import time

from common import *


class testCompatibilityContract(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def _query(self, graph_name, query):
        return self.conn.execute_command("GRAPH.QUERY", graph_name, query)

    def _expect_error(self, command, contains):
        try:
            command()
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertIn(contains, str(e))

    def _command_flags(self, command):
        info = self.conn.execute_command("COMMAND", "INFO", command)
        if isinstance(info, dict):
            command_info = next(iter(info.values()))
            return set(command_info["flags"])
        return set(info[0][2])

    def _assert_no_graph_version_stat(self, result):
        stats = result[-1]
        self.env.assertTrue(isinstance(stats, list))
        for stat in stats:
            self.env.assertFalse(stat.startswith("Graph version"))

    def test01_raw_response_contract(self):
        graph_name = "compat_raw_response"
        self.conn.delete(graph_name)

        result = self._query(graph_name, "RETURN 1 AS x")
        self.env.assertEqual(result[0], ["x"])
        self.env.assertEqual(result[1], [[1]])
        self._assert_no_graph_version_stat(result)

        result = self._query(graph_name, "RETURN [1,2] AS xs, {a:1,b:'x'} AS m")
        self.env.assertEqual(result[0], ["xs", "m"])
        self.env.assertEqual(result[1], [["[1, 2]", "{a: 1, b: x}"]])
        self._assert_no_graph_version_stat(result)

        self._query(graph_name, "CREATE (:N {a:1})")
        result = self._query(graph_name, "MATCH (n:N) RETURN n")
        node = dict(result[1][0][0])
        self.env.assertEqual(node["labels"], ["N"])
        self.env.assertEqual(node["properties"], [["a", 1]])
        self._assert_no_graph_version_stat(result)

        result = self._query(graph_name, "RETURN vecf32([1,2,3]) AS v")
        vector = result[1][0][0]
        self.env.assertTrue(re.fullmatch(r"<1\.0+, 2\.0+, 3\.0+>", vector) is not None)
        self._assert_no_graph_version_stat(result)

    def test02_mutation_stats_and_constraint_status(self):
        index_graph = "compat_index_stats"
        self.conn.delete(index_graph)

        result = self._query(index_graph, "CREATE INDEX FOR (n:N) ON (n.a)")
        stats = result[-1]
        self.env.assertIn("Indices created: 1", stats)
        self.env.assertNotIn("Labels added: 1", stats)
        self._assert_no_graph_version_stat(result)

        constraint_graph = "compat_constraint_status"
        self.conn.delete(constraint_graph)
        self._query(constraint_graph, "CREATE (:Person {id:1})")
        self._query(constraint_graph, "CREATE INDEX FOR (n:Person) ON (n.id)")
        result = self.conn.execute_command(
            "GRAPH.CONSTRAINT",
            "CREATE",
            constraint_graph,
            "UNIQUE",
            "NODE",
            "Person",
            "PROPERTIES",
            1,
            "id",
        )
        self.env.assertEqual(result, "PENDING")

        self._expect_error(
            lambda: self.conn.execute_command("GRAPH.CONSTRAINT", "LIST", constraint_graph),
            "wrong number of arguments for 'graph.CONSTRAINT' command",
        )

    def test03_command_metadata_flags(self):
        self.env.assertIn("denyoom", self._command_flags("graph.query"))
        self.env.assertIn("denyoom", self._command_flags("graph.profile"))
        self.env.assertIn("denyoom", self._command_flags("graph.constraint"))
        self.env.assertIn("denyoom", self._command_flags("graph.copy"))

        self.env.assertIn("allow_busy", self._command_flags("graph.config"))
        self.env.assertIn("allow_busy", self._command_flags("graph.slowlog"))
        self.env.assertNotIn("allow_busy", self._command_flags("graph.query"))

    def test04_plan_roots(self):
        graph_name = "compat_plan_roots"
        self.conn.delete(graph_name)
        self._query(graph_name, "RETURN 1")

        explain = self.conn.execute_command("GRAPH.EXPLAIN", graph_name, "RETURN 1")
        self.env.assertGreater(len(explain), 0)
        self.env.assertEqual(explain[0], "Results")

        profile = self.conn.execute_command("GRAPH.PROFILE", graph_name, "RETURN 1")
        self.env.assertGreater(len(profile), 0)
        self.env.assertTrue(profile[0].startswith("Results"))
        self.env.assertIn("Records produced", profile[0])

    def test05_function_catalog_contract(self):
        graph_name = "compat_functions"
        self.conn.delete(graph_name)
        self._query(graph_name, "RETURN 1")

        result = self.conn.execute_command(
            "GRAPH.RO_QUERY",
            graph_name,
            "CALL dbms.functions() YIELD name RETURN name ORDER BY name",
        )
        self.env.assertEqual(result[0], ["name"])
        names = {row[0] for row in result[1]}
        for name in [
            "endNode",
            "property",
            "randomuuid",
            "startNode",
            "string.matchRegEx",
            "typeof",
            "xor",
        ]:
            self.env.assertIn(name, names)
        self._assert_no_graph_version_stat(result)

    def test06_temporal_month_contract(self):
        graph_name = "compat_temporal"
        self.conn.delete(graph_name)

        result = self._query(
            graph_name,
            "RETURN duration('P1M') AS d, toString(duration('P1M')) AS s",
        )
        self.env.assertEqual(result[1], [["P1M", "P1M"]])

        result = self._query(
            graph_name,
            """
            RETURN date('2024-01-31') + duration('P1M') AS d,
                   localdatetime('2024-01-31T00:00:00') + duration('P1M') AS l
            """,
        )
        self.env.assertEqual(result[1], [["2024-03-02", "2024-03-02T00:00:00"]])

    def test07_error_and_boundary_contracts(self):
        graph_name = "compat_errors"
        self.conn.delete(graph_name)

        result = self._query(graph_name, "RETURN abs(-9223372036854775808) AS v")
        self.env.assertEqual(result[1], [[-9223372036854775808]])

        self._expect_error(
            lambda: self._query(graph_name, "RETURN 'abc' =~ 'a.*'"),
            "FalkorDB does not currently support =~",
        )
        self._expect_error(
            lambda: self.conn.execute_command("GRAPH.CONFIG", "SET", "ASYNC_DELETE", "1"),
            "Failed to set config value ASYNC_DELETE to 1",
        )
        for query in [" ", ";"]:
            self._expect_error(
                lambda q=query: self._query(graph_name, q),
                "Error: could not parse query",
            )

        non_graph_key = "compat_not_graph"
        self.conn.set(non_graph_key, 1)
        self._expect_error(
            lambda: self.conn.execute_command("GRAPH.DELETE", non_graph_key),
            "Invalid graph operation on empty key",
        )
        self.conn.delete(non_graph_key)

        self._query(graph_name, "RETURN 1")
        self._expect_error(
            lambda: self.conn.execute_command("GRAPH.COPY", graph_name, graph_name),
            "destination key already exists",
        )

        self._expect_error(
            lambda: self._query(graph_name, "CALL db.labels() YIELD missing RETURN missing"),
            "Procedure `db.labels` does not yield output `missing`",
        )
        self._expect_error(
            lambda: self._query(graph_name, "CALL db.labels() YIELD label, label RETURN label"),
            "Variable `label` already declared",
        )
        self._expect_error(
            lambda: self._query(graph_name, "CALL db.noSuchProc()"),
            "Procedure `db.noSuchProc` is not registered",
        )

    def test08_indexes_catalog_contract(self):
        graph_name = "compat_indexes"
        self.conn.delete(graph_name)

        self._query(graph_name, "CALL db.idx.fulltext.createNodeIndex('Person', 'name')")
        result = None
        for _ in range(20):
            result = self._query(graph_name, "CALL db.indexes()")
            if result[1] and dict(zip(result[0], result[1][0]))["status"] == "OPERATIONAL":
                break
            time.sleep(0.1)

        expected_header = [
            "label",
            "properties",
            "types",
            "options",
            "language",
            "stopwords",
            "entitytype",
            "status",
            "info",
        ]
        self.env.assertEqual(result[0][: len(expected_header)], expected_header)

        row_by_column = dict(zip(result[0], result[1][0]))
        self.env.assertEqual(row_by_column["label"], "Person")
        self.env.assertEqual(row_by_column["properties"], "[name]")
        self.env.assertEqual(row_by_column["types"], "{name: [FULLTEXT]}")
        self.env.assertEqual(row_by_column["options"], "{name: {}}")
        self.env.assertEqual(row_by_column["language"], "english")
        self.env.assertEqual(row_by_column["stopwords"], "[]")
        self.env.assertEqual(row_by_column["entitytype"], "NODE")
        self.env.assertEqual(row_by_column["status"], "OPERATIONAL")
        self._assert_no_graph_version_stat(result)

    def test09_slowlog_timestamp_contract(self):
        graph_name = "compat_slowlog"
        self.conn.delete(graph_name)
        self._query(graph_name, "RETURN 1")
        self.conn.execute_command("GRAPH.SLOWLOG", graph_name, "RESET")

        query = "UNWIND range(0, 200000) AS x RETURN max(x)"
        self._query(graph_name, query)

        slowlog = []
        for _ in range(10):
            slowlog = self.conn.execute_command("GRAPH.SLOWLOG", graph_name)
            if slowlog:
                break
            time.sleep(0.1)

        self.env.assertGreater(len(slowlog), 0)
        timestamp = slowlog[0][0]
        self.env.assertTrue(re.fullmatch(r"\d+", timestamp) is not None)
