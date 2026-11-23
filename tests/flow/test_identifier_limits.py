import csv

from common import *
from constraint_utils import create_constraint
from index_utils import *
from click.testing import CliRunner
from falkordb_bulk_loader.bulk_insert import bulk_insert

GRAPH_ID = "identifier_limits"


class testIdentifierLimits:
    def __init__(self):
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.con.delete(GRAPH_ID)
        self.graph = self.db.select_graph(GRAPH_ID)
        self.graph.query("CREATE (:Seed {v: 1})")
        self.port = self.env.envRunner.port

    def _assert_identifier_too_long(self, fn):
        try:
            fn()
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("exceeds maximum length of 512", str(e))

    # GRAPH.CONSTRAINT consumes command arguments directly (non-AST identifier path).
    def test01_constraint_command_identifier_length_limit(self):
        valid_label = "l" * 512
        valid_prop = "p" * 512
        long_name = "a" * 513

        # 512-byte label and property names are accepted.
        create_constraint(self.graph, "MANDATORY", "NODE", valid_label, "p")
        create_constraint(self.graph, "MANDATORY", "NODE", "L2", valid_prop)

        # 513-byte label and property names fail with an explicit query error.
        self._assert_identifier_too_long(
            lambda: create_constraint(self.graph, "MANDATORY", "NODE", long_name, "p")
        )
        self._assert_identifier_too_long(
            lambda: create_constraint(self.graph, "MANDATORY", "NODE", "L3", long_name)
        )

    # Property-map updates consume runtime map keys (non-AST identifier path).
    def test02_set_map_identifier_length_limit(self):
        valid_key_add = "k" * 512
        valid_key_replace = "m" * 512
        long_key = "z" * 513

        # 512-byte map keys are accepted.
        result = self.graph.query(
            "MATCH (n) SET n += $props RETURN n",
            params={"props": {valid_key_add: 1}},
        )
        self.env.assertEqual(result.properties_set, 1)

        result = self.graph.query(
            "MATCH (n) SET n = $props RETURN n",
            params={"props": {valid_key_replace: 2}},
        )
        self.env.assertEqual(result.properties_set, 1)

        # 513-byte map keys fail with an explicit query error.
        self._assert_identifier_too_long(
            lambda: self.graph.query(
                "MATCH (n) SET n += $props RETURN n",
                params={"props": {long_key: 1}},
            )
        )
        self._assert_identifier_too_long(
            lambda: self.graph.query(
                "MATCH (n) SET n = $props RETURN n",
                params={"props": {long_key: 1}},
            )
        )

    # Fulltext index creation enforces the same 512/513 boundary on label/field names.
    def test03_fulltext_procedure_identifier_length_limit(self):
        valid_label = "f" * 512
        valid_field = "g" * 512
        long_name = "h" * 513

        # 512-byte label and field names are accepted.
        result = create_node_fulltext_index(self.graph, valid_label, 'field')
        self.env.assertEqual(result.indices_created, 1)

        result = create_node_fulltext_index(self.graph, 'L_proc', valid_field)
        self.env.assertEqual(result.indices_created, 1)

        # 513-byte names fail with an explicit query error.
        self._assert_identifier_too_long(
            lambda: create_node_fulltext_index(self.graph, long_name, 'field')
        )
        self._assert_identifier_too_long(
            lambda: create_node_fulltext_index(self.graph, 'L_proc2', long_name)
        )

    # CREATE INDEX path should enforce the same 512/513 boundary.
    def test04_create_index_identifier_length_limit(self):
        valid_prop = "i" * 512
        long_prop = "i" * 513

        # 512-byte property name is accepted.
        result = create_node_range_index(self.graph, 'IndexLabel', valid_prop)
        self.env.assertEqual(result.indices_created, 1)

        # 513-byte property name fails.
        self._assert_identifier_too_long(
            lambda: create_node_range_index(self.graph, 'IndexLabel2', long_prop)
        )

    # Function-name validation should accept length 512 and reject 513.
    def test05_function_name_identifier_length_limit(self):
        self.db.udf_flush()

        lib = "Lib"
        valid_func = "f" * 508  # len("Lib") + 1 + 508 = 512
        long_func = "f" * 509  # len("Lib") + 1 + 509 = 513
        script = f"""
        function valid() {{ return 1; }}
        falkor.register('{valid_func}', valid);
        """

        # Register and call a function with a 512-byte qualified name.
        self.db.udf_load(lib, script, True)
        result = self.graph.query(f"RETURN {lib}.{valid_func}()")
        self.env.assertEqual(result.result_set, [[1]])

        # Calling a 513-byte function name fails at identifier-length validation.
        self._assert_identifier_too_long(
            lambda: self.graph.query(f"RETURN {lib}.{long_func}()")
        )

    # GRAPH.UDF LOAD should enforce 512/513 on library names.
    def test06_udf_library_name_identifier_length_limit(self):
        self.db.udf_flush()

        valid_lib = "L" * 512
        long_lib = "L" * 513

        # Use invalid JS script so both calls fail, but only 513 should fail
        # due to identifier length.
        invalid_script = "function broken( {"

        try:
            self.db.udf_load(valid_lib, invalid_script, True)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertFalse("Library name exceeds maximum length of 512" in str(e))

        self._assert_identifier_too_long(
            lambda: self.db.udf_load(long_lib, invalid_script, True)
        )

    # GRAPH.BULK headers should reject overlong label / rel-type / attribute names.
    def test07_bulk_insert_label_identifier_length_limit(self):
        runner = CliRunner()
        bulk_graph = f"{GRAPH_ID}_bulk"
        long_name = "a" * 513

        def assert_bulk_fail(res):
            self.env.assertNotEqual(res.exit_code, 0)
            combined = res.output + (str(res.exception) if res.exception else "")
            self.env.assertContains("exceeds maximum length of 512", combined)

        with runner.isolated_filesystem():
            # long label via --nodes-with-label
            with open('nodes.csv', 'w') as f:
                out = csv.writer(f)
                out.writerow(["name"])
                out.writerow(["node1"])

            res = runner.invoke(bulk_insert, [
                '--server-url', f"redis://{self.env.host}:{self.port}",
                '--nodes-with-label', long_name, 'nodes.csv',
                bulk_graph,
            ])
            assert_bulk_fail(res)

            # long attribute name via CSV column header
            with open('nodes_long_attr.csv', 'w') as f:
                out = csv.writer(f)
                out.writerow([long_name])
                out.writerow(["val1"])

            res = runner.invoke(bulk_insert, [
                '--server-url', f"redis://{self.env.host}:{self.port}",
                '--nodes', 'nodes_long_attr.csv',
                bulk_graph,
            ])
            assert_bulk_fail(res)

            # long rel-type via --relations-with-type
            with open('seed.csv', 'w') as f:
                out = csv.writer(f)
                out.writerow(["_identifier", "name"])
                out.writerow([0, "a"])
                out.writerow([1, "b"])

            with open('rels.csv', 'w') as f:
                out = csv.writer(f)
                out.writerow(["src", "dest"])
                out.writerow([0, 1])

            res = runner.invoke(bulk_insert, [
                '--server-url', f"redis://{self.env.host}:{self.port}",
                '--nodes', 'seed.csv',
                '--relations-with-type', long_name, 'rels.csv',
                bulk_graph,
            ])
            assert_bulk_fail(res)


# Every clause stacks an operator on the previous clause's, and the optimizer
# passes, the runtime's operator build, every `next()` pull and the operator
# tree's drop recurse once per operator. These queries are far taller than a
# 2 MiB worker stack allows without `planner::with_plan_stack` and
# `with_optimizer_stack`: before them, 2000 chained UNWINDs or ~285 OPTIONAL
# MATCH ... WHERE / CALL {} clauses took the server down with a stack overflow.
# C runs them.
#
# Kept as a flow test (not only in tests/test_e2e.py) so the sanitizer flow job
# runs them too: the stack budget is empirical, and an instrumented build has
# larger frames than a release one.

LONG_CHAINS_GRAPH_ID = "long_chains"


def long_chain_unwinds(n):
    return " ".join(f"UNWIND [1] AS x{i}" for i in range(n))


class testLongClauseChains:
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(LONG_CHAINS_GRAPH_ID)
        self.graph.query("CREATE (:N {v: 1})-[:R]->(:N {v: 2}), (:P {id: 0}), (:P {id: 1}), (:P {id: 2})")

    def server_is_up(self):
        self.env.assertEqual(self.graph.query("RETURN 1").result_set, [[1]])

    def test01_chained_long_chain_unwinds(self):
        for n in (500, 2000, 5000):
            res = self.graph.query(f"{long_chain_unwinds(n)} RETURN count(*)")
            self.env.assertEqual(res.result_set, [[1]], message=f"{n} UNWINDs")
        self.server_is_up()

    def test02_chained_with_and_aggregation(self):
        q = "UNWIND [1] AS x0 " + " ".join(
            f"WITH x{i} AS x{i + 1}, count(*) AS c{i}" for i in range(2000)
        )
        self.env.assertEqual(self.graph.query(f"{q} RETURN x2000").result_set, [[1]])
        self.server_is_up()

    def test03_chained_optional_match(self):
        for n, where in ((1000, ""), (1000, " WHERE b{i}.v > 0")):
            q = "MATCH (a:N {v: 1}) " + " ".join(
                f"OPTIONAL MATCH (a)-->(b{i})" + where.format(i=i) for i in range(n)
            )
            res = self.graph.query(f"{q} RETURN count(*)")
            self.env.assertEqual(res.result_set, [[1]], message=f"{n} OPTIONAL MATCHes {where!r}")
        self.server_is_up()

    def test04_chained_call_subqueries(self):
        q = "UNWIND [1] AS x0 " + " ".join(
            f"CALL {{ WITH x{i} RETURN x{i} + 1 AS x{i + 1} }}" for i in range(1000)
        )
        self.env.assertEqual(self.graph.query(f"{q} RETURN x1000").result_set, [[1001]])
        self.server_is_up()

    def test05_chained_merge(self):
        g = self.db.select_graph("long_chains_merge")
        q = " ".join(f"MERGE (a{i}:M {{k: {i}}})" for i in range(1000)) + " RETURN count(*)"
        res = g.query(q)
        self.env.assertEqual(res.result_set, [[1]])
        self.env.assertEqual(res.nodes_created, 1000)
        # the same chain again finds every node
        res = g.query(q)
        self.env.assertEqual(res.nodes_created, 0)
        self.env.assertEqual(g.query("MATCH (m:M) RETURN count(m)").result_set, [[1000]])
        g.delete()

    def test06_wide_cartesian_product_of_equality_joined_patterns(self):
        # N comma-separated patterns joined by a chain of equalities: the
        # optimizer turns it into a left-deep chain of N-1 hash joins, so it
        # gets taller than it comes in.
        def chain(n):
            pats = ", ".join(f"(a{i}:P)" for i in range(n))
            conds = " AND ".join(f"a{i}.id = a{i + 1}.id" for i in range(n - 1))
            return f"MATCH {pats} WHERE {conds} RETURN count(*)"

        for n in (100, 600):
            self.env.assertEqual(self.graph.query(chain(n)).result_set, [[3]], message=f"{n} patterns")
        # taller than anything the optimizer is allowed to build: a clean error
        try:
            self.graph.query(chain(3500))
            self.env.assertTrue(False, message="3500 joined patterns should be rejected")
        except ResponseError as e:
            self.env.assertContains("Query nesting exceeds the maximum depth of 10000", str(e))
        self.server_is_up()

    def test07_wide_union_is_not_a_tall_plan(self):
        # 2000 independent branches, each a two-way Cartesian product: wide, not tall.
        q = " UNION ALL ".join(["MATCH (a:P), (b:P) RETURN 1 AS x"] * 2000)
        res = self.graph.query(q)
        self.env.assertEqual(len(res.result_set), 2000 * 9)
        self.server_is_up()

    def test08_too_tall_is_a_clean_error(self):
        try:
            self.graph.query(f"{long_chain_unwinds(10001)} RETURN count(*)")
            self.env.assertTrue(False, message="10001 UNWINDs should be rejected")
        except ResponseError as e:
            self.env.assertContains("Query nesting exceeds the maximum depth of 10000", str(e))
        self.server_is_up()

    def test09_every_entry_point_and_inline_execution(self):
        q = f"{long_chain_unwinds(3000)} RETURN count(*)"
        self.env.assertEqual(self.graph.ro_query(q).result_set, [[1]])
        # raw replies: the Python client parses a plan recursively
        conn = self.env.getConnection()
        self.env.assertGreater(len(conn.execute_command("GRAPH.EXPLAIN", LONG_CHAINS_GRAPH_ID, q)), 3000)
        self.env.assertGreater(len(conn.execute_command("GRAPH.PROFILE", LONG_CHAINS_GRAPH_ID, q)), 3000)
        # inside MULTI/EXEC the query runs inline on the calling thread
        pipe = conn.pipeline(transaction=True)
        pipe.execute_command("GRAPH.QUERY", LONG_CHAINS_GRAPH_ID, q)
        pipe.execute_command("GRAPH.QUERY", LONG_CHAINS_GRAPH_ID, "RETURN 1")
        out = pipe.execute()
        self.env.assertEqual(out[0][1], [[1]])
        self.server_is_up()

    def test10_plan_cache_hit_runs_the_same_tall_plan(self):
        q = f"{long_chain_unwinds(2500)} RETURN count(*)"
        for _ in range(3):
            self.env.assertEqual(self.graph.query(q).result_set, [[1]])
        self.server_is_up()
