from common import *

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

GRAPH_ID = "long_chains"


def unwinds(n):
    return " ".join(f"UNWIND [1] AS x{i}" for i in range(n))


class testLongClauseChains(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.graph.query("CREATE (:N {v: 1})-[:R]->(:N {v: 2}), (:P {id: 0}), (:P {id: 1}), (:P {id: 2})")

    def server_is_up(self):
        self.env.assertEqual(self.graph.query("RETURN 1").result_set, [[1]])

    def test01_chained_unwinds(self):
        for n in (500, 2000, 5000):
            res = self.graph.query(f"{unwinds(n)} RETURN count(*)")
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
            self.graph.query(f"{unwinds(10001)} RETURN count(*)")
            self.env.assertTrue(False, message="10001 UNWINDs should be rejected")
        except ResponseError as e:
            self.env.assertContains("Query nesting exceeds the maximum depth of 10000", str(e))
        self.server_is_up()

    def test09_every_entry_point_and_inline_execution(self):
        q = f"{unwinds(3000)} RETURN count(*)"
        self.env.assertEqual(self.graph.ro_query(q).result_set, [[1]])
        # raw replies: the Python client parses a plan recursively
        conn = self.env.getConnection()
        self.env.assertGreater(len(conn.execute_command("GRAPH.EXPLAIN", GRAPH_ID, q)), 3000)
        self.env.assertGreater(len(conn.execute_command("GRAPH.PROFILE", GRAPH_ID, q)), 3000)
        # inside MULTI/EXEC the query runs inline on the calling thread
        pipe = conn.pipeline(transaction=True)
        pipe.execute_command("GRAPH.QUERY", GRAPH_ID, q)
        pipe.execute_command("GRAPH.QUERY", GRAPH_ID, "RETURN 1")
        out = pipe.execute()
        self.env.assertEqual(out[0][1], [[1]])
        self.server_is_up()

    def test10_plan_cache_hit_runs_the_same_tall_plan(self):
        q = f"{unwinds(2500)} RETURN count(*)"
        for _ in range(3):
            self.env.assertEqual(self.graph.query(q).result_set, [[1]])
        self.server_is_up()
