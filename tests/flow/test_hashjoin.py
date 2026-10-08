from common import *

GRAPH_ID = "hashjoin"


class testHashJoin(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def test_multi_hashjoins(self):
        # See issue https://github.com/RedisGraph/RedisGraph/issues/1124
        # Construct a 4 node graph, (v1),(v2),(v3),(v4)
        self.graph.query("CREATE ({val:1}), ({val:2}), ({val:3}), ({val:4})")

        # Find nodes a,b,c such that a.v = 1, a.v = b.v-1 and b.v = c.v-1
        q = "MATCH (a {val:1}), (b), (c) WHERE a.val = b.val-1 AND b.val = c.val-1 RETURN a.val, b.val, c.val"
        plan = str(self.graph.explain(q))

        # Make sure plan contains 2 Value Hash Join operations
        self.env.assertEqual(plan.count("Value Hash Join"), 2)

        # Validate results
        expected_result = [[1,2,3]]
        actual_result = self.graph.query(q)

        self.env.assertEqual(actual_result.result_set, expected_result)

    def test_argument_injection(self):
        # make sure ValueHashJoin is taken into account
        # when searching for Argument operations for data injection

        q = "CREATE (:A {id: 1}), (:B {id:2, A_id: 1})"
        self.graph.query(q)

        q = """UNWIND [1] AS current_id
        MATCH (a:A {id: current_id}), (b:B)
        WHERE b['A_id'] = a.id
        MERGE (a)-[:R]->(b)
        RETURN *
        """

        res = self.graph.query(q)
        self.env.assertEqual(res.relationships_created, 1)

        queries = [
            """UNWIND [1] AS current_id
            MATCH (a:A {id: current_id}), (b:B)
            WHERE b.A_id = a.id
            CREATE (a)-[:R]->(b)
            RETURN *
            """, 

             """UNWIND [1] AS current_id
            MATCH (a:A {id: current_id}), (b:B)
            WHERE b.A_id = a['id']
            CREATE (a)-[:R]->(b)
            RETURN *
            """, 

            """UNWIND [1] AS current_id
            MATCH (a:A {id: current_id}), (b:B)
            WHERE b['A_id'] = a['id']
            CREATE (a)-[:R]->(b)
            RETURN *
            """
        ]

        for q in queries:
            res = self.graph.query(q)
            self.env.assertEqual(res.relationships_created, 1)

    def test_probe_batch_boundary(self):
        # Regression: a probe (left) row whose matches exactly fill an output
        # batch (BATCH_SIZE = 1024) must not cause the following probe row to be
        # skipped. With L{k:1} producing exactly 1024 matches, draining them
        # fills the builder to the batch boundary and advances to the next probe
        # row before returning; a prior bug then re-advanced the probe cursor on
        # the next pull, dropping the L{k:2} group entirely.
        self.graph.query("CREATE (:L {k: 1}), (:L {k: 2})")
        self.graph.query("UNWIND range(1, 1024) AS i CREATE (:R {k: 1})")
        self.graph.query("CREATE (:R {k: 2})")

        q = """MATCH (a:L), (b:R)
               WHERE a.k = b.k
               RETURN a.k AS ak, count(b) AS cnt
               ORDER BY ak"""

        # Ensure the query is actually planned as a Value Hash Join so the
        # regression keeps exercising the batch-boundary path.
        plan = str(self.graph.explain(q))
        self.env.assertEqual(plan.count("Value Hash Join"), 1)

        actual_result = self.graph.query(q)
        self.env.assertEqual(actual_result.result_set, [[1, 1024], [2, 1]])

    def test_unique_and_duplicate_keys(self):
        """The build side keeps one slot per distinct key, inline for the common
           case of a key that appears once and spilling to the heap only for
           duplicates. Both have to answer identically, so this joins against a
           near-unique key and a heavily duplicated one over the same rows, and
           spans more than one batch either way.

           The inline case used to be a heap allocation per distinct key: over a
           10,000-row build side, 1,646,215 bytes allocated when every key was
           distinct against 12,423 when the same rows carried five."""

        g = self.graph
        n = 3000
        g.query(f"UNWIND range(1, {n}) AS i CREATE (:R {{uniq: i, few: i % 5}})")
        g.query("CREATE (:L {k: 7}), (:L {k: 2}), (:L {k: 999999})")

        # unique keys: one build slot each, matched one at a time
        res = g.query("MATCH (l:L) MATCH (r:R) WHERE r.uniq = l.k RETURN l.k, count(r) ORDER BY l.k")
        self.env.assertEqual(res.result_set, [[2, 1], [7, 1]])

        # duplicated keys: one slot holding many rows
        res = g.query("MATCH (l:L) MATCH (r:R) WHERE r.few = l.k RETURN l.k, count(r) ORDER BY l.k")
        self.env.assertEqual(res.result_set, [[2, n // 5]])

        # a key on no build row joins nothing
        res = g.query("MATCH (l:L) MATCH (r:R) WHERE r.uniq = 999999 RETURN count(r)")
        self.env.assertEqual(res.result_set, [[0]])

        # NULL never joins, on either side
        g.query("CREATE (:L {k: null})")
        res = g.query("MATCH (l:L) MATCH (r:R) WHERE r.uniq = l.k RETURN count(r)")
        self.env.assertEqual(res.result_set, [[2]])

        # a non-integer key promotes the table off the integer fast path and
        # must still agree
        g.query(f"UNWIND range(1, {n}) AS i CREATE (:S {{k: toString(i)}})")
        res = g.query("MATCH (s:S) MATCH (t:S) WHERE s.k = t.k RETURN count(s)")
        self.env.assertEqual(res.result_set, [[n]])

    def test_join_inside_batched_subplan_keeps_outer_rows_apart(self):
        # Under a batched Apply / Optional the join's argument batch holds
        # every outer row; a left row must only join right rows of its own
        # outer row. It joined across all of them (#3012).
        g = self.graph
        g.query("UNWIND [1, 2] AS i CREATE (:A {v: 1, w: i}), (:B {v: 1, w: i})")

        res = g.query("UNWIND [1, 2] AS x MATCH (a:A), (b:B) WHERE a.v = b.v RETURN count(*)")
        self.env.assertEqual(res.result_set, [[8]])

        res = g.query("""UNWIND [1, 2] AS x
                         OPTIONAL MATCH (a:A), (b:B) WHERE a.v = b.v AND a.w = x AND b.w = x
                         RETURN x, a.w, b.w ORDER BY x""")
        self.env.assertEqual(res.result_set, [[1, 1, 1], [2, 2, 2]])

        # more outer rows than one batch, correlated on both sides
        g.query("UNWIND range(1, 3000) AS i CREATE (:C {v: i % 10, w: i})")
        res = g.query("""UNWIND range(1, 3000) AS x
                         MATCH (a:C {w: x}), (b:C) WHERE b.v = a.v AND b.w <= 20
                         RETURN count(*)""")
        self.env.assertEqual(res.result_set, [[6000]])
