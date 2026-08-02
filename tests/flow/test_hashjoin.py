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

