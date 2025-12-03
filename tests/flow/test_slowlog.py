import asyncio
from common import *
from falkordb.asyncio import FalkorDB
from distutils.version import StrictVersion
from redis.asyncio import BlockingConnectionPool

GRAPH_ID = "slowlog_test"

class testSlowLog():
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def populate_slowlog(self, n):
        async def populate(self, n):
            pool = BlockingConnectionPool(max_connections=n, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            tasks = []
            for i in range(1, n):
                q = f"""UNWIND range(0, 100000) AS x
                       WITH x
                       WHERE x % {i} = 0
                       RETURN count(x)"""
                tasks.append(asyncio.create_task(g.query(q)))

            await asyncio.gather(*tasks)

            # close the connection pool
            await pool.aclose()

        asyncio.run(populate(self, n))

    def test01_slowlog(self):
        # Slowlog should fail when graph doesn't exists
        try:
            slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", "NONE_EXISTING_GRAPH")
        except ResponseError as e:
            self.env.assertIn("Invalid graph operation on empty key", str(e))

        # issue the same query twice
        q = "UNWIND range (0, 200000) AS x RETURN max(x)"
        self.graph.query(q)
        self.graph.query(q)

        # slow log should contain a single entry, no duplicates
        slowlog = self.graph.slowlog()
        self.env.assertEquals(len(slowlog), 1)

        # saturate slowlog
        self.populate_slowlog(20)
        A = self.graph.slowlog()
        B = self.graph.slowlog()

        # calling slowlog multiple times should preduce the same result
        self.env.assertEquals(A, B)
        self.env.assertEquals(len(A), 10)

        server = self.redis_con.info("Server")
        if StrictVersion(server["redis_version"]) < StrictVersion("6.2.0"):
            # redis < 6.2.0 not support slowlog time measure
            return

        # Issue a long running query, this should replace an existing entry in the slowlog.
        q = "UNWIND range(0, 1000) AS i UNWIND range(0, 1000) AS j WITH i, j WHERE i > 0 AND j < 500 RETURN SUM(i + j)"

        self.graph.query(q)
        B = self.graph.slowlog()

        self.env.assertNotEqual(A, B)

        # get redis slowlog
        slowlog = self.redis_con.slowlog_get()
        slowlog_commands = [log["command"].decode('ascii') for log in slowlog]

        # validate the command added to redis slowlog
        self.env.assertGreater(len(slowlog), 0)
        self.env.assertContains(f"GRAPH.QUERY slowlog_test {q} --compact", slowlog_commands)

    def test02_slowlog_reset(self):
        # reset none existing slowlog
        try:
            slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", "NONE_EXISTING_GRAPH", "RESET")
        except ResponseError as e:
            self.env.assertIn("Invalid graph operation on empty key", str(e))

        # issue an unknown slowlog sub command
        try:
            slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "UNKNOW_SUB_CMD")
        except ResponseError as e:
            self.env.assertIn("Unknown subcommand", str(e))

        # populate slowlog
        self.populate_slowlog(36)
        slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID)
        self.env.assertGreater(len(slowlog), 0)

        # clear slowlog
        # make sure there's no harm in double reseting
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        # expecting an empty slowlog
        slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID)
        self.env.assertEquals(len(slowlog), 0)

        # make sure slowlog repopulates after RESET
        self.populate_slowlog(36)
        slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID)
        self.env.assertGreater(len(slowlog), 0)

    def test03_cap_entry(self):
        # make sure slowlog entries are capped

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        #-----------------------------------------------------------------------
        # truncated query
        #-----------------------------------------------------------------------

        long_string = 'a' * 4000
        query = f"WITH '{long_string}' AS str UNWIND range(0, 200000) AS x RETURN count(x)"
        self.graph.query(query)

        slowlog = self.graph.slowlog()
        entry = slowlog[0]
        cmd     = entry[1]
        q       = entry[2]
        params  = entry[3]

        self.env.assertEquals(cmd, "GRAPH.QUERY")
        self.env.assertEquals(params, None)

        # reported query should be truncated
        self.env.assertIn("...", q)
        self.env.assertLess(len(q), len(query))

        #-----------------------------------------------------------------------
        # truncated params
        #-----------------------------------------------------------------------

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        query = "WITH $long_string AS str UNWIND range(0, 200000) AS x RETURN count(x)"
        self.graph.query(query, {'long_string': long_string})

        slowlog = self.graph.slowlog()
        entry = slowlog[0]
        cmd     = entry[1]
        q       = entry[2]
        params  = entry[3]

        self.env.assertEquals(cmd, "GRAPH.QUERY")
        self.env.assertEquals(query, q)
        self.env.assertIn("...", params)

        # reported param should be truncated
        self.env.assertLess(len(params), len(long_string))

        #-----------------------------------------------------------------------
        # truncated query & params
        #-----------------------------------------------------------------------

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        query = f"WITH $long_string as long_param, '{long_string}' AS long_string UNWIND range(0, 200000) AS x RETURN count(x)"
        self.graph.query(query, {'long_string': long_string})

        slowlog = self.graph.slowlog()
        entry = slowlog[0]
        cmd     = entry[1]
        q       = entry[2]
        params  = entry[3]

        self.env.assertEquals(cmd, "GRAPH.QUERY")

        # reported query should be truncated
        self.env.assertIn("...", q)
        self.env.assertLess(len(q), len(query))

        # reported param should be truncated
        self.env.assertIn("...", params)
        self.env.assertLess(len(params), len(long_string))

    def test04_same_query_diff_params(self):
        # make sure no new entries are added when the query remains the same
        # but the params change

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        query = f"UNWIND range(0, $i) AS x RETURN count(x)"
        self.graph.query(query, {'i': 200000})

        slowlog = self.graph.slowlog()
        self.env.assertEquals(len(slowlog), 1)

        entry = slowlog[0]
        q0 = entry[2]
        p0 = entry[3]

        # re-issue the same query but with different params
        query = f"UNWIND range(0, $i) AS x RETURN count(x)"
        self.graph.query(query, {'i': 400000})

        slowlog = self.graph.slowlog()
        self.env.assertEquals(len(slowlog), 1)

        entry = slowlog[0]
        q1 = entry[2]
        p1 = entry[3]

        # expecting the same query
        self.env.assertEquals(q0, q1)

        # expecting params to update
        self.env.assertNotEqual(p0, p1)
        self.env.assertIn('400000', p1)

    def test05_fast_queries(self):
        # make sure fast queries do not enter the slowlog

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        # query too fast for slowlog
        q = "RETURN 1"
        self.graph.query(q)

        slowlog = self.graph.slowlog()
        self.env.assertEquals(len(slowlog), 0)

    def test06_force_replace(self):
        # make sure slowlog entries get replcaed

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        # fill slowlog
        self.populate_slowlog(20)
        entries = self.graph.slowlog()

        # expecting 10 entries
        self.env.assertEquals(len(entries), 10)

        # issue 2 slower queries
        # expecting to have them replace existing entries

        q0 = "UNWIND range(0, 200000) AS x WITH x WHERE x % 1 = 0 RETURN count(x)"
        self.graph.query(q0)

        q1 = "UNWIND range(0, 250000) AS x WITH x WHERE x % 1 = 0 RETURN count(x)"
        self.graph.query(q1)

        entries = self.graph.slowlog()

        # expecting 10 entries
        self.env.assertEquals(len(entries), 10)

        # make sure both q0 & q1 are in entries
        queries = [entry[2] for entry in entries]
        self.env.assertIn (q0, queries)
        self.env.assertIn (q1, queries)

