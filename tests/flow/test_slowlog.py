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
                #q = f"UNWIND range(0, 100000) AS x WITH x WHERE x % {i} = 0 RETURN count(x)"
                q = """UNWIND range(0, 100000) AS x
                       WITH x
                       WHERE x % $i = 0
                       RETURN count(x)"""
                tasks.append(asyncio.create_task(g.query(q, {'i':i})))

            await asyncio.gather(*tasks)

            # close the connection pool
            await pool.aclose()

        asyncio.run(populate(self, n))

    def test01_slowlog(self):
        # Slowlog should fail when graph doesn't exists.
        try:
            slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", "NONE_EXISTING_GRAPH")
        except ResponseError as e:
            self.env.assertIn("Invalid graph operation on empty key", str(e))

        # Issue create query twice.
        self.graph.query("""CREATE ()""")
        self.graph.query("""CREATE ()""")

        # Slow log should contain a single entry, no duplicates.
        slowlog = self.graph.slowlog()
        self.env.assertEquals(len(slowlog), 1)

        # Saturate slowlog.
        self.populate_slowlog(128)
        A = self.graph.slowlog()
        B = self.graph.slowlog()

        # Calling slowlog multiple times should preduce the same result.
        self.env.assertEquals(A, B)

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
