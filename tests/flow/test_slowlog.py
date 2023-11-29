import asyncio
from common import *
from distutils.version import StrictVersion

GRAPH_ID = "slowlog_test"

class testSlowLog():
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def populate_slowlog(self):
        def _run_query():
            g = self.db.select_graph(GRAPH_ID)
            for i in range(1, 4):
                q = """UNWIND range(0, 1000000) AS x WITH x WHERE x % {mod} = 0 RETURN count(x)""".format(mod=i)
                g.query(q)

        if "to_thread" not in dir(asyncio):
            _run_query()
        else:
            loop = asyncio.get_event_loop()
            tasks = []
            for i in range(1, 6):
                tasks.append(loop.create_task(asyncio.to_thread(_run_query)))

            loop.run_until_complete(asyncio.wait(tasks))

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
        for i in range(1024):
            q = """CREATE ({v:%s})""" % i
            self.graph.query(q)
        A = self.graph.slowlog()
        B = self.graph.slowlog()

        # Calling slowlog multiple times should preduce the same result.
        self.env.assertEquals(A, B)

        server = self.redis_con.info("Server")
        if StrictVersion(server["redis_version"]) < StrictVersion("6.2.0"):
            # redis < 6.2.0 not support slowlog time measure
            return

        # Issue a long running query, this should replace an existing entry in the slowlog.
        q = """MATCH (n), (m) WHERE n.v > 0 AND n.v < 500 SET m.v = rand() WITH n, m RETURN SUM(n.v + m.v)"""
        self.graph.query(q)
        B = self.graph.slowlog()

        self.env.assertNotEqual(A, B)

        # get redis slowlog
        slowlog = self.redis_con.slowlog_get()
        slowlog_commands = [log["command"] for log in slowlog]

        # validate the command added to redis slowlog
        self.env.assertGreater(len(slowlog), 0)
        self.env.assertContains(b"GRAPH.QUERY slowlog_test MATCH (n), (m) WHERE n.v > 0 AND n.v < 500 SET m.v = rand() WITH n, m RETURN SUM(n.v + m.v) --compact", slowlog_commands)

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
        self.populate_slowlog()
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
        self.populate_slowlog()
        slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID)
        self.env.assertGreater(len(slowlog), 0)
