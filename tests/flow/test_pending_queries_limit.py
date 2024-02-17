from common import *
from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool
import asyncio

# 1.test getting and setting config
# 2. test overflowing the server when there's a limit
#    expect to get error!
# 3. test overflowing the server when there's no limit
#    expect not to get any exceptions

GRAPH_ID = "max_pending_queries"
SLOW_QUERY = "UNWIND range (0, 1000000) AS x WITH x WHERE (x / 2) = 50 RETURN x"


async def issue_query(self, g, q):
    try:
        await g.query(q)
        return False # no failures
    except Exception as e:
        self.env.assertIn("Max pending queries exceeded", str(e))
        return True # failed due to internal queries queue limit

class testPendingQueryLimit():
    def __init__(self):
        self.env, self.db = Env(moduleArgs="THREAD_COUNT 2")
        self.conn = self.env.getConnection()

    def stress_server(self):
        async def run(self):
            # connection pool with 16 connections
            # blocking when there's no connections available
            n = self.db.config_get("THREAD_COUNT") * 5
            pool = BlockingConnectionPool(max_connections=n, timeout=None)
            db = FalkorDB(host='localhost', port=self.env.port, connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            tasks = []
            for i in range(0, n):
                tasks.append(asyncio.create_task(issue_query(self, g, SLOW_QUERY)))

            results = await asyncio.gather(*tasks)

            # return if error encountered
            return any(results)

        return asyncio.run(run(self))

    def test_01_query_limit_config(self):
        # read max queued queries config
        max_queued_queries = self.db.config_get("MAX_QUEUED_QUERIES")
        self.env.assertEquals(max_queued_queries, 4294967295)

        # update configuration, set max queued queries
        self.db.config_set("MAX_QUEUED_QUERIES", 10)

        # re-read configuration
        result = self.conn.execute_command("GRAPH.CONFIG", "GET", "MAX_QUEUED_QUERIES")
        max_queued_queries = self.db.config_get("MAX_QUEUED_QUERIES")
        self.env.assertEquals(max_queued_queries, 10)

    def test_02_overflow_no_limit(self):
        # no limit on number of pending queries
        limit = 4294967295
        self.db.config_set("MAX_QUEUED_QUERIES", limit)

        error_encountered = self.stress_server()

        self.env.assertFalse(error_encountered)

    def test_03_overflow_with_limit(self):
        # limit number of pending queries
        limit = 1
        self.db.config_set("MAX_QUEUED_QUERIES", limit)

        error_encountered = self.stress_server()

        self.env.assertTrue(error_encountered)
