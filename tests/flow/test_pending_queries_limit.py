from common import *
from pathos.pools import ProcessPool as Pool

# 1.test getting and setting config
# 2. test overflowing the server when there's a limit
#    expect to get error!
# 3. test overflowing the server when there's no limit
#    expect not to get any exceptions

GRAPH_NAME = "max_pending_queries"
SLOW_QUERY = "UNWIND range (0, 1000000) AS x WITH x WHERE (x / 2) = 50 RETURN x"


def issue_query(conn, q):
    try:
        conn.execute_command("GRAPH.QUERY", GRAPH_NAME, q)
        return False
    except Exception as e:
        assert "Max pending queries exceeded" in str(e)
        return True

class testPendingQueryLimit():
    def __init__(self):
        self.env, self.db = Env(moduleArgs="THREAD_COUNT 2")
        # skip test if we're running under Valgrind
        if VALGRIND:
            self.env.skip() # valgrind is not working correctly with multi process

        self.conn = self.env.getConnection()

    def stress_server(self):
        threadpool_size = self.db.config_get("THREAD_COUNT")
        thread_count = threadpool_size * 5
        qs = [SLOW_QUERY] * thread_count
        connections = []
        pool = Pool(nodes=thread_count)

        # init connections
        for i in range(thread_count):
            connections.append(self.env.getConnection())

        # invoke queries
        result = pool.map(issue_query, connections, qs)
       
        pool.clear()

        # return if error encountered
        return any(result)

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
