from common import *
import random
import asyncio
from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool

# 1. test reading and setting query memory limit configuration

# 2. test overflowing the server when there's no limit,
#    expect no errors

# 3. test querying the server when there's a high memory limit and the queries
#    are all running under the current limit

# 4. test querying the server when there's a tight memory limit
#    expecting an out of memory error

# 5. test a mixture of queries, ~90% successful ones and the rest are expected
#    to fail due to out of memory error

GRAPH_ID          = "max_query_mem"
MEM_HOG_QUERY     = """UNWIND range(0, 100000) AS x RETURN x, count(x)"""
MEM_THRIFTY_QUERY = """RETURN 1"""

class testQueryMemoryLimit():
    def __init__(self):
        self.env, self.db = Env()

    def stress_server(self, queries):
        async def run(self, queries):
            qs           = []  # queries
            should_fails = []  # should query i fail
            thread_count = int(self.db.config_get("THREAD_COUNT"))

            # connection pool blocking when there's no available connections 
            pool = BlockingConnectionPool(max_connections=thread_count, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            for q in queries:
                qs.append(q[0])
                should_fails.append(q[1])

            tasks = []
            for q in qs:
                tasks.append(asyncio.create_task(g.query(q)))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # validate results
            for i, res in enumerate(results):
                if should_fails[i]:
                    # query should have failed
                    self.env.assertIn("Query's mem consumption exceeded capacity", str(res))
                else:
                    # make sure query did not throw an exception
                    self.env.assertNotEqual(type(res), redis.exceptions.ResponseError)

            # close the connection pool
            await pool.aclose()

        asyncio.run(run(self, queries))

    def test_01_read_memory_limit_config(self):
        # read configuration, test default value, expecting unlimited memory cap
        query_mem_capacity = int(self.db.config_get("QUERY_MEM_CAPACITY"))
        self.env.assertEquals(query_mem_capacity, 0)

        # update configuration, set memory limit to 1MB
        MB = 1024*1024
        self.db.config_set("QUERY_MEM_CAPACITY", MB)

        # re-read configuration
        query_mem_capacity = int(self.db.config_get("QUERY_MEM_CAPACITY"))
        self.env.assertEquals(query_mem_capacity, MB)

    def test_02_overflow_no_limit(self):
        # execute query on each one of the threads
        n_queries_to_execute = int(self.db.config_get("THREAD_COUNT"))

        # set query memory limit as UNLIMITED
        limit = 0
        self.db.config_set("QUERY_MEM_CAPACITY", limit) 

        self.stress_server([(MEM_HOG_QUERY, False)] * n_queries_to_execute)

    def test_03_no_overflow_with_limit(self):
        # execute query on each one of the threads
        n_queries_to_execute = int(self.db.config_get("THREAD_COUNT"))

        # set query memory limit to 1GB
        limit = 1024*1024*1024
        self.db.config_set("QUERY_MEM_CAPACITY", limit) 

        self.stress_server([(MEM_HOG_QUERY, False)] * n_queries_to_execute)

    def test_04_overflow_with_limit(self):
        # execute query on each one of the threads
        n_queries_to_execute = int(self.db.config_get("THREAD_COUNT"))

        # set query memory limit to 1MB
        limit = 1024*1024
        self.db.config_set("QUERY_MEM_CAPACITY", limit)

        self.stress_server([(MEM_HOG_QUERY, True)] * n_queries_to_execute)

    def test_05_test_mixed_queries(self):
        queries = []
        total_query_count = 100

        # Query the threadpool_size
        threadpool_size = int(self.db.config_get("THREAD_COUNT"))

        # set query memory limit to 1MB
        limit = 1024*1024
        self.db.config_set("QUERY_MEM_CAPACITY", limit)

        for i in range(total_query_count):
            should_fail = False
            q = MEM_THRIFTY_QUERY
            r = random.randint(0, 100)

            if r <= total_query_count * 0.1: # 10%
                q = MEM_HOG_QUERY
                should_fail = True

            queries.append((q, should_fail))

        self.stress_server(queries)

