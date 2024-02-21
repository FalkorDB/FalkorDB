import random
import asyncio
from common import *
from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool
from redis.asyncio import Redis as AsyncRedis

import asyncio

GRAPH_ID = "concurrent_query"       # Graph identifier.
SECONDERY_GRAPH_ID = GRAPH_ID + "2" # Secondery graph identifier.
CLIENT_COUNT = 16                   # Number of concurrent connections.
people = ["Roi", "Alon", "Ailon", "Boaz", "Tal", "Omri", "Ori"]

async def delete_graph(g):
    # Try to delete graph.
    try:
        await g.delete()
        return True
    except:
        # Graph deletion failed.
        return False

class testConcurrentQueryFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = redis.Redis("localhost", self.env.port)
        self.graph = self.db.select_graph(GRAPH_ID)

    def setUp(self):
        #self.conn.flushall()
        self.conn.delete(GRAPH_ID)
        self.conn.delete(SECONDERY_GRAPH_ID)

    def run_queries_concurrently(self, queries):
        async def run(self, queries):
            connection_kwargs = { 'decode_responses': True }
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            tasks = []
            for q in queries:
                tasks.append(asyncio.create_task(g.query(q)))

            # wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # close the connection pool
            await pool.aclose()

            return results

        return asyncio.run(run(self, queries))

    def populate_graph(self):
        nodes = {}

        # Create entities
        for idx, p in enumerate(people):
            alias = f"n_{idx}"
            node = Node(alias=alias, labels="person", properties={"name": p})
            nodes[alias] = node
        nodes_str = [str(node) for node in nodes.values()]

        # Fully connected graph
        edges = []
        for src in nodes:
            for dest in nodes:
                if src != dest:
                    edges.append(Edge(nodes[src], "know", nodes[dest]))
        edges_str = [str(edge) for edge in edges]

        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")

    # Count number of nodes in the graph
    def test_01_concurrent_aggregation(self):
        self.populate_graph()

        q = """MATCH (p:person) RETURN count(p)"""
        queries = [q] * CLIENT_COUNT
        results = self.run_queries_concurrently(queries)

        for result in results:
            person_count = result.result_set[0][0]
            self.env.assertEqual(person_count, len(people))
    
    # Concurrently get neighbors of every node.
    def test_02_retrieve_neighbors(self):
        self.populate_graph()

        q = """MATCH (p:person)-[know]->(n:person) RETURN n.name"""
        queries = [q] * CLIENT_COUNT
        results = self.run_queries_concurrently(queries)

        # Fully connected graph + header row.
        expected_resultset_size = len(people) * (len(people)-1)
        for result in results:
            self.env.assertEqual(len(result.result_set), expected_resultset_size)

    # Concurrent writes
    def test_03_concurrent_write(self):        
        self.populate_graph()

        q = """MATCH (p:person)-[know]->(n:person) RETURN n.name"""
        queries = ["CREATE (:country {id:%d})" % i for i in range(CLIENT_COUNT)]
        results = self.run_queries_concurrently(queries)
        for result in results:
            self.env.assertEqual(result.nodes_created, 1)
            self.env.assertEqual(result.properties_set, 1)
    
    # Try to delete graph multiple times.
    def test_04_concurrent_delete(self):
        async def run(self):
            self.graph.query("RETURN 1")
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            tasks = []
            for i in range(0, CLIENT_COUNT):
                tasks.append(asyncio.create_task(g.delete()))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Exactly one thread should have successfully deleted the graph.
            self.env.assertEquals(len(results) - sum(isinstance(res, ResponseError) for res in results), 1)

            # close the connection pool
            await pool.aclose()

        asyncio.run(run(self))

    # Try to delete a graph while multiple queries are executing.
    def test_05_concurrent_read_delete(self):
        async def run(self):
            async_conn = AsyncRedis(port=self.env.port)
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            #-------------------------------------------------------------------
            # Delete graph via Redis DEL key.
            #-------------------------------------------------------------------

            self.populate_graph()

            # invoke queries
            q = "UNWIND (range(0, 10000)) AS x WITH x AS x WHERE (x / 900) = 1 RETURN x"
            tasks = []
            del_task = None

            for i in range(CLIENT_COUNT):
                tasks.append(asyncio.create_task(g.query(q)))
                if i == CLIENT_COUNT / 2:
                    del_task = asyncio.create_task(async_conn.delete(GRAPH_ID))

            # wait for all async tasks
            results = await asyncio.gather(*tasks)
            await del_task

            # validate result.
            self.env.assertTrue(all(r.result_set[0][0] == 900 for r in results))

            # Make sure Graph is empty, e.g. graph was deleted.
            resultset = self.graph.query("MATCH (n) RETURN count(n)").result_set
            self.env.assertEquals(resultset[0][0], 0)

            #-------------------------------------------------------------------
            # Delete graph via GRAPH.DELETE.
            #-------------------------------------------------------------------

            self.populate_graph()

            # invoke queries
            tasks = []
            for i in range (CLIENT_COUNT):
                tasks.append(asyncio.create_task(g.query(q)))
                if i == CLIENT_COUNT / 2:
                    del_task = asyncio.create_task(g.delete())

            # wait for all async tasks
            results = await asyncio.gather(*tasks)
            await del_task

            # validate result.
            self.env.assertTrue(all(r.result_set[0][0] == 900 for r in results))

            # Make sure Graph is empty, e.g. graph was deleted.
            resultset = self.graph.query("MATCH (n) RETURN count(n)").result_set
            self.env.assertEquals(resultset[0][0], 0)

            # Close the connection
            await async_conn.close()

            # close the connection pool
            await pool.aclose()

        asyncio.run(run(self))

    def test_06_concurrent_write_delete(self):
        async def run(self):
            # connect to async graph via a connection pool
            # which will block if there are no available connections
            connection_kwargs = { 'decode_responses': True }
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)
            async_conn = AsyncRedis(port=self.env.port)

            # Test setup - validate that graph exists and possible results are None
            self.graph.query("RETURN 1")
            heavy_write_query = "UNWIND(range(0, 999999)) as x CREATE(n) RETURN count(1)"

            tasks = []
            tasks.append(asyncio.create_task(g.query(heavy_write_query)))
            tasks.append(asyncio.create_task(async_conn.delete(GRAPH_ID)))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            result = results[0]
            if type(result) is ResponseError:
                possible_exceptions = ["Encountered different graph value when opened key " + GRAPH_ID,
                                       "Encountered an empty key when opened key " + GRAPH_ID]
                self.env.assertIn(str(result), possible_exceptions)
            else:
                self.env.assertEquals(1000000, result.result_set[0][0])

            # close the connection pool
            await pool.aclose()

            # close async connection
            await async_conn.close()

        asyncio.run(run(self))
    
    def test_07_concurrent_write_rename(self):
        async def run(self):
            # connect to async graph via a connection pool
            # which will block if there are no available connections
            connection_kwargs = { 'decode_responses': True }
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # single async connection
            async_conn = AsyncRedis(port=self.env.port)

            # Test setup - validate that graph exists and possible results are None
            # Create new empty graph with ID SECONDERY_GRAPH_ID
            new_graph_id = SECONDERY_GRAPH_ID
            graph2 = self.db.select_graph(new_graph_id)
            graph2.query("RETURN 1")

            self.graph.query("MATCH (n) RETURN n")
            heavy_write_query = "UNWIND(range(0, 999999)) as x CREATE(n) RETURN count(1)"

            tasks = []
            tasks.append(asyncio.create_task(g.query(heavy_write_query)))
            tasks.append(asyncio.create_task(async_conn.rename(GRAPH_ID, new_graph_id)))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Possible scenarios:
            # 1. Rename is done before query is sent. The name in the graph context is new_graph,
            #    so when upon commit, when trying to open new_graph key,
            #    it will encounter an empty key since new_graph is not a valid key.
            #    Note: As from https://github.com/RedisGraph/RedisGraph/pull/820
            #          this may not be valid since the rename event handler might actually rename the graph key, before the query execution.    
            # 2. Rename is done during query executing, so when commiting and comparing stored graph context name (GRAPH_ID) to the retrived value graph context name (new_graph),
            #    the identifiers are not the same, since new_graph value is now stored at GRAPH_ID value.

            result = results[0]
            if type(result) is ResponseError:
                possible_exceptions = ["Encountered different graph value when opened key " + GRAPH_ID,
                                       "Encountered an empty key when opened key " + new_graph]
                self.env.assertIn(str(result), possible_exceptions)
            else:
                self.env.assertEquals(1000000, result.result_set[0][0])

            # close the connection pool
            await pool.aclose()

            # close async connection
            await async_conn.close()

        asyncio.run(run(self))

    def test_08_concurrent_write_replace(self):
        async def run(self):
            # connect to async graph via a connection pool
            # which will block if there are no available connections
            connection_kwargs = { 'decode_responses': True }
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # single async connection
            async_conn = AsyncRedis(port=self.env.port)

            # Test setup - validate that graph exists and possible results are None
            self.graph.query("MATCH (n) RETURN n")

            heavy_write_query = "UNWIND(range(0, 999999)) as x CREATE(n) RETURN count(1)"

            tasks = []
            tasks.append(asyncio.create_task(g.query(heavy_write_query)))
            tasks.append(asyncio.create_task(async_conn.set(GRAPH_ID, 1)))
            results = await asyncio.gather(*tasks, return_exceptions=True)

            result = results[0]
            if type(result) is ResponseError:
                possible_exceptions = ["Encountered a non-graph value type when opened key " + GRAPH_ID,
                                       "WRONGTYPE Operation against a key holding the wrong kind of value"]
                self.env.assertIn(str(result), possible_exceptions)
            else:
                self.env.assertEquals(1000000, result.result_set[0][0])

            # close the connection pool
            await pool.aclose()

            # close async connection
            await async_conn.close()

        asyncio.run(run(self))

    def test_09_concurrent_multiple_readers_after_big_write(self):
        # Test issue #890
        self.graph = Graph(self.conn, GRAPH_ID)
        self.graph.query("""UNWIND(range(0, 999)) as x CREATE ()-[r:R]->() RETURN count(r)""")

        read_query = """MATCH (n)-[r:R]->(m) RETURN count(r) AS res UNION RETURN 0 AS res"""
        self.graph.query(read_query)

        queries = [read_query] * CLIENT_COUNT
        results = self.run_queries_concurrently(queries)

        for result in results:
            self.env.assertEquals(1000, result.result_set[0][0])

    def test_10_write_starvation(self):
        # make sure write query do not starve
        # when issuing a large number of read queries
        # alongside a single write query
        # we dont want the write query to have to wait for
        # too long, consider the following sequence:
        # R, W, R, R, R, R, R, R, R...
        # if write is starved our write query might have to wait
        # for all queued read queries to complete while holding
        # Redis global lock, this will hurt performance
        #
        # this test issues a similar sequence of queries and
        # validates that the write query wasn't delayed too much

        async def run(self):
            self.graph.query("RETURN 1")

            connection_kwargs = { 'decode_responses': True }
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            Rq    = "UNWIND range(0, 10000)  AS x WITH x WHERE x = 9999   RETURN 'R', timestamp()"
            Wq    = "UNWIND range(0, 1000)   AS x WITH x WHERE x = 27     CREATE ({v:1}) RETURN 'W', timestamp()"
            Slowq = "UNWIND range(0, 100000) AS x WITH x WHERE x % 73 = 0 RETURN count(1)"

            # issue a number of slow queries, this will give us time to fill up
            # FalkorDB's internal threadpool queue
            slow_queries = []
            for i in range(0, CLIENT_COUNT * 5):
                slow_queries.append(asyncio.create_task(g.ro_query(Slowq)))

            # create a long sequence of read queries
            read_tasks = []

            # N Read queries
            for i in range(0, CLIENT_COUNT):
                read_tasks.append(asyncio.create_task(g.ro_query(Rq)))

            # Single Write query
            write_task = asyncio.create_task(g.query(Wq))

            # 9N Read queries
            for i in range(0, CLIENT_COUNT * 9):
                read_tasks.append(asyncio.create_task(g.ro_query(Rq)))

            # wait for all queries to return
            await asyncio.gather(*slow_queries)
            results = await asyncio.gather(*read_tasks)
            w_res   = await write_task

            # count how many queries completed before the write query
            write_ts = w_res.result_set[0][1]
            count = 0
            count = sum(1 for res in results if res.result_set[0][1] < write_ts)

            # make sure write query wasn't starved
            self.env.assertLessEqual(count, len(read_tasks) * 0.3)

            # close the connection pool
            await pool.aclose()

            # delete the key
            self.conn.delete(GRAPH_ID)

        return asyncio.run(run(self))

    def test_11_concurrent_resize_zero_matrix(self):
        async def run(self):
            connection_kwargs = { 'decode_responses': True }
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # make sure graph exists
            self.graph.query("RETURN 1")

            tasks = []
            read_q  = "MATCH (n:N)-[r:R]->() RETURN r"
            write_q = "UNWIND range(1, 10000) AS x CREATE (:M)"
            for i in range(1, 10):
                tasks.append(asyncio.create_task(g.query(write_q)))
                for j in range(1, 10):
                    tasks.append(asyncio.create_task(g.ro_query(read_q)))

            await asyncio.gather(*tasks)

            # close the connection pool
            await pool.aclose()

        asyncio.run(run(self))

