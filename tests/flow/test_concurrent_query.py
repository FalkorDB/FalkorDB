import random
import asyncio
from common import *
from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool
from redis.asyncio import Redis as AsyncRedis

GRAPH_ID = "concurrent_query"       # Graph identifier.
SECONDARY_GRAPH_ID = GRAPH_ID + "2" # Secondery graph identifier.
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
        self.conn.delete(GRAPH_ID)
        self.conn.delete(SECONDARY_GRAPH_ID)

    def run_queries_concurrently(self, queries):
        async def run(self, queries):            
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
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # single async connection
            async_conn = AsyncRedis(port=self.env.port)

            # Test setup - validate that graph exists and possible results are None
            # Create new empty graph with ID SECONDARY_GRAPH_ID
            new_graph_id = SECONDARY_GRAPH_ID
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

    def test_12_write_lock_yields_gil_while_waiting(self):
        # a writer that has to wait for the graph write lock must NOT pin
        # Redis' GIL for the whole wait: it should release the GIL every
        # WRITE_LOCK_POLL_INTERVAL ms and retry, so unrelated commands keep
        # being served while the in-flight reader drains.
        #
        # this test holds the graph READ lock with a ~2s read, issues a writer
        # that must wait behind it, and hammers the server with PINGs from an
        # unrelated connection. if the writer pinned the GIL the PINGs would
        # stall for the remainder of the read (~seconds); with GIL yielding
        # they return promptly. it also asserts the write ultimately lands.

        # timing assertions are unreliable under instrumentation
        if SANITIZER or VALGRIND:
            self.env.skip()

        # ensure graph exists with a single :N node
        self.graph.query("CREATE (:N {v:1})")

        async def run(self):
            pool = BlockingConnectionPool(max_connections=32, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # a long read (~2s) that holds the graph READ lock for its duration
            long_read = "UNWIND range(0, 40000000) AS x WITH x WHERE x % 40000001 = 0 RETURN count(x)"

            # a write that must wait for the read lock to drain before it commits
            write = "CREATE (:N {v:2}) RETURN 1"

            # unrelated connection used to probe server responsiveness
            probe = AsyncRedis(port=self.env.port, decode_responses=True)
            await probe.ping()

            # start the long read and let it acquire the READ lock
            read_task = asyncio.create_task(g.ro_query(long_read))
            await asyncio.sleep(0.2)

            # start the writer; it blocks waiting for the READ lock to drain
            write_task = asyncio.create_task(g.query(write))
            await asyncio.sleep(0.05)  # let the writer reach the lock wait

            # probe responsiveness while the writer is waiting: PINGs should
            # return promptly because the writer keeps yielding the GIL
            loop = asyncio.get_running_loop()
            latencies = []
            for _ in range(50):
                t0 = loop.time()
                await probe.ping()
                latencies.append(loop.time() - t0)
                await asyncio.sleep(0.01)

            # the read completes and the write lands
            await read_task
            await write_task

            # server stayed responsive: with the GIL pinned (old behavior) every
            # PING would stall ~the full 2s read, so the median would be ~2s.
            # with GIL yielding the vast majority are served in a few ms; assert
            # on the median so a rare scheduling outlier under CI load (a single
            # slow PING) can't flake the test while still clearly separating the
            # responsive behavior from the frozen one
            latencies.sort()
            median_latency = latencies[len(latencies) // 2]
            self.env.assertLess(median_latency, 0.2)

            # the write took effect
            res = await g.ro_query("MATCH (n:N {v:2}) RETURN count(n)")
            self.env.assertEqual(res.result_set[0][0], 1)

            await probe.aclose()
            await pool.aclose()
            self.conn.delete(GRAPH_ID)

        asyncio.run(run(self))

    def test_13_writer_gate_prevents_starvation(self):
        # the writer gate is the anti-starvation mechanism: while a writer waits
        # for the lock, new reader-queries park at the gate instead of taking the
        # read lock, so they can't slip in during the GIL hand-offs. the writer
        # therefore waits only for the reads already holding the lock to drain,
        # not for the whole flood. (before the gate this scenario took ~3s.)

        # timing assertions are unreliable under instrumentation
        if SANITIZER or VALGRIND:
            self.env.skip()

        self.graph.query("CREATE (:N {v:1})")

        async def run(self):
            pool = BlockingConnectionPool(max_connections=64, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # a short read (~20ms) that holds the graph READ lock, then returns
            # the SERVER's clock (timestamp()) so we can prove ordering from the
            # server side instead of trusting client wall-clock. kept short so
            # the in-flight batch drains quickly
            read_q  = "UNWIND range(0, 500000) AS x WITH count(x) AS c RETURN timestamp()"
            write_q = "CREATE (:N {v:2}) RETURN timestamp()"

            loop = asyncio.get_running_loop()
            FLOOD_SECONDS = 6.0
            deadline = loop.time() + FLOOD_SECONDS
            reads    = []

            # keep ~32 reads in flight so the read lock stays continuously
            # contended for the whole window
            async def flood():
                while loop.time() < deadline:
                    while (sum(1 for r in reads if not r.done()) < 32
                           and loop.time() < deadline):
                        reads.append(asyncio.create_task(g.ro_query(read_q)))
                    await asyncio.sleep(0.005)

            flood_task = asyncio.create_task(flood())
            await asyncio.sleep(0.3)  # let the flood saturate the read lock

            # the writer reports the server-side time at which it committed
            writer_ts = (await g.query(write_q)).result_set[0][0]

            await flood_task
            # surface any reader failure: a dead flood must fail the test rather
            # than silently let the writer "win" an uncontended lock
            results = await asyncio.gather(*reads)

            # prove interleaving from the SERVER clock, not client timing: each
            # flood read reports when it actually ran. if the gate works the
            # writer commits *amid* the flood, so some reads run before it and a
            # large share run after it. without the gate the writer waits the
            # whole flood out and almost no read would run after it.
            read_ts = [r.result_set[0][0] for r in results]
            reads_before = sum(1 for t in read_ts if t < writer_ts)
            reads_after  = sum(1 for t in read_ts if t > writer_ts)
            self.env.assertGreater(reads_before, 0)
            self.env.assertGreater(reads_after, len(read_ts) * 0.3)

            # the write took effect
            res = await g.ro_query("MATCH (n:N {v:2}) RETURN count(n)")
            self.env.assertEqual(res.result_set[0][0], 1)

            await pool.aclose()
            self.conn.delete(GRAPH_ID)

        asyncio.run(run(self))

    def test_14_reader_respects_timeout_at_writer_gate(self):
        # a reader parked at the writer gate must still honor its own query
        # timeout: if the in-flight writer doesn't clear within the reader's
        # budget, the reader aborts with a timeout error rather than waiting for
        # the whole write (or the read blocking it) to resolve.
        #
        # setup: a long read holds the read lock, a writer behind it is blocked
        # acquiring the write lock (so the gate stays up for the whole read),
        # and a probe reader with a short timeout must abort at the gate.

        if SANITIZER or VALGRIND:
            self.env.skip()

        self.graph.query("CREATE (:N {v:1})")

        async def run(self):
            # socket_timeout generous so the ~3s long read / blocked writer that
            # we await below don't trip a client-side read timeout
            pool = BlockingConnectionPool(max_connections=32, timeout=None, socket_timeout=30, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # a reliably-long read (~3s) holds the read lock; wide margins so the
            # probe below lands well inside the window regardless of scheduling
            long_read = "UNWIND range(0, 60000000) AS x WITH x WHERE x % 60000001 = 0 RETURN count(x)"

            loop = asyncio.get_running_loop()

            # synchronize on observable server state instead of fixed sleeps:
            # poll GRAPH.INFO for the counts of running / waiting queries. the
            # sync self.conn call runs off the event loop so it doesn't block the
            # read / write tasks awaiting on their own connections
            async def inflight():
                res = await loop.run_in_executor(None, lambda:
                    self.conn.execute_command(
                        "GRAPH.INFO", "RunningQueries", "WaitingQueries"))
                return len(res[1]), len(res[3])  # (running, waiting)

            async def wait_until(pred, timeout=20.0):
                deadline = loop.time() + timeout
                while loop.time() < deadline:
                    if pred(*(await inflight())):
                        return True
                    await asyncio.sleep(0.02)
                return False

            read_task = asyncio.create_task(g.ro_query(long_read))
            # the long read must actually be executing (holding the READ lock)
            # before we introduce the writer
            self.env.assertTrue(await wait_until(lambda r, w: r >= 1))

            write_task = asyncio.create_task(g.query("CREATE (:N {v:2}) RETURN 1"))
            # the writer must be in-flight behind it (blocked acquiring the write
            # lock, so the gate is raised) before we probe
            self.env.assertTrue(await wait_until(lambda r, w: r + w >= 2))

            # a reader with a 200ms timeout must abort at the gate within its
            # budget, not block for the full read
            t0 = loop.time()
            timed_out = False
            try:
                await g.ro_query("MATCH (n) RETURN count(n)", timeout=200)
            except redis.exceptions.ResponseError as e:
                timed_out = "timed out" in str(e).lower()
            probe_wait = loop.time() - t0

            await read_task
            await write_task

            # core property: the reader aborted (timed out) instead of slipping
            # through or hanging for the whole read
            self.env.assertTrue(timed_out)
            # and it aborted promptly - nowhere near the full read
            self.env.assertLess(probe_wait, 2.0)

            await pool.aclose()
            self.conn.delete(GRAPH_ID)

        asyncio.run(run(self))

