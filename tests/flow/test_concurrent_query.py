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

    def test_14_reader_parks_at_gate_without_timeout_penalty(self):
        # a reader parked at the writer gate is WAITING for a lock it does not yet
        # hold - that wait must NOT be charged against the query's timeout. the
        # reader parks for as long as needed (the same courtesy a queued writer
        # gets) and then runs; it is never aborted while merely waiting at the
        # gate, and the parked time is accounted as wait time, not execution time.
        #
        # setup: a long read holds the read lock, a writer behind it is blocked
        # acquiring the write lock (so the gate stays up for the whole read), and a
        # probe reader with a tiny timeout must still SUCCEED - parking through the
        # whole write, far past its own timeout budget.

        if SANITIZER or VALGRIND:
            self.env.skip()

        self.graph.query("CREATE (:N {v:1})")

        outcome = {}

        async def run(self):
            # socket_timeout generous so the ~5s long read / blocked writer that
            # we await below don't trip a client-side read timeout
            pool = BlockingConnectionPool(max_connections=32, timeout=None, socket_timeout=60, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # a reliably-long read (~5s) holds the read lock; wide margins so the
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

            # a reader with a 200ms timeout, launched as a task so we can observe
            # it parking. the gate stays up for the whole ~5s read - far beyond
            # this budget.
            t0 = loop.time()
            probe_task = asyncio.create_task(g.ro_query("MATCH (n) RETURN count(n)", timeout=200))

            # give it 1s - 5x its own timeout. three outcomes:
            #  - still pending  => PARKED at the gate without being timed out (the
            #    property under test); let the writer clear and assert it succeeds.
            #  - done with a "timed out" error => it was aborted AT THE GATE: the
            #    regression this test guards -> record so we FAIL below.
            #  - done successfully => it slipped through because the gate never
            #    engaged (e.g. macOS lacks the writer-preference the gate relies on)
            #    -> can't exercise the gate here, record so we SKIP below.
            await asyncio.sleep(1.0)
            if not probe_task.done():
                outcome['parked'] = True
                probe_err = None ; probe_res = None
                try:
                    probe_res = await probe_task
                except redis.exceptions.ResponseError as e:
                    probe_err = str(e)
                outcome['wait'] = loop.time() - t0
                outcome['err'] = probe_err
                outcome['res'] = probe_res
            else:
                # finished within 1s: a "timed out" error => aborted at the gate
                # (regression); any other error => unexpected failure; a clean
                # success => the gate never engaged. only the last may skip.
                try:
                    await probe_task
                    outcome['gate_not_engaged'] = True
                except redis.exceptions.ResponseError as e:
                    if "timed out" in str(e).lower():
                        outcome['timed_out_at_gate'] = True
                    else:
                        outcome['early_error'] = str(e)

            await read_task
            await write_task
            if not probe_task.done():
                probe_task.cancel()
            await pool.aclose()
            self.conn.delete(GRAPH_ID)

        asyncio.run(run(self))

        # a reader aborted WHILE PARKED at the gate -> the behavior regressed. fail
        # loudly (this is the whole point of the test), never skip past it.
        self.env.assertFalse(outcome.get('timed_out_at_gate', False),
            message="reader was timed out while parked at the writer gate")
        # any other probe failure is also a real failure, not a reason to skip
        self.env.assertFalse(outcome.get('early_error'),
            message="probe failed unexpectedly: %s" % outcome.get('early_error'))
        # the gate is a Linux (PREFER_WRITER) mechanism; if it never engaged the
        # scenario can't be built here - skip rather than false-fail
        if not outcome.get('parked'):
            self.env.skip()
            return

        # 1. the probe did NOT time out at the gate - it parked ~5s (>> its 200ms
        #    budget) and still completed successfully
        self.env.assertIsNone(outcome['err'])
        self.env.assertIsNotNone(outcome['res'])
        self.env.assertGreater(outcome['wait'], 1.0)
        # 2. the parked time is WAIT time, not execution time: the reported query
        #    run time excludes the multi-second park (real execution is sub-ms)
        self.env.assertLess(outcome['res'].run_time_ms, 1000)


    def test_15_server_responsive_under_heavy_mixed_load(self):
        # under a SUSTAINED heavy mix of reads and writes the server (Redis main
        # thread) must keep answering unrelated commands (PING) promptly: neither
        # the writer gate nor the write-lock wait may pin the GIL or let the event
        # loop freeze under a flood. this guards the responsiveness the reentrant
        # write lock exists to provide - under the realistic mixed workload, not
        # just the single waiting writer of test_12.

        # timing assertions are unreliable under instrumentation
        if SANITIZER or VALGRIND:
            self.env.skip()

        self.graph.query("UNWIND range(1, 100000) AS i CREATE (:N {v:i})")

        async def run(self):
            pool = BlockingConnectionPool(max_connections=64, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # a moderate read (holds the read lock briefly) + a small write, mixed
            # ~3:1 and kept continuously in flight to saturate the graph locks
            read_q  = "MATCH (n:N) WHERE n.v < 2000 RETURN count(n)"
            write_q = "CREATE (:N {v:1})"

            probe = AsyncRedis(port=self.env.port, decode_responses=True)
            await probe.ping()

            loop = asyncio.get_running_loop()
            DURATION = 4.0
            deadline = loop.time() + DURATION

            # track in-flight work with a counter + done-callback (O(1) per task)
            # instead of rescanning a growing list each iteration - that would be
            # quadratic and stall the very event loop we use to measure PING. the
            # callback also records any query failure, so a broken workload (which
            # wouldn't actually contend the locks) fails the test instead of
            # leaving PING trivially "responsive".
            inflight = 0
            errors = []

            def _on_done(t):
                nonlocal inflight
                inflight -= 1
                if not t.cancelled() and t.exception() is not None:
                    errors.append(t.exception())

            async def flood():
                nonlocal inflight
                i = 0
                while loop.time() < deadline:
                    while inflight < 48 and loop.time() < deadline:
                        i += 1
                        q = g.query(write_q) if (i % 4 == 0) else g.ro_query(read_q)
                        t = asyncio.create_task(q)
                        inflight += 1
                        t.add_done_callback(_on_done)
                    await asyncio.sleep(0.002)

            flood_task = asyncio.create_task(flood())
            await asyncio.sleep(0.3)  # let the flood saturate the locks

            # probe PING latency from an unrelated connection during the flood
            latencies = []
            while loop.time() < deadline - 0.3:
                t0 = loop.time()
                await probe.ping()
                latencies.append(loop.time() - t0)
                await asyncio.sleep(0.01)

            await flood_task
            # drain the last wave, then fail if any flood query actually errored
            while inflight > 0:
                await asyncio.sleep(0.01)
            self.env.assertEqual(errors, [], message="flood query failures: %r" % errors[:3])

            latencies.sort()
            median = latencies[len(latencies) // 2]
            p99    = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]

            # a frozen event loop would show PING latencies of seconds; with the
            # GIL yielded between slices PINGs are served in a few ms. assert on
            # the median (robust to a rare outlier) plus a generous p99
            self.env.assertLess(median, 0.1)
            self.env.assertLess(p99, 1.0)

            await probe.aclose()
            await pool.aclose()
            self.conn.delete(GRAPH_ID)

        asyncio.run(run(self))

    def test_16_concurrent_readers_park_at_gate(self):
        # many reader-queries can be parked at the writer gate at once, each with a
        # tiny timeout. none of them may be aborted for waiting at the gate: they
        # ALL park through the in-flight write and then succeed - no hang, no crash,
        # no spurious timeout - and the server stays healthy and the write lands.
        # this stresses the concurrent gate-parking path (where readers funnel in,
        # historically a use-after-free risk) now that the park is exempt from the
        # query timeout.

        if SANITIZER or VALGRIND:
            self.env.skip()

        self.graph.query("CREATE (:N {v:1})")

        outcome = {}

        async def run(self):
            pool = BlockingConnectionPool(max_connections=64, timeout=None, socket_timeout=60, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # a reliably-long read (~5s) holds the read lock; a writer behind it
            # blocks acquiring the write lock, raising the gate for the duration
            long_read = "UNWIND range(0, 60000000) AS x WITH x WHERE x % 60000001 = 0 RETURN count(x)"

            loop = asyncio.get_running_loop()

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
            self.env.assertTrue(await wait_until(lambda r, w: r >= 1))
            write_task = asyncio.create_task(g.query("CREATE (:N {v:2}) RETURN 1"))
            # writer in-flight behind the read (gate up) before we probe
            self.env.assertTrue(await wait_until(lambda r, w: r + w >= 2))

            # 20 readers, each with a 200ms timeout, all parking at the gate at once.
            async def probe():
                try:
                    await g.ro_query("MATCH (n) RETURN count(n)", timeout=200)
                    return "ok"
                except redis.exceptions.ResponseError as e:
                    return "timeout" if "timed out" in str(e).lower() else ("ERR:" + str(e))

            t0 = loop.time()
            probes = [asyncio.create_task(probe()) for _ in range(20)]

            # give them 1s - 5x their budget. if all are still pending they are
            # PARKED at the gate without being timed out (the property under test).
            # otherwise inspect the ones that finished: a "timeout" means a reader
            # was aborted AT THE GATE -> the regression this test guards (FAIL); a
            # clean finish means the gate never engaged on this platform (e.g. macOS
            # lacks the Linux writer-preference) -> can't exercise here (SKIP).
            await asyncio.sleep(1.0)
            if all(not p.done() for p in probes):
                outcome['parked'] = True
                outcome['outcomes'] = await asyncio.gather(*probes)
                outcome['wait'] = loop.time() - t0
            else:
                early = await asyncio.gather(*[p for p in probes if p.done()])
                # a parked reader must never be aborted: a "timeout" is the gate
                # regression and any "ERR:" is an unexpected failure - both must fail
                # the test. only clean early successes mean the gate didn't engage.
                outcome['timed_out_at_gate'] = any(o == "timeout" for o in early)
                outcome['early_errors'] = [o for o in early if o not in ("ok", "timeout")]

            await read_task
            await write_task
            for p in probes:
                if not p.done():
                    p.cancel()
            # server stayed healthy and the write landed
            outcome['ping'] = self.conn.ping()
            res = await g.ro_query("MATCH (n:N {v:2}) RETURN count(n)")
            outcome['v2'] = res.result_set[0][0]

            await pool.aclose()
            self.conn.delete(GRAPH_ID)

        asyncio.run(run(self))

        # any reader aborted while parked at the gate -> regression -> FAIL
        self.env.assertFalse(outcome.get('timed_out_at_gate', False),
            message="a reader was timed out while parked at the writer gate")
        # any other early probe failure is a real failure, not a reason to skip
        self.env.assertEqual(outcome.get('early_errors', []), [],
            message="unexpected probe errors: %r" % outcome.get('early_errors'))
        # writer gate is a Linux (PREFER_WRITER) mechanism; skip where it can't engage
        if not outcome.get('parked'):
            self.env.skip()
            return

        # every one of the 20 parked through the write and SUCCEEDED - none was
        # timed out at the gate, none errored or crashed
        self.env.assertEqual(outcome['outcomes'], ["ok"] * 20, message=str(outcome['outcomes']))
        # they genuinely parked: the batch waited well past the 200ms budget
        self.env.assertGreater(outcome['wait'], 1.0)
        # server stayed healthy and the write landed
        self.env.assertTrue(outcome['ping'])
        self.env.assertEqual(outcome['v2'], 1)
