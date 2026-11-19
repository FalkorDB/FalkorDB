import random
import asyncio
import time
from common import *
from long_read_utils import *
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
        self.conn = redis.Redis(self.env.host, self.env.port)
        self.graph = self.db.select_graph(GRAPH_ID)

    def setUp(self):
        self.conn.delete(GRAPH_ID)
        self.conn.delete(SECONDARY_GRAPH_ID)

    def run_queries_concurrently(self, queries):
        async def run(self, queries):            
            pool = BlockingConnectionPool(
                max_connections=16,
                timeout=None,
                host=self.env.host,
                port=self.env.port,
                decode_responses=True,
                socket_timeout=SOCKET_TIMEOUT,
            )
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
            pool = BlockingConnectionPool(
                max_connections=16,
                timeout=None,
                host=self.env.host,
                port=self.env.port,
                decode_responses=True,
                socket_timeout=SOCKET_TIMEOUT,
            )
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            tasks = []
            for i in range(0, CLIENT_COUNT):
                tasks.append(asyncio.create_task(g.delete()))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Exactly one thread should have successfully deleted the graph.
            self.env.assertEqual(len(results) - sum(isinstance(res, ResponseError) for res in results), 1)

            # close the connection pool
            await pool.aclose()

        asyncio.run(run(self))

    # Try to delete a graph while multiple queries are executing.
    def test_05_concurrent_read_delete(self):
        async def run(self):
            async_conn = AsyncRedis(host=self.env.host, port=self.env.port)
            pool = BlockingConnectionPool(
                max_connections=16,
                timeout=None,
                host=self.env.host,
                port=self.env.port,
                decode_responses=True,
                socket_timeout=SOCKET_TIMEOUT,
            )
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
            self.env.assertEqual(resultset[0][0], 0)

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
            self.env.assertEqual(resultset[0][0], 0)

            # Close the connection
            await async_conn.close()

            # close the connection pool
            await pool.aclose()

        asyncio.run(run(self))

    def test_06_concurrent_write_delete(self):
        async def run(self):
            # connect to async graph via a connection pool
            # which will block if there are no available connections
            pool = BlockingConnectionPool(
                max_connections=16,
                timeout=None,
                host=self.env.host,
                port=self.env.port,
                decode_responses=True,
                socket_timeout=SOCKET_TIMEOUT,
            )
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)
            async_conn = AsyncRedis(host=self.env.host, port=self.env.port)

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
                                       "Encountered an empty key when opened key " + GRAPH_ID,
                                       # The write escalated to writer mode and found the key no
                                       # longer holds this graph, so it aborted before mutating.
                                       "graph was deleted or replaced while the query was running, aborting"]
                self.env.assertContains(str(result), possible_exceptions)
            else:
                self.env.assertEqual(1000000, result.result_set[0][0])

            # close the connection pool
            await pool.aclose()

            # close async connection
            await async_conn.close()

        asyncio.run(run(self))
    
    def test_07_concurrent_write_rename(self):
        async def run(self):
            # connect to async graph via a connection pool
            # which will block if there are no available connections
            pool = BlockingConnectionPool(
                max_connections=16,
                timeout=None,
                host=self.env.host,
                port=self.env.port,
                decode_responses=True,
                socket_timeout=SOCKET_TIMEOUT,
            )
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # single async connection
            async_conn = AsyncRedis(host=self.env.host, port=self.env.port)

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
                self.env.assertContains(str(result), possible_exceptions)
            else:
                self.env.assertEqual(1000000, result.result_set[0][0])

            # close the connection pool
            await pool.aclose()

            # close async connection
            await async_conn.close()

        asyncio.run(run(self))

    def test_08_concurrent_write_replace(self):
        async def run(self):
            # connect to async graph via a connection pool
            # which will block if there are no available connections
            pool = BlockingConnectionPool(
                max_connections=16,
                timeout=None,
                host=self.env.host,
                port=self.env.port,
                decode_responses=True,
                socket_timeout=SOCKET_TIMEOUT,
            )
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            # single async connection
            async_conn = AsyncRedis(host=self.env.host, port=self.env.port)

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
                                       "WRONGTYPE Operation against a key holding the wrong kind of value",
                                       "Existing key has wrong Redis type",
                                       # The write escalated to writer mode and found the key no
                                       # longer holds this graph, so it aborted before mutating.
                                       "graph was deleted or replaced while the query was running, aborting"]
                self.env.assertContains(str(result), possible_exceptions)
            else:
                self.env.assertEqual(1000000, result.result_set[0][0])

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
            self.env.assertEqual(1000, result.result_set[0][0])

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

            pool = BlockingConnectionPool(
                max_connections=16,
                timeout=None,
                host=self.env.host,
                port=self.env.port,
                decode_responses=True,
                socket_timeout=SOCKET_TIMEOUT,
            )
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
            pool = BlockingConnectionPool(
                max_connections=16,
                timeout=None,
                host=self.env.host,
                port=self.env.port,
                decode_responses=True,
                socket_timeout=SOCKET_TIMEOUT,
            )
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


class testWriteLockWaitFlow(FlowTestsBase):
    # a writer waiting for a graph's write lock while reads hold it: the server
    # must keep answering (the writer gives the GIL back between slices of
    # WRITE_LOCK_POLL_INTERVAL ms), the writer must not starve (the writer gate
    # holds new readers back), and readers held at the gate must not pay for it
    # in their timeout. C's test_concurrent_query tests 12-16.
    #
    # with a fixed THREAD_COUNT, not the machine's core count: what the gate
    # prevents only happens when enough reads overlap. measured with the gate
    # disabled, a writer behind a flood of ~20ms reads was not starved with 2
    # workers, starved now and then with 4, and starved every time with 10.
    # test_03 also needs workers left over for its probes beyond the long read
    # and the writer.
    def __init__(self):
        self.env, self.db = Env(moduleArgs='THREAD_COUNT 10')

        # timing assertions are unreliable under instrumentation
        if SANITIZER:
            self.env.skip()

        self.conn = redis.Redis(self.env.host, self.env.port,
                                socket_timeout=SOCKET_TIMEOUT, retry=NO_RETRY)
        self.graph = self.db.select_graph(GRAPH_ID)

    def setUp(self):
        self.conn.delete(GRAPH_ID)

    def _async_graph(self, max_connections):
        # the graph on an async client, and the pool to close when done
        pool = async_pool(self.env, max_connections, SOCKET_TIMEOUT)
        return pool, FalkorDB(connection_pool=pool).select_graph(GRAPH_ID)

    def _inflight_waiter(self, loop):
        # (running, waiting) query counts from GRAPH.INFO, polled off the event
        # loop so it doesn't block the tasks awaiting on their own connections
        async def inflight():
            res = await loop.run_in_executor(None, lambda:
                self.conn.execute_command(
                    "GRAPH.INFO", "RunningQueries", "WaitingQueries"))
            return len(res[1]), len(res[3])

        async def wait_until(pred, timeout=20.0):
            deadline = loop.time() + timeout
            while loop.time() < deadline:
                if pred(*(await inflight())):
                    return True
                await asyncio.sleep(0.02)
            return False

        return wait_until

    async def _writer_behind_long_read(self, g, n, wait_until):
        # a long read takes the READ lock, then a write is sent behind it. both
        # are executing, and the write has waited long enough to close the
        # writer gate (two 5 ms slices), when this returns (read, write) tasks
        read_task = asyncio.create_task(g.ro_query(long_read_query(n)))
        # the long read must be executing (holding the READ lock) before the
        # writer is introduced
        self.env.assertTrue(await wait_until(lambda r, w: r >= 1))
        write_task = asyncio.create_task(g.query("CREATE (:N {v:2}) RETURN 1"))
        # running, not waiting in a queue: a worker has picked the write up.
        # from here 0.2 s is about 40 slices of its wait
        self.env.assertTrue(await wait_until(lambda r, w: r >= 2))
        await asyncio.sleep(0.2)
        return read_task, write_task

    def test_01_write_lock_yields_gil_while_waiting(self):
        # a writer that has to wait for the graph write lock must NOT hold the
        # GIL for the whole wait: it gives the GIL back every
        # WRITE_LOCK_POLL_INTERVAL ms and tries again, so unrelated commands
        # keep being served while the read in flight drains.
        #
        # a long read holds the graph READ lock, a writer waits behind it, and a
        # PING probe on an unrelated connection samples the server throughout.
        # if the writer held the GIL, one PING would wait out the rest of the
        # read. the read is sized for this machine: a fixed one can end before
        # the PINGs that would catch the freeze are sent.
        #
        # GRAPH.INFO is polled only before the writer is sent: a writer holding
        # the GIL would block the poll too, and the test would fail on the poll
        # instead of on the PING.
        self.graph.query("CREATE (:N {v:1})")
        n, read_sec = calibrate_long_read(self.graph, 3.0)

        async def run(self):
            pool, g = self._async_graph(32)
            wait_until = self._inflight_waiter(asyncio.get_running_loop())

            probe = PingProbe(self.env.port, host=self.env.host)
            await probe.start()

            # the long read takes the READ lock; the writer then waits behind it
            read_task = asyncio.create_task(g.ro_query(long_read_query(n)))
            self.env.assertTrue(await wait_until(lambda r, w: r >= 1))
            write_task = asyncio.create_task(g.query("CREATE (:N {v:2}) RETURN 1"))

            await read_task
            await write_task
            worst = await probe.stop()

            # holding the GIL, the worst PING is the rest of the read (seconds);
            # giving it back, a slice (ms). the bound leaves room for a slow
            # runner and stays far below the freeze
            self.env.assertLess(worst, read_sec / 3,
                message="worst PING %.3fs during a %.1fs read" % (worst, read_sec))

            # the write took effect
            res = await g.ro_query("MATCH (n:N {v:2}) RETURN count(n)")
            self.env.assertEqual(res.result_set[0][0], 1)

            await pool.aclose()
            self.conn.delete(GRAPH_ID)

        asyncio.run(run(self))

    def test_02_writer_gate_prevents_starvation(self):
        # the writer gate is the anti-starvation mechanism: while a writer waits
        # for the lock, new reader-queries wait at the gate instead of taking the
        # read lock, so they can't slip in while the writer has given the GIL
        # back. the writer therefore waits only for the reads already holding the
        # lock, not for the whole flood.
        #
        # the reads must outlast a WRITE_LOCK_POLL_INTERVAL slice: during a slice
        # the lock itself holds new readers back, so reads shorter than a slice
        # drain within it and no writer starves, gate or not. so the read is
        # sized to ~30ms on this machine rather than fixed, with no minimum: in
        # a slow build (coverage) a larger read runs for a second or more, and
        # the ~30 reads sent before the write then fill the window by
        # themselves, gate or not.
        self.graph.query("CREATE (:N {v:1})")

        def flood_read(n):
            return f"UNWIND range(0, {n}) AS x WITH count(x) AS c RETURN timestamp()"

        n = 2000000
        durations = []
        for _ in range(3):
            t0 = time.monotonic()
            self.graph.ro_query(flood_read(n))
            durations.append(time.monotonic() - t0)
        # the fastest of three: a stall only ever makes a timing longer
        n = min(int(n * 0.03 / min(durations)), 50000000)

        async def run(self):
            pool, g = self._async_graph(64)

            # a read that holds the graph READ lock, then returns the SERVER's
            # clock (timestamp()) so ordering is proven from the server side
            # instead of trusting client wall-clock. the write's RETURN is
            # evaluated above its Commit, so its timestamp is taken once it holds
            # the write lock
            write_q = "CREATE (:N {v:2}) RETURN timestamp()"

            loop = asyncio.get_running_loop()
            deadline = loop.time() + 6.0

            # keep ~32 reads in flight so the read lock stays continuously
            # contended for the whole window
            flood = ReadLockHolder(g, flood_read(n), concurrency=32)
            flood.start()

            # the flood ends at the deadline whether or not the write is done:
            # a starved writer then gets the lock as the flood drains, and the
            # split below fails, rather than the write waiting on a flood that
            # never ends
            async def end_flood():
                await asyncio.sleep(max(0.0, deadline - loop.time()))
                await flood.stop()
            ender = asyncio.create_task(end_flood())

            await asyncio.sleep(0.3)  # let the flood saturate the read lock

            writer_ts = (await g.query(write_q)).result_set[0][0]

            await ender
            # surface any reader failure: a dead flood must fail the test rather
            # than silently let the writer "win" an uncontended lock
            read_errors = await flood.drain()
            self.env.assertEqual(read_errors, [], message=str(read_errors[:3]))

            # if the gate works the writer gets the lock *amid* the flood, so some
            # reads run before it and a large share after it. without the gate it
            # waits the flood out and almost no read runs after it
            read_ts = [r.result().result_set[0][0] for r in flood.reads
                       if r.exception() is None]
            reads_before = sum(1 for t in read_ts if t < writer_ts)
            reads_after  = sum(1 for t in read_ts if t > writer_ts)
            split = "%d reads before the write, %d after, of %d" % (
                reads_before, reads_after, len(read_ts))
            self.env.assertGreater(reads_before, 0, message=split)
            self.env.assertGreater(reads_after, len(read_ts) * 0.3, message=split)

            # the write took effect
            res = await g.ro_query("MATCH (n:N {v:2}) RETURN count(n)")
            self.env.assertEqual(res.result_set[0][0], 1)

            await pool.aclose()
            self.conn.delete(GRAPH_ID)

        asyncio.run(run(self))

    def test_03_readers_park_at_gate_without_timeout_penalty(self):
        # readers parked at the writer gate are WAITING for a lock they do not
        # yet hold - that wait must NOT be charged against their timeout. many
        # park at once, each with a tiny timeout: they wait out the write, then
        # run and succeed, and the parked time is not reported as execution
        # time. no hang, no crash, no spurious timeout, and the write lands.
        #
        # setup: a long read holds the read lock, a writer behind it has closed
        # the gate, and the probe readers are sent behind both.
        self.graph.query("CREATE (:N {v:1})")
        n, _ = calibrate_long_read(self.graph, 5.0)

        PROBES = 20
        outcome = {}

        async def run(self):
            pool, g = self._async_graph(64)
            loop = asyncio.get_running_loop()
            wait_until = self._inflight_waiter(loop)
            read_task, write_task = await self._writer_behind_long_read(
                g, n, wait_until)

            async def probe():
                # a 200ms timeout; the gate stays closed for the rest of the
                # read - far beyond this budget
                try:
                    return await g.ro_query("MATCH (n) RETURN count(n)", timeout=200)
                except redis.exceptions.ResponseError as e:
                    return e

            t0 = loop.time()
            probes = [asyncio.create_task(probe()) for _ in range(PROBES)]

            # 1s is 5x their budget: all still pending means they are parked
            # without being timed out (the property under test)
            await asyncio.sleep(1.0)
            outcome['parked'] = all(not p.done() for p in probes)
            outcome['results'] = await asyncio.gather(*probes)
            outcome['wait'] = loop.time() - t0

            await read_task
            await write_task
            outcome['ping'] = self.conn.ping()
            res = await g.ro_query("MATCH (n:N {v:2}) RETURN count(n)")
            outcome['v2'] = res.result_set[0][0]

            await pool.aclose()
            self.conn.delete(GRAPH_ID)

        asyncio.run(run(self))

        # every probe waited out the write and SUCCEEDED
        errors = [r for r in outcome['results'] if isinstance(r, Exception)]
        self.env.assertEqual(errors, [], message=str(errors[:3]))
        # they were genuinely held back behind the waiting writer: the lock
        # holds new readers back on every platform, so a probe that finished
        # early means the scenario never formed
        self.env.assertTrue(outcome['parked'],
            message="a probe was not held back behind the waiting writer")
        self.env.assertGreater(outcome['wait'], 1.0)
        # the parked time is WAIT time, not execution time
        run_times = [r.run_time_ms for r in outcome['results']
                     if not isinstance(r, Exception)]
        self.env.assertLess(max(run_times, default=0), 1000,
            message="probe run times (ms): %r" % sorted(run_times)[-3:])
        # the server stayed healthy and the write landed
        self.env.assertTrue(outcome['ping'])
        self.env.assertEqual(outcome['v2'], 1)

    def test_04_server_responsive_under_heavy_mixed_load(self):
        # under a SUSTAINED heavy mix of reads and writes the server must keep
        # answering unrelated commands (PING) promptly. a smoke test of the gate
        # and the sliced wait under load: its reads take milliseconds, so a
        # writer holding the GIL for its whole wait would pass it too, and
        # test_01 is the one that catches that.
        self.graph.query("UNWIND range(1, 100000) AS i CREATE (:N {v:i})")

        async def run(self):
            pool, g = self._async_graph(64)

            # a moderate read (holds the read lock briefly) + a small write, mixed
            # ~3:1 and kept continuously in flight to saturate the graph locks
            read_q  = "MATCH (n:N) WHERE n.v < 2000 RETURN count(n)"
            write_q = "CREATE (:N {v:1})"

            loop = asyncio.get_running_loop()
            DURATION = 4.0
            deadline = loop.time() + DURATION

            # in-flight work tracked with a counter + done-callback (O(1) per
            # task), not by rescanning a growing list, which would stall the very
            # event loop PING is measured on. the callback also records query
            # failures, so a broken workload can't leave PING trivially fast
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

            # the probe connects before the workload starts and samples it from
            # the first query, ramp included, until 0.3 s before the deadline:
            # its first PING isn't recorded, so sent under load it could hide a
            # stall
            probe = PingProbe(self.env.port, host=self.env.host)
            await probe.start()

            flood_task = asyncio.create_task(flood())
            await asyncio.sleep(max(0.0, deadline - 0.3 - loop.time()))
            # a failed PING is recorded as inf: none may fail, not just 99%
            worst = await probe.stop()
            self.env.assertLess(worst, float('inf'), message="a PING failed")

            await flood_task
            while inflight > 0:
                await asyncio.sleep(0.01)
            self.env.assertEqual(errors, [], message="flood query failures: %r" % errors[:3])

            latencies = sorted(probe.latencies)
            median = latencies[len(latencies) // 2]
            p99    = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]

            # a frozen event loop shows PING latencies of seconds; with the GIL
            # given back between slices PINGs are served in a few ms
            self.env.assertLess(median, 0.1)
            self.env.assertLess(p99, 1.0)

            await pool.aclose()
            self.conn.delete(GRAPH_ID)

        asyncio.run(run(self))

    def _write_aborted_while_waiting(self, action, expected_err):
        # a long read holds the graph READ lock and a write waits behind it;
        # `action` runs on another connection while the write waits. between
        # slices the writer holds no lock and not the GIL either, so the
        # action lands mid-wait, and the next slice's checks must refuse the
        # write. a writer holding the GIL for its whole wait would freeze the
        # action out until it had committed.
        self.graph.query("CREATE (:N {v:1})")
        n, _ = calibrate_long_read(self.graph, 3.0)

        async def run(self):
            pool, g = self._async_graph(16)
            admin = AsyncRedis(host=self.env.host, port=self.env.port, decode_responses=True)
            loop = asyncio.get_running_loop()
            wait_until = self._inflight_waiter(loop)
            read_task, write_task = await self._writer_behind_long_read(
                g, n, wait_until)

            t0 = loop.time()
            await action(admin)
            try:
                reply = await write_task
            except redis.exceptions.ResponseError as e:
                reply = e
            refused_after = loop.time() - t0
            await asyncio.gather(read_task, return_exceptions=True)

            await admin.aclose()
            await pool.aclose()
            return reply, refused_after

        reply, refused_after = asyncio.run(run(self))
        self.env.assertTrue(isinstance(reply, redis.exceptions.ResponseError),
            message="the write was not refused: %r" % (reply,))
        self.env.assertContains(expected_err, str(reply))
        # refused at its next slice, not once the read had ended
        self.env.assertLess(refused_after, 1.0)

    def test_05_graph_deleted_while_waiting(self):
        async def delete(admin):
            await admin.delete(GRAPH_ID)

        self._write_aborted_while_waiting(delete, "graph was deleted or replaced")
        # the refused write did not bring the graph back
        self.env.assertEqual(self.conn.exists(GRAPH_ID), 0)

    def test_06_client_pause_while_waiting(self):
        async def pause(admin):
            # far longer than the wait; lifted as soon as the write replies
            await admin.execute_command("CLIENT", "PAUSE", 60000, "WRITE")

        try:
            self._write_aborted_while_waiting(pause, "replica traffic is currently paused")
        finally:
            self.conn.execute_command("CLIENT", "UNPAUSE")
        self.env.assertTrue(self.conn.ping())
        # the write did not land
        res = self.graph.ro_query("MATCH (n:N {v:2}) RETURN count(n)")
        self.env.assertEqual(res.result_set[0][0], 0)
