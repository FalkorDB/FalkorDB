from common import Env, SOCKET_TIMEOUT
from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool
import asyncio
import os
import resource
import signal

# 1.test getting and setting config
#
# 2. test overflowing the server when there's a limit
#    expect to get error!
#
# 3. test overflowing the server when there's no limit
#    expect not to get any exceptions
#
# 4. flood the server with more queued writes than any fixed queue capacity
#    expect every write to complete and the server to stay responsive
#
# 5. queue writes / reads behind a long-running query with a low limit
#    expect the excess to be rejected, the rest to complete, and the
#    admission count to return to zero
#
# 6. change the limit at runtime
#    expect the new bound to apply to the next queries

GRAPH_ID = "max_pending_queries"
SLOW_QUERY = "UNWIND range (0, 1000000) AS x WITH x WHERE (x / 2) = 50 RETURN x"
FLOOD_NODES = 1000
FLOOD_QUERY = f"UNWIND range(1, {FLOOD_NODES}) AS x CREATE (:N)"
# Long enough to keep a graph's writer (or a pool worker) busy while a burst is
# queued behind it — about a second on a release build.
HOLD_ROWS = 100_000_000
HOLD_WRITE = f"UNWIND range(1, {HOLD_ROWS}) AS x WITH x WHERE x < 0 CREATE (:Hold)"
HOLD_READ = f"UNWIND range(1, {HOLD_ROWS}) AS x WITH x WHERE x < 0 RETURN count(x)"
DEFAULT_MAX_QUEUED_QUERIES = 10000
MAX_PENDING_ERR = b"-Max pending queries exceeded\r\n"


def enc(*args):
    out = f"*{len(args)}\r\n"
    for a in args:
        out += f"${len(a.encode())}\r\n{a}\r\n"
    return out.encode()


def raise_fd_limit(n):
    """Allow about n concurrent client sockets; returns how many fit."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = n + 256
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE,
                           (want if hard == resource.RLIM_INFINITY else min(want, hard), hard))
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    return min(n, soft - 256)


async def issue_query(self, g, q):
    try:
        res = await g.ro_query(q)
        return False # no failures
    except Exception as e:
        self.env.assertContains("Max pending queries exceeded", str(e))
        return True # failed due to internal queries queue limit

class testPendingQueryLimit():
    def __init__(self):
        self.env, self.db = Env(moduleArgs="THREAD_COUNT 2")
        # create graph
        self.g = self.db.select_graph(GRAPH_ID)
        self.g.query("RETURN 3")

    def stress_server(self):
        async def run(self):
            # connection pool with 16 connections
            # blocking when there's no connections available
            n = self.db.config_get("THREAD_COUNT") * 5
            limit = self.db.config_get("MAX_QUEUED_QUERIES")
            pool = BlockingConnectionPool(max_connections=n, timeout=None, host=self.env.host, port=self.env.port, decode_responses=True, socket_timeout=SOCKET_TIMEOUT)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            tasks = []
            for i in range(0, n):
                tasks.append(asyncio.create_task(issue_query(self, g, SLOW_QUERY)))

            results = await asyncio.gather(*tasks)

            # close the connection pool
            await pool.aclose()

            # return if error encountered
            res = any(results)
            return res

        return asyncio.run(run(self))

    def test_01_query_limit_config(self):
        # read max queued queries config
        max_queued_queries = self.db.config_get("MAX_QUEUED_QUERIES")
        self.env.assertEqual(max_queued_queries, DEFAULT_MAX_QUEUED_QUERIES)

        # update configuration, set max queued queries
        self.db.config_set("MAX_QUEUED_QUERIES", 10)

        # re-read configuration
        max_queued_queries = self.db.config_get("MAX_QUEUED_QUERIES")
        self.env.assertEqual(max_queued_queries, 10)

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

    def burst(self, host_queries, queries, timeout=120):
        """Send every query on its own connection, all at once, behind the
        `host_queries` (sent first, given time to start). Returns the raw
        first reply line of each of `queries`."""
        async def one(graph, cmd, q):
            r, w = await asyncio.open_connection(self.env.host, self.env.port)
            try:
                w.write(enc(cmd, graph, q))
                await w.drain()
                return await r.readline()
            finally:
                w.close()

        async def run():
            hosts = [asyncio.create_task(one(*h)) for h in host_queries]
            if hosts:
                await asyncio.sleep(0.2)
            replies = await asyncio.gather(*[one(*q) for q in queries])
            await asyncio.gather(*hosts)
            return replies

        return asyncio.run(asyncio.wait_for(run(), timeout=timeout))

    def assert_bounded(self, replies, limit):
        rejected = [r for r in replies if r.startswith(b"-")]
        completed = [r for r in replies if r.startswith(b"*")]
        # nothing but the C error text, and every query got a reply
        self.env.assertEqual(set(rejected), {MAX_PENDING_ERR})
        self.env.assertEqual(len(rejected) + len(completed), len(replies))
        self.env.assertGreater(len(rejected), 0)
        self.env.assertGreaterEqual(len(completed), limit)
        self.env.assertTrue(self.db.connection.ping())
        return len(completed)

    def assert_admission_drained(self, limit, cmd, graph, q):
        # Every slot taken above was given back: with nothing running, `limit`
        # simultaneous queries are all admitted.
        replies = self.burst([], [(graph, cmd, q)] * limit)
        self.env.assertEqual([r for r in replies if not r.startswith(b"*")], [])

    def test_04_flood_beyond_queue_capacity(self):
        # More concurrent writes than the old 1024-slot thread-pool queue and
        # the old 1024-slot per-graph write queue could hold. With bounded
        # queues the Redis main thread blocked on a full pool queue while
        # holding the GIL that running writers wait for, and pool workers
        # blocked on a full write queue while holding the graph read lock the
        # write drainer waits for: the server hung forever, PING included.
        # Runs with the default limit, which is above n: nothing is rejected.
        self.db.config_set("MAX_QUEUED_QUERIES", DEFAULT_MAX_QUEUED_QUERIES)

        n = raise_fd_limit(3000)
        graphs = ["flood_single"] + [f"flood_{i}" for i in range(8)]

        async def one(graph):
            r, w = await asyncio.open_connection(self.env.host, self.env.port)
            try:
                w.write(enc("GRAPH.QUERY", graph, FLOOD_QUERY))
                await w.drain()
                return await r.readline()
            finally:
                w.close()

        async def flood(targets):
            return await asyncio.wait_for(
                asyncio.gather(*[one(g) for g in targets]), timeout=120)

        # all writes on one graph, then spread over several graphs
        for targets in ([graphs[0]] * n,
                        [graphs[1 + i % 8] for i in range(n)]):
            hung = True
            try:
                replies = asyncio.run(flood(targets))
                hung = False
            except asyncio.TimeoutError:
                pass
            finally:
                if hung:
                    # A deadlocked server never answers the teardown either:
                    # kill it before reporting, so the failure cannot hang.
                    try:
                        os.kill(self.env.envRunner.masterProcess.pid, signal.SIGKILL)
                    except Exception:
                        # already gone; nothing left to kill
                        pass
            if hung:
                self.env.assertTrue(False, message="server hung under a write flood")
                return
            errors = [r for r in replies if not r.startswith(b"*")]
            self.env.assertEqual(errors, [])

        self.env.assertTrue(self.db.connection.ping())
        count = self.db.select_graph(graphs[0]).ro_query(
            "MATCH (n:N) RETURN count(n)").result_set[0][0]
        self.env.assertEqual(count, n * FLOOD_NODES)
        total = sum(self.db.select_graph(g).ro_query(
            "MATCH (n:N) RETURN count(n)").result_set[0][0] for g in graphs[1:])
        self.env.assertEqual(total, n * FLOOD_NODES)

    def test_05_queued_writes_and_reads_are_bounded(self):
        # Writes: one long write owns the graph's writer, so every later write
        # waits in the graph's write queue, which the limit must cover too.
        limit = 50
        n = raise_fd_limit(200)
        self.db.config_set("MAX_QUEUED_QUERIES", limit)
        graph = "bounded_writes"
        replies = self.burst([(graph, "GRAPH.QUERY", HOLD_WRITE)],
                             [(graph, "GRAPH.QUERY", "CREATE (:W)")] * n)
        completed = self.assert_bounded(replies, limit)
        count = self.db.select_graph(graph).ro_query(
            "MATCH (w:W) RETURN count(w)").result_set[0][0]
        self.env.assertEqual(count, completed)
        self.assert_admission_drained(limit, "GRAPH.QUERY", graph, "CREATE (:W)")

        # Reads: long reads occupy both pool workers (THREAD_COUNT 2), so later
        # reads wait in the pool queue.
        graph = "bounded_reads"
        self.db.select_graph(graph).query("CREATE ()")
        holds = [(graph, "GRAPH.RO_QUERY", HOLD_READ)] * 2
        replies = self.burst(holds, [(graph, "GRAPH.RO_QUERY", "MATCH (n) RETURN count(n)")] * n)
        self.assert_bounded(replies, limit)
        self.assert_admission_drained(limit, "GRAPH.RO_QUERY", graph,
                                      "MATCH (n) RETURN count(n)")

    def test_06_limit_changes_at_runtime(self):
        n = raise_fd_limit(200)
        graph = "runtime_limit"
        writes = [(graph, "GRAPH.QUERY", "CREATE (:R)")] * n

        self.db.config_set("MAX_QUEUED_QUERIES", 20)
        replies = self.burst([(graph, "GRAPH.QUERY", HOLD_WRITE)], writes)
        self.assert_bounded(replies, 20)

        # Raised above the burst: the same burst is now admitted in full.
        self.db.config_set("MAX_QUEUED_QUERIES", n + 1)
        replies = self.burst([(graph, "GRAPH.QUERY", HOLD_WRITE)], writes)
        self.env.assertEqual([r for r in replies if not r.startswith(b"*")], [])

        self.db.config_set("MAX_QUEUED_QUERIES", DEFAULT_MAX_QUEUED_QUERIES)
