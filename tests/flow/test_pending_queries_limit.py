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

GRAPH_ID = "max_pending_queries"
SLOW_QUERY = "UNWIND range (0, 1000000) AS x WITH x WHERE (x / 2) = 50 RETURN x"
FLOOD_NODES = 1000
FLOOD_QUERY = f"UNWIND range(1, {FLOOD_NODES}) AS x CREATE (:N)"


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
        self.env.assertEqual(max_queued_queries, 4294967295)

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

    def test_04_flood_beyond_queue_capacity(self):
        # More concurrent writes than the old 1024-slot thread-pool queue and
        # the old 1024-slot per-graph write queue could hold. With bounded
        # queues the Redis main thread blocked on a full pool queue while
        # holding the GIL that running writers wait for, and pool workers
        # blocked on a full write queue while holding the graph read lock the
        # write drainer waits for: the server hung forever, PING included.
        self.db.config_set("MAX_QUEUED_QUERIES", 4294967295)

        n = 2500
        graphs = ["flood_single"] + [f"flood_{i}" for i in range(8)]
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = n + 256
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (want if hard == resource.RLIM_INFINITY else min(want, hard), hard))
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        n = min(n, soft - 256)

        def enc(*args):
            out = f"*{len(args)}\r\n"
            for a in args:
                out += f"${len(a.encode())}\r\n{a}\r\n"
            return out.encode()

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
            try:
                replies = asyncio.run(flood(targets))
            except asyncio.TimeoutError:
                self.env.assertTrue(False, message="server hung under a write flood")
                # a deadlocked server never answers the teardown either
                try:
                    os.kill(self.env.envRunner.masterProcess.pid, signal.SIGKILL)
                except Exception:
                    pass
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
