import math
import time
import asyncio
from common import *
from constraint_utils import *
from index_utils import wait_for_indices_to_sync
from falkordb.asyncio import FalkorDB as AsyncFalkorDB
from redis.asyncio import BlockingConnectionPool
from redis.asyncio import Redis as AsyncRedis

# GRAPH.CONSTRAINT CREATE / DROP run on a worker thread and need both the GIL
# and the graph WRITE lock. taking the GIL and then blocking on the write lock
# froze the whole server (PING included) for as long as a long read held the
# READ lock (#3753). the command now waits for the write lock in GIL-yielding
# slices, so the main thread keeps running while it waits - and can change
# state the command relies on.
#
# every test keeps a few long reads holding the READ lock, sends the command
# behind them, and checks either that the server stayed responsive, or that a
# change made on the main thread during the wait (CLIENT PAUSE, DEL, a second
# CREATE) is re-checked before the command mutates the graph and replicates.

GRAPH_ID = "constraint_write_lock"

# target duration of one lock-holding read, calibrated per machine: a fixed
# read size either takes minutes on a slow runner or finishes so fast that a
# fixed PING bound passes even with the freeze
READ_TARGET_SEC = 3.0

def _long_read(n):
    # nested UNWINDs keep memory bounded. the predicate MUST depend on both loop
    # vars, else the planner pushes it below the inner UNWIND and the read
    # finishes in ~ms holding no lock (see test_19 in test_concurrent_query.py)
    return (f"UNWIND range(0, {n}) AS a UNWIND range(0, {n}) AS b "
            f"WITH a, b WHERE a + b < 0 RETURN count(a)")

# shared harness: calibrated long reads + running commands behind them
class ConstraintBehindLongReads():
    def __init__(self, **env_args):
        # enough worker threads that 3 reads + the commands never queue on a
        # small runner
        self.env, self.db = Env(moduleArgs="THREAD_COUNT 8", **env_args)

        # timing assertions are unreliable under instrumentation
        if SANITIZER or VALGRIND:
            self.env.skip()

        self.port  = self.env.envRunner.port
        self.conn  = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self._calibrate()

    def setUp(self):
        self.conn.delete(GRAPH_ID)

    # size the long read so a single one takes ~READ_TARGET_SEC on this machine
    def _calibrate(self):
        def timed(n):
            t0 = time.monotonic()
            self.graph.ro_query(_long_read(n))
            return time.monotonic() - t0

        self.graph.query("CREATE ()")  # a read needs an existing graph
        n = 3000
        dur = timed(n)
        # cost grows with n^2
        n = int(n * math.sqrt(READ_TARGET_SEC / max(dur, 0.01)))
        n = max(1000, min(n, 40000))
        self.read_n   = n
        self.read_dur = timed(n)
        self.conn.delete(GRAPH_ID)
        self.env.debugPrint(f"long read: n={n} dur={self.read_dur:.2f}s",
                            force=True)

    def _populate(self, unique_index=False, without_v=False):
        self.graph.query("UNWIND range(1, 50000) AS x CREATE (:P {v:x})")
        if without_v:
            # MANDATORY enforcement of v ends FAILED, so the indexer never
            # re-announces the constraint. keeps that replication (#2458) out
            # of the pause test, which is about the command's own GRAPH.EFFECT
            self.graph.query("CREATE (:P)")
        if unique_index:
            self.graph.query("CREATE INDEX FOR (p:P) ON (p.v)")
            wait_for_indices_to_sync(self.graph)

    # keep 3 long reads holding the READ lock, send `cmds` concurrently behind
    # them, optionally run `action` on another connection while they wait, and
    # (with probe=True) sample PING on its own cadence throughout.
    #
    # the flood stops before `action`: the reads already running keep holding
    # the lock, so the commands are still waiting when the action lands, and no
    # new read hits a deleted / paused graph.
    #
    # returns (replies, latencies, worst_ping)
    def _behind_long_reads(self, cmds, action=None, probe=False):
        long_read = _long_read(self.read_n)

        async def run():
            pool = BlockingConnectionPool(max_connections=64, timeout=None,
                    socket_timeout=120, port=self.port, decode_responses=True)
            g = AsyncFalkorDB(connection_pool=pool).select_graph(GRAPH_ID)
            admin = AsyncRedis(port=self.port, decode_responses=True,
                               socket_timeout=120)
            clients = [AsyncRedis(port=self.port, decode_responses=True,
                                  socket_timeout=120) for _ in cmds]
            ping = AsyncRedis(port=self.port, decode_responses=True)
            await ping.ping()
            loop = asyncio.get_running_loop()

            stop_flood = False
            reads = []
            async def flood():
                while not stop_flood:
                    while sum(1 for r in reads if not r.done()) < 3 and not stop_flood:
                        reads.append(asyncio.create_task(g.ro_query(long_read)))
                    await asyncio.sleep(0.01)
            flood_task = asyncio.create_task(flood())
            await asyncio.sleep(0.5)  # the reads hold the read lock

            # sample PING on a fixed cadence in its own task, so a freeze lands
            # in a recorded sample. a failed PING (server died) counts as inf
            pings = []
            stop_ping = False
            async def ping_probe():
                while not stop_ping:
                    t0 = loop.time()
                    try:
                        await ping.ping()
                        pings.append(loop.time() - t0)
                    except Exception:
                        pings.append(float('inf'))
                    await asyncio.sleep(0.01)
            ping_task = asyncio.create_task(ping_probe()) if probe else None

            async def send(client, cmd):
                t0 = loop.time()
                try:
                    reply = await client.execute_command(*cmd)
                except Exception as e:
                    reply = e
                return reply, loop.time() - t0
            cmd_tasks = []
            for client, cmd in zip(clients, cmds):
                cmd_tasks.append(asyncio.create_task(send(client, cmd)))
                await asyncio.sleep(0.05)  # keep the send order deterministic

            if action is not None:
                await asyncio.sleep(0.5)  # the commands are waiting
                stop_flood = True
                await flood_task
                await action(admin)

            results = await asyncio.gather(*cmd_tasks)

            # keep probing a little past the replies
            await asyncio.sleep(0.3)
            stop_ping = True
            stop_flood = True
            if ping_task is not None:
                await ping_task
            await flood_task
            read_results = await asyncio.gather(*reads, return_exceptions=True)

            for c in [admin, ping] + clients:
                await c.aclose()
            await pool.aclose()
            return results, read_results, pings

        results, read_results, pings = asyncio.run(run())

        # the reads exist only to hold the read lock; an error means the setup
        # broke (or the server died), which would make the assertions vacuous
        read_errors = [r for r in read_results if isinstance(r, Exception)]
        self.env.assertEqual(read_errors, [], message=str(read_errors[:3]))

        replies   = [r for r, _ in results]
        latencies = [l for _, l in results]
        worst     = max(pings) if pings else 0.0
        self.env.debugPrint(f"replies={replies!r} latencies="
                f"{[round(l, 2) for l in latencies]} "
                f"worst_ping={worst*1000:.1f}ms", force=True)
        return replies, latencies, worst

    # the command waited behind the reads, and no PING stalled for a large part
    # of that wait. with the freeze, the worst PING ~= the command's latency
    def _assert_stayed_responsive(self, latency, worst_ping):
        # non-vacuity: the command really waited for the read lock to drain
        self.env.assertGreater(latency, 0.3 * self.read_dur)
        self.env.assertLess(worst_ping, 0.5 * latency)

    def _constraint_count(self):
        res = self.graph.ro_query("CALL db.constraints() YIELD type RETURN count(1)")
        return res.result_set[0][0]

class testConstraintWriteLock(ConstraintBehindLongReads):
    def __init__(self):
        super().__init__()

    def test_01_create_unique_stays_responsive(self):
        self._populate(unique_index=True)
        cmd = ["GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "UNIQUE", "NODE", "P",
               "PROPERTIES", "1", "v"]
        replies, latencies, worst = self._behind_long_reads([cmd], probe=True)

        self.env.assertEqual(replies[0], "PENDING")
        self._assert_stayed_responsive(latencies[0], worst)

        wait_on_constraint(self.graph, 'UNIQUE', 'NODE', 'P', 'v')
        c = get_constraint(self.graph, 'UNIQUE', 'NODE', 'P', 'v')
        self.env.assertEqual(c.status, 'OPERATIONAL')

    def test_02_create_mandatory_stays_responsive(self):
        self._populate()
        cmd = ["GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "P",
               "PROPERTIES", "1", "v"]
        replies, latencies, worst = self._behind_long_reads([cmd], probe=True)

        self.env.assertEqual(replies[0], "PENDING")
        self._assert_stayed_responsive(latencies[0], worst)

        wait_on_constraint(self.graph, 'MANDATORY', 'NODE', 'P', 'v')
        c = get_constraint(self.graph, 'MANDATORY', 'NODE', 'P', 'v')
        self.env.assertEqual(c.status, 'OPERATIONAL')

    def test_03_drop_stays_responsive(self):
        self._populate(unique_index=True)
        create_unique_node_constraint(self.graph, 'P', 'v', sync=True)
        cmd = ["GRAPH.CONSTRAINT", "DROP", GRAPH_ID, "UNIQUE", "NODE", "P",
               "PROPERTIES", "1", "v"]
        replies, latencies, worst = self._behind_long_reads([cmd], probe=True)

        self.env.assertEqual(replies[0], "OK")
        self._assert_stayed_responsive(latencies[0], worst)
        self.env.assertEqual(self._constraint_count(), 0)

    def test_04_duplicate_create_during_wait(self):
        # two identical CREATEs queue behind the same reads, both passing the
        # read-locked precheck. the second must be rejected once it holds the
        # write lock, without disturbing the first
        self._populate(unique_index=True)
        cmd = ["GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "UNIQUE", "NODE", "P",
               "PROPERTIES", "1", "v"]
        replies, _, _ = self._behind_long_reads([cmd, cmd])

        ok   = [r for r in replies if r == "PENDING"]
        errs = [r for r in replies if isinstance(r, redis.exceptions.ResponseError)]
        self.env.assertEqual(len(ok), 1)
        self.env.assertEqual(len(errs), 1)
        self.env.assertContains("Constraint already exists", str(errs[0]))

        wait_on_constraint(self.graph, 'UNIQUE', 'NODE', 'P', 'v')
        c = get_constraint(self.graph, 'UNIQUE', 'NODE', 'P', 'v')
        self.env.assertEqual(c.status, 'OPERATIONAL')
        self.env.assertEqual(self._constraint_count(), 1)

        # the graph still accepts writes and enforces the constraint
        self.graph.query("CREATE (:P {v:-1})")
        try:
            self.graph.query("CREATE (:P {v:-1})")
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("unique constraint violation", str(e))

# a server of its own with AOF on: a GRAPH.EFFECT replicated at the wrong time
# either crashes Redis (inside a pause) or is replayed by DEBUG LOADAOF. no
# constraint is ever created here, so the AOF holds no enforcement
# re-announcements, which aren't re-checked against a deleted graph either
class testConstraintWriteLockStateChanges(ConstraintBehindLongReads):
    def __init__(self):
        super().__init__(useAof=True, enableDebugCommand=True)

    def test_01_client_pause_during_wait(self):
        # a CLIENT PAUSE WRITE window opens while CREATE waits. CREATE must not
        # replicate GRAPH.EFFECT inside it: with AOF on, Redis asserts on
        # propagation during a pause and the server crashes
        self._populate(without_v=True)
        cmd = ["GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "P",
               "PROPERTIES", "1", "v"]
        # long enough to cover the whole wait; lifted right after CREATE replies
        pause_ms = int((3 * self.read_dur + 10) * 1000)
        async def pause(admin):
            await admin.execute_command("CLIENT", "PAUSE", pause_ms, "WRITE")

        try:
            replies, _, _ = self._behind_long_reads([cmd], action=pause)
        finally:
            self.conn.execute_command("CLIENT", "UNPAUSE")

        self.env.assertTrue(isinstance(replies[0], redis.exceptions.ResponseError))
        self.env.assertContains("replica traffic is currently paused",
                                str(replies[0]))
        self.env.assertTrue(self.conn.ping())
        self.env.assertEqual(self._constraint_count(), 0)

    def test_02_graph_deleted_during_wait(self):
        # the graph is deleted while CREATE waits. CREATE must not replicate a
        # GRAPH.EFFECT for it: GRAPH.EFFECT creates a missing graph, so an AOF
        # replay (or a replica) would bring back an empty graph and apply the
        # constraint to it
        self._populate()
        cmd = ["GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "P",
               "PROPERTIES", "1", "v"]
        async def delete(admin):
            await admin.delete(GRAPH_ID)

        replies, _, _ = self._behind_long_reads([cmd], action=delete)

        self.env.assertTrue(isinstance(replies[0], redis.exceptions.ResponseError))
        self.env.assertContains("deleted or replaced", str(replies[0]))
        self.env.assertEqual(self.conn.exists(GRAPH_ID), 0)

        # replaying the AOF must neither crash nor resurrect the graph
        self.conn.execute_command("DEBUG", "LOADAOF")
        self.env.assertEqual(self.conn.exists(GRAPH_ID), 0)
