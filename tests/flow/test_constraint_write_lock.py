import asyncio
from common import *
from constraint_utils import *
from long_read_utils import *
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

# target duration of one lock-holding read
READ_TARGET_SEC = 3.0

CREATE_UNIQUE    = constraint_cmd("CREATE", GRAPH_ID, "UNIQUE",    "NODE", "P", "v")
CREATE_MANDATORY = constraint_cmd("CREATE", GRAPH_ID, "MANDATORY", "NODE", "P", "v")
DROP_UNIQUE      = constraint_cmd("DROP",   GRAPH_ID, "UNIQUE",    "NODE", "P", "v")

# shared harness: runs constraint commands behind calibrated long reads
class ConstraintBehindLongReads():
    def __init__(self, **env_args):
        # enough worker threads that the reads + the commands never queue on a
        # small runner
        self.env, self.db = Env(moduleArgs="THREAD_COUNT 8", **env_args)

        # timing assertions are unreliable under instrumentation
        if SANITIZER or VALGRIND:
            self.env.skip()

        self.port  = self.env.envRunner.port
        self.conn  = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

        self.graph.query("CREATE ()")  # a read needs an existing graph
        n, self.read_dur = calibrate_long_read(self.graph, READ_TARGET_SEC)
        self.long_read = long_read_query(n)
        self.env.debugPrint(f"long read: n={n} dur={self.read_dur:.2f}s",
                            force=True)

    def setUp(self):
        self.conn.delete(GRAPH_ID)

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
    # (with probe=True) sample PING throughout.
    #
    # new reads stop before `action`: the ones in flight keep holding the lock,
    # so the commands are still waiting when the action lands, and no new read
    # hits a deleted / paused graph.
    #
    # returns (replies, latencies, worst_ping)
    def _behind_long_reads(self, cmds, action=None, probe=False):
        async def run():
            pool = BlockingConnectionPool(max_connections=64, timeout=None,
                    socket_timeout=120, port=self.port, decode_responses=True)
            g = AsyncFalkorDB(connection_pool=pool).select_graph(GRAPH_ID)
            # one client per command, plus one to run `action`
            clients = [AsyncRedis(port=self.port, decode_responses=True,
                                  socket_timeout=120) for _ in cmds + [None]]
            admin = clients.pop()
            loop = asyncio.get_running_loop()

            ping = PingProbe(self.port) if probe else None
            if ping:
                await ping.start()

            reads = ReadLockHolder(g, self.long_read)
            reads.start()
            await asyncio.sleep(0.5)  # the reads hold the read lock

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
                await reads.stop()
                await action(admin)
            results = await asyncio.gather(*cmd_tasks)

            await asyncio.sleep(0.3)  # keep probing a little past the replies
            worst = await ping.stop() if ping else 0.0
            read_errors = await reads.drain()

            for c in clients + [admin]:
                await c.aclose()
            await pool.aclose()
            return results, read_errors, worst

        results, read_errors, worst = asyncio.run(run())

        # the reads exist only to hold the lock; an error means the setup broke
        # (or the server died), which would make the assertions vacuous
        self.env.assertEqual(read_errors, [], message=str(read_errors[:3]))

        replies   = [r for r, _ in results]
        latencies = [l for _, l in results]
        self.env.debugPrint(f"replies={replies!r} latencies="
                f"{[round(l, 2) for l in latencies]} "
                f"worst_ping={worst*1000:.1f}ms", force=True)
        return replies, latencies, worst

    # run `cmd` behind the reads: it must reply `expected`, having really
    # waited for the read lock, and no PING may stall for a large part of that
    # wait (with the freeze, the worst PING ~= the command's latency)
    def _assert_waits_without_freezing(self, cmd, expected):
        replies, latencies, worst = self._behind_long_reads([cmd], probe=True)
        self.env.assertEqual(replies[0], expected)
        self.env.assertGreater(latencies[0], 0.3 * self.read_dur)
        self.env.assertLess(worst, 0.5 * latencies[0])

    # run `cmd` behind the reads with `action` landing while it waits: it must
    # fail with an error containing `expected_err`
    def _assert_aborted_by(self, cmd, action, expected_err):
        replies, _, _ = self._behind_long_reads([cmd], action=action)
        self.env.assertTrue(isinstance(replies[0], ResponseError))
        self.env.assertContains(expected_err, str(replies[0]))

    def _constraint_count(self):
        res = self.graph.ro_query("CALL db.constraints() YIELD type RETURN count(1)")
        return res.result_set[0][0]

    def _assert_operational(self, ct_type):
        wait_on_constraint(self.graph, ct_type, 'NODE', 'P', 'v')
        c = get_constraint(self.graph, ct_type, 'NODE', 'P', 'v')
        self.env.assertEqual(c.status, 'OPERATIONAL')

class testConstraintWriteLock(ConstraintBehindLongReads):
    def __init__(self):
        super().__init__()

    def test_01_create_unique_stays_responsive(self):
        self._populate(unique_index=True)
        self._assert_waits_without_freezing(CREATE_UNIQUE, "PENDING")
        self._assert_operational('UNIQUE')

    def test_02_create_mandatory_stays_responsive(self):
        self._populate()
        self._assert_waits_without_freezing(CREATE_MANDATORY, "PENDING")
        self._assert_operational('MANDATORY')

    def test_03_drop_stays_responsive(self):
        self._populate(unique_index=True)
        create_unique_node_constraint(self.graph, 'P', 'v', sync=True)
        self._assert_waits_without_freezing(DROP_UNIQUE, "OK")
        self.env.assertEqual(self._constraint_count(), 0)

    def test_04_duplicate_create_during_wait(self):
        # two identical CREATEs queue behind the same reads, both passing the
        # read-locked precheck. the second must be rejected once it holds the
        # write lock, without disturbing the first
        self._populate(unique_index=True)
        replies, _, _ = self._behind_long_reads([CREATE_UNIQUE, CREATE_UNIQUE])

        errs = [r for r in replies if isinstance(r, ResponseError)]
        self.env.assertEqual(replies.count("PENDING"), 1)
        self.env.assertEqual(len(errs), 1)
        self.env.assertContains("Constraint already exists", str(errs[0]))

        self._assert_operational('UNIQUE')
        self.env.assertEqual(self._constraint_count(), 1)

        # the graph still accepts writes and enforces the constraint
        self.graph.query("CREATE (:P {v:-1})")
        try:
            self.graph.query("CREATE (:P {v:-1})")
            self.env.assertTrue(False)
        except ResponseError as e:
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

        # long enough to cover the whole wait; lifted right after CREATE replies
        pause_ms = int((3 * self.read_dur + 10) * 1000)
        async def pause(admin):
            await admin.execute_command("CLIENT", "PAUSE", pause_ms, "WRITE")

        try:
            self._assert_aborted_by(CREATE_MANDATORY, pause,
                                    "replica traffic is currently paused")
        finally:
            self.conn.execute_command("CLIENT", "UNPAUSE")

        self.env.assertTrue(self.conn.ping())
        self.env.assertEqual(self._constraint_count(), 0)

    def test_02_graph_deleted_during_wait(self):
        # the graph is deleted while CREATE waits. CREATE must not replicate a
        # GRAPH.EFFECT for it: GRAPH.EFFECT creates a missing graph, so an AOF
        # replay (or a replica) would bring back an empty graph and apply the
        # constraint to it
        self._populate()
        async def delete(admin):
            await admin.delete(GRAPH_ID)

        self._assert_aborted_by(CREATE_MANDATORY, delete, "empty key")
        self.env.assertEqual(self.conn.exists(GRAPH_ID), 0)

        # replaying the AOF must neither crash nor resurrect the graph
        self.conn.execute_command("DEBUG", "LOADAOF")
        self.env.assertEqual(self.conn.exists(GRAPH_ID), 0)
