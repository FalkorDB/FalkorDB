import time
import threading
from common import *

GRAPH_ID = "role_change_race"

WRITER_THREADS      = 24   # concurrent connections hammering the master with writes
STRESS_DURATION_SEC = 3    # wall-clock budget to try to hit the race
POLL_INTERVAL_SEC   = 0.1  # how often we check whether the master died
DEMOTE_AT_SEC       = 1.0  # demote this far into the stress window

# A demotion is driven by pointing the master at a master that does not exist:
# it becomes a replica (READONLY / SLAVE set, master_link_status:down) without
# disturbing the real replica RLTest attached, and `REPLICAOF NO ONE` restores it.
DEAD_MASTER_HOST = "127.0.0.1"
DEAD_MASTER_PORT = "1"

# Expected, transient rejections a writer can hit across a demotion.
#
# Two of these come from different layers and both are correct:
#   * "READONLY You can't write against a read only replica" - Redis rejecting a
#     *new* command at dispatch, once the instance is already a replica;
#   * "Write query aborted: this instance is not a master" - our own guard in
#     QuerySession::reauthorize_write, for a write that was already admitted on a
#     master and only reaches its commit after the demotion. That is the window
#     this test exists for, and the one Redis cannot close for us.
_EXPECTED_WRITE_ERRORS = (
    "this instance is not a master",
    # redis-py strips the `READONLY` error-code prefix, so match the text only
    "You can't write against a read only replica",
    "replica traffic is currently paused",
    "Max pending queries exceeded",
    "another write is in progress, retry the query",
    "graph was deleted or replaced while the query was running",
)

# Redis force-unblocks every blocked client when the role changes. The reply says
# the *blocking operation* was aborted - it is not a statement about whether the
# write landed, and measurement says it sometimes had: a run with 3 of these ended
# with exactly 3 more nodes than replies reported success. So these are counted
# separately and their writes treated as of unknown outcome.
_UNBLOCKED_ERROR = "UNBLOCKED force unblock from blocking operation"

stop_event = threading.Event()


def _writer(env, idx, create_counts, unblocked_counts, unexpected):
    conn = env.getConnection()
    while not stop_event.is_set():
        try:
            conn.execute_command("GRAPH.QUERY", GRAPH_ID, "CREATE (n:P) RETURN 1")
            create_counts[idx] += 1
        except ResponseError as e:
            msg = str(e)
            if _UNBLOCKED_ERROR in msg:
                unblocked_counts[idx] += 1
                continue
            if any(m in msg for m in _EXPECTED_WRITE_ERRORS):
                continue
            unexpected.append(msg)
            return
        except Exception:
            # connection-level failure - most likely the master died; the main
            # thread notices via the liveness check
            return


def _master_died(env, probe_conn):
    proc = getattr(env.envRunner, "masterProcess", None)
    if proc is not None:
        return proc.poll() is not None
    try:
        probe_conn.ping()
        return False
    except Exception:
        return True


class testRoleChangeRace(FlowTestsBase):
    def __init__(self):
        # replication timing doesn't play well with sanitizers
        if SANITIZER:
            Environment.skip(None)

        # useSlaves so the master has a replica attached: propagation is only
        # attempted at all when there is somewhere to propagate to
        # (Redis `shouldPropagate`), which is what makes these paths live.
        self.env, self.db = Env(env='oss', useSlaves=True)

    def tearDown(self):
        # Always hand the instance back as a master, or RLTest's teardown and
        # every later class inherit a read-only server.
        try:
            self.env.getConnection().execute_command("REPLICAOF", "NO", "ONE")
        except Exception:
            pass

    def test01_demotion_during_inflight_write_does_not_crash_master(self):
        env = self.env
        stop_event.clear()

        probe_conn = env.getConnection()
        admin_conn = env.getConnection()

        create_counts = [0] * WRITER_THREADS
        unblocked_counts = [0] * WRITER_THREADS
        unexpected = []
        writers = [threading.Thread(target=_writer,
                                    args=(env, i, create_counts,
                                          unblocked_counts, unexpected),
                                    daemon=True)
                   for i in range(WRITER_THREADS)]
        for t in writers:
            t.start()

        demoted = False
        crashed = False
        started = time.time()
        deadline = started + STRESS_DURATION_SEC
        while time.time() < deadline:
            if _master_died(env, probe_conn):
                crashed = True
                break
            if not demoted and time.time() - started >= DEMOTE_AT_SEC:
                # writes are in flight right now; some were admitted as master and
                # will not reach their commit until after this returns
                admin_conn.execute_command("REPLICAOF", DEAD_MASTER_HOST, DEAD_MASTER_PORT)
                demoted = True
            time.sleep(POLL_INTERVAL_SEC)

        stop_event.set()
        for t in writers:
            t.join(timeout=5)

        if crashed and getattr(env.envRunner, "masterProcess", None) is not None:
            # keep RLTest's teardown away from the dead process
            env.envRunner.masterProcess = None

        env.assertFalse(crashed,
            message="master crashed when demoted with writes in flight - "
                    "see the master log for details")
        env.assertTrue(demoted,
            message="the demotion never ran, so this test proved nothing")
        env.assertEqual(unexpected, [],
            message=f"writers saw unexpected errors: {unexpected[:3]}")

        # Back to master, and verify the instance is actually usable again rather
        # than merely alive.
        admin_conn.execute_command("REPLICAOF", "NO", "ONE")
        graph = self.db.select_graph(GRAPH_ID)
        graph.query("CREATE (:P)")

        # Data integrity across the demotion, stated as a range rather than an
        # equality, because a force-unblocked write has a genuinely unknown outcome:
        # Redis told the client its blocking operation was aborted, which says
        # nothing about whether the write had already been applied.
        #
        #   lower bound - every write that *reported success* must have landed:
        #                 anything less is silent loss
        #   upper bound - nothing may land twice: at most one node per reported
        #                 success plus one per force-unblocked write of unknown
        #                 outcome. Anything more is duplication.
        confirmed = sum(create_counts) + 1  # +1 for the CREATE just above
        ambiguous = sum(unblocked_counts)
        actual = graph.query("MATCH (n:P) RETURN count(n)").result_set[0][0]
        env.assertTrue(actual >= confirmed,
            message=f"writes were lost: {actual} nodes for {confirmed} successful "
                    f"replies")
        env.assertTrue(actual <= confirmed + ambiguous,
            message=f"writes were duplicated: {actual} nodes exceeds "
                    f"{confirmed} successful replies plus {ambiguous} "
                    f"force-unblocked writes of unknown outcome")

    def test02_telemetry_batch_held_by_a_pause_is_dropped_on_demotion(self):
        # The flusher's pause check holds prepared XADDs until replica traffic
        # resumes. That makes one interleaving the *likely* one rather than a
        # corner case: a FAILOVER opens the pause, the batch is held for its
        # duration, and the window closes at the moment this instance has become a
        # replica. Dispatching the held batch then would write telemetry directly
        # to a replica - which `enqueue_entry` refuses to do on the hot path,
        # because the master replicates its own entries to us.
        #
        # Deterministic, unlike test01: the pause guarantees the batch is still
        # pending when the demotion lands.
        env = self.env
        conn = env.getConnection()
        graph = self.db.select_graph(GRAPH_ID)
        stream = f"telemetry{{{GRAPH_ID}}}"

        # a graph to read from, and a settled telemetry stream to measure against
        graph.query("CREATE (:P {v: 1})")
        time.sleep(0.5)
        before = conn.execute_command("XLEN", stream)

        # PAUSE WRITE sets PAUSE_ACTION_REPLICA, so the flusher holds its batch.
        # Reads still run, and every query enqueues a telemetry entry.
        conn.execute_command("CLIENT", "PAUSE", "3000", "WRITE")
        for _ in range(20):
            graph.ro_query("MATCH (n:P) RETURN count(n)")
        held = conn.execute_command("XLEN", stream)
        env.assertEqual(held, before,
            message="the flusher dispatched during a pause - the pause guard "
                    "is not holding the batch, so this test cannot prove anything "
                    "about what happens to a held batch")

        # Demote while that batch is still held, then let the pause lapse so the
        # flusher wakes up as a replica.
        conn.execute_command("REPLICAOF", DEAD_MASTER_HOST, DEAD_MASTER_PORT)
        time.sleep(4)

        after = conn.execute_command("XLEN", stream)
        env.assertEqual(after, before,
            message="telemetry entries were written after this instance became a "
                    "replica - the held batch was dispatched instead of discarded")

        conn.execute_command("REPLICAOF", "NO", "ONE")
