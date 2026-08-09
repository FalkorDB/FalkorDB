import time
import threading
from common import *

GRAPH_ID = "pause_replication_race"

WRITER_THREADS      = 24   # concurrent connections hammering the master with writes
STRESS_DURATION_SEC = 3    # wall-clock budget to try to hit the race
POLL_INTERVAL_SEC   = 0.1  # how often we check whether the master died
PAUSE_MS            = 10   # duration of each CLIENT PAUSE pulse
PAUSE_GAP_SEC       = 0.01 # gap between pulses, so postponed writes get a chance to drain

# expected, transient rejections a writer can hit while stressing the pause
# window / role-check this test targets - retrying past these keeps write
# pressure up for the full stress budget; anything else is unexpected and
# likely means the master died, so the writer gives up
_EXPECTED_WRITE_ERRORS = (
    "replica traffic is currently paused",
    "this instance is not a master",
    "Max pending queries exceeded",
)

stop_event = threading.Event()


def _writer(env, idx, create_counts):
    conn = env.getConnection()
    while not stop_event.is_set():
        try:
            conn.execute_command("GRAPH.QUERY", GRAPH_ID, "CREATE (n:P) RETURN 1")
            create_counts[idx] += 1
        except ResponseError as e:
            if any(msg in str(e) for msg in _EXPECTED_WRITE_ERRORS):
                continue
            return
        except Exception:
            # a real connection-level failure - most likely the master
            # crashed; the main thread will notice via the process check
            return


def _pauser(env):
    conn = env.getConnection()
    while not stop_event.is_set():
        try:
            # PAUSE ... WRITE sets PAUSE_ACTION_REPLICA, the same bit a
            # FAILOVER / CLUSTER FAILOVER opens for its pause window
            conn.execute_command("CLIENT", "PAUSE", str(PAUSE_MS), "WRITE")
        except Exception:
            return
        time.sleep(PAUSE_GAP_SEC)


# Reproduces a master crash on:
#   server.c:3600 '!(isPausedActions(PAUSE_ACTION_REPLICA) &&
#                    (!server.client_pause_in_transaction))'
#
# GRAPH.QUERY writes are admitted on the Redis main thread but actually
# mutate the graph and replicate later, on a background writer thread
# (see enter_writer_loop / _ExecuteQuery in cmd_query.c). If a
# CLIENT PAUSE / FAILOVER opens a replica-pause window in between admission
# and that later write+replicate step, nothing re-checks the pause state -
# the write proceeds anyway and crashes on propagateNow()'s invariant that
# nothing may propagate while replica traffic is paused.
#
# This continuously hammers the master with concurrent writes while pulsing
# CLIENT PAUSE ... WRITE, trying to land a pause pulse inside that window.
class testPauseReplicationRace():
    def __init__(self):
        # replication timing doesn't play well with Valgrind, and the crash
        # handler this test relies on to read a bug report is disabled
        # under sanitizers (see _test_crash_handler.py)
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env(env='oss', useSlaves=True,
                                 enableDebugCommand=True)

    def test_pause_during_inflight_write_does_not_crash_master(self):
        env = self.env
        stop_event.clear()

        create_counts = [0] * WRITER_THREADS
        writers = [threading.Thread(target=_writer, args=(env, i, create_counts),
                                     daemon=True)
                   for i in range(WRITER_THREADS)]
        pauser = threading.Thread(target=_pauser, args=(env,), daemon=True)

        for t in writers:
            t.start()
        pauser.start()

        crashed = False
        deadline = time.time() + STRESS_DURATION_SEC
        while time.time() < deadline:
            if env.envRunner.masterProcess.poll() is not None:
                crashed = True
                break
            time.sleep(POLL_INTERVAL_SEC)

        stop_event.set()
        for t in writers:
            t.join(timeout=5)
        pauser.join(timeout=5)

        if crashed:
            # don't let RLTest's teardown try to interact with the dead
            # process / print its own crash report
            env.envRunner.masterProcess = None

        env.assertFalse(crashed,
            message="master crashed under concurrent CLIENT PAUSE + "
                    "GRAPH.QUERY writes - see the master log for details")

        # data integrity: every write that came back successful must have
        # landed exactly once - no writes silently duplicated (double
        # applied around a postponed commit) or lost (dropped despite a
        # success reply) under the pause/race stress
        expected_count = sum(create_counts)
        env.assertTrue(expected_count > 0,
            message="no writes ever succeeded - the stress loop isn't "
                    "exercising the write path")

        master_graph = Graph(env.getConnection(), GRAPH_ID)
        actual_count = master_graph.ro_query(
            "MATCH (n:P) RETURN count(n)").result_set[0][0]
        env.assertEquals(actual_count, expected_count,
            message="node count diverged from the number of writes "
                    "reported successful - possible duplication or loss "
                    "under pause/race stress")
