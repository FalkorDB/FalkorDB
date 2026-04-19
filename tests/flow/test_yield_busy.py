import time
import threading

import redis
from common import *

# Regression test for https://github.com/FalkorDB/FalkorDB/issues/1893
# (PR https://github.com/FalkorDB/FalkorDB/pull/1877).
#
# In v4.18.1 (and any version since v4.16.0 / PR #1166) write queries
# executing on the Redis main thread - e.g. inside MULTI/EXEC - would
# unconditionally call RedisModule_Yield(REDISMODULE_YIELD_FLAG_CLIENTS).
# That call puts Redis into a "busy" state and dispatches buffered
# commands from other clients, which then receive BUSY errors
# ("BUSY Redis is busy ..."), surfaced as JedisBusyException in
# Jedis-based clients.
#
# After PR #1877 the yield is restricted to AOF/RDB loading only, so
# concurrent clients should never observe BUSY while a write query runs
# inside a MULTI/EXEC transaction on the main thread.

GRAPH_ID = "yield_busy"


class testYieldBusy(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    def test_no_busy_during_multi_exec_write(self):
        # MULTI/EXEC is not supported in cluster mode
        if self.env.isCluster():
            self.env.skip()

        # client A drives the MULTI/EXEC transactions
        con_a = self.env.getConnection()
        # client B observes BUSY responses from concurrent commands
        con_b = self.env.getConnection()

        # make sure both clients are connected
        self.env.assertEqual(con_a.ping(), True)
        self.env.assertEqual(con_b.ping(), True)

        # reset graph
        con_a.delete(GRAPH_ID)

        # write payload that does enough work to make the per-EXEC
        # window big enough for a concurrent command to land inside
        # the buggy yield.
        write_q = "UNWIND range(1, 200) AS i CREATE (:N {i: i})"

        stop = threading.Event()
        busy_errors = []
        other_errors = []

        def hammer():
            # continuously send a non-busy-allowed command
            # (SET is denied while Redis is in busy state)
            i = 0
            while not stop.is_set():
                try:
                    con_b.execute_command("SET", "yield_busy:probe", str(i))
                except redis.exceptions.BusyLoadingError as e:
                    busy_errors.append(str(e))
                except redis.exceptions.ResponseError as e:
                    msg = str(e)
                    if msg.startswith("BUSY"):
                        busy_errors.append(msg)
                    else:
                        other_errors.append(msg)
                except Exception as e:
                    other_errors.append(repr(e))
                i += 1

        t = threading.Thread(target=hammer)
        t.start()

        try:
            # repeatedly run write queries inside MULTI/EXEC on the
            # main thread; this is the code path that triggered the
            # unconditional RedisModule_Yield in v4.18.1.
            iterations = 100
            for _ in range(iterations):
                pipe = con_a.pipeline(transaction=True)
                # a few writes per transaction to widen the yield window
                for _ in range(3):
                    pipe.execute_command("GRAPH.QUERY", GRAPH_ID, write_q)
                pipe.execute()
        finally:
            # give the hammer thread a moment to observe any late
            # BUSY responses from in-flight commands, then stop it.
            time.sleep(0.05)
            stop.set()
            t.join(timeout=10)

        # no command issued by client B should ever come back with a
        # BUSY reply while client A is running write queries on the
        # main thread.
        self.env.assertEqual(
            busy_errors,
            [],
            message="received BUSY responses from concurrent client "
                    "while a write query was running in MULTI/EXEC: "
                    f"{busy_errors[:3]} (total {len(busy_errors)})",
        )
        # surface any other unexpected errors too
        self.env.assertEqual(
            other_errors,
            [],
            message=f"unexpected errors on concurrent client: "
                    f"{other_errors[:3]} (total {len(other_errors)})",
        )
