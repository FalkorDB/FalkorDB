from common import *
from time import sleep

GRAPH_ID = "aof_replay"


class testAOFReplay():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

    def tearDown(self):
        # Leave AOF off, and the preamble on its default, for whatever runs
        # next in this environment.
        try:
            self.conn.config_set("appendonly", "no")
            self.conn.config_set("aof-use-rdb-preamble", "yes")
        except Exception:
            pass
        self.conn.flushall()

    def _wait_for_aof_rewrite(self, timeout=30):
        # Turning AOF on kicks off a background rewrite. Writes issued while it runs
        # still land in the incr file, but waiting keeps the test deterministic.
        while timeout > 0:
            if self.conn.info("persistence").get("aof_rewrite_in_progress", 0) == 0:
                return
            sleep(0.1)
            timeout -= 0.1
        raise RuntimeError("AOF rewrite did not finish in time")

    def test01_verbatim_query_survives_aof_replay(self):
        # AOF replay drives a fake client that carries CLIENT_DENY_BLOCKING but *not*
        # CLIENT_MASTER, so REPLICATED is not set for it. The write path used to guard
        # on MULTI || REPLICATED only, so that client took the blocking path — and
        # Redis asserts `(fakeClient->flags & CLIENT_BLOCKED) == 0` (aof.c) while
        # loading, killing the server with SIGSEGV. Any AOF-enabled server therefore
        # failed to restart (#2421).
        #
        # The crash needs a *verbatim* GRAPH.QUERY in the AOF. At the 300us default for
        # EFFECTS_THRESHOLD, whether a write replicates verbatim or as GRAPH.EFFECT
        # depends on timing, so push the threshold up to pin the verbatim branch.
        self.db.config_set("EFFECTS_THRESHOLD", 999999)

        # Start from a known-empty AOF. In CI this environment may be a container shared
        # with earlier test classes in the same matrix cell, and DEBUG LOADAOF replays
        # the *whole* AOF — so without this the replay would also re-apply their
        # commands. FLUSHALL clears the data, and toggling appendonly off then on starts
        # a fresh AOF sequence whose base is that empty dataset, bounding the replay
        # below to the two writes this test makes.
        self.conn.flushall()
        self.conn.config_set("appendonly", "no")
        self.conn.config_set("appendonly", "yes")
        self._wait_for_aof_rewrite()

        graph = self.db.select_graph(GRAPH_ID)
        graph.query("CREATE (:P {n:1})")
        graph.query("CREATE (:P {n:2})")

        # DEBUG LOADAOF empties the dataset and replays the AOF in-process, driving the
        # same fake loading client a restart would — it reproduces the crash exactly
        # (verified: SIGSEGV, aof.c:1692) without needing to stop the server.
        #
        # Deliberately not RLTest's restartAndReload(): it calls BGREWRITEAOF first,
        # and with aof-use-rdb-preamble on (the default) the rewritten AOF is an RDB
        # preamble, so the graph returns via RDB load and the command is never
        # replayed. Such a test passes against the unfixed build and proves nothing.
        self.conn.execute_command("DEBUG", "LOADAOF")

        # The server survived the replay...
        self.env.assertTrue(self.conn.ping())

        # ...and the graph came back through it, so the command really was replayed
        # rather than skipped.
        result = graph.query("MATCH (n) RETURN count(n)")
        self.env.assertEqual(result.result_set[0][0], 2)

    def test02_bgrewriteaof_survives_without_rdb_preamble(self):
        # With aof-use-rdb-preamble on (the default, test01's control) BGREWRITEAOF
        # writes an RDB preamble and never calls the type's aof_rewrite callback.
        # With it off, Redis rewrites every key as commands and calls
        # aof_rewrite unconditionally. Neither engine registered one (#2710), so
        # the rewrite forked a child that dereferenced a null function pointer
        # and segfaulted every time, forever, since auto-aof-rewrite-percentage
        # keeps re-triggering a rewrite that can never succeed.
        self.conn.flushall()
        self.conn.config_set("appendonly", "no")
        self.conn.config_set("aof-use-rdb-preamble", "no")
        self.conn.config_set("appendonly", "yes")
        self._wait_for_aof_rewrite()

        graph = self.db.select_graph(GRAPH_ID)
        graph.query("CREATE (:P {n:1})")
        graph.query("CREATE (:P {n:2})")

        self.conn.execute_command("BGREWRITEAOF")
        self._wait_for_aof_rewrite()

        # The child that ran the rewrite must not have crashed.
        self.env.assertTrue(self.conn.ping())
        status = self.conn.info("persistence").get("aof_last_bgrewrite_status")
        self.env.assertEqual(status, "ok")

        # The rewritten AOF must actually reconstruct the graph, not just avoid
        # crashing while producing an empty one.
        self.conn.execute_command("DEBUG", "LOADAOF")
        self.env.assertTrue(self.conn.ping())
        result = graph.query("MATCH (n) RETURN count(n)")
        self.env.assertEqual(result.result_set[0][0], 2)
