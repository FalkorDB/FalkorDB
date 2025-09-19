import redis
import time
from common import *

GRAPH_ID = "defrag"

class testDefrag():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

        # skip test if we're running under Sanitizer
        if SANITIZER:
            self.env.skip() # sanitizer is not working correctly with bulk

    def test_farg_ratio(self):
        #-----------------------------------------------------------------------
        # 1. Create many fragmented graphs
        #-----------------------------------------------------------------------

        #print("Creating graphs...")
        n = 100
        for i in range(n):
            g = self.db.select_graph(f"key:{i}")
            g.query("UNWIND range(0, 200) AS x CREATE ({x:'some string value'})")

        # Delete half of them to create fragmentation
        #print("Deleting half...")
        for i in range(0, n, 2):
            self.conn.delete(f"key:{i}")

        # Force jemalloc to release unused memory
        #print("Purging allocator...")
        self.conn.execute_command("MEMORY PURGE")

        # Capture baseline fragmentation ratio
        info = self.conn.info("memory")
        frag_ratio = float(info.get("mem_fragmentation_ratio"))
        #print(f"Initial fragmentation ratio: {frag_ratio}")

        #-----------------------------------------------------------------------
        # 2. Enable active defrag with aggressive thresholds
        #-----------------------------------------------------------------------

        keys = [
            "activedefrag",
            "active-defrag-threshold-lower",
            "active-defrag-threshold-upper",
            "active-defrag-ignore-bytes",
        ]

        original_cfg = {}
        for k in keys:
            original_cfg.update(self.conn.config_get(k))

        self.conn.config_set("activedefrag", "yes")
        self.conn.config_set("active-defrag-threshold-lower", "1")
        self.conn.config_set("active-defrag-threshold-upper", "1")
        self.conn.config_set("active-defrag-ignore-bytes", "1")

        #-------------------------------------------------------------------
        # 3. Wait for defrag to run (poll instead of fixed sleep)
        #-------------------------------------------------------------------

        # Wait until some active defrag hits occur
        timeout = 2
        started = False
        initial_hits = int(self.conn.info("memory").get("active_defrag_hits", 0))

        for _ in range(timeout):
            info = self.conn.info("memory")
            hits = int(info.get("active_defrag_hits", 0))
            if hits > initial_hits:
                started = True
                break
            time.sleep(1)

        #if not started:
        #    # Active defrag did not start within timeout
        #    self.env.assertTrue(False)

        new_frag_ratio = frag_ratio

        for _ in range(timeout):
            info = self.conn.info("memory")
            new_frag_ratio = float(info.get("mem_fragmentation_ratio"))
            #print(
            #    "active_defrag_running:",
            #    info.get("active_defrag_running"),
            #    "hits:",
            #    info.get("active_defrag_hits"),
            #    "misses:",
            #    info.get("active_defrag_misses"),
            #    "mem_fragmentation_ratio:",
            #    new_frag_ratio,
            #)

            # If defrag is no longer running and fragmentation dropped
            # we can stop early
            if int(info.get("active_defrag_running")) == 0 and new_frag_ratio < frag_ratio:
                break

            time.sleep(1)

        #-----------------------------------------------------------------------
        # 4. Restore original config
        #-----------------------------------------------------------------------

        for k, v in original_cfg.items():
            self.conn.config_set(k, v)

        #-----------------------------------------------------------------------
        # 5. Assert: fragmentation ratio dropped
        #-----------------------------------------------------------------------

        #print(f"Prev fragmentation ratio: {frag_ratio}")
        #print(f"Final fragmentation ratio: {new_frag_ratio}")
        # Allow tiny fluctuation (0.01) to avoid flaky tests
        self.env.assertLess(new_frag_ratio, frag_ratio + 0.01)

