import time
import redis
import random
import datetime
from dateutil.relativedelta import relativedelta
from common import *

GRAPH_ID = "defrag"

class testDefrag():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

        # skip test if we're running under Sanitizer
        if SANITIZER:
            self.env.skip() # sanitizer is not working correctly with bulk

    def test_frag_ratio(self):
        #-----------------------------------------------------------------------
        # 1. Create many fragmented graphs
        #-----------------------------------------------------------------------

        #print("Creating graphs...")
        n = 100
        for i in range(n):
            g = self.db.select_graph(f"key:{i}")

            params = {
                    'a': 1,
                    'b': 2.3,
                    'c': 'some string value',
                    'd': [1,2,3, 'four', 'five'],
                    'e': True,
                    'f':{'latitude': 30, 'longitude': -27}
            }

            q = """UNWIND range(0, 200) AS x
                   CREATE ({a: $a, b: $b, c0:$c, c1: intern($c), d0: $d,
                            d1: vecf32($d[..3]), e: $e, f:point($f),
                            g:date('2025-09-15'),
                            h: localtime('07:00:00'),
                            i: localdatetime('2025-06-29T13:45:00'),
                            j: duration('P3DT12H')})"""

            g.query(q, params)

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

        try:
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

        except ResponseError:
            # Active defragmentation not supported on this build
            self.env.skip()
            return

        finally:
            #-------------------------------------------------------------------
            # 4. Restore original config
            #-------------------------------------------------------------------

            for k, v in original_cfg.items():
                self.conn.config_set(k, v)

        #-----------------------------------------------------------------------
        # 5. Assert: graphs are intact
        #-----------------------------------------------------------------------

        g = self.db.select_graph(f"key:11")

        props = {'a': 1, 'b': 2.3, 'c0': 'some string value',
                 'c1': 'some string value', 'd0': [1, 2, 3, 'four', 'five'],
                 'd1': [1.0, 2.0, 3.0], 'e': True,
                 'f': {'latitude': 30.0, 'longitude': -27.0},
                 'g': datetime.date(2025, 9, 15),
                 'h': datetime.time(7, 0),
                 'i': datetime.datetime(2025, 6, 29, 13, 45),
                 'j': relativedelta(days=3, hours=12)}

        q = """MATCH (n)
               RETURN properties(n)"""

        res = g.query(q, params).result_set
        self.env.assertGreater(len(res), 0)

        for row in res:
            self.env.assertEquals(props, row[0])

