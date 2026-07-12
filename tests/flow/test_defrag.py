import time
import redis
import random
import string
import datetime
import threading
from dateutil.relativedelta import relativedelta
from common import *

GRAPH_ID = "defrag"

class testDefrag():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

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

            q = """
                   WITH {a: $a, b: $b, c0:$c, c1: intern($c), d0: $d,
                            d1: vecf32($d[..3]), e: $e, f:point($f),
                            g:date('2025-09-15'),
                            h: localtime('07:00:00'),
                            i: localdatetime('2025-06-29T13:45:00'),
                            j: duration('P3DT12H')} as props
                   UNWIND range(0, 200) AS x
                   CREATE (s)-[e:R]->()
                   SET s = props, e = props"""

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

        q = """MATCH (n)-[e]->()
               RETURN properties(n), properties(e)"""

        res = g.query(q, params).result_set
        self.env.assertGreater(len(res), 0)

        for row in res:
            self.env.assertEquals(props, row[0])
            self.env.assertEquals(props, row[1])

    def test_concurrent_writers_during_defrag(self):
        """Regression test for issue #1831: writer queries must hold a graph
        READ lock during the match phase to prevent active defrag from
        relocating entity memory mid-execution."""

        conn = self.env.getConnection()

        #----------------------------------------------------------------------
        # 1. Seed the graph with data to defragment
        #----------------------------------------------------------------------

        g = self.db.select_graph("defrag_race")

        # create nodes with properties to give defrag something to relocate
        g.query("""UNWIND range(0, 500) AS i
                   CREATE (:Item {id: i, name: 'item_' + toString(i),
                                  value: i * 1.5, tag: 'bulk'})""")

        # delete half to create fragmentation
        g.query("MATCH (n:Item) WHERE n.id % 2 = 0 DELETE n")
        conn.execute_command("MEMORY PURGE")

        #----------------------------------------------------------------------
        # 2. Enable aggressive active defrag
        #----------------------------------------------------------------------

        keys = [
            "activedefrag",
            "active-defrag-threshold-lower",
            "active-defrag-threshold-upper",
            "active-defrag-ignore-bytes",
        ]

        original_cfg = {}
        for k in keys:
            original_cfg.update(conn.config_get(k))

        try:
            conn.config_set("activedefrag", "yes")
            conn.config_set("active-defrag-threshold-lower", "1")
            conn.config_set("active-defrag-threshold-upper", "1")
            conn.config_set("active-defrag-ignore-bytes", "1")
        except ResponseError:
            self.env.skip()
            return

        #----------------------------------------------------------------------
        # 3. Run concurrent MERGE+SET writers while defrag is active
        #----------------------------------------------------------------------

        errors = []
        stop_event = threading.Event()

        def writer_thread(thread_id):
            """Run MERGE+SET queries concurrently with defrag."""
            try:
                tg = self.db.select_graph("defrag_race")
                i = 0
                while not stop_event.is_set():
                    try:
                        tg.query(
                            "UNWIND $rows AS row "
                            "MERGE (n:Item {id: row.id}) "
                            "SET n.name = row.name, n.value = row.value",
                            {"rows": [
                                {"id": thread_id * 1000 + i,
                                 "name": f"t{thread_id}_{i}",
                                 "value": float(i)},
                                {"id": thread_id * 1000 + i + 1,
                                 "name": f"t{thread_id}_{i+1}",
                                 "value": float(i + 1)},
                            ]}
                        )
                    except Exception as e:
                        err_str = str(e)
                        if "connection" not in err_str.lower():
                            errors.append(f"Thread {thread_id}: {e}")
                        else:
                            # server crash — connection lost
                            errors.append(f"Thread {thread_id}: server crashed: {e}")
                            return
                    i += 2
            except Exception as e:
                errors.append(f"Thread {thread_id} setup: {e}")

        # start writer threads
        num_threads = 4
        threads = []
        for t in range(num_threads):
            th = threading.Thread(target=writer_thread, args=(t,))
            th.daemon = True
            th.start()
            threads.append(th)

        try:
            # let writers and defrag run concurrently
            time.sleep(3)

            # signal threads to stop
            stop_event.set()
            for th in threads:
                th.join(timeout=5)

            #------------------------------------------------------------------
            # 4. Verify server is alive and data is consistent
            #------------------------------------------------------------------

            # server should still respond
            self.env.assertTrue(conn.ping())

            # query should return valid results
            res = g.query("MATCH (n:Item) RETURN count(n)").result_set
            self.env.assertGreater(res[0][0], 0)

            # verify no connection-lost errors (would indicate a crash)
            crash_errors = [e for e in errors if "crash" in e.lower()]
            self.env.assertEquals(len(crash_errors), 0)

        finally:
            #------------------------------------------------------------------
            # 5. Restore original config
            #------------------------------------------------------------------

            stop_event.set()
            for th in threads:
                th.join(timeout=5)

            for k, v in original_cfg.items():
                try:
                    conn.config_set(k, v)
                except Exception:
                    pass

# Regression test for a use-after-free between Redis active defragmentation and
# staged SET / ON MATCH SET updates.
#
# A staged update builds its pending attribute-set with AttributeSet_ShallowClone
# (staged_updates.c): the unchanged heap-valued attributes only *alias* the live
# set. The staged set is created under the graph READ lock but committed *after*
# the writer's read->write lock-upgrade gap (query_ctx.c). Active defrag runs on
# the main thread and relocates the live attribute-set values inside that gap, so
# at commit AttributeSet_TransferOwnership sees a stale pointer:
#   - debug build   -> assertion abort at attribute_set.c:903
#   - release build -> a dangling pointer is installed -> SIGSEGV
#
# The #1834 read-lock fix protects the *match* phase but NOT this staged-set
# aliasing across the upgrade gap, so this test currently REPRODUCES the crash on
# master. It should pass once staged updates deep-persist their borrowed values
# (AttributeSet_Clone instead of ShallowClone) before the lock upgrade.
#
# This is a *soak-style* reproduction of a timing race: it drives concurrent
# writers + fragmentation across several graphs under aggressive active defrag
# and waits for the crash (which lands within tens of seconds on master).
#
# Requires a jemalloc redis-server (active defrag is a no-op under libc malloc,
# e.g. stock macOS) -> the test skips otherwise.

GRAPH_PREFIX = "defrag_staged"
NUM_GRAPHS = 6
STR_PROPS = ["s0", "s1", "s2", "s3", "s4", "s5"]

def _rnd(n):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))

class testDefragStagedUpdate():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

    def test_staged_update_uaf_during_defrag(self):
        test_start = time.time()  # bound total test time (setup + soak) to ~10s
        conn = self.conn
        # fewer distinct names => more writer/defrag contention on each entity
        names = [f"e{i}" for i in range(80)]
        graph_ids = [f"{GRAPH_PREFIX}_{i}" for i in range(NUM_GRAPHS)]

        #----------------------------------------------------------------------
        # 1. Seed entities carrying several string properties + a blob, so a
        #    single-property SET carries the *other* heap values over as
        #    shallow-cloned (aliasing) attributes -- the UAF trigger. Spread
        #    across several graphs to give defrag more memory to relocate.
        #----------------------------------------------------------------------

        sets = ", ".join([f"n.{sp}=${sp}" for sp in STR_PROPS])
        for gid in graph_ids:
            g = self.db.select_graph(gid)
            try:
                g.query("CREATE INDEX FOR (n:Ent) ON (n.name)")
            except ResponseError:
                pass
            for nm in names:
                p = {"name": nm, "blob": _rnd(1024)}
                for sp in STR_PROPS:
                    p[sp] = _rnd(random.randint(64, 256))
                g.query(f"MERGE (n:Ent {{name:$name}}) SET {sets}, n.blob=$blob", p)
            # churn to fragment the heap
            g.query("UNWIND range(0, 800) AS i CREATE (:Frag {i:i, s:'frag_' + toString(i)})")
            g.query("MATCH (n:Frag) WHERE n.i % 2 = 0 DELETE n")
        conn.execute_command("MEMORY PURGE")

        #----------------------------------------------------------------------
        # 2. Enable aggressive active defrag (skip if unsupported / not jemalloc)
        #----------------------------------------------------------------------

        # `hz` is the dominant lever: it sets how often serverCron (and thus the
        # defrag cycle) fires. RLTest launches redis at the default hz=10; bump it
        # so defrag runs ~every 10ms and reliably lands in a writer's upgrade gap.
        keys = ["activedefrag", "active-defrag-threshold-lower",
                "active-defrag-threshold-upper", "active-defrag-ignore-bytes",
                "active-defrag-cycle-min", "active-defrag-cycle-max", "hz"]
        original_cfg = {}
        for k in keys:
            try:
                original_cfg.update(conn.config_get(k))
            except ResponseError:
                pass

        try:
            conn.config_set("hz", "100")
            conn.config_set("activedefrag", "yes")
            conn.config_set("active-defrag-ignore-bytes", "100kb")
            conn.config_set("active-defrag-threshold-lower", "1")
            conn.config_set("active-defrag-threshold-upper", "1")
            conn.config_set("active-defrag-cycle-min", "25")
            conn.config_set("active-defrag-cycle-max", "75")
        except ResponseError:
            self.env.skip()
            return

        if "jemalloc" not in conn.info("memory").get("mem_allocator", ""):
            self.env.skip()  # active defrag can't relocate under libc malloc
            return

        #----------------------------------------------------------------------
        # 3. Concurrent staged-SET writers while defrag relocates memory
        #----------------------------------------------------------------------

        stop = threading.Event()
        crashed = threading.Event()
        errors = []

        def writer(tid):
            graphs = [self.db.select_graph(gid) for gid in graph_ids]
            while not stop.is_set():
                r = random.random()
                nm = random.choice(names)
                tg = random.choice(graphs)
                try:
                    if r < 0.5:
                        # first-writer SET of ONE string; carries s1..s5+blob as aliases
                        sp = random.choice(STR_PROPS)
                        tg.query(f"MATCH (n:Ent {{name:$name}}) SET n.{sp}=$v RETURN ID(n)",
                                 {"name": nm, "v": _rnd(random.randint(48, 192))})
                    elif r < 0.8:
                        # borrow-and-set: read one heap value, write another
                        a, b = random.sample(STR_PROPS, 2)
                        tg.query(f"MATCH (n:Ent {{name:$name}}) SET n.{a} = n.{b}", {"name": nm})
                    else:
                        tg.query("MERGE (n:Ent {name:$name}) ON MATCH SET n.tag=$t",
                                 {"name": nm, "t": _rnd(random.randint(48, 160))})
                except redis.exceptions.ResponseError:
                    pass  # cypher/runtime error, not a crash
                except Exception as e:  # noqa: BLE001
                    if "connection" in str(e).lower() or isinstance(
                            e, (redis.exceptions.ConnectionError, ConnectionResetError)):
                        crashed.set()
                        errors.append(f"t{tid}: server crashed: {e}")
                        return
                    errors.append(f"t{tid}: {e}")

        def churn():
            # Create nodes carrying several big blobs in the SAME jemalloc size
            # classes as the Ent string/blob properties, then delete most. This
            # fragments those classes so active defrag actually *relocates* the
            # Ent attribute values a committing writer aliases (not just scans).
            graphs = [self.db.select_graph(gid) for gid in graph_ids]
            props = ", ".join([f"b{j}:$b{j}" for j in range(8)])
            i = 0
            while not stop.is_set():
                cg = random.choice(graphs)
                try:
                    p = {f"b{j}": _rnd(random.randint(64, 1024)) for j in range(8)}
                    cg.query(f"UNWIND range(0,20) AS x CREATE (:Frag {{x:x, {props}}})", p)
                    cg.query("MATCH (n:Frag) WITH n LIMIT 12 DELETE n")
                    i += 1
                    if i % 5 == 0:            # purge occasionally, not every cycle
                        conn.execute_command("MEMORY PURGE")
                except redis.exceptions.ResponseError:
                    pass
                except Exception:  # noqa: BLE001
                    return
                time.sleep(0.01)

        threads = [threading.Thread(target=writer, args=(t,), daemon=True) for t in range(12)]
        threads += [threading.Thread(target=churn, daemon=True) for _ in range(3)]
        for th in threads:
            th.start()

        try:
            # soak for a crash, but keep the *whole* test (setup included) under
            # ~10s -- deadline is anchored to test_start, not to "now".
            deadline = test_start + 10
            while time.time() < deadline and not crashed.is_set():
                try:
                    conn.ping()
                except Exception:  # noqa: BLE001
                    crashed.set()
                    break
                time.sleep(0.5)

            stop.set()
            for th in threads:
                th.join(timeout=5)

            #------------------------------------------------------------------
            # 4. The server must still be alive (no defrag-induced UAF)
            #------------------------------------------------------------------
            self.env.assertFalse(crashed.is_set())
            self.env.assertTrue(conn.ping())
            res = self.db.select_graph(graph_ids[0]).query(
                "MATCH (n:Ent) RETURN count(n)").result_set
            self.env.assertGreater(res[0][0], 0)

        finally:
            stop.set()
            for th in threads:
                th.join(timeout=5)
            for k, v in original_cfg.items():
                try:
                    conn.config_set(k, v)
                except Exception:  # noqa: BLE001
                    pass


# Regression test: active defrag must not orphan a delegated write.
#
# A write delegated to the queue while defrag transiently owns the write election
# was never drained (defrag runs on the main thread and doesn't drain the queue),
# so the client hung forever. Stream a CREATE load while aggressive defrag churns
# the same graph; assert it finishes (no hang) with every write landed once.
# Requires jemalloc (active defrag is a no-op under libc malloc).

class testDefragWriteHang():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

    def test_write_not_orphaned_during_defrag(self):
        conn = self.conn

        # active defrag is a no-op without jemalloc (e.g. sanitizer/libc builds);
        # skip up-front, before the expensive fragmentation setup
        if "jemalloc" not in conn.info("memory").get("mem_allocator", ""):
            self.env.skip()
            return

        gname = "defrag_write_hang"
        g = self.db.select_graph(gname)

        # fragment the SAME graph the loader writes to, so defrag keeps
        # relocating its memory while writes stream in (the orphan trigger)
        for _ in range(40):
            g.query("UNWIND range(0, 4000) AS x "
                    "CREATE (:Seed {x:x, s0:'a_' + toString(x), "
                    "s1:'b_' + toString(x * 3), s2:'c_' + toString(x * 7)})"
                    "-[:LINK {w:'e_' + toString(x)}]->(:Seed {x:-x})")
        g.query("MATCH (n:Seed) WHERE n.x % 4 <> 0 DELETE n")
        conn.execute_command("MEMORY PURGE")

        # snapshot config we're about to change so the finally can restore it
        keys = ["activedefrag", "active-defrag-threshold-lower",
                "active-defrag-threshold-upper", "active-defrag-ignore-bytes",
                "active-defrag-cycle-min", "active-defrag-cycle-max", "hz"]
        original_cfg = {}
        for k in keys:
            original_cfg.update(conn.config_get(k))

        BATCHES = 150
        BATCH   = 5000
        t = None
        # one outer try/finally so config is always restored, even on a partial
        # config_set failure or an assertion failure
        try:
            try:
                conn.config_set("hz", "100")            # fires defrag ~every 10ms
                conn.config_set("activedefrag", "yes")
                conn.config_set("active-defrag-ignore-bytes", "1")
                conn.config_set("active-defrag-threshold-lower", "1")
                conn.config_set("active-defrag-threshold-upper", "1")
                conn.config_set("active-defrag-cycle-min", "25")
                conn.config_set("active-defrag-cycle-max", "75")
            except ResponseError:
                self.env.skip()
                return

            # stream the load on a thread; an orphaned batch blocks it forever,
            # which the bounded wait on `done` below detects
            state = {"completed": 0, "error": None}
            done = threading.Event()

            def loader():
                try:
                    wg = self.db.select_graph(gname)
                    for b in range(BATCHES):
                        lo = b * BATCH
                        wg.query(
                            "UNWIND range($lo, $hi) AS i CREATE (:Node {id:i})",
                            {"lo": lo, "hi": lo + BATCH - 1})
                        state["completed"] = b + 1
                except Exception as e:  # noqa: BLE001
                    state["error"] = str(e)
                finally:
                    done.set()

            t = threading.Thread(target=loader, daemon=True)
            t.start()

            # a healthy load returns in seconds; only a hang waits out the deadline
            finished = done.wait(timeout=90)

            # load must complete (no hung write) ...
            self.env.assertTrue(finished)            # False => a write hung
            self.env.assertIsNone(state["error"])
            self.env.assertEquals(state["completed"], BATCHES)

            # ... with every write landed exactly once
            if finished and state["error"] is None:
                res = self.db.select_graph(gname).query(
                    "MATCH (n:Node) RETURN count(n)").result_set
                self.env.assertEquals(res[0][0], BATCHES * BATCH)

        finally:
            if t is not None:
                t.join(timeout=5)
            # restore every key; a genuine restore failure should surface
            for k, v in original_cfg.items():
                conn.config_set(k, v)

