from common import *
import random
import threading

# End-to-end flow tests for Customizable Contraction Hierarchies:
#   algo.CCH        -- builds SHORTCUT edges + rank/middle properties into the graph
#   algo.CCH.query  -- rank-aware bidirectional Dijkstra + shortcut unpacking
#
# Correctness contract: a CCH query returns the SAME weight as algo.SPpaths, and
# its path is a genuine road path (every hop an original edge, contiguous, edge
# weights summing to the total). Exact edge-for-edge equality with SPpaths is only
# asserted where the shortest path is unique (ties legitimately differ).

CFG = ("relTypes:['ROAD'], weightProp:'w', shortcutRelType:'SHORTCUT', "
       "rankProp:'rank', middleProp:'mid'")
BUILD = f"CALL algo.CCH({{{CFG}}}) YIELD shortcutsCreated RETURN shortcutsCreated"
QCFG  = ("relTypes:['ROAD'], shortcutRelType:'SHORTCUT', weightProp:'w', "
         "rankProp:'rank', middleProp:'mid'")


class testCCH(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    # ---- helpers -----------------------------------------------------------
    def _reset(self, name):
        # fully drop the graph so no schema/edges survive from a prior run
        g = self.db.select_graph(name)
        try:
            g.delete()
        except Exception:
            pass
        return self.db.select_graph(name)

    def _build(self, g, cfg=CFG):
        return g.query(f"CALL algo.CCH({{{cfg}}}) YIELD shortcutsCreated "
                       "RETURN shortcutsCreated").result_set[0][0]

    def _cch(self, g, s, t, qcfg=QCFG):
        q = (f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL algo.CCH.query({{sourceNode:a,"
             f"targetNode:b,{qcfg}}}) YIELD pathWeight, path "
             "RETURN pathWeight, [n IN nodes(path)|n.v] AS vs, "
             "[r IN relationships(path)|type(r)] AS ts, "
             "reduce(x=0.0, r IN relationships(path)|x + r.w) AS psum")
        r = g.query(q).result_set
        return r[0] if r else None

    def _sp(self, g, s, t, rels="['ROAD']"):
        r = g.query(f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL algo.SPpaths("
                    f"{{sourceNode:a,targetNode:b,relTypes:{rels},weightProp:'w'}}) "
                    "YIELD pathWeight, path RETURN pathWeight, [n IN nodes(path)|n.v]").result_set
        return (r[0][0], r[0][1]) if r else (None, None)

    def _assert_valid_road_path(self, g, s, t, row, rels=('ROAD',)):
        exp, _ = self._sp(g, s, t)
        gw = row[0] if row else None
        self.env.assertEquals(gw is None, exp is None)
        if gw is None:
            return
        self.env.assertAlmostEqual(gw, exp, delta=1e-9)
        _, vs, ts, psum = row
        self.env.assertEquals(vs[0], s)
        self.env.assertEquals(vs[-1], t)
        self.env.assertTrue(all(x in rels for x in ts))
        self.env.assertAlmostEqual(psum, gw, delta=1e-9)

    def _rand_graph(self, g, n, m, seed, wmax=9):
        random.seed(seed)
        g.query(f"UNWIND range(0,{n-1}) AS i CREATE (:N {{v:i}})")
        best = {}
        for _ in range(m):
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v:
                best[(u, v)] = min(random.randint(1, wmax), best.get((u, v), 999))
        if best:
            payload = ",".join(f"[{u},{v},{w}]" for (u, v), w in best.items())
            g.query(f"UNWIND [{payload}] AS e MATCH (a:N{{v:e[0]}}),(b:N{{v:e[1]}}) "
                    f"CREATE (a)-[:ROAD {{w:e[2]}}]->(b)")

    # ---- config validation -------------------------------------------------
    def test01_build_validation(self):
        g = self._reset("cch_v1")
        # a ROAD edge so 'ROAD' is a known rel type; the weightProp/attribute
        # cases below must reach their own checks, not fail earlier on relTypes
        g.query("CREATE (:N {v:0})-[:ROAD {w:1}]->(:N {v:1})")
        bad = [
            ("CALL algo.CCH({relTypes:['ROAD']})", "exactly 5 keys"),
            ("CALL algo.CCH({relTypes:['NOPE'], weightProp:'w', shortcutRelType:'SC',"
             " rankProp:'rank', middleProp:'mid'})", "non-existent relationship type"),
            ("CALL algo.CCH({relTypes:'ROAD', weightProp:'w', shortcutRelType:'SC',"
             " rankProp:'rank', middleProp:'mid'})", "array of strings"),
            ("CALL algo.CCH({relTypes:['ROAD'], weightProp:5, shortcutRelType:'SC',"
             " rankProp:'rank', middleProp:'mid'})", "should be a string"),
            ("CALL algo.CCH({relTypes:['ROAD'], weightProp:'nope', shortcutRelType:'SC',"
             " rankProp:'rank', middleProp:'mid'})", "unknown attribute"),
        ]
        for q, msg in bad:
            try:
                g.query(q)
                self.env.assertTrue(False)
            except Exception as e:
                self.env.assertContains(msg, str(e))

    def test02_query_validation(self):
        g = self._reset("cch_v2")
        g.query("CREATE (:N {v:0})-[:ROAD{w:1}]->(:N {v:1})")
        self._build(g)
        base = "MATCH (a:N{v:0}),(b:N{v:1}) "
        bad = [
            (base + "CALL algo.CCH.query({sourceNode:a})", "targetNode"),
            (base + "CALL algo.CCH.query({sourceNode:1, targetNode:b, " + QCFG + "})", "must be nodes"),
            (base + "CALL algo.CCH.query({sourceNode:a, targetNode:b, relTypes:['ROAD'],"
                    " shortcutRelType:'NOPE', weightProp:'w', rankProp:'rank', middleProp:'mid'})",
             "unknown shortcutRelType"),
            (base + "CALL algo.CCH.query({sourceNode:a, targetNode:b, relTypes:['ROAD'],"
                    " shortcutRelType:'SHORTCUT', weightProp:'nope', rankProp:'rank', middleProp:'mid'})",
             "unknown attribute"),
        ]
        for q, msg in bad:
            try:
                g.query(q)
                self.env.assertTrue(False)
            except Exception as e:
                self.env.assertContains(msg, str(e))

    # ---- structure written to the graph -----------------------------------
    def test03_structure(self):
        g = self._reset("cch_struct")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                   (a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(a),
                   (c)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c)""")
        created = self._build(g)
        sc = g.query("MATCH ()-[r:SHORTCUT]->() RETURN count(r)").result_set[0][0]
        self.env.assertEquals(created, sc)                    # yield matches reality
        self.env.assertTrue(sc >= 2)
        ranked = g.query("MATCH (n:N) WHERE n.rank IS NOT NULL RETURN count(n)").result_set[0][0]
        self.env.assertEquals(ranked, 3)                      # every node ranked
        withmid = g.query("MATCH ()-[r:SHORTCUT]->() WHERE r.mid IS NOT NULL "
                          "RETURN count(r)").result_set[0][0]
        self.env.assertEquals(withmid, sc)                    # every shortcut has a middle

    # ---- correctness fixtures ---------------------------------------------
    def test04_valley(self):
        # a-b road is 10, but a-c-b is 2: shortcut a->b must carry 2 and unpack to a-c-b
        g = self._reset("cch_valley")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                   (a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(a),
                   (c)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c)""")
        self._build(g)
        w, vs, ts, psum = self._cch(g, 0, 1)
        self.env.assertEquals(w, 2)
        self.env.assertEquals(vs, [0, 2, 1])
        self.env.assertEquals(ts, ['ROAD', 'ROAD'])
        self.env.assertAlmostEqual(psum, 2, delta=1e-9)

    def test05_directed_asymmetric(self):
        # opposite directions carry different weights; both must be correct
        g = self._reset("cch_dir")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),(d:N{v:3}),
                   (a)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:9}]->(a),
                   (b)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:9}]->(b),
                   (c)-[:ROAD{w:1}]->(d),(d)-[:ROAD{w:9}]->(c),
                   (a)-[:ROAD{w:9}]->(d),(d)-[:ROAD{w:1}]->(a)""")
        self._build(g)
        for s in range(4):
            for t in range(4):
                if s != t:
                    self._assert_valid_road_path(g, s, t, self._cch(g, s, t))

    def test06_parallel_edges(self):
        # two ROAD edges between the same pair: CCH must use the cheaper
        g = self._reset("cch_par")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:5}]->(b),(a)-[:ROAD{w:2}]->(b),
                   (b)-[:ROAD{w:5}]->(c),(b)-[:ROAD{w:1}]->(c)""")
        self._build(g)
        w, vs, ts, psum = self._cch(g, 0, 2)
        self.env.assertAlmostEqual(w, 3, delta=1e-9)          # 2 + 1
        self.env.assertAlmostEqual(psum, w, delta=1e-9)

    def test07_self_loops(self):
        # self-loops (as in real road exports) must be ignored, not crash
        g = self._reset("cch_loop")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:0.5}]->(a),(b)-[:ROAD{w:0.5}]->(b),
                   (a)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(a),
                   (b)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(b)""")
        self._build(g)
        for s in range(3):
            for t in range(3):
                if s != t:
                    self._assert_valid_road_path(g, s, t, self._cch(g, s, t))

    def test08_multiple_reltypes(self):
        # CCH over ROAD + FERRY; query must consider both layers
        g = self._reset("cch_multi")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                   (b)-[:ROAD{w:10}]->(c),(c)-[:ROAD{w:10}]->(b),
                   (a)-[:FERRY{w:1}]->(c),(c)-[:FERRY{w:1}]->(a)""")
        self._build(g, cfg=("relTypes:['ROAD','FERRY'], weightProp:'w', "
                            "shortcutRelType:'SHORTCUT', rankProp:'rank', middleProp:'mid'"))
        qcfg = ("relTypes:['ROAD','FERRY'], shortcutRelType:'SHORTCUT', weightProp:'w', "
                "rankProp:'rank', middleProp:'mid'")
        row = self._cch(g, 0, 2, qcfg=qcfg)
        exp, _ = self._sp(g, 0, 2, rels="['ROAD','FERRY']")
        self.env.assertAlmostEqual(row[0], exp, delta=1e-9)   # ferry shortcut wins (1)
        self.env.assertAlmostEqual(row[0], 1, delta=1e-9)
        self.env.assertTrue(all(x in ('ROAD', 'FERRY') for x in row[2]))

    # ---- randomized sweep vs the exact baseline ---------------------------
    def test09_random_vs_sppaths(self):
        for ti, (n, m) in enumerate([(8, 22), (15, 60), (25, 120), (35, 200)]):
            for seed in range(3):
                g = self._reset(f"cch_rnd_{ti}_{seed}")
                self._rand_graph(g, n, m, seed * 131 + ti)
                self._build(g)
                pairs = [(s, t) for s in range(n) for t in range(n) if s != t]
                if len(pairs) > 80:
                    random.seed(seed)
                    pairs = random.sample(pairs, 80)
                for s, t in pairs:
                    self._assert_valid_road_path(g, s, t, self._cch(g, s, t))

    def test10_unique_shortest_path_exact(self):
        # distinct power-of-two weights -> unique shortest path -> exact edge match
        g = self._reset("cch_unique")
        g.query("""CREATE (n0:N{v:0}),(n1:N{v:1}),(n2:N{v:2}),(n3:N{v:3}),(n4:N{v:4}),
                   (n0)-[:ROAD{w:1}]->(n1),(n1)-[:ROAD{w:2}]->(n2),
                   (n2)-[:ROAD{w:4}]->(n3),(n3)-[:ROAD{w:8}]->(n4),
                   (n0)-[:ROAD{w:100}]->(n4),(n1)-[:ROAD{w:100}]->(n3)""")
        self._build(g)
        for s, t in [(0, 4), (0, 3), (1, 4), (0, 2)]:
            exp_w, exp_vs = self._sp(g, s, t)
            row = self._cch(g, s, t)
            self.env.assertTrue(row is not None)
            self.env.assertAlmostEqual(row[0], exp_w, delta=1e-9)
            self.env.assertEquals(row[1], exp_vs)             # exact node sequence

    def test11_edge_cases(self):
        g = self._reset("cch_edge")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(z:N{v:2}),
                   (a)-[:ROAD{w:3}]->(b),(b)-[:ROAD{w:3}]->(a)""")
        self._build(g)
        w, vs, ts, psum = self._cch(g, 0, 0)                  # src == dst
        self.env.assertEquals(w, 0)
        self.env.assertEquals(vs, [0])
        self.env.assertEquals(ts, [])
        self.env.assertTrue(self._cch(g, 0, 2) is None)       # unreachable -> no rows

    def test12_recustomize_second_metric(self):
        # a second CCH on the same graph with a different weight + own rel/props
        # must be independent and correct (side-by-side hierarchies)
        g = self._reset("cch_recust")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:1, t:10}]->(b),(b)-[:ROAD{w:1, t:10}]->(a),
                   (b)-[:ROAD{w:1, t:1}]->(c),(c)-[:ROAD{w:1, t:1}]->(b),
                   (a)-[:ROAD{w:1, t:1}]->(c),(c)-[:ROAD{w:1, t:1}]->(a)""")
        self._build(g)                                         # metric w
        g.query("CALL algo.CCH({relTypes:['ROAD'], weightProp:'t', "
                "shortcutRelType:'SC_T', rankProp:'rank_t', middleProp:'mid_t'}) "
                "YIELD shortcutsCreated RETURN shortcutsCreated")
        # query the time hierarchy: a->b optimal on t is a-c-b (1+1=2) not direct (10)
        q = ("MATCH (a:N{v:0}),(b:N{v:1}) CALL algo.CCH.query({sourceNode:a,targetNode:b,"
             "relTypes:['ROAD'],shortcutRelType:'SC_T',weightProp:'t',rankProp:'rank_t',"
             "middleProp:'mid_t'}) YIELD pathWeight RETURN pathWeight")
        wt = g.query(q).result_set[0][0]
        self.env.assertAlmostEqual(wt, 2, delta=1e-9)

    # ---- graph.ro_query must reject the write procedure --------------------
    def test13_ro_query_rejects_build(self):
        # algo.CCH modifies the graph, so it is registered as a write procedure;
        # running it through GRAPH.RO_QUERY must be refused. The read-only
        # algo.CCH.query, by contrast, is allowed through RO_QUERY.
        g = self._reset("cch_roq")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                   (a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(a),
                   (c)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c)""")

        # the write build via RO_QUERY -> rejected
        try:
            g.ro_query(BUILD)
            self.env.assertTrue(False)              # must not succeed
        except Exception as e:
            self.env.assertContains("read-only", str(e))

        # a normal (write-capable) query builds the hierarchy
        self._build(g)

        # the read-only query via RO_QUERY -> allowed
        q = ("MATCH (a:N{v:0}),(b:N{v:1}) CALL algo.CCH.query({sourceNode:a,"
             "targetNode:b," + QCFG + "}) YIELD pathWeight RETURN pathWeight")
        w = g.ro_query(q).result_set[0][0]
        self.env.assertAlmostEqual(w, 2, delta=1e-9)


# CCH is a write procedure: running algo.CCH on the master must replicate every
# modification it makes -- SHORTCUT edges (with weight + middle node) and per-node
# rank properties -- so a replica ends up with an identical, queryable hierarchy.
class testCCHReplication(FlowTestsBase):
    def __init__(self):
        # replication isn't reliable under Valgrind/sanitizer
        if VALGRIND or SANITIZER:
            Environment.skip(None)
        self.env, self.db = Env(env='oss', useSlaves=True)

    def test01_build_replicates(self):
        env = self.env
        master_con  = env.getConnection()
        replica_con = env.getSlaveConnection()

        # all FalkorDB commands are registered as write commands; allow the
        # replica to serve reads back to us
        replica_con.config_set("slave-read-only", "no")

        master  = Graph(master_con,  "cch_repl")
        replica = Graph(replica_con, "cch_repl")

        # valley graph: direct a->b costs 10 but a->c->b costs 2, so CCH must
        # create an improving SHORTCUT a->b of weight 2 whose middle node is c
        master.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                        (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                        (a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(a),
                        (c)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c)""")

        created = master.query(BUILD).result_set[0][0]
        env.assertTrue(created >= 1)

        # force master->replica sync to complete
        master_con.execute_command("WAIT", "1", "0")

        # -- SHORTCUT edges (weight + middle) identical on master and replica --
        sc_q = ("MATCH (a:N)-[r:SHORTCUT]->(b:N) "
                "RETURN a.v, b.v, r.w, r.mid ORDER BY a.v, b.v, r.w")
        master_sc  = master.ro_query(sc_q).result_set
        replica_sc = replica.ro_query(sc_q).result_set
        env.assertEquals(len(master_sc), created)      # yield == materialized
        env.assertEquals(replica_sc, master_sc)        # every shortcut replicated
        env.assertTrue(all(r[3] is not None for r in replica_sc))  # middles too
        # the improving a->b shortcut of weight 2 (unpacking via c) is present
        ab = [r for r in replica_sc if r[0] == 0 and r[1] == 1]
        env.assertTrue(any(abs(r[2] - 2) < 1e-9 for r in ab))

        # -- per-node rank properties identical on master and replica --
        rank_q = "MATCH (n:N) RETURN n.v, n.rank ORDER BY n.v"
        master_rank  = master.ro_query(rank_q).result_set
        replica_rank = replica.ro_query(rank_q).result_set
        env.assertEquals(replica_rank, master_rank)
        env.assertTrue(all(row[1] is not None for row in replica_rank))

        # -- the replicated hierarchy is functional: a CCH query on the REPLICA
        #    returns the valley path a->c->b of weight 2 --
        qrow = replica.ro_query(
            "MATCH (a:N{v:0}),(b:N{v:1}) CALL algo.CCH.query({sourceNode:a,"
            "targetNode:b," + QCFG + "}) YIELD pathWeight, path "
            "RETURN pathWeight, [n IN nodes(path)|n.v]").result_set[0]
        env.assertAlmostEqual(qrow[0], 2, delta=1e-9)
        env.assertEquals(qrow[1], [0, 2, 1])


# algo.CCH.query is a read procedure with no shared/global state (all scratch is
# per-invocation), so many threads must be able to query the SAME graph in
# parallel. This test builds the hierarchy once, then hammers algo.CCH.query
# from several threads at once and checks every answer against the single-
# threaded algo.SPpaths ground truth -- a data race would surface as a wrong
# weight or an error.
class testCCHConcurrentQuery(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    def test01_concurrent_queries(self):
        gname = "cch_concurrent"
        g = self.db.select_graph(gname)
        try:
            g.delete()
        except Exception:
            pass
        g = self.db.select_graph(gname)

        # a random digraph with enough edges to keep most pairs reachable
        n, m = 100, 500
        random.seed(99)
        g.query(f"UNWIND range(0,{n-1}) AS i CREATE (:N {{v:i}})")
        best = {}
        for _ in range(m):
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v:
                best[(u, v)] = min(random.randint(1, 9), best.get((u, v), 999))
        payload = ",".join(f"[{u},{v},{w}]" for (u, v), w in best.items())
        g.query(f"UNWIND [{payload}] AS e MATCH (a:N{{v:e[0]}}),(b:N{{v:e[1]}}) "
                f"CREATE (a)-[:ROAD {{w:e[2]}}]->(b)")
        g.query(BUILD)

        # ground-truth weights for a set of reachable pairs (single threaded)
        pairs = []
        random.seed(3)
        while len(pairs) < 40:
            s, t = random.randint(0, n - 1), random.randint(0, n - 1)
            if s == t:
                continue
            r = g.query(f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL algo.SPpaths("
                        f"{{sourceNode:a,targetNode:b,relTypes:['ROAD'],weightProp:'w'}}) "
                        "YIELD pathWeight RETURN pathWeight").result_set
            if r:
                pairs.append((s, t, r[0][0]))

        # hammer algo.CCH.query from several threads against the same graph
        THREADS, ITERS = 8, 50
        failures = []
        barrier = threading.Barrier(THREADS)

        def worker(tid):
            try:
                tg = Graph(self.env.getConnection(), gname)
                rng = random.Random(tid * 17 + 1)
            except Exception as e:
                failures.append(f"t{tid} setup: {e}")
                try: barrier.abort()
                except Exception: pass
                return
            try:
                barrier.wait(timeout=30)      # release all threads together
            except Exception:
                pass
            for _ in range(ITERS):
                s, t, exp = pairs[rng.randrange(len(pairs))]
                try:
                    res = tg.ro_query(
                        f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL algo.CCH.query("
                        f"{{sourceNode:a,targetNode:b,{QCFG}}}) YIELD pathWeight "
                        "RETURN pathWeight").result_set
                except Exception as e:
                    failures.append(f"t{tid} {s}->{t} error: {e}")
                    return
                if not res or abs(res[0][0] - exp) > 1e-9:
                    failures.append(f"t{tid} {s}->{t}: got {res} want {exp}")
                    return

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(THREADS)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        # zero mismatches / errors across all concurrent queries
        self.env.assertEquals(failures, [])
