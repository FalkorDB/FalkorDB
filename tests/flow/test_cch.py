from common import *
import random
import threading
import time

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

    def _assert_hops_forward_directed(self, g, vs, rels=('ROAD',)):
        # every consecutive (u,v) in the returned node sequence must be realized
        # by a REAL directed edge u->v of one of 'rels' -- i.e. the path never
        # walks an edge backwards. This is the direct guard against a one-way
        # street being traversed in its forbidden direction.
        rel_pattern = "|".join(f":{r}" for r in rels)
        for i in range(len(vs) - 1):
            u, v = vs[i], vs[i + 1]
            cnt = g.query(f"MATCH (a:N{{v:{u}}})-[{rel_pattern}]->(b:N{{v:{v}}}) "
                          "RETURN count(*)").result_set[0][0]
            self.env.assertTrue(cnt > 0)   # a forward edge u->v really exists

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

    # ---- inconsistent shortcut layer must error, never crash ---------------
    def test14_corrupt_shortcut_layer_errors_gracefully(self):
        # a middleProp that SHORTCUT edges don't actually carry makes shortcut
        # unpacking fail. That path must raise a graceful error (or, when the
        # reconstruction happens to use only road hops, succeed) -- it must never
        # crash the server or return garbage. In release builds the old ASSERTs
        # in _unpack/_best_subedge were no-ops, so this exercises the runtime
        # validation that replaced them.
        g = self._reset("cch_corrupt")
        n, m = 60, 320
        random.seed(5)
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

        # 'rank' is a real attribute (on nodes) but SHORTCUT edges don't carry it,
        # so unpacking any shortcut hop fails its middle-node lookup.
        graceful = 0
        random.seed(9)
        for _ in range(60):
            s, t = random.randint(0, n - 1), random.randint(0, n - 1)
            if s == t:
                continue
            q = (f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL algo.CCH.query({{"
                 f"sourceNode:a, targetNode:b, relTypes:['ROAD'], "
                 f"shortcutRelType:'SHORTCUT', weightProp:'w', rankProp:'rank', "
                 f"middleProp:'rank'}}) YIELD pathWeight RETURN pathWeight")
            try:
                g.ro_query(q)               # road-only reconstruction is fine
            except Exception as e:
                # a crash would surface as a connection error, not this message
                self.env.assertContains("inconsistent shortcut layer", str(e))
                graceful += 1

        # at least one pair exercised the error path, and the server is alive
        self.env.assertTrue(graceful > 0)
        self.env.assertEquals(g.query("RETURN 1").result_set[0][0], 1)

    # ---- one-way streets: direction must be respected end to end -----------
    def test15_one_way_streets(self):
        # A one-way street is an edge present in only ONE direction. CCH -- both
        # the build (shortcut customization) and the query (bidirectional search
        # + unpacking) -- must never route a leg the wrong way down such an edge.
        #
        # Layout (weights all 1):
        #   0 -> 1                      ONE-WAY street (no 1 -> 0)
        #   1 <-> 2, 2 <-> 3, 3 <-> 0   a two-way ring around the back
        #   0 -> 4                      ONE-WAY into a dead-end sink (4 has no
        #                               outgoing edges at all)
        g = self._reset("cch_oneway")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),(d:N{v:3}),(e:N{v:4}),
                   (a)-[:ROAD{w:1}]->(b),
                   (b)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(b),
                   (c)-[:ROAD{w:1}]->(d),(d)-[:ROAD{w:1}]->(c),
                   (d)-[:ROAD{w:1}]->(a),(a)-[:ROAD{w:1}]->(d),
                   (a)-[:ROAD{w:1}]->(e)""")
        self._build(g)

        # forward: 0 -> 1 takes the one-way street directly (weight 1), not the
        # 3-hop back-ring detour 0 -> 3 -> 2 -> 1
        row = self._cch(g, 0, 1)
        self._assert_valid_road_path(g, 0, 1, row)
        w, vs, ts, psum = row
        self.env.assertAlmostEqual(w, 1, delta=1e-9)
        self.env.assertEquals(vs, [0, 1])
        self._assert_hops_forward_directed(g, vs)

        # reverse: 1 -> 0 must NOT reuse the 0 -> 1 street backwards; it is
        # forced onto the legal detour 1 -> 2 -> 3 -> 0 (weight 3)
        row = self._cch(g, 1, 0)
        self._assert_valid_road_path(g, 1, 0, row)
        w, vs, ts, psum = row
        self.env.assertAlmostEqual(w, 3, delta=1e-9)
        self.env.assertEquals(vs, [1, 2, 3, 0])
        self._assert_hops_forward_directed(g, vs)

        # a node reachable only via a one-way edge (0 -> 4, and 4 is a sink) is
        # reachable one way but has NO path back -- CCH must not fabricate a
        # 4 -> 0 leg out of the 0 -> 4 edge
        self.env.assertAlmostEqual(self._cch(g, 0, 4)[0], 1, delta=1e-9)
        self.env.assertTrue(self._cch(g, 4, 0) is None)

        # exhaustive: every CCH answer matches the SPpaths oracle (itself
        # direction-respecting), and every hop of every returned path is a real
        # forward-directed edge -- so no reverse traversal can slip through
        for s in range(5):
            for t in range(5):
                if s == t:
                    continue
                row = self._cch(g, s, t)
                self._assert_valid_road_path(g, s, t, row)
                if row is not None and row[0] is not None:
                    self._assert_hops_forward_directed(g, row[1])


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


# End-to-end tests for the CCH *path index* (CREATE/DROP CCH INDEX DDL + db.idx.cch.query),
# the internal-state form of CCH: no SHORTCUT edges or rank properties are written
# to the graph; the hierarchy lives inside the index. The correctness contract is
# the same as algo.CCH -- a query returns the SAME weight as algo.SPpaths -- but
# these also exercise INCREMENTAL MAINTENANCE: an edge-weight change re-customizes
# the index (scoped to the affected arcs) at query commit, and the index must stay
# exactly in sync with the graph. Convention note: for source == target this query
# returns weight 0 + a single-node path, where algo.SPpaths returns nothing.
class testCCHIndex(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    # ---- helpers -----------------------------------------------------------
    def _reset(self, name):
        g = self.db.select_graph(name)
        try:
            g.delete()
        except Exception:
            pass
        return self.db.select_graph(name)

    # convert a Cypher-list string of rel types ("['ROAD','FERRY']") to the
    # relationship-pattern form used by the DDL (ROAD|FERRY)
    def _rels_pattern(self, rels):
        types = [t.strip().strip("'\"")
                 for t in rels.strip().strip("[]").split(",") if t.strip()]
        return "|".join(types)

    # CREATE CCH INDEX DDL; returns the number of indices created
    def _create(self, g, rels="['ROAD']", wp="w"):
        return g.query(f"CREATE CCH INDEX FOR ()-[e:{self._rels_pattern(rels)}]->() "
                       f"ON (e.{wp})").indices_created

    # DROP CCH INDEX DDL
    def _drop(self, g, rels="['ROAD']", wp="w"):
        return g.query(f"DROP CCH INDEX FOR ()-[e:{self._rels_pattern(rels)}]->() "
                       f"ON (e.{wp})")

    def _idx(self, g, s, t, rels="['ROAD']", wp="w"):
        q = (f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL db.idx.cch.query({{sourceNode:a,"
             f"targetNode:b,relTypes:{rels},weightProp:'{wp}'}}) YIELD pathWeight, path "
             "RETURN pathWeight, [n IN nodes(path)|n.v] AS vs, "
             f"reduce(x=0.0, r IN relationships(path)|x + r.{wp}) AS psum")
        r = g.query(q).result_set
        return r[0] if r else None

    def _sp(self, g, s, t, rels="['ROAD']", wp="w"):
        r = g.query(f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL algo.SPpaths("
                    f"{{sourceNode:a,targetNode:b,relTypes:{rels},weightProp:'{wp}'}}) "
                    "YIELD pathWeight RETURN pathWeight").result_set
        return r[0][0] if r else None

    # assert the index query agrees with the SPpaths oracle (skip s==t, whose
    # convention differs); when a path exists, its edge weights must sum to it
    def _check(self, g, s, t):
        row = self._idx(g, s, t)
        exp = self._sp(g, s, t)
        gw  = row[0] if row else None
        self.env.assertEquals(gw is None, exp is None)
        if gw is not None:
            self.env.assertAlmostEqual(gw, exp, delta=1e-6)
            self.env.assertAlmostEqual(row[2], gw, delta=1e-6)   # psum == weight

    def _rand_graph(self, g, n, m, seed, wmax=9):
        random.seed(seed)
        g.query(f"UNWIND range(0,{n-1}) AS i CREATE (:N {{v:i}})")
        best = {}
        for _ in range(m):
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v:
                best[(u, v)] = min(random.randint(1, wmax), best.get((u, v), 999))
        payload = ",".join(f"[{u},{v},{w}]" for (u, v), w in best.items())
        g.query(f"UNWIND [{payload}] AS e MATCH (a:N{{v:e[0]}}),(b:N{{v:e[1]}}) "
                f"CREATE (a)-[:ROAD {{w:e[2]}}]->(b)")
        return list(best.keys())

    # ---- basic create / query / drop --------------------------------------
    def test01_create_query_drop(self):
        g = self._reset("cchi_basic")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                   (a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(a),
                   (c)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c)""")
        self.env.assertEquals(self._create(g), 1)

        # nothing leaked into the user graph (no SHORTCUT edges / rank props)
        self.env.assertEquals(
            g.query("MATCH ()-[r:SHORTCUT]->() RETURN count(r)").result_set[0][0], 0)

        # valley: direct a->b is 10 but a->c->b is 2
        w, vs, psum = self._idx(g, 0, 1)
        self.env.assertAlmostEqual(w, 2, delta=1e-9)
        self.env.assertEquals(vs, [0, 2, 1])

        # src == target -> weight 0, single-node path
        w, vs, psum = self._idx(g, 0, 0)
        self.env.assertEquals(w, 0)
        self.env.assertEquals(vs, [0])

        # drop, then query must error (no such index)
        self._drop(g)
        try:
            self._idx(g, 0, 1)
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("no CCH index", str(e))

    def test02_validation(self):
        g = self._reset("cchi_valid")
        g.query("CREATE (:N {v:0})-[:ROAD{w:1}]->(:N {v:1})")
        bad = [
            # CCH is a relationship pathfinding index -- a node pattern is refused
            ("CREATE CCH INDEX FOR (n:N) ON (n.w)",
             "only supported on relationships"),
            # exactly one property, the edge weight
            ("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.w, e.x)",
             "exactly one property"),
            # an unknown index-type keyword is rejected
            ("CREATE FOO INDEX FOR (n:N) ON (n.w)", "Unknown index type"),
            # a multi-label node pattern is a syntax error
            ("CREATE INDEX FOR (a:A|B) ON (a.x)", "Invalid input"),
        ]
        for q, msg in bad:
            try:
                g.query(q)
                self.env.assertTrue(False)
            except Exception as e:
                self.env.assertContains(msg, str(e))

        # duplicate create is rejected
        self._create(g)
        try:
            self._create(g)
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("already exists", str(e))

    def test03_listed_in_db_indexes(self):
        g = self._reset("cchi_list")
        g.query("CREATE (:N {v:0})-[:ROAD{w:1}]->(:N {v:1})")
        self._create(g)
        rows = g.query("CALL db.indexes() YIELD label, properties, types, entitytype "
                       "RETURN label, properties, types, entitytype").result_set
        cch = [r for r in rows if r[2] and 'w' in r[2] and r[2]['w'] == ['CCH']]
        self.env.assertEquals(len(cch), 1)
        self.env.assertEquals(cch[0][3], 'RELATIONSHIP')

    # ---- incremental maintenance: weight change -> scoped recustomize ------
    def test04_weight_update_recustomize(self):
        g = self._reset("cchi_reweight")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                   (a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(a),
                   (c)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c)""")
        self._create(g)
        self.env.assertAlmostEqual(self._idx(g, 0, 1)[0], 2, delta=1e-9)   # via c

        # raise c->b so the valley detour is no longer improving: a->b becomes 10
        g.query("MATCH (:N{v:2})-[r:ROAD]->(:N{v:1}) SET r.w=100")
        self.env.assertAlmostEqual(self._idx(g, 0, 1)[0], 10, delta=1e-9)
        self._check(g, 0, 1)

        # lower it back below the direct edge: valley wins again
        g.query("MATCH (:N{v:2})-[r:ROAD]->(:N{v:1}) SET r.w=0.5")
        self.env.assertAlmostEqual(self._idx(g, 0, 1)[0], 1.5, delta=1e-9)
        self._check(g, 0, 1)

    # ---- batched weight changes in one commit ------------------------------
    def test05_batch_weight_update(self):
        g = self._reset("cchi_batch")
        edges = self._rand_graph(g, 30, 150, 7)
        self._create(g)
        rng = random.Random(1)
        for _ in range(15):
            # change several edges in a single query (one commit -> one recustomize
            # over several dirty arcs)
            picks = [rng.choice(edges) for _ in range(4)]
            sets = [f"MATCH (:N{{v:{u}}})-[r{j}:ROAD]->(:N{{v:{v}}}) SET r{j}.w={rng.uniform(1,12):.2f}"
                    for j, (u, v) in enumerate(picks)]
            g.query(" WITH 1 AS _ ".join(sets))
        for s in range(30):
            for t in range(30):
                if s != t:
                    self._check(g, s, t)

    # ---- a bulk change must trip the full-customization fallback -----------
    def test06_large_batch_fallback(self):
        # rewriting every edge's weight in one query marks far more than n/8 arcs
        # dirty, so CCHIndex_Recustomize takes the full-customize branch. Result
        # must still match the oracle.
        g = self._reset("cchi_bulk")
        self._rand_graph(g, 40, 260, 3)
        self._create(g)
        g.query("MATCH ()-[r:ROAD]->() SET r.w = (r.w % 7) + 1")   # touches all edges
        for s in range(40):
            for t in range(0, 40, 3):
                if s != t:
                    self._check(g, s, t)

    # ---- randomized sweep: build + repeated re-weighting vs SPpaths --------
    def test07_fuzz_reweight_vs_sppaths(self):
        for ti, (n, m) in enumerate([(10, 30), (20, 90), (35, 200)]):
            for seed in range(2):
                g = self._reset(f"cchi_fz_{ti}_{seed}")
                edges = self._rand_graph(g, n, m, seed * 71 + ti)
                self._create(g)
                rng = random.Random(seed * 13 + ti)
                for _ in range(12):
                    # single or small batch weight change (scoped recustomize)
                    b = rng.randint(1, 3)
                    sets = [f"MATCH (:N{{v:{u}}})-[r{j}:ROAD]->(:N{{v:{v}}}) SET r{j}.w={rng.uniform(1,15):.2f}"
                            for j, (u, v) in enumerate(rng.choice(edges) for _ in range(b))]
                    g.query(" WITH 1 AS _ ".join(sets) if b > 1 else sets[0])
                    pairs = [(s, t) for s in range(n) for t in range(n) if s != t]
                    for s, t in random.sample(pairs, min(25, len(pairs))):
                        self._check(g, s, t)

    # ---- Phase A: incremental topology maintenance -------------------------
    def test08_edge_delete_and_readd(self):
        # deleting an edge keeps its chordal arc (re-seeded); re-adding it revives
        # the same arc -- both are scoped recustomizations, and both must leave the
        # index in sync with the graph
        g = self._reset("cchi_deledge")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                   (a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(a),
                   (c)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c)""")
        self._create(g)
        self.env.assertAlmostEqual(self._idx(g, 0, 1)[0], 2, delta=1e-9)   # via c

        # delete c->b: the valley detour is gone, a->b falls back to the direct 10
        g.query("MATCH (:N{v:2})-[r:ROAD]->(:N{v:1}) DELETE r")
        self.env.assertAlmostEqual(self._idx(g, 0, 1)[0], 10, delta=1e-9)
        self._check(g, 0, 1)

        # re-create c->b cheaper than before: the valley detour returns
        g.query("MATCH (a:N{v:2}),(b:N{v:1}) CREATE (a)-[:ROAD{w:0.5}]->(b)")
        self.env.assertAlmostEqual(self._idx(g, 0, 1)[0], 1.5, delta=1e-9)
        self._check(g, 0, 1)

    def test09_edge_add_new_adjacency(self):
        # adding an edge between a pair with no chordal arc introduces a new
        # adjacency -> the index escalates to a full rebuild and stays correct
        g = self._reset("cchi_newedge")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),(d:N{v:3}),
                   (a)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(d)""")
        self._create(g)
        self.env.assertAlmostEqual(self._idx(g, 0, 3)[0], 3, delta=1e-9)   # a-b-c-d

        # a brand-new shortcut edge a->d of weight 1 (no prior a,d adjacency)
        g.query("MATCH (a:N{v:0}),(d:N{v:3}) CREATE (a)-[:ROAD{w:1}]->(d)")
        self.env.assertAlmostEqual(self._idx(g, 0, 3)[0], 1, delta=1e-9)
        self._check(g, 0, 3)

    def test10_topology_fuzz_vs_sppaths(self):
        # random mix of weight change / delete / new-adjacency add / re-add,
        # checked against the SPpaths oracle after every mutation
        for ti, (n, m) in enumerate([(12, 40), (25, 120), (45, 260)]):
            g = self._reset(f"cchi_topo_{ti}")
            edges = {e: random.randint(1, 9) for e in self._rand_graph_map(g, n, m, ti * 17)}
            self._create(g)
            rng = random.Random(ti * 7 + 1)
            deleted = []
            for _ in range(40):
                op = rng.choice(["w", "w", "del", "del", "add", "readd"])
                if op == "w" and edges:
                    u, v = rng.choice(list(edges)); w = rng.randint(1, 15)
                    edges[(u, v)] = w
                    g.query(f"MATCH (:N{{v:{u}}})-[r:ROAD]->(:N{{v:{v}}}) SET r.w={w}")
                elif op == "del" and edges:
                    u, v = rng.choice(list(edges)); edges.pop((u, v))
                    g.query(f"MATCH (:N{{v:{u}}})-[r:ROAD]->(:N{{v:{v}}}) WITH r LIMIT 1 DELETE r")
                    deleted.append((u, v))
                elif op == "add":
                    p = None
                    for _ in range(20):
                        u, v = rng.randint(0, n - 1), rng.randint(0, n - 1)
                        if u != v and (u, v) not in edges: p = (u, v); break
                    if p is None: continue
                    w = rng.randint(1, 9); edges[p] = w
                    g.query(f"MATCH (a:N{{v:{p[0]}}}),(b:N{{v:{p[1]}}}) CREATE (a)-[:ROAD{{w:{w}}}]->(b)")
                elif op == "readd" and deleted:
                    u, v = deleted.pop(rng.randrange(len(deleted)))
                    if (u, v) in edges: continue
                    w = rng.randint(1, 12); edges[(u, v)] = w
                    g.query(f"MATCH (a:N{{v:{u}}}),(b:N{{v:{v}}}) CREATE (a)-[:ROAD{{w:{w}}}]->(b)")
                else:
                    continue
                allp = [(s, t) for s in range(n) for t in range(n) if s != t]
                for s, t in random.sample(allp, min(15, len(allp))):
                    self._check(g, s, t)

    def test11_staleness_valve(self):
        # deleting many edges keeps stale arcs until the valve rebuilds; correctness
        # must hold across the threshold (1024 deletions for a small graph)
        g = self._reset("cchi_valve")
        n = 50
        rng = random.Random(9)
        edges = self._rand_graph_map(g, n, 1500, 3)
        self._create(g)
        for i, (u, v) in enumerate(edges[:1100]):   # crosses the 1024 valve threshold
            g.query(f"MATCH (:N{{v:{u}}})-[r:ROAD]->(:N{{v:{v}}}) WITH r LIMIT 1 DELETE r")
            if i % 200 == 0:
                for _ in range(6):
                    s, t = rng.randint(0, n - 1), rng.randint(0, n - 1)
                    if s != t:
                        self._check(g, s, t)

    # build a random graph and return its edge list (parallel helper to _rand_graph)
    def _rand_graph_map(self, g, n, m, seed):
        return self._rand_graph(g, n, m, seed)

    # ---- query-side validation / access control ---------------------------
    def test12_query_validation(self):
        g = self._reset("cchi_qvalid")
        # two metrics so we can request one that has no index
        g.query("CREATE (:N{v:0})-[:ROAD{w:1, t:2}]->(:N{v:1})")
        self._create(g, wp="w")
        base = "MATCH (a:N{v:0}),(b:N{v:1}) "
        bad = [
            (base + "CALL db.idx.cch.query({sourceNode:a})", "sourceNode and targetNode"),
            (base + "CALL db.idx.cch.query({sourceNode:1, targetNode:b, relTypes:['ROAD'],"
                    " weightProp:'w'})", "must be nodes"),
            (base + "CALL db.idx.cch.query({sourceNode:a, targetNode:b, relTypes:['NOPE'],"
                    " weightProp:'w'})", "unknown relationship type"),
            (base + "CALL db.idx.cch.query({sourceNode:a, targetNode:b, relTypes:['ROAD'],"
                    " weightProp:'nope'})", "unknown attribute"),
            # 't' is a real attribute but no CCH index is built over it
            (base + "CALL db.idx.cch.query({sourceNode:a, targetNode:b, relTypes:['ROAD'],"
                    " weightProp:'t'})", "no CCH index"),
        ]
        for q, msg in bad:
            try:
                g.query(q)
                self.env.assertTrue(False)
            except Exception as e:
                self.env.assertContains(msg, str(e))

    def test13_ro_query_access(self):
        # create + drop mutate index state (write procs) -> refused via RO_QUERY;
        # the read-only query is allowed through
        g = self._reset("cchi_roq")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                   (a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(a),
                   (c)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c)""")
        try:
            g.ro_query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("read-only", str(e))

        self._create(g)
        w = g.ro_query("MATCH (a:N{v:0}),(b:N{v:1}) CALL db.idx.cch.query({sourceNode:a,"
                       "targetNode:b,relTypes:['ROAD'],weightProp:'w'}) YIELD pathWeight "
                       "RETURN pathWeight").result_set[0][0]
        self.env.assertAlmostEqual(w, 2, delta=1e-9)

        try:
            g.ro_query("DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("read-only", str(e))

    # ---- graph-shape correctness through the index query -------------------
    def test14_parallel_selfloop_unreachable(self):
        g = self._reset("cchi_par")
        # parallel edges (cheapest wins), a self-loop (ignored), an isolated node
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),(z:N{v:3}),
                   (a)-[:ROAD{w:5}]->(b),(a)-[:ROAD{w:2}]->(b),
                   (b)-[:ROAD{w:5}]->(c),(b)-[:ROAD{w:1}]->(c),
                   (a)-[:ROAD{w:0.5}]->(a)""")
        self._create(g)
        self.env.assertAlmostEqual(self._idx(g, 0, 2)[0], 3, delta=1e-9)   # 2 + 1
        self._check(g, 0, 2)
        self._check(g, 0, 1)                                # self-loop ignored, no crash
        self.env.assertTrue(self._idx(g, 0, 3) is None)     # unreachable -> no rows

    def test15_one_way_streets(self):
        # direction must be respected: a one-way edge is never walked backwards
        g = self._reset("cchi_oneway")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),(d:N{v:3}),(e:N{v:4}),
                   (a)-[:ROAD{w:1}]->(b),
                   (b)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(b),
                   (c)-[:ROAD{w:1}]->(d),(d)-[:ROAD{w:1}]->(c),
                   (d)-[:ROAD{w:1}]->(a),(a)-[:ROAD{w:1}]->(d),
                   (a)-[:ROAD{w:1}]->(e)""")
        self._create(g)
        self.env.assertAlmostEqual(self._idx(g, 0, 1)[0], 1, delta=1e-9)   # direct one-way
        self.env.assertAlmostEqual(self._idx(g, 1, 0)[0], 3, delta=1e-9)   # forced detour
        self.env.assertAlmostEqual(self._idx(g, 0, 4)[0], 1, delta=1e-9)   # into a sink
        self.env.assertTrue(self._idx(g, 4, 0) is None)                    # no way back
        for s in range(5):
            for t in range(5):
                if s != t:
                    self._check(g, s, t)

    def test16_multiple_reltypes(self):
        # a CCH index over ROAD + FERRY; the ferry shortcut must win, and a weight
        # change on either relationship type maintains the index
        g = self._reset("cchi_multi")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                   (b)-[:ROAD{w:10}]->(c),(c)-[:ROAD{w:10}]->(b),
                   (a)-[:FERRY{w:1}]->(c),(c)-[:FERRY{w:1}]->(a)""")
        self._create(g, rels="['ROAD','FERRY']")
        self.env.assertAlmostEqual(
            self._idx(g, 0, 2, rels="['ROAD','FERRY']")[0], 1, delta=1e-9)   # ferry (1)

        # raise the ferry: now the road path a-b-c (20) is optimal -> maintenance
        # must react to a change on the FERRY relationship type
        g.query("MATCH (:N{v:0})-[r:FERRY]->(:N{v:2}) SET r.w=100")
        row = self._idx(g, 0, 2, rels="['ROAD','FERRY']")
        exp = self._sp(g, 0, 2, rels="['ROAD','FERRY']")
        self.env.assertAlmostEqual(row[0], exp, delta=1e-6)
        self.env.assertAlmostEqual(row[0], 20, delta=1e-9)

    def test17_multiple_indices_independent(self):
        # two CCH indices over the same relationship type but different metrics must
        # be independent; a change to one metric maintains both (correctly) and a
        # drop of one leaves the other intact
        g = self._reset("cchi_two")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:1, t:10}]->(b),(b)-[:ROAD{w:1, t:10}]->(a),
                   (b)-[:ROAD{w:1, t:1}]->(c),(c)-[:ROAD{w:1, t:1}]->(b),
                   (a)-[:ROAD{w:1, t:1}]->(c),(c)-[:ROAD{w:1, t:1}]->(a)""")
        self._create(g, wp="w")
        self._create(g, wp="t")
        self.env.assertAlmostEqual(self._idx(g, 0, 1, wp="w")[0], 1, delta=1e-9)   # direct
        self.env.assertAlmostEqual(self._idx(g, 0, 1, wp="t")[0], 2, delta=1e-9)   # via c

        # change the 't' metric: the t-index updates, the w-index is unaffected
        g.query("MATCH (:N{v:2})-[r:ROAD]->(:N{v:1}) SET r.t=100")
        self.env.assertAlmostEqual(self._idx(g, 0, 1, wp="t")[0], 10, delta=1e-9)  # detour gone
        self.env.assertAlmostEqual(self._idx(g, 0, 1, wp="w")[0], 1, delta=1e-9)   # intact

        # drop the w-index; the t-index still answers
        g.query("DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)")
        self.env.assertAlmostEqual(self._idx(g, 0, 1, wp="t")[0], 10, delta=1e-9)
        try:
            self._idx(g, 0, 1, wp="w")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("no CCH index", str(e))

    def test18_node_add_delete_rebuilds(self):
        # node add / delete change the id-space -> full rebuild; the index must stay
        # in sync with the graph across both
        g = self._reset("cchi_node")
        self._rand_graph(g, 20, 90, 5)
        self._create(g)
        # add a node and splice it into a path
        g.query("MATCH (u:N{v:0}) CREATE (u)-[:ROAD{w:2}]->(:N{v:100})")
        g.query("MATCH (u:N{v:100}),(w:N{v:1}) CREATE (u)-[:ROAD{w:2}]->(w)")
        for s in (0, 1, 2, 100):
            for t in (0, 1, 2, 100):
                if s != t:
                    self._check(g, s, t)
        # delete a node
        g.query("MATCH (v:N{v:5}) DETACH DELETE v")
        for s in range(0, 20, 3):
            for t in range(0, 20, 4):
                if s != t and s != 5 and t != 5:
                    self._check(g, s, t)

    def test20_counted_in_memory_report(self):
        # the CCH index owns a resident hierarchy (not RediSearch-backed); it must
        # be accounted for in GRAPH.MEMORY's indices_sz_mb. use a grid big enough
        # that the hierarchy is several MB so the MB-rounded report moves.
        g = self._reset("cchi_mem")
        N = 80
        g.query("CREATE INDEX FOR (n:N) ON (n.v)")           # fast grid construction
        g.query(f"UNWIND range(0,{N*N-1}) AS i CREATE (:N {{v:i}})")
        edges = []
        for x in range(N):
            for y in range(N):
                for dx, dy in ((1, 0), (0, 1)):
                    nx, ny = x + dx, y + dy
                    if nx < N and ny < N:
                        a, b = x * N + y, nx * N + ny
                        edges.append((a, b)); edges.append((b, a))
        B = 20000
        for i in range(0, len(edges), B):
            lit = "[" + ",".join(f"[{a},{b}]" for a, b in edges[i:i + B]) + "]"
            g.query(f"UNWIND {lit} AS e MATCH (a:N{{v:e[0]}}),(b:N{{v:e[1]}}) "
                    f"CREATE (a)-[:ROAD {{w:1.0}}]->(b)")

        conn = self.env.getConnection()

        def indices_mb():
            res = conn.execute_command("GRAPH.MEMORY", "USAGE", "cchi_mem",
                                       "SAMPLES", 100)
            if isinstance(res, dict):
                return res["indices_sz_mb"]
            return res[res.index("indices_sz_mb") + 1]     # flat [k,v,...] reply

        before = indices_mb()
        self._create(g)
        after = indices_mb()
        self.env.assertTrue(after > before)                 # hierarchy is accounted for

        g.query("DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)")
        self.env.assertTrue(indices_mb() < after)           # freed on drop

    def test19_bulk_edge_create(self):
        # several edges created in ONE query -- the bulk CreateEdges path -- with an
        # index present: a batch containing new adjacencies rebuilds (via the
        # pending-rebuild short-circuit), a batch touching only existing arcs
        # re-customizes. either way the index must match the oracle
        g = self._reset("cchi_bulk_edge")
        self._rand_graph(g, 25, 120, 8)
        self._create(g)
        # a batch of (mostly new) adjacencies in a single query
        g.query("""UNWIND [[0,10],[1,11],[2,12],[3,13],[4,14]] AS e
                   MATCH (a:N{v:e[0]}),(b:N{v:e[1]}) CREATE (a)-[:ROAD{w:2}]->(b)""")
        # a batch of parallel edges on those same pairs (arcs now exist)
        g.query("""UNWIND [[0,10],[1,11],[2,12]] AS e
                   MATCH (a:N{v:e[0]}),(b:N{v:e[1]}) CREATE (a)-[:ROAD{w:0.5}]->(b)""")
        for s in range(0, 25, 3):
            for t in range(0, 25, 4):
                if s != t:
                    self._check(g, s, t)

    def test21_multi_reltype_key_exactness(self):
        # a CCH index is keyed by its FULL (relationship-type set, weight attr).
        # {ROAD}, {PATH} and {ROAD,PATH} over the same weight are three distinct,
        # simultaneously-live indices; a query must match the exact key -- an index
        # over {ROAD,PATH} is NOT usable by a query for just {ROAD}.
        g = self._reset("cchi_key")
        # 0->1 costs: ROAD-only 2 (0-2-1), PATH-only 1.5 (direct),
        # ROAD+PATH 1.3 (ROAD 0->2 + PATH 2->1) -- three different answers, so a
        # query provably uses only its own index's relationship types
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(b),
                   (a)-[:PATH{w:1.5}]->(b),(c)-[:PATH{w:0.3}]->(b)""")

        self.env.assertEquals(self._create(g, rels="['ROAD']"), 1)
        self.env.assertEquals(self._create(g, rels="['PATH']"), 1)
        self.env.assertEquals(self._create(g, rels="['ROAD','PATH']"), 1)

        # all three coexist
        rows = g.query("CALL db.indexes() YIELD types RETURN types").result_set
        self.env.assertEquals(len([r for r in rows if r[0] and r[0].get('w') == ['CCH']]), 3)

        # each key answers over its own relationship-type set, matching SPpaths
        self.env.assertAlmostEqual(self._idx(g, 0, 1, rels="['ROAD']")[0], 2, delta=1e-9)
        self.env.assertAlmostEqual(self._idx(g, 0, 1, rels="['PATH']")[0], 1.5, delta=1e-9)
        self.env.assertAlmostEqual(
            self._idx(g, 0, 1, rels="['ROAD','PATH']")[0], 1.3, delta=1e-9)
        for rels in ("['ROAD']", "['PATH']", "['ROAD','PATH']"):
            row = self._idx(g, 0, 1, rels=rels)
            self.env.assertAlmostEqual(row[0], self._sp(g, 0, 1, rels=rels), delta=1e-6)

        # dropping {ROAD} leaves {PATH} and {ROAD,PATH} intact (drop is keyed too)
        self._drop(g, rels="['ROAD']")
        rows = g.query("CALL db.indexes() YIELD types RETURN types").result_set
        self.env.assertEquals(len([r for r in rows if r[0] and r[0].get('w') == ['CCH']]), 2)
        self.env.assertAlmostEqual(self._idx(g, 0, 1, rels="['PATH']")[0], 1.5, delta=1e-9)
        self.env.assertAlmostEqual(
            self._idx(g, 0, 1, rels="['ROAD','PATH']")[0], 1.3, delta=1e-9)

        # with no {ROAD} index left, a query for just {ROAD} must NOT fall back to
        # the {ROAD,PATH} index -- it errors
        try:
            self._idx(g, 0, 1, rels="['ROAD']")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("no CCH index", str(e))

        # likewise a drop must match the exact key
        try:
            self._drop(g, rels="['ROAD']")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("no CCH index", str(e))

    def test22_weight_attr_is_part_of_key(self):
        # same relationship type, different weight attributes -> distinct indices
        g = self._reset("cchi_wkey")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:1, t:10}]->(b),(b)-[:ROAD{w:1, t:1}]->(c),
                   (c)-[:ROAD{w:1, t:1}]->(b),(a)-[:ROAD{w:1, t:1}]->(c)""")
        self.env.assertEquals(self._create(g, wp="w"), 1)
        self.env.assertEquals(self._create(g, wp="t"), 1)   # not a duplicate

        rows = g.query("CALL db.indexes() YIELD properties RETURN properties").result_set
        cch_props = sorted(p[0][0] for p in rows if p[0])
        self.env.assertEquals(cch_props, ["t", "w"])

        # a query must match the exact weight attribute
        self.env.assertAlmostEqual(self._idx(g, 0, 1, wp="w")[0], 1, delta=1e-9)   # direct
        self.env.assertAlmostEqual(self._idx(g, 0, 1, wp="t")[0], 2, delta=1e-9)   # via c

    def test23_creates_missing_reltypes_and_attr(self):
        # like any CREATE INDEX, a CCH create may introduce brand-new relationship
        # types and a new weight attribute -- they are created (via GraphHub, so
        # they get their own schema/attribute effects) rather than rejected
        g = self._reset("cchi_new")
        g.query("CREATE (:N{v:0}),(:N{v:1})")   # nodes only; HWY/SEA/km don't exist

        def reltypes():
            return {r[0] for r in g.query(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType").result_set}
        def propkeys():
            return {r[0] for r in g.query(
                "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey").result_set}

        self.env.assertTrue("HWY" not in reltypes())
        self.env.assertTrue("km" not in propkeys())

        # create over two new reltypes + a new weight attribute
        self.env.assertEquals(self._create(g, rels="['HWY','SEA']", wp="km"), 1)

        # all were created
        self.env.assertTrue("HWY" in reltypes())
        self.env.assertTrue("SEA" in reltypes())
        self.env.assertTrue("km" in propkeys())

        # the index is live and queryable by its (new) key
        rows = g.query("CALL db.indexes() YIELD label, types RETURN label, types").result_set
        self.env.assertTrue(any(t and t.get('km') == ['CCH'] for (_, t) in rows))
        # empty subgraph -> src reaches only itself; unreachable target -> no rows
        self.env.assertTrue(self._idx(g, 0, 1, rels="['HWY','SEA']", wp="km") is None)

    def test24_duplicate_reltypes_canonicalized(self):
        # the relationship-type set is the index identity, so a repeated type is
        # de-duplicated: ROAD|ROAD == {ROAD}. it is NOT a second, distinct index.
        g = self._reset("cchi_dup")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(b),
                   (a)-[:PATH{w:1}]->(b)""")
        self._create(g, rels="['ROAD']")

        # ROAD|ROAD canonicalizes to {ROAD} -> a duplicate, not a new index
        try:
            self._create(g, rels="['ROAD','ROAD']")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("already exists", str(e))

        # exactly one CCH index, keyed {ROAD}
        rows = g.query("CALL db.indexes() YIELD label, types RETURN label, types").result_set
        cch = [lbl for (lbl, t) in rows if t and t.get('w') == ['CCH']]
        self.env.assertEquals(cch, ["ROAD"])

        # a query / drop with a duplicated key resolves to the same {ROAD} index
        self.env.assertAlmostEqual(
            self._idx(g, 0, 1, rels="['ROAD','ROAD']")[0], 2, delta=1e-9)   # via c
        self._drop(g, rels="['ROAD','ROAD']")
        self.env.assertTrue(all(t.get('w') != ['CCH']
            for (_, t) in g.query("CALL db.indexes() YIELD label, types "
                                  "RETURN label, types").result_set if t))

        # a repeat within a multi-type set collapses too: ROAD|PATH|ROAD == {ROAD,PATH}
        self._create(g, rels="['ROAD','PATH','ROAD']")
        rows = g.query("CALL db.indexes() YIELD label RETURN label").result_set
        # label is the joined sorted rel-type set -> "PATH,ROAD" or "ROAD,PATH"
        cch_label = [r[0] for r in rows][0]
        self.env.assertEquals(sorted(cch_label.split(",")), ["PATH", "ROAD"])
        # queryable by the de-duplicated multi-type key, matching SPpaths
        row = self._idx(g, 0, 1, rels="['ROAD','PATH','ROAD']")
        self.env.assertAlmostEqual(
            row[0], self._sp(g, 0, 1, rels="['ROAD','PATH']"), delta=1e-6)

    def test25_reltype_key_is_order_agnostic(self):
        # the relationship-type set is order-agnostic: an index created over A|B
        # is the same index whether create / query / drop list it as A|B or B|A.
        g = self._reset("cchi_order")
        # a->b best (1.3) uses ROAD 0->2 then PATH 2->1, so the answer genuinely
        # depends on BOTH types being in the index
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(a)-[:ROAD{w:1}]->(c),(c)-[:PATH{w:0.3}]->(b)""")
        self._create(g, rels="['ROAD','PATH']")

        # creating the same set in reverse order is a duplicate, not a new index
        try:
            self._create(g, rels="['PATH','ROAD']")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("already exists", str(e))
        cch = [t for (_, t) in g.query("CALL db.indexes() YIELD label, types "
               "RETURN label, types").result_set if t and t.get('w') == ['CCH']]
        self.env.assertEquals(len(cch), 1)

        # querying either order hits the same index with the same answer
        w_ab = self._idx(g, 0, 1, rels="['ROAD','PATH']")[0]
        w_ba = self._idx(g, 0, 1, rels="['PATH','ROAD']")[0]
        self.env.assertAlmostEqual(w_ab, 1.3, delta=1e-9)
        self.env.assertAlmostEqual(w_ba, 1.3, delta=1e-9)

        # dropping with the reversed order removes that same index
        self._drop(g, rels="['PATH','ROAD']")
        self.env.assertTrue(all(t.get('w') != ['CCH']
            for (_, t) in g.query("CALL db.indexes() YIELD label, types "
                                  "RETURN label, types").result_set if t))

    def test26_drop_key_is_a_set(self):
        # DROP resolves the index by the same canonical (set, weight) key as
        # create/query: order-agnostic and de-duplicated, and it removes EXACTLY
        # the matching index -- not a sibling that shares some relationship types.
        g = self._reset("cchi_dropkey")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:1}]->(c),(c)-[:PATH{w:1}]->(b)""")
        self._create(g, rels="['ROAD']")
        self._create(g, rels="['PATH']")
        self._create(g, rels="['ROAD','PATH']")

        def cch_keys():
            return sorted(tuple(sorted(lbl.split(","))) for (lbl, t) in
                g.query("CALL db.indexes() YIELD label, types "
                        "RETURN label, types").result_set
                if t and t.get('w') == ['CCH'])

        self.env.assertEquals(cch_keys(), [("PATH",), ("PATH", "ROAD"), ("ROAD",)])

        # drop {ROAD,PATH} using a reversed AND duplicated key -> removes exactly
        # that index; the {ROAD} and {PATH} indices are untouched
        self._drop(g, rels="['PATH','ROAD','PATH']")
        self.env.assertEquals(cch_keys(), [("PATH",), ("ROAD",)])

        # a drop whose exact set doesn't exist errors (e.g. {ROAD,PATH} is gone)
        try:
            self._drop(g, rels="['ROAD','PATH']")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("no CCH index", str(e))


# A CCH path index is serialized WHOLE into the RDB -- its full hierarchy, not
# just the definition -- so a reload restores it directly with no rebuild. The
# hierarchy lives in its own encode state (once per graph), while the graph spans
# several virtual keys under a small VKEY size. After an RDB round-trip the index
# must be listed again and answer queries IDENTICALLY (byte-for-byte the same
# hierarchy, not a re-derived one), still agree with algo.SPpaths, and remain
# maintainable.
class testCCHIndexPersistence(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env(moduleArgs="VKEY_MAX_ENTITY_COUNT 50",
                                enableDebugCommand=True)

    def _idxw(self, g, s, t):
        r = g.query(f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL db.idx.cch.query({{"
                    f"sourceNode:a,targetNode:b,relTypes:['ROAD'],weightProp:'w'}}) "
                    "YIELD pathWeight RETURN pathWeight").result_set
        return r[0][0] if r else None

    def _spw(self, g, s, t):
        r = g.query(f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL algo.SPpaths({{"
                    f"sourceNode:a,targetNode:b,relTypes:['ROAD'],weightProp:'w'}}) "
                    "YIELD pathWeight RETURN pathWeight").result_set
        return r[0][0] if r else None

    def test01_reload_restores_index(self):
        g = self.db.select_graph("cchi_persist")
        n, m = 60, 260                       # > 50 entities -> multiple virtual keys
        random.seed(4)
        g.query(f"UNWIND range(0,{n-1}) AS i CREATE (:N {{v:i}})")
        best = {}
        for _ in range(m):
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v:
                best[(u, v)] = min(random.randint(1, 9), best.get((u, v), 999))
        payload = ",".join(f"[{u},{v},{w}]" for (u, v), w in best.items())
        g.query(f"UNWIND [{payload}] AS e MATCH (a:N{{v:e[0]}}),(b:N{{v:e[1]}}) "
                f"CREATE (a)-[:ROAD {{w:e[2]}}]->(b)")
        g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)")
        # a weight change before saving: maintenance state must not leak to disk
        g.query("MATCH ()-[r:ROAD]->() WITH r LIMIT 1 SET r.w=7")

        pairs = [(s, t) for s in range(0, n, 7) for t in range(0, n, 5) if s != t]
        before = {p: self._idxw(g, *p) for p in pairs}

        # RDB round-trip: the whole hierarchy is restored from disk (no rebuild)
        self.env.getConnection().execute_command("DEBUG", "RELOAD")

        # still listed as a CCH index
        rows = g.query("CALL db.indexes() YIELD types RETURN types").result_set
        self.env.assertTrue(any(t and t.get('w') == ['CCH'] for (t,) in rows))

        # every answer identical after reload, and still matches the oracle
        for p in pairs:
            after = self._idxw(g, *p)
            self.env.assertEquals(after, before[p])
            self.env.assertEquals(after is None, self._spw(g, *p) is None)
            if after is not None:
                self.env.assertAlmostEqual(after, self._spw(g, *p), delta=1e-6)

        # the reloaded index is fully live: incremental maintenance still works
        g.query("MATCH ()-[r:ROAD]->() WITH r LIMIT 1 SET r.w=123")
        g.query("MATCH (a:N{v:0}),(b:N{v:1}) CREATE (a)-[:ROAD{w:0.25}]->(b)")
        for p in pairs:
            self.env.assertEquals(self._idxw(g, *p) is None, self._spw(g, *p) is None)
            if self._idxw(g, *p) is not None:
                self.env.assertAlmostEqual(self._idxw(g, *p), self._spw(g, *p), delta=1e-6)

    # ---- helpers for the multi-index case (arbitrary rel-set + weight attr) ----
    def _mk(self, g, rels, wp):
        types = "|".join(x.strip().strip("'\"") for x in rels.strip("[]").split(","))
        return g.query(
            f"CREATE CCH INDEX FOR ()-[e:{types}]->() ON (e.{wp})").indices_created

    def _qidx(self, g, s, t, rels, wp):
        r = g.query(
            f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL db.idx.cch.query({{sourceNode:a,"
            f"targetNode:b,relTypes:{rels},weightProp:'{wp}'}}) YIELD pathWeight "
            "RETURN pathWeight").result_set
        return r[0][0] if r else None

    def _qsp(self, g, s, t, rels, wp):
        r = g.query(
            f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL algo.SPpaths({{sourceNode:a,"
            f"targetNode:b,relTypes:{rels},weightProp:'{wp}'}}) YIELD pathWeight "
            "RETURN pathWeight").result_set
        return r[0][0] if r else None

    def _count_cch(self, g):
        rows = g.query("CALL db.indexes() YIELD types RETURN types").result_set
        return len([1 for (t,) in rows
                    if t and any('CCH' in v for v in t.values())])

    @staticmethod
    def _same(a, b):
        return (a is None and b is None) or \
               (a is not None and b is not None and abs(a - b) < 1e-6)

    def test02_reload_restores_multiple_indices(self):
        # a graph can hold several DISTINCT CCH indices at once, keyed by the full
        # (relationship-type set, weight attr). ALL of them must survive an RDB
        # round-trip: each restored and listed, queryable by its EXACT key,
        # answering identically to before the reload AND agreeing with its own
        # SPpaths oracle -- i.e. no two hierarchies got merged or mis-keyed during
        # (de)serialization, and the weight attr / rel-type set stay part of the key.
        g = self.db.select_graph("cchi_multi")
        try:
            g.delete()
        except Exception:
            pass
        g = self.db.select_graph("cchi_multi")
        n = 60                              # > 50 entities -> multiple virtual keys
        random.seed(8)
        g.query(f"UNWIND range(0,{n-1}) AS i CREATE (:N {{v:i}})")
        # two relationship types, each edge carrying two weight attributes (w, t)
        for rel in ("ROAD", "PATH"):
            best = {}
            for _ in range(220):
                u, v = random.randint(0, n - 1), random.randint(0, n - 1)
                if u != v:
                    best[(u, v)] = (random.randint(1, 9), random.randint(1, 9))
            payload = ",".join(f"[{u},{v},{w},{t}]" for (u, v), (w, t) in best.items())
            g.query(f"UNWIND [{payload}] AS e MATCH (a:N{{v:e[0]}}),(b:N{{v:e[1]}}) "
                    f"CREATE (a)-[:{rel} {{w:e[2], t:e[3]}}]->(b)")

        # four distinct keys: {ROAD}/w, {PATH}/w, {ROAD,PATH}/w and {ROAD}/t
        specs = [("['ROAD']", "w"), ("['PATH']", "w"),
                 ("['ROAD','PATH']", "w"), ("['ROAD']", "t")]
        for rels, wp in specs:
            self.env.assertEquals(self._mk(g, rels, wp), 1)
        self.env.assertEquals(self._count_cch(g), len(specs))

        pairs = [(s, t) for s in range(0, n, 11) for t in range(0, n, 13) if s != t]
        # snapshot every index's answers before the reload
        before = {(rels, wp): {p: self._qidx(g, p[0], p[1], rels, wp) for p in pairs}
                  for rels, wp in specs}

        # RDB round-trip: all four hierarchies are serialized and restored (no rebuild)
        self.env.getConnection().execute_command("DEBUG", "RELOAD")

        # all four survived; the CCH hierarchy is internal (no shortcut edges shipped)
        self.env.assertEquals(self._count_cch(g), len(specs))
        self.env.assertEquals(
            g.query("MATCH ()-[r:SHORTCUT]->() RETURN count(r)").result_set[0][0], 0)

        # every key answers identically to before AND matches its own oracle -- so
        # no index leaked into another during (de)serialization
        for rels, wp in specs:
            for p in pairs:
                after = self._qidx(g, p[0], p[1], rels, wp)
                self.env.assertTrue(self._same(after, before[(rels, wp)][p]))
                self.env.assertTrue(self._same(after, self._qsp(g, p[0], p[1], rels, wp)))

        # every reloaded index is independently LIVE: change a ROAD 'w' weight, then
        # all four still agree with their oracles (the {ROAD}/w index re-customizes;
        # the {PATH}/w, {ROAD,PATH}/w and {ROAD}/t indices are unaffected but valid)
        g.query("MATCH ()-[r:ROAD]->() WITH r LIMIT 1 SET r.w=99")
        for rels, wp in specs:
            for p in pairs:
                self.env.assertTrue(self._same(
                    self._qidx(g, p[0], p[1], rels, wp),
                    self._qsp(g, p[0], p[1], rels, wp)))


# CREATE CCH INDEX writes NOTHING to the graph, so its replication rides on the
# index-creation result-set statistic (which marks the query as a modification)
# plus a definition-only create/drop EFFECT: each replica rebuilds its OWN
# hierarchy (no shortcut edges are shipped), and a subsequent weight change
# replicates as an effect whose application re-customizes the replica's index.
class testCCHIndexReplication(FlowTestsBase):
    def __init__(self):
        # replication isn't reliable under Valgrind/sanitizer
        if VALGRIND or SANITIZER:
            Environment.skip(None)
        self.env, self.db = Env(env='oss', useSlaves=True)

    def test01_create_update_drop_replicate(self):
        env = self.env
        master_con  = env.getConnection()
        replica_con = env.getSlaveConnection()
        replica_con.config_set("slave-read-only", "no")

        master  = Graph(master_con,  "cchi_repl")
        replica = Graph(replica_con, "cchi_repl")

        # valley graph: a->b direct 10 but a->c->b is 2
        master.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                        (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                        (a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(a),
                        (c)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c)""")
        created = master.query(
            "CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)").indices_created
        env.assertEquals(created, 1)
        master_con.execute_command("WAIT", "1", "0")

        def rq(s, t):
            return replica.ro_query(
                f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL db.idx.cch.query({{"
                f"sourceNode:a,targetNode:b,relTypes:['ROAD'],weightProp:'w'}}) "
                "YIELD pathWeight RETURN pathWeight").result_set[0][0]

        # the definition replicated: listed on the replica, which built its own
        # queryable hierarchy -- and NO SHORTCUT edges were shipped into the graph
        rows = replica.ro_query("CALL db.indexes() YIELD types RETURN types").result_set
        env.assertEquals(len([r for r in rows if r[0] and r[0].get('w') == ['CCH']]), 1)
        env.assertEquals(
            replica.ro_query("MATCH ()-[r:SHORTCUT]->() RETURN count(r)").result_set[0][0], 0)
        env.assertAlmostEqual(rq(0, 1), 2, delta=1e-9)             # valley via c

        # a weight change replicates (as an effect) and the replica re-customizes
        master.query("MATCH (:N{v:2})-[r:ROAD]->(:N{v:1}) SET r.w=100")
        master_con.execute_command("WAIT", "1", "0")
        env.assertAlmostEqual(rq(0, 1), 10, delta=1e-9)            # detour gone

        # drop replicates -> query on the replica now errors
        master.query("DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)")
        master_con.execute_command("WAIT", "1", "0")
        try:
            rq(0, 1)
            env.assertTrue(False)
        except Exception as e:
            env.assertContains("no CCH index", str(e))

    def test02_full_key_replicates(self):
        # the create/drop effect is keyed by the FULL (relationship-type set,
        # weight attr): three distinct indices over {ROAD}, {PATH}, {ROAD,PATH}
        # must each replicate as their own index, queryable by their exact key.
        env = self.env
        master_con  = env.getConnection()
        replica_con = env.getSlaveConnection()
        replica_con.config_set("slave-read-only", "no")
        master  = Graph(master_con,  "cchi_repl_key")
        replica = Graph(replica_con, "cchi_repl_key")

        master.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                        (a)-[:ROAD{w:10}]->(b),(a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(b),
                        (a)-[:PATH{w:1.5}]->(b),(c)-[:PATH{w:0.3}]->(b)""")
        master.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)")
        master.query("CREATE CCH INDEX FOR ()-[e:PATH]->() ON (e.w)")
        master.query("CREATE CCH INDEX FOR ()-[e:ROAD|PATH]->() ON (e.w)")
        master_con.execute_command("WAIT", "1", "0")

        # replica has exactly the same three CCH indices
        rows = replica.ro_query("CALL db.indexes() YIELD types RETURN types").result_set
        env.assertEquals(len([r for r in rows if r[0] and r[0].get('w') == ['CCH']]), 3)

        # each key answers over its own relationship-type set on the replica
        def rq(rels):
            return replica.ro_query(
                f"MATCH (a:N{{v:0}}),(b:N{{v:1}}) CALL db.idx.cch.query({{sourceNode:a,"
                f"targetNode:b,relTypes:{rels},weightProp:'w'}}) YIELD pathWeight "
                "RETURN pathWeight").result_set[0][0]
        env.assertAlmostEqual(rq("['ROAD']"), 2, delta=1e-9)
        env.assertAlmostEqual(rq("['PATH']"), 1.5, delta=1e-9)
        env.assertAlmostEqual(rq("['ROAD','PATH']"), 1.3, delta=1e-9)

        # dropping one key on the master replicates as dropping only that key
        master.query("DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)")
        master_con.execute_command("WAIT", "1", "0")
        rows = replica.ro_query("CALL db.indexes() YIELD types RETURN types").result_set
        env.assertEquals(len([r for r in rows if r[0] and r[0].get('w') == ['CCH']]), 2)

    def test03_new_reltypes_and_attr_replicate_before_index(self):
        # a CCH create over brand-new reltypes / weight attribute creates them via
        # GraphHub, so their schema/attribute effects replicate BEFORE the CCH
        # create effect -- the replica applies them in order (its ApplyCreateCCH
        # verifies the schema+attribute exist), never diverging.
        env = self.env
        master_con  = env.getConnection()
        replica_con = env.getSlaveConnection()
        replica_con.config_set("slave-read-only", "no")
        master  = Graph(master_con,  "cchi_repl_new")
        replica = Graph(replica_con, "cchi_repl_new")

        master.query("CREATE (:N{v:0}),(:N{v:1})")   # HWY / km don't exist yet
        master.query("CREATE CCH INDEX FOR ()-[e:HWY]->() ON (e.km)")
        master_con.execute_command("WAIT", "1", "0")

        # the reltype, the attribute and the index all replicated (in order)
        rtypes = {r[0] for r in replica.ro_query(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN relationshipType").result_set}
        env.assertTrue("HWY" in rtypes)
        pkeys = {r[0] for r in replica.ro_query(
            "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey").result_set}
        env.assertTrue("km" in pkeys)
        rows = replica.ro_query("CALL db.indexes() YIELD types RETURN types").result_set
        env.assertEquals(len([r for r in rows if r[0] and r[0].get('km') == ['CCH']]), 1)

        # the replica's index is functional (queryable without divergence)
        res = replica.ro_query(
            "MATCH (a:N{v:0}),(b:N{v:1}) CALL db.idx.cch.query({sourceNode:a,"
            "targetNode:b,relTypes:['HWY'],weightProp:'km'}) YIELD pathWeight "
            "RETURN pathWeight").result_set
        env.assertEquals(res, [])   # no HWY edges -> unreachable, no rows (no error)


# A full RESYNC ships the master's live state to the replica as an RDB, so unlike
# the create-EFFECT path above (each replica rebuilds its own hierarchy) it drives
# the CCH RDB encode/decode end to end: the master forks and RdbSaveCCH_v20
# serializes the whole hierarchy, the replica loads it through RdbLoadCCH_v20 (no
# rebuild, no shortcut edges shipped into the user graph). Cycling REPLICAOF would
# disturb the incremental tests, so this lives in its own class/Env.
class testCCHIndexFullSync(FlowTestsBase):
    def __init__(self):
        # replication timing doesn't play well with Valgrind/sanitizers
        if VALGRIND or SANITIZER:
            Environment.skip(None)
        self.env, self.db = Env(env='oss', useSlaves=True, enableDebugCommand=True)

    def test01_full_sync_ships_cch_index(self):
        env     = self.env
        master  = env.getConnection()
        replica = env.getSlaveConnection()

        master_graph  = Graph(master,  "cchi_fullsync")
        replica_graph = Graph(replica, "cchi_fullsync")

        # remember where the replica replicates from, then detach it so the graph
        # AND its index reach it via a fresh FULL SYNC (an RDB load) rather than
        # the incremental effect stream once we re-attach
        repl_info   = replica.info()
        master_host = repl_info["master_host"]
        master_port = repl_info["master_port"]
        replica.execute_command("REPLICAOF", "NO", "ONE")

        # build a road network on the master and index it WHILE the replica is
        # detached, so nothing about the index reaches it as an effect -- the only
        # way it can arrive is inside the RDB of the coming full sync
        n, m = 60, 260
        random.seed(11)
        master_graph.query(f"UNWIND range(0,{n-1}) AS i CREATE (:N {{v:i}})")
        best = {}
        for _ in range(m):
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v:
                best[(u, v)] = min(random.randint(1, 9), best.get((u, v), 999))
        payload = ",".join(f"[{u},{v},{w}]" for (u, v), w in best.items())
        master_graph.query(
            f"UNWIND [{payload}] AS e MATCH (a:N{{v:e[0]}}),(b:N{{v:e[1]}}) "
            f"CREATE (a)-[:ROAD {{w:e[2]}}]->(b)")
        created = master_graph.query(
            "CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)").indices_created
        env.assertEquals(created, 1)

        # snapshot the full-sync counter so we can confirm a FULLRESYNC really
        # happened (not a partial PSYNC CONTINUE)
        sync_full_before = master.info()["sync_full"]

        # re-attach -> full resync: master forks, RdbSaveCCH_v20 serializes the
        # whole hierarchy, the replica loads it via RdbLoadCCH_v20
        replica.execute_command("REPLICAOF", master_host, master_port)

        deadline = time.time() + 60
        synced   = False
        while time.time() < deadline:
            if (master.info()["sync_full"] > sync_full_before and
                    replica.info()["master_link_status"] == "up"):
                synced = True
                break
            time.sleep(0.5)
        env.assertTrue(synced)
        master.execute_command("WAIT", "1", "10000")

        # the index arrived via the RDB: listed as CCH on the replica, and NOT a
        # single shortcut edge was shipped into the user graph
        rows = replica_graph.ro_query(
            "CALL db.indexes() YIELD types RETURN types").result_set
        env.assertEquals(
            len([r for r in rows if r[0] and r[0].get('w') == ['CCH']]), 1)
        env.assertEquals(replica_graph.ro_query(
            "MATCH ()-[r:SHORTCUT]->() RETURN count(r)").result_set[0][0], 0)

        # the loaded hierarchy answers correctly: every pair matches the master's
        # own CCH answer AND the algo.SPpaths oracle (computed on the master, which
        # is writable; the replica is queried read-only)
        def cch(g, s, t, ro):
            q = g.ro_query if ro else g.query
            r = q(f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL db.idx.cch.query({{"
                  f"sourceNode:a,targetNode:b,relTypes:['ROAD'],weightProp:'w'}}) "
                  "YIELD pathWeight RETURN pathWeight").result_set
            return r[0][0] if r else None

        def sp(g, s, t):
            r = g.query(f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL algo.SPpaths({{"
                        f"sourceNode:a,targetNode:b,relTypes:['ROAD'],weightProp:'w'}}) "
                        "YIELD pathWeight RETURN pathWeight").result_set
            return r[0][0] if r else None

        pairs = [(s, t) for s in range(0, n, 7) for t in range(0, n, 5) if s != t]
        for s, t in pairs:
            m_cch = cch(master_graph,  s, t, ro=False)
            r_cch = cch(replica_graph, s, t, ro=True)
            m_sp  = sp(master_graph,   s, t)
            env.assertEquals(r_cch, m_cch)                  # replica index == master
            env.assertEquals(r_cch is None, m_sp is None)   # agree on reachability
            if r_cch is not None:
                env.assertAlmostEqual(r_cch, m_sp, delta=1e-6)  # absolute oracle


# With AOF on, CREATE/DROP CCH INDEX and edge-weight changes are persisted as
# GRAPH.EFFECT / GRAPH.QUERY entries. Replaying the AOF must rebuild the index
# from the create effect and reflect every subsequent maintenance effect -- the
# index is definition-only on disk, never its materialized hierarchy.
class testCCHIndexAOF(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env(useAof=True, enableDebugCommand=True)

    def test01_aof_replay_rebuilds_index(self):
        con = self.env.getConnection()
        g = self.db.select_graph("cchi_aof")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                   (a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(a),
                   (c)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c)""")
        g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)")

        def ab():
            return g.query("MATCH (a:N{v:0}),(b:N{v:1}) CALL db.idx.cch.query({"
                           "sourceNode:a,targetNode:b,relTypes:['ROAD'],weightProp:'w'}) "
                           "YIELD pathWeight RETURN pathWeight").result_set[0][0]

        self.env.assertAlmostEqual(ab(), 2, delta=1e-9)             # valley via c
        g.query("MATCH (:N{v:2})-[r:ROAD]->(:N{v:1}) SET r.w=100")  # update -> AOF effect
        self.env.assertAlmostEqual(ab(), 10, delta=1e-9)

        # reload the dataset purely from the AOF
        con.execute_command("DEBUG", "LOADAOF")

        # index rebuilt from the replayed create effect; the update is reflected
        rows = g.query("CALL db.indexes() YIELD types RETURN types").result_set
        self.env.assertTrue(any(t and t.get('w') == ['CCH'] for (t,) in rows))
        self.env.assertAlmostEqual(ab(), 10, delta=1e-9)


# db.idx.cch.query is read-only with no shared/global state (all scratch is
# per-invocation), so many threads must query the same index in parallel. Build
# once, hammer db.idx.cch.query from several threads, and check every answer
# against the single-threaded algo.SPpaths ground truth.
class testCCHIndexConcurrentQuery(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    def test01_concurrent_queries(self):
        gname = "cchi_concurrent"
        g = self.db.select_graph(gname)
        try:
            g.delete()
        except Exception:
            pass
        g = self.db.select_graph(gname)

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
        g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.w)")

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
                barrier.wait(timeout=30)
            except Exception:
                pass
            for _ in range(ITERS):
                s, t, exp = pairs[rng.randrange(len(pairs))]
                try:
                    res = tg.ro_query(
                        f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL db.idx.cch.query("
                        f"{{sourceNode:a,targetNode:b,relTypes:['ROAD'],weightProp:'w'}}) "
                        "YIELD pathWeight RETURN pathWeight").result_set
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

        self.env.assertEquals(failures, [])
