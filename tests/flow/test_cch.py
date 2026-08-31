from common import *
import random

# End-to-end tests for Customizable Contraction Hierarchies:
#   algo.CCH        -- builds SHORTCUT edges + rank/middle properties
#   algo.CCH.query  -- rank-aware bidirectional Dijkstra + shortcut unpacking
#
# The correctness contract (see the branch design notes): a CCH query must
# return the SAME WEIGHT as algo.SPpaths, and its path must be a genuine ROAD
# path (every hop a ROAD edge, contiguous, weights summing to the total).
# Exact edge-for-edge equality with SPpaths is only asserted on a graph
# engineered to have a unique shortest path (ties legitimately differ).

BUILD = ("CALL algo.CCH({relTypes:['ROAD'], weightProp:'w', "
         "shortcutRelType:'SHORTCUT', rankProperty:'rank', middleProp:'mid'}) "
         "YIELD shortcutsCreated RETURN shortcutsCreated")

QUERY = ("CALL algo.CCH.query({sourceNode:a, targetNode:b, relTypes:['ROAD'], "
         "shortcutRelType:'SHORTCUT', weightProp:'w', rankProperty:'rank', "
         "middleProp:'mid'}) YIELD pathWeight, path "
         "RETURN pathWeight, [n IN nodes(path) | n.v] AS vs, "
         "[r IN relationships(path) | type(r)] AS ts, "
         "reduce(s=0.0, r IN relationships(path) | s + r.w) AS psum")


class testCCH(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    def _reset(self, name):
        g = self.db.select_graph(name)
        try:
            g.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass
        return g

    def _cch_query(self, g, s, t):
        r = g.query(f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) " + QUERY).result_set
        return r[0] if r else None

    def _sp(self, g, s, t):
        r = g.query(f"MATCH (a:N{{v:{s}}}),(b:N{{v:{t}}}) CALL algo.SPpaths("
                    f"{{sourceNode:a,targetNode:b,relTypes:['ROAD'],"
                    f"weightProp:'w'}}) YIELD pathWeight, path "
                    f"RETURN pathWeight, [n IN nodes(path) | n.v]").result_set
        return (r[0][0], r[0][1]) if r else (None, None)

    def test01_validations(self):
        g = self._reset("cch_valid")
        g.query("CREATE (:N {v:0}), (:N {v:1})")

        # algo.CCH: wrong key count
        try:
            g.query("CALL algo.CCH({relTypes:['ROAD']})")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("exactly 5 keys", str(e))

        # algo.CCH: unknown relationship type
        try:
            g.query("CALL algo.CCH({relTypes:['NOPE'], weightProp:'w', "
                    "shortcutRelType:'SC', rankProperty:'rank', "
                    "middleProp:'mid'})")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("non-existent relationship type", str(e))

    def test02_valley_fixture(self):
        # a-b road is 10, but a-c-b is 2: the shortcut a->b must carry 2, and
        # the query must return the unpacked road path a-c-b.
        g = self._reset("cch_valley")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(c:N{v:2}),
                   (a)-[:ROAD{w:10}]->(b),(b)-[:ROAD{w:10}]->(a),
                   (a)-[:ROAD{w:1}]->(c),(c)-[:ROAD{w:1}]->(a),
                   (c)-[:ROAD{w:1}]->(b),(b)-[:ROAD{w:1}]->(c)""")
        sc = g.query(BUILD).result_set[0][0]
        self.env.assertTrue(sc >= 2)

        w, vs, ts, psum = self._cch_query(g, 0, 1)
        self.env.assertEquals(w, 2)
        self.env.assertEquals(vs, [0, 2, 1])          # a -> c -> b
        self.env.assertEquals(ts, ['ROAD', 'ROAD'])   # unpacked to road edges
        self.env.assertAlmostEqual(psum, 2, delta=1e-9)

    def _random_graph(self, g, n, m, seed):
        random.seed(seed)
        g.query(f"UNWIND range(0,{n-1}) AS i CREATE (:N {{v:i}})")
        best = {}
        for _ in range(m):
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v:
                best[(u, v)] = min(random.randint(1, 9), best.get((u, v), 99))
        if best:
            payload = ",".join(f"[{u},{v},{w}]" for (u, v), w in best.items())
            g.query(f"UNWIND [{payload}] AS e MATCH (a:N{{v:e[0]}}),(b:N{{v:e[1]}}) "
                    f"CREATE (a)-[:ROAD {{w:e[2]}}]->(b)")

    def test03_random_vs_sppaths(self):
        # for many random digraphs and pairs: CCH weight == SPpaths weight, and
        # the CCH path is a valid ROAD path.
        for ti, (n, m) in enumerate([(8, 20), (15, 60), (25, 120)]):
            for seed in range(3):
                g = self._reset(f"cch_rand_{ti}_{seed}")
                self._random_graph(g, n, m, seed * 131 + ti)
                g.query(BUILD)

                pairs = [(s, t) for s in range(n) for t in range(n) if s != t]
                if len(pairs) > 90:
                    random.seed(seed)
                    pairs = random.sample(pairs, 90)

                for s, t in pairs:
                    exp, _ = self._sp(g, s, t)
                    row = self._cch_query(g, s, t)
                    gw = row[0] if row else None

                    # reachability + weight agree with SPpaths
                    self.env.assertEquals(gw is None, exp is None)
                    if gw is None:
                        continue
                    self.env.assertAlmostEqual(gw, exp, delta=1e-9)

                    # path is a genuine ROAD path
                    _, vs, ts, psum = row
                    self.env.assertEquals(vs[0], s)
                    self.env.assertEquals(vs[-1], t)
                    self.env.assertTrue(all(x == 'ROAD' for x in ts))
                    self.env.assertAlmostEqual(psum, gw, delta=1e-9)

    def test04_unique_shortest_path_exact(self):
        # a graph whose s->t shortest path is unique, so the CCH path must match
        # SPpaths edge-for-edge. distinct powers-of-two-ish weights guarantee
        # every distinct edge subset has a distinct sum -> no ties.
        g = self._reset("cch_unique")
        g.query("""CREATE (n0:N{v:0}),(n1:N{v:1}),(n2:N{v:2}),
                          (n3:N{v:3}),(n4:N{v:4}),
                   (n0)-[:ROAD{w:1}]->(n1),
                   (n1)-[:ROAD{w:2}]->(n2),
                   (n2)-[:ROAD{w:4}]->(n3),
                   (n3)-[:ROAD{w:8}]->(n4),
                   (n0)-[:ROAD{w:100}]->(n4),
                   (n1)-[:ROAD{w:100}]->(n3)""")
        g.query(BUILD)

        for s, t in [(0, 4), (0, 3), (1, 4), (0, 2)]:
            exp_w, exp_vs = self._sp(g, s, t)
            row = self._cch_query(g, s, t)
            self.env.assertTrue(row is not None)
            gw, vs, ts, psum = row
            self.env.assertAlmostEqual(gw, exp_w, delta=1e-9)
            self.env.assertEquals(vs, exp_vs)             # exact node sequence
            self.env.assertTrue(all(x == 'ROAD' for x in ts))

    def test05_edge_cases(self):
        g = self._reset("cch_edge")
        g.query("""CREATE (a:N{v:0}),(b:N{v:1}),(z:N{v:2}),
                   (a)-[:ROAD{w:3}]->(b),(b)-[:ROAD{w:3}]->(a)""")
        g.query(BUILD)

        # src == dst -> weight 0, single-node path
        w, vs, ts, psum = self._cch_query(g, 0, 0)
        self.env.assertEquals(w, 0)
        self.env.assertEquals(vs, [0])
        self.env.assertEquals(ts, [])

        # unreachable (isolated node z) -> no rows
        row = self._cch_query(g, 0, 2)
        self.env.assertTrue(row is None)
