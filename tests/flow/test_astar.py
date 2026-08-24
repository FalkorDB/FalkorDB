from common import *
import heapq
import math

# haversine great-circle distance in meters between two (lat, lon) pairs
# given in degrees. Mirrors the formula in src/arithmetic/point_funcs/
# point_funcs.c (AR_DISTANCE) / src/algorithms/AStar.c (_haversine_meters).
def _haversine(lat1, lon1, lat2, lon2):
    R = 6378140.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2) - math.radians(lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class testAStar(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    def test01_astar_validations(self):
        g = self.db.select_graph("astar_validations")
        g.query("CREATE (:AK {id: 0, lat: 0.0, lon: 0.0}), (:AK {id: 1, lat: 0.0, lon: 0.0})")

        # missing sourceNode/targetNode
        queries = [
            """CALL algo.AStar({})""",
            """MATCH (n:AK {id: 0}) CALL algo.AStar({sourceNode: n})""",
            """MATCH (n:AK {id: 0}) CALL algo.AStar({targetNode: n})"""
        ]
        for query in queries:
            try:
                g.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("sourceNode and targetNode are required", str(e))

        # wrong type for sourceNode/targetNode
        queries = [
            """MATCH (n:AK {id: 0}) CALL algo.AStar({sourceNode: 1, targetNode: 1})""",
            """MATCH (n:AK {id: 0}) CALL algo.AStar({sourceNode: 1, targetNode: n})""",
            """MATCH (n:AK {id: 0}) CALL algo.AStar({sourceNode: n, targetNode: 1})"""
        ]
        for query in queries:
            try:
                g.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("sourceNode and targetNode must be of type Node", str(e))

        # missing latitudeProperty/longitudeProperty (either or both)
        queries = [
            """MATCH (n:AK {id: 0}), (m:AK {id: 1}) CALL algo.AStar({sourceNode: n, targetNode: m})""",
            """MATCH (n:AK {id: 0}), (m:AK {id: 1}) CALL algo.AStar({sourceNode: n, targetNode: m, latitudeProperty: 'lat'})""",
            """MATCH (n:AK {id: 0}), (m:AK {id: 1}) CALL algo.AStar({sourceNode: n, targetNode: m, longitudeProperty: 'lon'})"""
        ]
        for query in queries:
            try:
                g.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("latitudeProperty and longitudeProperty are required", str(e))

        # wrong type for latitudeProperty/longitudeProperty
        query = """MATCH (n:AK {id: 0}), (m:AK {id: 1}) CALL algo.AStar({sourceNode: n, targetNode: m, latitudeProperty: 1, longitudeProperty: 'lon'})"""
        try:
            g.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("latitudeProperty/longitudeProperty must be string", str(e))

        # bad relDirection
        query = """MATCH (n:AK {id: 0}), (m:AK {id: 1}) CALL algo.AStar({sourceNode: n, targetNode: m, latitudeProperty: 'lat', longitudeProperty: 'lon', relDirection: 'a'})"""
        try:
            g.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("relDirection values must be 'incoming', 'outgoing' or 'both'", str(e))

        # bad relTypes
        query = """MATCH (n:AK {id: 0}), (m:AK {id: 1}) CALL algo.AStar({sourceNode: n, targetNode: m, latitudeProperty: 'lat', longitudeProperty: 'lon', relTypes: 1})"""
        try:
            g.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("relTypes must be array of strings", str(e))

        # bad weightProp
        query = """MATCH (n:AK {id: 0}), (m:AK {id: 1}) CALL algo.AStar({sourceNode: n, targetNode: m, latitudeProperty: 'lat', longitudeProperty: 'lon', weightProp: 1})"""
        try:
            g.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("weightProp must be string", str(e))

    def _dijkstra_all_pairs(self, n_nodes, edges):
        # independent reference implementation: all-pairs shortest
        # distances over a directed, non-negatively weighted graph given
        # as a list of (u, v, weight) edges.
        adj = [[] for _ in range(n_nodes)]
        for u, v, w in edges:
            adj[u].append((v, w))

        all_dist = {}
        for src in range(n_nodes):
            dist = {src: 0}
            pq = [(0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, float('inf')):
                    continue
                for v, w in adj[u]:
                    nd = d + w
                    if nd < dist.get(v, float('inf')):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
            all_dist[src] = dist

        return all_dist

    def _verify_astar_all_pairs(self, graph_name, n_nodes, coords, edge_pairs):
        # build a directed graph with real lat/lon node coordinates, where
        # every edge's weight is set to the haversine distance between
        # that edge's own two endpoints. This guarantees the haversine
        # heuristic stays admissible for *any* topology built this way,
        # regardless of how the nodes/edges are arranged: the triangle
        # inequality guarantees the sum of any path's segment distances
        # can never be less than the straight-line distance between its
        # ends. So any mismatch against the independent Dijkstra reference
        # below would indicate a genuine AStar bug, not a violated
        # precondition of the test graph itself.
        g = self.db.select_graph(graph_name)
        g.query(f"UNWIND range(0, {n_nodes - 1}) AS x CREATE (:AK {{id: x}})")

        for i, (lat, lon) in enumerate(coords):
            g.query(f"MATCH (n:AK {{id: {i}}}) SET n.lat = {lat}, n.lon = {lon}")

        edges = []
        for u, v in edge_pairs:
            w = _haversine(coords[u][0], coords[u][1], coords[v][0], coords[v][1])
            edges.append((u, v, w))

        if edges:
            rows = ", ".join(f"[{u}, {v}, {w}]" for u, v, w in edges)
            g.query(f"""
                UNWIND [{rows}] AS e
                MATCH (a:AK {{id: e[0]}}), (b:AK {{id: e[1]}})
                CREATE (a)-[:AE {{weight: e[2]}}]->(b)
            """)

        result = g.query("""
            MATCH (n:AK), (m:AK)
            WHERE n.id <> m.id
            CALL algo.AStar({
                sourceNode: n,
                targetNode: m,
                weightProp: 'weight',
                latitudeProperty: 'lat',
                longitudeProperty: 'lon'
            }) YIELD pathWeight
            RETURN n.id, m.id, pathWeight
        """)

        actual = {(row[0], row[1]): row[2] for row in result.result_set}
        expected = self._dijkstra_all_pairs(n_nodes, edges)

        for src in range(n_nodes):
            for dst in range(n_nodes):
                if src == dst:
                    continue

                key = (src, dst)
                exp_weight = expected[src].get(dst)

                if exp_weight is None:
                    self.env.assertNotContains(key, actual)
                else:
                    self.env.assertContains(key, actual)
                    self.env.assertAlmostEqual(actual[key], exp_weight, delta=1e-6)

    def test02_astar_line_graph(self):
        # simple directed line 0->1->2->...->(n-1), each hop a small real
        # geographic step. exactly one path exists between any (src, dst)
        # pair, and only "forward" pairs are reachable at all.
        n = 8
        coords = [(37.0 + 0.01 * i, -122.0 + 0.01 * i) for i in range(n)]
        edge_pairs = [(i, i + 1) for i in range(n - 1)]
        self._verify_astar_all_pairs("astar_line", n, coords, edge_pairs)

    def test03_astar_diamond_graph(self):
        # two chained diamonds, each offering a direct waypoint and a
        # geographically detouring one between the same pair of nodes.
        # Since every edge's weight is the true geographic distance
        # between its own endpoints, the detour branch is both physically
        # longer and heavier -- AStar must still find the correct minimum
        # via real relaxation, not just by following the heuristic blindly.
        coords = [
            (37.00, -122.00),  # 0
            (37.01, -122.00),  # 1 direct branch waypoint
            (37.02, -121.90),  # 2 detour branch waypoint
            (37.03, -122.00),  # 3
            (37.04, -122.00),  # 4 direct branch waypoint
            (37.05, -121.90),  # 5 detour branch waypoint
            (37.06, -122.00),  # 6
            (37.07, -122.00),  # 7
        ]
        edge_pairs = [
            (0, 1), (0, 2),
            (1, 3), (2, 3),
            (3, 4), (3, 5),
            (4, 6), (5, 6),
            (6, 7),
        ]
        self._verify_astar_all_pairs("astar_diamond", 8, coords, edge_pairs)

    def test04_astar_grid_graph(self):
        # a 4x4 grid with only rightward/downward edges: many equal-length
        # alternative routes exist between nodes on the same diagonal.
        rows, cols = 4, 4

        def node_id(r, c):
            return r * cols + c

        coords = [(37.0 + 0.01 * r, -122.0 + 0.01 * c)
                  for r in range(rows) for c in range(cols)]
        edge_pairs = []
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    edge_pairs.append((node_id(r, c), node_id(r, c + 1)))
                if r + 1 < rows:
                    edge_pairs.append((node_id(r, c), node_id(r + 1, c)))

        self._verify_astar_all_pairs("astar_grid", rows * cols, coords, edge_pairs)

    def test05_astar_agrees_with_dijkstra(self):
        # direct cross-check on the same graph/weights: algo.SPpaths'
        # Dijkstra fast path (pathCount: 1, no maxCost) and algo.AStar
        # must agree on pathWeight -- the single most important
        # correctness check, since it validates AStar's admissible-
        # heuristic optimality against a trusted, independently
        # implemented baseline rather than only the hand-rolled Python
        # reference used above.
        coords = [
            (37.00, -122.00),
            (37.01, -122.00),
            (37.02, -121.90),
            (37.03, -122.00),
        ]
        edge_pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]

        g = self.db.select_graph("astar_vs_dijkstra")
        g.query("UNWIND range(0, 3) AS x CREATE (:AK {id: x})")
        for i, (lat, lon) in enumerate(coords):
            g.query(f"MATCH (n:AK {{id: {i}}}) SET n.lat = {lat}, n.lon = {lon}")

        rows = ", ".join(
            f"[{u}, {v}, {_haversine(*coords[u], *coords[v])}]" for u, v in edge_pairs)
        g.query(f"""
            UNWIND [{rows}] AS e
            MATCH (a:AK {{id: e[0]}}), (b:AK {{id: e[1]}})
            CREATE (a)-[:AE {{weight: e[2]}}]->(b)
        """)

        dijkstra_result = g.query("""
            MATCH (n:AK {id: 0}), (m:AK {id: 3})
            CALL algo.SPpaths({sourceNode: n, targetNode: m, weightProp: 'weight', pathCount: 1})
            YIELD pathWeight
            RETURN pathWeight
        """)
        astar_result = g.query("""
            MATCH (n:AK {id: 0}), (m:AK {id: 3})
            CALL algo.AStar({sourceNode: n, targetNode: m, weightProp: 'weight',
                              latitudeProperty: 'lat', longitudeProperty: 'lon'})
            YIELD pathWeight
            RETURN pathWeight
        """)

        self.env.assertEquals(len(dijkstra_result.result_set), 1)
        self.env.assertEquals(len(astar_result.result_set), 1)
        self.env.assertAlmostEqual(
            astar_result.result_set[0][0], dijkstra_result.result_set[0][0], delta=1e-9)

    def test06_astar_unreachable(self):
        g = self.db.select_graph("astar_unreachable")
        g.query("""CREATE (:AK {id: 0, lat: 37.0, lon: -122.0}),
                           (:AK {id: 1, lat: 38.0, lon: -121.0})""")

        result = g.query("""
            MATCH (n:AK {id: 0}), (m:AK {id: 1})
            CALL algo.AStar({sourceNode: n, targetNode: m, weightProp: 'weight',
                              latitudeProperty: 'lat', longitudeProperty: 'lon'})
            YIELD path
            RETURN path
        """)
        self.env.assertEquals(len(result.result_set), 0)

    def test07_astar_src_eq_dst(self):
        # sourceNode == targetNode is degenerate: a path needs at least
        # one edge, so this must always return no results, even though a
        # trivial zero-edge/zero-weight "path" would otherwise be found
        # instantly by the search's own seeding step.
        g = self.db.select_graph("astar_src_eq_dst")
        g.query("CREATE (:AK {id: 0, lat: 37.0, lon: -122.0})")

        result = g.query("""
            MATCH (n:AK {id: 0})
            CALL algo.AStar({sourceNode: n, targetNode: n, weightProp: 'weight',
                              latitudeProperty: 'lat', longitudeProperty: 'lon'})
            YIELD path
            RETURN path
        """)
        self.env.assertEquals(len(result.result_set), 0)

    def test08_astar_missing_latlon_on_node(self):
        # a node reachable mid-path with no lat/lon properties at all: its
        # heuristic must degrade to 0 (still admissible) rather than
        # error, and the true shortest (lowest-weight) path must still be
        # found -- here the 2-hop route through 'b' (weight 10) beats the
        # direct edge to 'c' (weight 20).
        g = self.db.select_graph("astar_missing_latlon")
        g.query("""
            CREATE (a:AK {id: 0, lat: 37.00, lon: -122.00}),
                   (b:AK {id: 1}),
                   (c:AK {id: 2, lat: 37.02, lon: -122.00}),
                   (a)-[:AE {weight: 5}]->(b),
                   (b)-[:AE {weight: 5}]->(c),
                   (a)-[:AE {weight: 20}]->(c)
        """)

        result = g.query("""
            MATCH (n:AK {id: 0}), (m:AK {id: 2})
            CALL algo.AStar({sourceNode: n, targetNode: m, weightProp: 'weight',
                              latitudeProperty: 'lat', longitudeProperty: 'lon'})
            YIELD pathWeight, path
            RETURN pathWeight, length(path)
        """)

        self.env.assertEquals(len(result.result_set), 1)
        self.env.assertAlmostEqual(result.result_set[0][0], 10, delta=1e-9)
        self.env.assertEquals(result.result_set[0][1], 2)

    def test09_astar_k_shortest_agrees_with_sppaths(self):
        # A* k-shortest (Yen driven by A* spur searches) must return the same
        # set of paths, by weight, as algo.SPpaths' k-shortest (Yen driven by
        # Dijkstra) on the same graph -- both are exact k-shortest-loopless by
        # weight. two chained diamonds give four distinct 0->7 routes.
        coords = [
            (37.00, -122.00),  # 0
            (37.01, -122.00),  # 1
            (37.02, -121.90),  # 2
            (37.03, -122.00),  # 3
            (37.04, -122.00),  # 4
            (37.05, -121.90),  # 5
            (37.06, -122.00),  # 6
            (37.07, -122.00),  # 7
        ]
        edge_pairs = [
            (0, 1), (0, 2),
            (1, 3), (2, 3),
            (3, 4), (3, 5),
            (4, 6), (5, 6),
            (6, 7),
        ]

        g = self.db.select_graph("astar_k_vs_sppaths")
        g.query("UNWIND range(0, 7) AS x CREATE (:AK {id: x})")
        for i, (lat, lon) in enumerate(coords):
            g.query(f"MATCH (n:AK {{id: {i}}}) SET n.lat = {lat}, n.lon = {lon}")
        rows = ", ".join(
            f"[{u}, {v}, {_haversine(*coords[u], *coords[v])}]" for u, v in edge_pairs)
        g.query(f"""
            UNWIND [{rows}] AS e
            MATCH (a:AK {{id: e[0]}}), (b:AK {{id: e[1]}})
            CREATE (a)-[:AE {{weight: e[2]}}]->(b)
        """)

        def k_paths(call):
            result = g.query(f"""
                MATCH (n:AK {{id: 0}}), (m:AK {{id: 7}})
                {call}
                YIELD path, pathWeight
                RETURN [nd IN nodes(path) | nd.id] AS ids, pathWeight
            """)
            return [(tuple(r[0]), r[1]) for r in result.result_set]

        total_paths = 4
        for k in (1, 2, 3, 4, 5):
            astar = k_paths(f"""CALL algo.AStar({{sourceNode: n, targetNode: m,
                weightProp: 'weight', latitudeProperty: 'lat',
                longitudeProperty: 'lon', pathCount: {k}}})""")
            sppaths = k_paths(f"""CALL algo.SPpaths({{sourceNode: n,
                targetNode: m, weightProp: 'weight', pathCount: {k}}})""")

            self.env.assertEquals(len(astar), min(k, total_paths))
            self.env.assertEquals(len(sppaths), min(k, total_paths))

            # same weights (multiset), and same set of paths
            aw = sorted(w for _, w in astar)
            dw = sorted(w for _, w in sppaths)
            for a, d in zip(aw, dw):
                self.env.assertAlmostEqual(a, d, delta=1e-9)
            self.env.assertEquals(set(ids for ids, _ in astar),
                                  set(ids for ids, _ in sppaths))

            # each A* path is loopless and weakly ordered by ascending weight
            weights = [w for _, w in astar]
            self.env.assertEquals(weights, sorted(weights))
            for ids, _ in astar:
                self.env.assertEquals(len(ids), len(set(ids)))

    def test10_astar_k_shortest_validations(self):
        # pathCount must be a positive integer (unlike algo.SPpaths, A* has no
        # all-minimal mode, so 0 is rejected too).
        g = self.db.select_graph("astar_k_validations")
        g.query("""CREATE (:AK {id: 0, lat: 37.0, lon: -122.0}),
                           (:AK {id: 1, lat: 37.1, lon: -122.0})""")

        for bad in ("'a'", "1.5", "0", "-1"):
            query = f"""MATCH (n:AK {{id: 0}}), (m:AK {{id: 1}})
                CALL algo.AStar({{sourceNode: n, targetNode: m,
                    latitudeProperty: 'lat', longitudeProperty: 'lon',
                    pathCount: {bad}}}) YIELD path RETURN path"""
            try:
                g.query(query)
                self.env.assertTrue(False)  # should have raised
            except Exception as e:
                self.env.assertContains("pathCount", str(e))

    #-------------------------------------------------------------------------
    # Additional A* coverage: K-shortest on a geographic grid, traversal
    # direction, a destination with no coordinates, multigraph parallel edges,
    # and relTypes filtering.
    #
    # Oracle: algo.SPpaths (Dijkstra / Dijkstra-Yen), which A* must match
    # exactly since both are exact. Weights are compared always; the exact path
    # set is compared only for full enumeration (pathCount large), where there
    # is no k-th-place tie for the two engines to break differently. Where the
    # A* heuristic is not the thing under test the nodes are given identical
    # coordinates, so h == 0 everywhere (admissible) and A* reduces to Dijkstra
    # over the logic being exercised; heuristic effectiveness is covered by the
    # geographic parity tests (test02-04).
    #-------------------------------------------------------------------------

    def _sp_rows(self, g, query):
        r = g.query(query)
        return sorted((tuple(row[0]), round(row[1], 6)) for row in r.result_set)

    def _astar_vs_sppaths(self, g, src, dst, path_count, rel_dir=None,
                          rel_types=None):
        cfg = ["sourceNode: u", "targetNode: v", "weightProp: 'weight'",
               f"pathCount: {path_count}"]
        if rel_dir is not None:
            cfg.append(f"relDirection: '{rel_dir}'")
        if rel_types is not None:
            cfg.append(f"relTypes: {rel_types}")
        acfg = cfg + ["latitudeProperty: 'lat'", "longitudeProperty: 'lon'"]
        m = f"MATCH (u:AK {{id: {src}}}), (v:AK {{id: {dst}}})"
        ret = ("YIELD path, pathWeight "
               "RETURN [n IN nodes(path)|n.id] AS ids, pathWeight")
        astar = self._sp_rows(g, f"{m} CALL algo.AStar({{{', '.join(acfg)}}}) {ret}")
        sp    = self._sp_rows(g, f"{m} CALL algo.SPpaths({{{', '.join(cfg)}}}) {ret}")

        # both are optimal, so the weight multiset must always match
        self.env.assertEquals(sorted(w for _, w in astar),
                              sorted(w for _, w in sp))
        # exact identities agree only when the whole set is returned
        if path_count >= 100:
            self.env.assertEquals(astar, sp)
        return astar

    def test11_astar_k_shortest_grid(self):
        # A*-Yen vs Dijkstra-Yen on a geographic grid with haversine-distance
        # weights (heuristic active and admissible); ask for more paths than
        # exist so both return the full simple-path set.
        rows, cols = 4, 4

        def nid(r, c):
            return r * cols + c

        coords = [(37.0 + 0.01 * r, -122.0 + 0.01 * c)
                  for r in range(rows) for c in range(cols)]
        edge_pairs = []
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    edge_pairs.append((nid(r, c), nid(r, c + 1)))
                if r + 1 < rows:
                    edge_pairs.append((nid(r, c), nid(r + 1, c)))

        g = self.db.select_graph("astar_k_grid")
        g.query(f"UNWIND range(0, {rows * cols - 1}) AS x CREATE (:AK {{id: x}})")
        for i, (lat, lon) in enumerate(coords):
            g.query(f"MATCH (n:AK {{id: {i}}}) SET n.lat = {lat}, n.lon = {lon}")
        er = ", ".join(
            f"[{u}, {v}, {_haversine(*coords[u], *coords[v])}]" for u, v in edge_pairs)
        g.query(f"""
            UNWIND [{er}] AS e
            MATCH (a:AK {{id: e[0]}}), (b:AK {{id: e[1]}})
            CREATE (a)-[:AE {{weight: e[2]}}]->(b)
        """)

        for dst in (5, 10, 15):
            self._astar_vs_sppaths(g, 0, dst, 100)

    def test12_astar_direction(self):
        # incoming and both traversal, colocated coords (h == 0, admissible).
        g = self.db.select_graph("astar_dir")
        g.query("""
            CREATE (a:AK {id:0, lat:37.0, lon:-122.0}),
                   (b:AK {id:1, lat:37.0, lon:-122.0}),
                   (c:AK {id:2, lat:37.0, lon:-122.0}),
                   (d:AK {id:3, lat:37.0, lon:-122.0}),
                   (a)-[:AE {weight:1}]->(b), (b)-[:AE {weight:1}]->(d),
                   (a)-[:AE {weight:1}]->(c), (c)-[:AE {weight:1}]->(d),
                   (a)-[:AE {weight:5}]->(d)
        """)
        self._astar_vs_sppaths(g, 3, 0, 1,   rel_dir="incoming")
        self._astar_vs_sppaths(g, 3, 0, 100, rel_dir="incoming")
        self._astar_vs_sppaths(g, 0, 3, 100, rel_dir="both")

    def test13_astar_dst_missing_coords(self):
        # the destination has no lat/lon => the heuristic is 0 graph-wide
        # (degrades to Dijkstra) but the result must still be optimal. Also
        # exercises parallel edges into that destination.
        g = self.db.select_graph("astar_no_dst_coords")
        g.query("""
            CREATE (a:AK {id:0, lat:37.0,  lon:-122.0}),
                   (b:AK {id:1, lat:37.01, lon:-122.0}),
                   (c:AK {id:2}),
                   (a)-[:AE {weight:1}]->(b), (b)-[:AE {weight:1}]->(c),
                   (a)-[:AE {weight:5}]->(c),
                   (a)-[:AE {weight:1}]->(c)
        """)
        self._astar_vs_sppaths(g, 0, 2, 1)
        self._astar_vs_sppaths(g, 0, 2, 100)

    def test14_astar_parallel_edges(self):
        # parallel edges are distinct paths for both single and K-shortest A*.
        g = self.db.select_graph("astar_parallel")
        g.query("""
            CREATE (a:AK {id:0, lat:37.0, lon:-122.0}),
                   (b:AK {id:1, lat:37.0, lon:-122.0}),
                   (c:AK {id:2, lat:37.0, lon:-122.0}),
                   (a)-[:AE {weight:3}]->(b),
                   (a)-[:AE {weight:5}]->(b),
                   (b)-[:AE {weight:1}]->(c),
                   (a)-[:AE {weight:4}]->(c)
        """)
        self._astar_vs_sppaths(g, 0, 2, 1)
        got = self._astar_vs_sppaths(g, 0, 2, 100)
        # a-b-c(4), a->c(4), a-b-c via heavy edge(6) => 3 distinct routes
        self.env.assertEquals(len(got), 3)

    def test15_astar_reltypes(self):
        # relTypes must restrict traversal for both single and K-shortest A*.
        g = self.db.select_graph("astar_reltypes")
        g.query("""
            CREATE (a:AK {id:0, lat:37.0, lon:-122.0}),
                   (b:AK {id:1, lat:37.0, lon:-122.0}),
                   (c:AK {id:2, lat:37.0, lon:-122.0}),
                   (a)-[:AR {weight:1}]->(b), (b)-[:AR {weight:1}]->(c),
                   (a)-[:AS {weight:1}]->(c)
        """)
        for rt in (["AR"], ["AS"], ["AR", "AS"]):
            self._astar_vs_sppaths(g, 0, 2, 1, rel_types=rt)
            self._astar_vs_sppaths(g, 0, 2, 100, rel_types=rt)
