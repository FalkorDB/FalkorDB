import time

from common import *
from index_utils import *
from functools import cmp_to_key

NODES = 20    # node count
EDGES = 200   # edge count

GRAPH_ID = "path_algos"

class testAllShortestPaths():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()
        self.init()

    def populate_graph(self):
        create_node_range_index(self.graph, 'L', 'v', sync=True)
        self.graph.query(f"UNWIND range(1, {NODES}) AS x CREATE (:L{{v: x}})")
        self.graph.query(f"""UNWIND range(1, {EDGES}) AS i
                             WITH ToInteger(rand() * {NODES}) AS x, ToInteger(rand() * {NODES}) AS y
                             MATCH (a:L{{v: x}}), (b:L{{v: y}})
                             CREATE (a)-[:E {{weight: ToInteger(rand()*5) + 1, cost: ToInteger(rand()*10) + 3}}]->(b)""")

    def init(self):
        self.n = 0                   # start node ID
        self.m = 0                   # end node ID
        self.sp_paths = []           # paths between (n)->(m)
        self.incoming_sp_paths = []  # paths between (m)<-(n)
        self.ss_paths = []           # all paths expand from (n)

        # look for nodes `i` and `j` with at least 10 different paths
        # between them, stop once found
        for i in range(1, NODES):
            for j in range(1, NODES):
                if i == j:
                    continue

                query = f"""
                MATCH (n:L {{v: {i}}}), (m:L {{v: {j}}})
                MATCH p=(n)-[:E*1..3]->(m)
                RETURN p,
                       reduce(weight = 0, r in relationships(p) | weight + r.weight) AS weight,
                       reduce(cost = 0, r in relationships(p) | cost + r.cost) AS cost,
                       length(p) as pathLen"""

                result = self.graph.query(query)
                l = len(result.result_set)
                if l > 10:
                    # found nodes `i` and `j` with multiple paths
                    self.n = i
                    self.m = j
                    self.sp_paths = result.result_set

                    query = f"""
                    MATCH (n:L {{v: {i}}})
                    MATCH p=(n)-[:E*1..3]->(m)
                    RETURN p,
                           reduce(weight = 0, r in relationships(p) | weight + r.weight) AS weight,
                           reduce(cost = 0, r in relationships(p) | cost + r.cost) AS cost,
                           length(p) as pathLen"""

                    result = self.graph.query(query)
                    self.ss_paths = result.result_set

                    query = f"""
                    MATCH (n:L {{v: {i}}}), (m:L {{v: {j}}})
                    MATCH p=(m)<-[:E*1..3]-(n)
                    RETURN p,
                           reduce(weight = 0, r in relationships(p) | weight + r.weight) AS weight,
                           reduce(cost = 0, r in relationships(p) | cost + r.cost) AS cost,
                           length(p) as pathLen"""

                    result = self.graph.query(query)
                    self.incoming_sp_paths = result.result_set
                    break

        # expecting `cost` to be at p[2]
        def compare_cost(p1, p2):
            return p1[2] - p2[2]

        def compare_full(p1, p2):
            # p[1] - weight
            # p[2] - cost
            # p[3] - length
            if p1[1] == p2[1]:
                if p1[2] == p2[2]:
                    return p1[3] - p2[3]
                return p1[2] - p2[2]
            return p1[1] - p2[1]

        # sort shortest paths by cost
        self.sp_paths.sort(key=cmp_to_key(compare_cost))
        self.max_cost = self.sp_paths[7][2]

        # filter
        self.sp_paths = [p for p in self.sp_paths if p[2] <= self.max_cost and len(p[0].nodes()) == len(set([n.id for n in p[0].nodes()]))]
        self.ss_paths = [p for p in self.ss_paths if p[2] <= self.max_cost and len(p[0].nodes()) == len(set([n.id for n in p[0].nodes()]))]
        self.incoming_sp_paths = [p for p in self.incoming_sp_paths if p[2] <= self.max_cost and len(p[0].nodes()) == len(set([n.id for n in p[0].nodes()]))]

        # sort
        self.sp_paths.sort(key=cmp_to_key(compare_full))
        self.ss_paths.sort(key=cmp_to_key(compare_full))
        self.incoming_sp_paths.sort(key=cmp_to_key(compare_full))

        # for p in self.sp_paths:
        #     print(p)
        #     print(p[0])

    def test01_SPpaths_validations(self):
        # all queries should produce a run-time errors
        queries = [
            """CALL algo.SPpaths({})""",
            """MATCH (n:L {v: 1}) CALL algo.SPpaths({sourceNode: n})""",
            """MATCH (n:L {v: 1}) CALL algo.SPpaths({targetNode: n})"""
        ]

        # validate we're getting an exception
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                self.env.assertContains("sourceNode and targetNode are required", str(e))

        # all queries should produce a run-time errors
        queries = [
            """MATCH (n:L {v: 1}) CALL algo.SPpaths({sourceNode: 1, targetNode: 1})""",
            """MATCH (n:L {v: 1}) CALL algo.SPpaths({sourceNode: 1, targetNode: n})""",
            """MATCH (n:L {v: 1}) CALL algo.SPpaths({sourceNode: n, targetNode: 1})"""
        ]

        # validate we're getting an exception
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                self.env.assertContains("sourceNode and targetNode must be of type Node", str(e))

        # all queries should produce a run-time errors
        queries = [
            """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, relTypes: 1})""",
            """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, relTypes: [1]})""",
            """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, relTypes: ['a', 1]})"""
        ]

        # validate we're getting an exception
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                self.env.assertContains("relTypes must be array of strings", str(e))

        # all queries should produce a run-time errors
        queries = [
            """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, relDirection: 1})""",
            """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, relDirection: 'a'})"""
        ]

        # validate we're getting an exception
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                self.env.assertContains("relDirection values must be 'incoming', 'outgoing' or 'both'", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, maxLen: 'a'})"""

        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("maxLen must be integer", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, weightProp: 1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("weightProp must be string", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, costProp: 1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("costProp must be string", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, maxCost: '1'})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("maxCost must be numeric", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, pathCount: '1'})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("pathCount must be integer", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, pathCount: -1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("pathCount must be a non-negative integer", str(e))

    def test01_SSpaths_validations(self):
        query = """CALL algo.SSpaths({})"""

        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("sourceNode is required", str(e))

        query = """MATCH (n:L {v: 1}) CALL algo.SSpaths({sourceNode: 1})"""

        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("sourceNode must be of type Node", str(e))

        # all queries should produce a run-time errors
        queries = [
            """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, relTypes: 1})""",
            """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, relTypes: [1]})""",
            """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, relTypes: ['a', 1]})"""
        ]

        # validate we're getting an exception
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                self.env.assertContains("relTypes must be array of strings", str(e))

        # all queries should produce a run-time errors
        queries = [
            """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, relDirection: 1})""",
            """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, relDirection: 'a'})"""
        ]

        # validate we're getting an exception
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                self.env.assertContains("relDirection values must be 'incoming', 'outgoing' or 'both'", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, maxLen: 'a'})"""

        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("maxLen must be integer", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, weightProp: 1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("weightProp must be string", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, costProp: 1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("costProp must be string", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, maxCost: '1'})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("maxCost must be numeric", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, pathCount: '1'})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("pathCount must be integer", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, pathCount: -1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError as e:
            self.env.assertContains("pathCount must be a non-negative integer", str(e))

    def sp_query(self, source, target, relTypes, maxLen, maxCost, pathCount, relDirection):
        args = ["sourceNode: n",
                "targetNode: m",
                "weightProp: 'weight'",
                "costProp: 'cost'"]
        if relTypes is not None:
            args.append(f"relTypes: {relTypes}")
        if maxLen is not None:
            args.append(f"maxLen: {maxLen}")
        if maxCost is not None:
            args.append(f"maxCost: {maxCost}")
        if pathCount is not None:
            args.append(f"pathCount: {pathCount}")
        if relDirection is not None:
            args.append(f"relDirection: '{relDirection}'")
        query = f"""
        MATCH (n:L {{v: {source}}}), (m:L {{v: {target}}})
        CALL algo.SPpaths({{{", ".join(args)}}}) YIELD path, pathWeight, pathCost
        RETURN path, pathWeight, pathCost, length(path)"""

        return self.graph.query(query)

    def test02_sp_single_path(self):
        results = [
            self.sp_query(self.n, self.m, ["E"], 3, self.max_cost, 1, None),
            self.sp_query(self.n, self.m, None, 3, self.max_cost, 1, None)
        ]

        for result in results:
            self.env.assertEqual(len(result.result_set), 1)

            all_minimal = [p for p in self.sp_paths if p[1]
                           == self.sp_paths[0][1]]
            self.env.assertContains(result.result_set[0], all_minimal)

        results = [
            self.sp_query(self.m, self.n, ["E"], 3, self.max_cost, 1, "incoming"),
            self.sp_query(self.m, self.n, None, 3, self.max_cost, 1, "incoming")
        ]

        for result in results:
            self.env.assertEqual(len(result.result_set), 1)

            all_minimal = [p for p in self.incoming_sp_paths if p[1]
                           == self.incoming_sp_paths[0][1]]
            self.env.assertContains(result.result_set[0], all_minimal)

    def test03_sp_all_minimal_paths(self):
        results = [
            self.sp_query(self.n, self.m, ["E"], 3, self.max_cost, 0, None),
            self.sp_query(self.n, self.m, None, 3, self.max_cost, 0, None)
        ]

        for result in results:
            all_minimal = [p for p in self.sp_paths if p[1] == self.sp_paths[0][1]]
            self.env.assertEqual(len(result.result_set), len(all_minimal))
            for i in range(0, len(all_minimal)):
                self.env.assertContains(result.result_set[i], all_minimal)

        results = [
            self.sp_query(self.m, self.n, ["E"], 3, self.max_cost, 0, "incoming"),
            self.sp_query(self.m, self.n, None, 3, self.max_cost, 0, "incoming")
        ]

        for result in results:
            all_minimal = [p for p in self.incoming_sp_paths if p[1] == self.incoming_sp_paths[0][1]]
            self.env.assertEqual(len(result.result_set), len(all_minimal))
            for i in range(0, len(all_minimal)):
                self.env.assertContains(result.result_set[i], all_minimal)

    def test04_sp_k_minimal_paths(self):
        results = [
            self.sp_query(self.n, self.m, ["E"], 3, self.max_cost, 5, None),
            self.sp_query(self.n, self.m, None, 3, self.max_cost, 5, None)
        ]

        for result in results:
            expected_len = min(len(self.sp_paths), 5)
            self.env.assertEqual(len(result.result_set), expected_len)
            for i in range(0, expected_len):
                self.env.assertContains(result.result_set[i], self.sp_paths)

        results = [
            self.sp_query(self.m, self.n, ["E"], 3, self.max_cost, 5, "incoming"),
            self.sp_query(self.m, self.n, None, 3, self.max_cost, 5, "incoming")
        ]

        for result in results:
            self.env.assertEqual(len(result.result_set), expected_len)
            for i in range(0, expected_len):
                self.env.assertContains(result.result_set[i], self.incoming_sp_paths)

    def ss_query(self, source, relTypes, maxLen, maxCost, pathCount, relDirection):
        args = ["sourceNode: n",
                "weightProp: 'weight'",
                "costProp: 'cost'"]
        if relTypes is not None:
            args.append(f"relTypes: {relTypes}")
        if maxLen is not None:
            args.append(f"maxLen: {maxLen}")
        if maxCost is not None:
            args.append(f"maxCost: {maxCost}")
        if pathCount is not None:
            args.append(f"pathCount: {pathCount}")
        if relDirection is not None:
            args.append(f"relDirection: '{relDirection}'")
        query = f"""
        MATCH (n:L {{v: {source}}})
        CALL algo.SSpaths({{{", ".join(args)}}}) YIELD path, pathWeight, pathCost
        RETURN path, pathWeight, pathCost, length(path)"""

        return self.graph.query(query)

    def test05_ss_single_path(self):
        results = [
            self.ss_query(self.n, ["E"], 3, self.max_cost, 1, None),
            self.ss_query(self.n, None, 3, self.max_cost, 1, None)
        ]

        # `ss_paths` is sorted by (weight, cost, length); when several paths tie
        # on all three, which one the search reports is implementation-defined
        # (depth-first order here, as in the C engine), so assert membership in
        # the tied set rather than a specific path — the same way test02..test04
        # already do. Asserting `ss_paths[0]` exactly made this flake whenever
        # the fixture happened to generate a tie at the minimum.
        best = self.ss_paths[0]
        tied = [p for p in self.ss_paths
                if (p[1], p[2], p[3]) == (best[1], best[2], best[3])]

        for result in results:
            self.env.assertEqual(len(result.result_set), 1)
            self.env.assertContains(result.result_set[0], tied)

    def test06_ss_all_minimal_paths(self):
        results = [
            self.ss_query(self.n, ["E"], 3, self.max_cost, 0, None),
            self.ss_query(self.n, None, 3, self.max_cost, 0, None)
        ]

        for result in results:
            all_minimal = [p for p in self.ss_paths if p[1] == self.ss_paths[0][1]]
            self.env.assertEqual(len(result.result_set), len(all_minimal))
            for i in range(0, len(all_minimal)):
                self.env.assertContains(result.result_set[i], all_minimal)

    def test07_ss_k_minimal_paths(self):
        results = [
            self.ss_query(self.n, ["E"], 3, self.max_cost, 5, None),
            self.ss_query(self.n, None, 3, self.max_cost, 5, None)
        ]

        for result in results:
            self.env.assertEqual(len(result.result_set), 5)
            for i in range(0, 5):
                self.env.assertContains(result.result_set[i], self.ss_paths)

    def test08_fractional_weights(self):
        # Regression test: path_cmp used to return (int)(weight_a - weight_b),
        # truncating differences in the range (-1.0, 1.0) to 0 and treating
        # paths with different fractional weights as equal, causing incorrect ordering.
        # The fix uses (a > b) - (a < b) which correctly returns -1, 0, or 1.
        g = self.db.select_graph("frac_weight_graph")

        # Build a graph with two paths from A to C:
        #   Path 1: (A) -[w=0.9]-> (C)                 total weight: 0.9
        #   Path 2: (A) -[w=0.1]-> (B) -[w=0.1]-> (C)  total weight: 0.2
        # Correct ordering: Path 2 (0.2) before Path 1 (0.9).
        # Buggy behaviour: int(0.9 - 0.2) = int(0.7) = 0, treating them equal.
        g.query("""
            CREATE (a:FN {id: 'A'}),
                   (b:FN {id: 'B'}),
                   (c:FN {id: 'C'}),
                   (a)-[:FR {weight: 0.9}]->(c),
                   (a)-[:FR {weight: 0.1}]->(b),
                   (b)-[:FR {weight: 0.1}]->(c)
        """)

        # SPpaths: both paths should be returned ordered by ascending weight
        result = g.query("""
            MATCH (src:FN {id: 'A'}), (dst:FN {id: 'C'})
            CALL algo.SPpaths({
                sourceNode: src,
                targetNode: dst,
                weightProp: 'weight',
                maxLen: 3,
                pathCount: 2
            }) YIELD pathWeight
            RETURN pathWeight
            ORDER BY pathWeight DESC
        """)

        self.env.assertEqual(len(result.result_set), 2)
        # first the heavier path (A->C, weight 0.9), then the lighter
        # path (A->B->C, weight 0.2).
        self.env.assertGreater(result.result_set[0][0], result.result_set[1][0])
        self.env.assertAlmostEqual(result.result_set[0][0], 0.9, delta=1e-9)
        self.env.assertAlmostEqual(result.result_set[1][0], 0.2, delta=1e-9)

        # SSpaths: same graph, verify ordering for single-source paths.
        # From A the three reachable paths are: A->B (0.1), A->B->C (0.2), A->C (0.9).
        # With pathCount=2 the two lightest are A->B (0.1) and A->B->C (0.2).
        # Their weight difference is 0.1 - a fractional value the old bug would
        # truncate to 0, making them appear equal and breaking the heap order.
        ss_result = g.query("""
            MATCH (src:FN {id: 'A'})
            CALL algo.SSpaths({
                sourceNode: src,
                weightProp: 'weight',
                maxLen: 3,
                pathCount: 2
            }) YIELD pathWeight
            RETURN pathWeight
            ORDER BY pathWeight DESC
        """)

        self.env.assertEqual(len(ss_result.result_set), 2)
        self.env.assertGreater(ss_result.result_set[0][0], ss_result.result_set[1][0])
        self.env.assertAlmostEqual(ss_result.result_set[0][0], 0.2, delta=1e-9)
        self.env.assertAlmostEqual(ss_result.result_set[1][0], 0.1, delta=1e-9)

    def _populate_branchy_graph(self, g, blob, fanout):
        """A densely connected blob of `blob` nodes, plus one node nothing points
        at so that a path into it can never exist.

        The fan-out is a deterministic modular hash rather than rand(), so the
        hop counts asserted below are stable across runs and across engines."""

        create_node_range_index(g, 'B', 'v', sync=True)
        g.query(f"UNWIND range(1, {blob}) AS x CREATE (:B {{v: x}})")
        g.query("CREATE (:B {v: 0})")  # unreachable: no incoming edges

        g.query(f"""
            UNWIND range(1, {blob}) AS x
            UNWIND range(1, {fanout}) AS k
            WITH x, 1 + (x * 7919 + k * 104729) % {blob} AS y
            MATCH (a:B {{v: x}}), (b:B {{v: y}})
            CREATE (a)-[:BE]->(b)""")

    def test09_unreachable_target_terminates(self):
        """Regression: algo.SPpaths used to search over *paths* rather than
        nodes, holding one heap entry per partial path with its own cloned
        visited-set and edge-list. With no route to the target it enumerated
        every simple path in the reachable component, so a 12 MB dataset drove
        used_memory past 4.9 GB and OOM-killed the server (the FalkorDB
        benchmark's `shortest_path` query, which passes no maxLen).

        A per-node reachability sweep answers "unreachable" in one pass. Pre-fix
        this test kills the server outright; the timeout keeps a milder
        regression from hanging the suite instead of failing it."""

        g = self.db.select_graph("unreachable_target")
        self._populate_branchy_graph(g, blob=200, fanout=8)

        # (v:0) has no incoming edges, so no path can reach it from the blob.
        # Each shape takes a different branch: no-maxLen is the single-shortest
        # path search, the rest go through the enumeration.
        for extra in ("",
                      ", maxLen: 12",
                      ", pathCount: 0",
                      ", pathCount: 4",
                      ", maxCost: 500, costProp: 'nonexistent'"):
            result = g.query(f"""
                MATCH (s:B {{v: 1}}), (t:B {{v: 0}})
                CALL algo.SPpaths({{sourceNode: s, targetNode: t,
                                    relTypes: ['BE'], relDirection: 'outgoing'
                                    {extra}}})
                YIELD path
                RETURN length(path)""", timeout=60000)
            self.env.assertEqual(result.result_set, [])

    def test10_deep_search_results(self):
        """Covers the deep-but-reachable case: the half of the benchmark OOM
        that still returned a correct answer (pre-fix this peaked at +139 MB on
        this fixture, and at 2.2 GB on pokec-small).

        Asserts the hop counts so the hardcoded pairs cannot quietly drift into
        no-ops, and so the unbounded-depth search — which no other test in this
        file reaches, every other one pins maxLen — stays covered."""

        g = self.db.select_graph("deep_search")
        self._populate_branchy_graph(g, blob=3000, fanout=8)

        def lengths(src, dst, extra=""):
            return sorted(row[0] for row in g.query(f"""
                MATCH (s:B {{v: {src}}}), (t:B {{v: {dst}}})
                CALL algo.SPpaths({{sourceNode: s, targetNode: t,
                                    relTypes: ['BE'], relDirection: 'outgoing'
                                    {extra}}})
                YIELD path
                RETURN length(path)""", timeout=60000).result_set)

        # unbounded maxLen: the benchmark's own shape, single cheapest path
        self.env.assertEqual(lengths(1, 137), [6])
        self.env.assertEqual(lengths(42, 299), [5])
        self.env.assertEqual(lengths(7, 88), [4])

        # bounded depth reaches the same answers through the enumeration
        self.env.assertEqual(lengths(1, 137, ", maxLen: 6"), [6])
        self.env.assertEqual(lengths(42, 299, ", maxLen: 6"), [5])
        self.env.assertEqual(lengths(7, 88, ", maxLen: 6"), [4])

        # all-minimal and k-minimal
        self.env.assertEqual(lengths(7, 88, ", maxLen: 5, pathCount: 0"), [4, 4])
        self.env.assertEqual(lengths(7, 88, ", maxLen: 6, pathCount: 3"), [4, 4, 6])
        # a shorter cap than the true distance must find nothing
        self.env.assertEqual(lengths(1, 137, ", maxLen: 5, pathCount: 0"), [])

    def test11_fractional_weights_through_enumeration(self):
        """Regression: the enumeration accumulates weight source->target while
        the reachability pre-pass used to sum its seed bound along the reverse
        parent walk. Float addition is not associative, so the seed could land
        an ULP below the forward sum of the very same edges and the
        `weight > bound` prune then dropped the only path — SPpaths silently
        returned nothing on ordinary decimal weights.

        0.1+0.2+0.3 forward is 0.6000000000000001, reverse is 0.6. Every shape
        that leaves the single-shortest-path fast path was affected."""

        g = self.db.select_graph("frac_enum")
        g.query("""
            CREATE (a:FE {id: 'A'}), (b:FE {id: 'B'}),
                   (c:FE {id: 'C'}), (d:FE {id: 'D'}),
                   (a)-[:FE {weight: 0.1}]->(b),
                   (b)-[:FE {weight: 0.2}]->(c),
                   (c)-[:FE {weight: 0.3}]->(d)""")

        for extra in ("",
                      ", maxLen: 10",
                      ", maxLen: 3",
                      ", pathCount: 0, maxLen: 10",
                      ", pathCount: 2, maxLen: 10",
                      ", maxCost: 100, costProp: 'weight'"):
            result = g.query(f"""
                MATCH (s:FE {{id: 'A'}}), (t:FE {{id: 'D'}})
                CALL algo.SPpaths({{sourceNode: s, targetNode: t,
                                    weightProp: 'weight'{extra}}})
                YIELD pathWeight
                RETURN pathWeight""", timeout=60000)
            self.env.assertEqual(
                len(result.result_set), 1,
                message=f"shape '{extra}' lost the only path")
            self.env.assertAlmostEqual(result.result_set[0][0], 0.6, delta=1e-9)

        # Same defect reached a second way: restoring the running total by
        # subtraction on backtrack does not round-trip in f64, so an unrelated
        # dead-end sibling could push the real path over the bound. The result
        # then depended on edge insertion order.
        g2 = self.db.select_graph("frac_sibling")
        g2.query("""
            CREATE (s:FS {id: 'S'}), (a:FS {id: 'A'}),
                   (t:FS {id: 'T'}), (y:FS {id: 'Y'}),
                   (s)-[:FS {weight: 0.1}]->(a),
                   (a)-[:FS {weight: 0.5}]->(t),
                   (a)-[:FS {weight: 0.2}]->(y)""")

        result = g2.query("""
            MATCH (s:FS {id: 'S'}), (t:FS {id: 'T'})
            CALL algo.SPpaths({sourceNode: s, targetNode: t,
                               weightProp: 'weight', maxLen: 10})
            YIELD pathWeight
            RETURN pathWeight""", timeout=60000)
        self.env.assertEqual(len(result.result_set), 1)
        self.env.assertAlmostEqual(result.result_set[0][0], 0.6, delta=1e-9)

    def test12_shortest_path_fast_path_results(self):
        """The unconstrained single-shortest-path shape (no maxLen, no maxCost,
        pathCount 1) takes a dedicated Dijkstra branch. Every other behavioural
        test in this file pins maxLen and so never reaches it, which left the
        parent-chain walk, the unit-weight default and the cost fold unasserted."""

        g = self.db.select_graph("fast_path")
        # A->C directly at 0.9, or A->B->C at 0.1+0.1: the cheaper route is the
        # longer one, so a hop-count search would pick the wrong path.
        g.query("""
            CREATE (a:FP {id: 'A'}), (b:FP {id: 'B'}), (c:FP {id: 'C'}),
                   (a)-[:FR {weight: 0.9, cost: 5}]->(c),
                   (a)-[:FR {weight: 0.1, cost: 1}]->(b),
                   (b)-[:FR {weight: 0.1, cost: 1}]->(c)""")

        result = g.query("""
            MATCH (s:FP {id: 'A'}), (t:FP {id: 'C'})
            CALL algo.SPpaths({sourceNode: s, targetNode: t,
                               weightProp: 'weight', costProp: 'cost'})
            YIELD path, pathWeight, pathCost
            RETURN [n IN nodes(path) | n.id], pathWeight, pathCost,
                   length(path)""", timeout=60000)

        self.env.assertEqual(len(result.result_set), 1)
        ids, weight, cost, hops = result.result_set[0]
        self.env.assertEqual(ids, ['A', 'B', 'C'])
        self.env.assertAlmostEqual(weight, 0.2, delta=1e-9)
        self.env.assertEqual(cost, 2)
        self.env.assertEqual(hops, 2)

        # unweighted: falls back to hop count, so the direct edge wins
        unweighted = g.query("""
            MATCH (s:FP {id: 'A'}), (t:FP {id: 'C'})
            CALL algo.SPpaths({sourceNode: s, targetNode: t})
            YIELD path, pathWeight
            RETURN [n IN nodes(path) | n.id], pathWeight""", timeout=60000)
        self.env.assertEqual(unweighted.result_set, [[['A', 'C'], 1]])

        # unreachable in the requested direction, reachable in the other
        self.env.assertEqual(
            g.query("""
                MATCH (s:FP {id: 'C'}), (t:FP {id: 'A'})
                CALL algo.SPpaths({sourceNode: s, targetNode: t})
                YIELD path RETURN length(path)""", timeout=60000).result_set,
            [])

    def test13_rel_direction_both(self):
        """`relDirection: 'both'` had no behavioural coverage: a swapped branch
        in the direction handling passed the whole suite. C->A exists only by
        traversing the two edges backwards."""

        g = self.db.select_graph("dir_both")
        g.query("""
            CREATE (a:DB {id: 'A'}), (b:DB {id: 'B'}), (c:DB {id: 'C'}),
                   (a)-[:DR]->(b), (b)-[:DR]->(c)""")

        def hops(src, dst, direction, extra=""):
            return g.query(f"""
                MATCH (s:DB {{id: '{src}'}}), (t:DB {{id: '{dst}'}})
                CALL algo.SPpaths({{sourceNode: s, targetNode: t,
                                    relTypes: ['DR'],
                                    relDirection: '{direction}'{extra}}})
                YIELD path RETURN length(path)""",
                timeout=60000).result_set

        self.env.assertEqual(hops('A', 'C', 'outgoing'), [[2]])
        self.env.assertEqual(hops('C', 'A', 'outgoing'), [])
        self.env.assertEqual(hops('C', 'A', 'incoming'), [[2]])
        self.env.assertEqual(hops('C', 'A', 'both'), [[2]])
        self.env.assertEqual(hops('A', 'C', 'both'), [[2]])
        # and through the enumeration branch, not just the fast path
        self.env.assertEqual(hops('C', 'A', 'both', ", maxLen: 5"), [[2]])
        self.env.assertEqual(hops('C', 'A', 'incoming', ", maxLen: 5"), [[2]])

    def test14_long_enumeration_honours_timeout(self):
        """The searches poll the query deadline from inside their loops. Without
        that, an enumeration that cannot finish is uninterruptible and the
        client-visible symptom is a hang until the server dies, rather than an
        error.

        `pathCount` this large keeps the pruning bound at infinity, so the
        depth-first walk cannot terminate on this fixture in any usable time."""

        g = self.db.select_graph("timeout_enum")
        self._populate_branchy_graph(g, blob=200, fanout=8)

        start = time.time()
        try:
            g.query("""
                MATCH (s:B {v: 1}), (t:B {v: 137})
                CALL algo.SPpaths({sourceNode: s, targetNode: t,
                                   relTypes: ['BE'], relDirection: 'outgoing',
                                   maxLen: 30, pathCount: 1000000})
                YIELD path RETURN count(path)""", timeout=2000)
            self.env.assertTrue(
                False, message="expected the query to hit its timeout")
        except redis.ResponseError as e:
            self.env.assertContains("Query timed out", str(e))

        # the deadline must actually cut the search short, not merely be
        # reported once it finishes on its own
        self.env.assertLess(time.time() - start, 30)
