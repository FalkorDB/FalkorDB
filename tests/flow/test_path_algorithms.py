from common import *
from index_utils import *
from functools import cmp_to_key
import heapq
import random

NODES = 20    # node count
EDGES = 200   # edge count

GRAPH_ID = "path_algos"

class testAllShortestPaths():
    def __init__(self):
        self.env, self.db = Env()
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
            except redis.exceptions.ResponseError as e:
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
            except redis.exceptions.ResponseError as e:
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
            except redis.exceptions.ResponseError as e:
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
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("relDirection values must be 'incoming', 'outgoing' or 'both'", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, maxLen: 'a'})"""

        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("maxLen must be integer", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, weightProp: 1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("weightProp must be string", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, costProp: 1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("costProp must be string", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, maxCost: '1'})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("maxCost must be numeric", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, pathCount: '1'})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("pathCount must be integer", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SPpaths({sourceNode: n, targetNode: m, pathCount: -1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("pathCount must be a non-negative integer", str(e))

    def test01_SSpaths_validations(self):
        query = """CALL algo.SSpaths({})"""

        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("sourceNode is required", str(e))

        query = """MATCH (n:L {v: 1}) CALL algo.SSpaths({sourceNode: 1})"""

        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
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
            except redis.exceptions.ResponseError as e:
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
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("relDirection values must be 'incoming', 'outgoing' or 'both'", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, maxLen: 'a'})"""

        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("maxLen must be integer", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, weightProp: 1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("weightProp must be string", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, costProp: 1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("costProp must be string", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, maxCost: '1'})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("maxCost must be numeric", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, pathCount: '1'})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("pathCount must be integer", str(e))

        query = """MATCH (n:L {v: 1}), (m:L {v: 5}) CALL algo.SSpaths({sourceNode: n, pathCount: -1})"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
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
            self.env.assertEquals(len(result.result_set), 1)

            all_minimal = [p for p in self.sp_paths if p[1]
                           == self.sp_paths[0][1]]
            self.env.assertContains(result.result_set[0], all_minimal)

        results = [
            self.sp_query(self.m, self.n, ["E"], 3, self.max_cost, 1, "incoming"),
            self.sp_query(self.m, self.n, None, 3, self.max_cost, 1, "incoming")
        ]

        for result in results:
            self.env.assertEquals(len(result.result_set), 1)

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
            self.env.assertEquals(len(result.result_set), len(all_minimal))
            for i in range(0, len(all_minimal)):
                self.env.assertContains(result.result_set[i], all_minimal)

        results = [
            self.sp_query(self.m, self.n, ["E"], 3, self.max_cost, 0, "incoming"),
            self.sp_query(self.m, self.n, None, 3, self.max_cost, 0, "incoming")
        ]

        for result in results:
            all_minimal = [p for p in self.incoming_sp_paths if p[1] == self.incoming_sp_paths[0][1]]
            self.env.assertEquals(len(result.result_set), len(all_minimal))
            for i in range(0, len(all_minimal)):
                self.env.assertContains(result.result_set[i], all_minimal)

    def test04_sp_k_minimal_paths(self):
        results = [
            self.sp_query(self.n, self.m, ["E"], 3, self.max_cost, 5, None),
            self.sp_query(self.n, self.m, None, 3, self.max_cost, 5, None)
        ]

        for result in results:
            expected_len = min(len(self.sp_paths), 5)
            self.env.assertEquals(len(result.result_set), expected_len)
            for i in range(0, expected_len):
                self.env.assertContains(result.result_set[i], self.sp_paths)

        results = [
            self.sp_query(self.m, self.n, ["E"], 3, self.max_cost, 5, "incoming"),
            self.sp_query(self.m, self.n, None, 3, self.max_cost, 5, "incoming")
        ]

        for result in results:
            self.env.assertEquals(len(result.result_set), expected_len)
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

        for result in results:
            self.env.assertEquals(len(result.result_set), 1)
            self.env.assertEquals(result.result_set[0], self.ss_paths[0])

    def test06_ss_all_minimal_paths(self):
        results = [
            self.ss_query(self.n, ["E"], 3, self.max_cost, 0, None),
            self.ss_query(self.n, None, 3, self.max_cost, 0, None)
        ]

        for result in results:
            all_minimal = [p for p in self.ss_paths if p[1] == self.ss_paths[0][1]]
            self.env.assertEquals(len(result.result_set), len(all_minimal))
            for i in range(0, len(all_minimal)):
                self.env.assertContains(result.result_set[i], all_minimal)

    def test07_ss_k_minimal_paths(self):
        results = [
            self.ss_query(self.n, ["E"], 3, self.max_cost, 5, None),
            self.ss_query(self.n, None, 3, self.max_cost, 5, None)
        ]

        for result in results:
            self.env.assertEquals(len(result.result_set), 5)
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

        self.env.assertEquals(len(result.result_set), 2)
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

        self.env.assertEquals(len(ss_result.result_set), 2)
        self.env.assertGreater(ss_result.result_set[0][0], ss_result.result_set[1][0])
        self.env.assertAlmostEqual(ss_result.result_set[0][0], 0.2, delta=1e-9)
        self.env.assertAlmostEqual(ss_result.result_set[1][0], 0.1, delta=1e-9)

    def test09_sp_dijkstra_unconstrained_cost(self):
        # pathCount == 1 with no maxCost set routes through the Dijkstra
        # fast path (SPpaths_dijkstra_single) instead of the DFS-based
        # search used everywhere else in this file. Verify it agrees with
        # the true minimum-weight path, computed independently via a fully
        # unconstrained Cypher variable-length match (no cost filtering).
        query = f"""
        MATCH (n:L {{v: {self.n}}}), (m:L {{v: {self.m}}})
        MATCH p=(n)-[:E*1..3]->(m)
        RETURN p,
               reduce(weight = 0, r in relationships(p) | weight + r.weight) AS weight"""
        result = self.graph.query(query)
        candidates = [p for p in result.result_set
                      if len(p[0].nodes()) == len(set(node.id for node in p[0].nodes()))]
        min_weight = min(p[1] for p in candidates)

        results = [
            self.sp_query(self.n, self.m, ["E"], 3, None, 1, None),
            self.sp_query(self.n, self.m, None, 3, None, 1, None)
        ]

        for result in results:
            self.env.assertEquals(len(result.result_set), 1)
            self.env.assertAlmostEqual(result.result_set[0][1], min_weight, delta=1e-9)

    def test10_sp_unreachable(self):
        # two nodes with no path between them at all: the BFS bound
        # pre-pass (finite maxCost) and Dijkstra (unconstrained maxCost)
        # must both report "no path" cleanly rather than erroring or
        # hanging, regardless of pathCount.
        g = self.db.select_graph("unreachable_graph")
        g.query("CREATE (x:UR {id: 'X'}), (y:UR {id: 'Y'})")

        def ur_query(path_count, max_cost):
            args = ["sourceNode: src", "targetNode: dst", "weightProp: 'w'"]
            if max_cost is not None:
                args.append(f"maxCost: {max_cost}")
            args.append(f"pathCount: {path_count}")
            return g.query(f"""
                MATCH (src:UR {{id: 'X'}}), (dst:UR {{id: 'Y'}})
                CALL algo.SPpaths({{{", ".join(args)}}}) YIELD path
                RETURN path
            """)

        # pathCount==1, unconstrained cost -> Dijkstra "not found" branch
        self.env.assertEquals(len(ur_query(1, None).result_set), 0)
        # pathCount==1, finite maxCost -> pre-pass short-circuit
        self.env.assertEquals(len(ur_query(1, 100).result_set), 0)
        # pathCount==0 (all-minimal), finite maxCost -> pre-pass short-circuit
        self.env.assertEquals(len(ur_query(0, 100).result_set), 0)
        # pathCount>1 (k-minimal), finite maxCost -> pre-pass short-circuit
        self.env.assertEquals(len(ur_query(3, 100).result_set), 0)

    def test11_sp_duplicate_edges(self):
        # (a) a self-loop combined with relDirection: 'both' can surface
        # the same edge as a candidate via both the outgoing and incoming
        # scan; it must be rejected like any other cycle, not corrupt or
        # duplicate the result.
        g = self.db.select_graph("selfloop_graph")
        g.query("""
            CREATE (a:SL {id: 'A'}),
                   (b:SL {id: 'B'}),
                   (a)-[:SLR {weight: 1}]->(a),
                   (a)-[:SLR {weight: 1}]->(b)
        """)

        result = g.query("""
            MATCH (src:SL {id: 'A'}), (dst:SL {id: 'B'})
            CALL algo.SPpaths({
                sourceNode: src,
                targetNode: dst,
                weightProp: 'weight',
                relDirection: 'both',
                pathCount: 1
            }) YIELD path, pathWeight
            RETURN pathWeight, length(path)
        """)

        self.env.assertEquals(len(result.result_set), 1)
        self.env.assertAlmostEqual(result.result_set[0][0], 1, delta=1e-9)
        self.env.assertEquals(result.result_set[0][1], 1)

        # (b) listing the same relationship type twice in relTypes must
        # not double-count a path in the k-minimal results (previously,
        # duplicate relation ids could surface a node/edge as a candidate
        # twice, letting the same path be re-derived and returned again
        # after the DFS backtracked through it).
        expected_len = min(len(self.sp_paths), 3)
        dup = self.sp_query(self.n, self.m, ["E", "E"], 3, self.max_cost, 3, None)
        single = self.sp_query(self.n, self.m, ["E"], 3, self.max_cost, 3, None)

        self.env.assertEquals(len(dup.result_set), expected_len)
        self.env.assertEquals(len(single.result_set), expected_len)
        for row in dup.result_set:
            self.env.assertContains(row, single.result_set)

    def test12_sp_src_eq_dst(self):
        # sourceNode == targetNode is degenerate: minLen==1 requires at
        # least one edge, so this must always return no results, whether
        # or not a self-loop exists, and regardless of which code path
        # handles it (Dijkstra fast path vs. DFS).
        result = self.sp_query(self.n, self.n, ["E"], 3, self.max_cost, 1, None)
        self.env.assertEquals(len(result.result_set), 0)

        # unconstrained cost -- this is exactly the case SPpaths_dijkstra_single
        # would otherwise mishandle (trivially "finding" src at distance 0).
        result = self.sp_query(self.n, self.n, None, 3, None, 1, None)
        self.env.assertEquals(len(result.result_set), 0)

    #-------------------------------------------------------------------------
    # Dijkstra fast-path (SPpaths_dijkstra_single) correctness on structured
    # graphs: pathCount==1 with no maxCost is exactly the condition that
    # routes into the Dijkstra fast path (see the dispatch check in
    # Proc_SPpathsInvoke). Each test below builds a graph with a known,
    # hand-designed topology, computes all-pairs shortest distances with an
    # independent Python Dijkstra implementation, then asks algo.SPpaths for
    # every (src, dst) combination in a single query and checks that every
    # reachable pair's pathWeight matches the reference exactly, and that
    # every unreachable pair yields no result at all.
    #-------------------------------------------------------------------------

    def _dijkstra_all_pairs(self, n_nodes, edges):
        # independent reference implementation: all-pairs shortest distances
        # over a directed, non-negatively weighted graph given as a list of
        # (u, v, weight) edges. returns {src: {dst: dist, ...}, ...}, where
        # unreachable dst's are simply absent from the inner dict.
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
                    continue  # stale heap entry, already finalized cheaper
                for v, w in adj[u]:
                    nd = d + w
                    if nd < dist.get(v, float('inf')):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
            all_dist[src] = dist

        return all_dist

    def _verify_dijkstra_all_pairs(self, graph_name, n_nodes, edges):
        # build a directed weighted graph from `edges` (nodes labeled :DK
        # with an 'id' property 0..n_nodes-1, edges typed :DE with a
        # 'weight' property), then validate algo.SPpaths' Dijkstra fast
        # path (pathCount: 1, no maxCost) against the independent reference
        # in self._dijkstra_all_pairs for every (src, dst) combination.
        g = self.db.select_graph(graph_name)
        g.query(f"UNWIND range(0, {n_nodes - 1}) AS x CREATE (:DK {{id: x}})")

        if edges:
            rows = ", ".join(f"[{u}, {v}, {w}]" for u, v, w in edges)
            g.query(f"""
                UNWIND [{rows}] AS e
                MATCH (a:DK {{id: e[0]}}), (b:DK {{id: e[1]}})
                CREATE (a)-[:DE {{weight: e[2]}}]->(b)
            """)

        # one query drives algo.SPpaths once per (n, m) row produced by the
        # outer MATCH, covering every ordered pair in a single round trip;
        # unreachable pairs simply yield no row (CALL acts like an inner
        # join), so they never appear in 'actual' below.
        result = g.query("""
            MATCH (n:DK), (m:DK)
            WHERE n.id <> m.id
            CALL algo.SPpaths({
                sourceNode: n,
                targetNode: m,
                weightProp: 'weight',
                pathCount: 1
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
                    # no path should have been found for this pair
                    self.env.assertNotContains(key, actual)
                else:
                    self.env.assertContains(key, actual)
                    self.env.assertAlmostEqual(actual[key], exp_weight, delta=1e-9)

    def test13_dijkstra_line_graph(self):
        # simple directed line 0->1->2->...->(n-1) with strictly increasing
        # weights. exactly one path exists between any (src, dst) pair, and
        # only "forward" pairs are reachable at all -- a baseline sanity
        # check for path composition, weight accumulation, and correct
        # rejection of unreachable (backward) pairs.
        n = 8
        edges = [(i, i + 1, i + 1) for i in range(n - 1)]
        self._verify_dijkstra_all_pairs("dijkstra_line", n, edges)

    def test14_dijkstra_diamond_graph(self):
        # a DAG made of two chained "diamonds", each offering a cheap and an
        # expensive parallel route between the same pair of nodes:
        #   0 -> 1 -> 3 -> 4 -> 6 -> 7   (cheap branch: weight 1 each hop)
        #   0 -> 2 -> 3 -> 5 -> 6 -> 7   (expensive branch: weight 5 each hop)
        # Dijkstra must independently pick the cheap branch at each diamond.
        edges = [
            (0, 1, 1), (0, 2, 5),
            (1, 3, 1), (2, 3, 1),
            (3, 4, 1), (3, 5, 5),
            (4, 6, 1), (5, 6, 1),
            (6, 7, 1),
        ]
        self._verify_dijkstra_all_pairs("dijkstra_diamond", 8, edges)

    def test15_dijkstra_grid_graph(self):
        # a 4x4 grid with only rightward/downward edges and randomized
        # weights: many equal-length alternative routes exist between any
        # two nodes on the same diagonal, forcing Dijkstra to actually
        # compare weights rather than just hop count. reversed (backward)
        # pairs are unreachable, exercising that path too.
        random.seed(1234)
        rows, cols = 4, 4

        def node_id(r, c):
            return r * cols + c

        edges = []
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    edges.append((node_id(r, c), node_id(r, c + 1), random.randint(1, 9)))
                if r + 1 < rows:
                    edges.append((node_id(r, c), node_id(r + 1, c), random.randint(1, 9)))

        self._verify_dijkstra_all_pairs("dijkstra_grid", rows * cols, edges)

    def test16_dijkstra_dense_cyclic_graph(self):
        # a dense graph with edges in both directions between most node
        # pairs, forming many cycles. exercises Dijkstra's finalize-once
        # invariant and lazy heap-deletion under heavy relabeling: many
        # nodes get repeatedly relaxed and re-queued before the true
        # minimum is found.
        n = 10
        edges = []
        for u in range(n):
            for v in range(n):
                if u != v and (u + v) % 3 != 0:
                    edges.append((u, v, ((u * 7 + v * 13) % 11) + 1))

        self._verify_dijkstra_all_pairs("dijkstra_dense", n, edges)
