from common import *
import threading


class testBFS(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("proc_bfs")
        self.populate_graph()

    def populate_graph(self):
        # Construct a graph with the form:
        # (a)-[:E1]->(b:B)-[:E1]->(c), (b)-[:E2]->(d)-[:E1]->(e)

        q = """
        CREATE (a {v: 'a'})
        CREATE (b {v: 'b'})
        CREATE (c {v: 'c'})
        CREATE (d {v: 'd'})
        CREATE (e {v: 'e'})
        CREATE (a)-[ab:E1 {v:'b'}]->(b)
        CREATE (b)-[bc:E1 {v:'c'}]->(c)
        CREATE (b)-[bd:E2 {v:'d'}]->(d)
        CREATE (d)-[de:E1 {v:'e'}]->(e)
        RETURN a, b, c, d, e, ab, bc, bd, de"""

        res = self.graph.query(q).result_set

    # Verify that the contents of two arrays are equal without respect to order.
    def compare_unsorted_arrays(self, a, b):
        self.env.assertEqual(len(a), len(b))
        for elem in a:
            # Each element in a should appear in b exactly once.
            self.env.assertEqual(b.count(elem), 1)

    def test_invalid_invocation(self):
        invalid_queries = [

            # array relationship type
            """MATCH (a {v: 'a'}) CALL algo.bfs(a, -1, {'E1'}) YIELD nodes""",         

            # unexpected extra parameters
            """MATCH (a {v: 'a'}) CALL algo.bfs(a, -1, NULL, 'extra') YIELD nodes""",         

            # invalid configuration type
            """MATCH (a {v: 'a'}) CALL algo.bfs('invalid') YIELD nodes""",

            # integer values in relationshipType
            """MATCH (a {v: 'a'}) CALL algo.bfs(a, -1, 0) YIELD nodes""",

            # non-existent yield field
            """MATCH (a {v: 'a'}) CALL algo.bfs(null) YIELD nodes, invalidField""",

            # max depth is expected to an integer
            """MATCH (a {v: 'a'}) CALL algo.bfs(a, 'hello', NULL)""",
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except:
                pass
    # Test BFS from a single source without specifying a relationship type.
    def test01_bfs_single_source_all_reltypes(self):
        # Test BFS algorithm for node collection.
        # The results array must be sorted, since the order is non-deterministic (due to creations occurring in any order).
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, NULL) YIELD nodes
                   UNWIND nodes AS n
                   WITH n.v AS v
                   ORDER BY n.v
                   RETURN COLLECT(v)"""

        actual_result = self.graph.query(query)
        expected_result = [[['b', 'c', 'd', 'e']]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, NULL) YIELD nodes
                   RETURN nodes"""
        self.compare_unsorted_arrays(actual_result.result_set[0][0], expected_result[0][0])

        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, NULL) YIELD nodes
                   UNWIND nodes AS n
                   WITH a, n.v AS v
                   ORDER BY n.v
                   RETURN a.v, COLLECT(v)"""

        actual_result = self.graph.query(query)
        expected_result = [['a', ['b', 'c', 'd', 'e']]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # Test BFS algorithm for node and edge collection.
        # Parity between nodes and edges can be validated by testing the properties of each for equality,
        # as edges have the same property as their destination node.
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, NULL) YIELD nodes, edges
                   RETURN [n IN nodes | n.v], [e IN edges | e.v]"""

        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set[0][0], actual_result.result_set[0][1])

    # Test BFS from a single source traversing a single relationship type.
    def test02_bfs_single_source_restricted_reltype(self):
        # Test BFS algorithm for node collection.
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, 'E1') YIELD nodes
                   RETURN [n IN nodes | n.v]"""

        actual_result = self.graph.query(query)
        expected_result = ['b', 'c']
        self.compare_unsorted_arrays(actual_result.result_set[0][0], expected_result)

        # Test BFS algorithm for node and edge collection.
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, -1, 'E1') YIELD nodes, edges
                   RETURN [n IN nodes | n.v], [e IN edges | e.v]"""

        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set[0][0], actual_result.result_set[0][1])

    # Test BFS from all sources traversing a single relationship type, ignoring 0-hop paths.
    def test03_bfs_all_sources_restricted_reltype(self):
        # We only expect to see 'd' as a source node, as it is connected as a destination by an 'E2' edge.
        query = """MATCH (a)
                   CALL algo.BFS(a, -1, 'E1') YIELD nodes
                   RETURN a.v, [n IN nodes | n.v]
                   ORDER BY a.v"""

        actual_result = self.graph.query(query)
        expected_result = [['a', ['b', 'c']],
                           ['b', ['c']],
                           ['d', ['e']]]
        for idx, row in enumerate(actual_result.result_set):
            self.env.assertEqual(row[0], expected_result[idx][0])
            self.compare_unsorted_arrays(row[1], expected_result[idx][1])

        # Test BFS path-tracking algorithm
        query = """MATCH (a)
                   CALL algo.BFS(a, -1, 'E1') YIELD nodes, edges
                   RETURN a.v, [n IN nodes | n.v], [e IN edges | e.v]
                   ORDER BY a.v"""

        actual_result = self.graph.query(query)
        expected_result = [['a', ['b', 'c'], ['b', 'c']],
                           ['b', ['c'], ['c']],
                           ['d', ['e'], ['e']]]
        for idx, row in enumerate(actual_result.result_set):
            self.env.assertEqual(row[0], expected_result[idx][0])
            self.compare_unsorted_arrays(row[1], expected_result[idx][1])
            self.compare_unsorted_arrays(row[2], expected_result[idx][2])

    # Test BFS from a single source with a maximum depth.
    def test04_bfs_single_source_max_depth(self):
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, 1, NULL) YIELD nodes
                   RETURN [n IN nodes | n.v]"""

        actual_result = self.graph.query(query)
        expected_result = [[['b']]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        query = """MATCH (a {v: 'a'}) CALL algo.BFS(a, 1, NULL) YIELD nodes, edges RETURN [n IN nodes | n.v], [e IN edges | e.v]"""
        actual_result = self.graph.query(query)
        expected_result = [[['b'], ['b']]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Test BFS from all sources with a maximum depeth.
    def test05_bfs_all_sources_max_depth(self):
        query = """MATCH (a)
                   CALL algo.BFS(a, 1, NULL) YIELD nodes
                   RETURN a.v, [n IN nodes | n.v]
                   ORDER BY a.v"""

        actual_result = self.graph.query(query)
        expected_result = [['a', ['b']],
                           ['b', ['c', 'd']],
                           ['d', ['e']]]
        for idx, row in enumerate(actual_result.result_set):
            self.env.assertEqual(row[0], expected_result[idx][0])
            self.compare_unsorted_arrays(row[1], expected_result[idx][1])

        query = """MATCH (a)
                   CALL algo.BFS(a, 1, NULL) YIELD nodes, edges
                   RETURN a.v, [n IN nodes | n.v], [e IN edges | e.v]
                   ORDER BY a.v"""

        actual_result = self.graph.query(query)
        expected_result = [['a', ['b'], ['b']],
                           ['b', ['c', 'd'], ['c', 'd']],
                           ['d', ['e'], ['e']]]
        for idx, row in enumerate(actual_result.result_set):
            self.env.assertEqual(row[0], expected_result[idx][0])
            self.compare_unsorted_arrays(row[1], expected_result[idx][1])
            self.compare_unsorted_arrays(row[2], expected_result[idx][2])

    def test06_bfs_no_results(self):
        empty_result_set = []
        # Missing relationship type
        query = """MATCH (a)
                   CALL algo.BFS(a, -1, 'NONE_EXISTING_RELATION') YIELD nodes"""

        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, empty_result_set)

        # Leaf node
        query = """MATCH (leaf {v:'e'})
                   CALL algo.BFS(leaf, -1, NULL) YIELD nodes"""

        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, empty_result_set)

        # NULL node
        query = """OPTIONAL MATCH (n:NONE_EXISTING_LABEL)
                   CALL algo.BFS(n, -1, NULL) YIELD nodes"""

        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, empty_result_set)

    # test a query which calls BFS multiple times in the same scope
    def test07_multiple_bfs_calls(self):
        query = """MATCH (a {v: 'a'})
                   CALL algo.BFS(a, 1, NULL) YIELD nodes as n1
                   MATCH (b {v: 'd'})
                   CALL algo.BFS(b, 1, NULL) YIELD nodes as n2
                   RETURN [n IN n1 | n.v] AS x, [n IN n2 | n.v] AS y
                   ORDER BY x"""

        actual_result = self.graph.query(query)
        expected_result = [[['b'], ['e']]]
        self.env.assertEqual(actual_result.result_set, expected_result)




class testBFSEdgeChanges(FlowTestsBase):
    def __init__(self):
        # enableDebugCommand: dumpAndReload goes through DEBUG RELOAD.
        self.env, self.db = Env(enableDebugCommand=True)
        self.graph = self.db.select_graph("proc_bfs_changes")

    def reach(self, max_depth=-1):
        q = f"""MATCH (s:N {{v: 0}})
                CALL algo.BFS(s, {max_depth}, 'E') YIELD nodes
                RETURN [n IN nodes | n.v]"""
        return sorted(self.graph.query(q).result_set[0][0])

    def test01_bfs_sees_edge_changes(self):
        # BFS reads the stored relationship matrix in place when it holds no
        # pending change (as right after a load) and materializes it
        # otherwise. Each step below flips between the two, and every
        # committed delete and create must be visible to the next call.
        g = self.graph
        g.query("UNWIND range(0, 4) AS i CREATE (:N {v: i})")
        g.query("""UNWIND range(0, 3) AS i
                   MATCH (a:N {v: i}), (b:N {v: i + 1})
                   CREATE (a)-[:E]->(b)""")
        self.env.assertEqual(self.reach(), [1, 2, 3, 4])

        self.env.dumpAndReload()
        self.env.assertEqual(self.reach(), [1, 2, 3, 4])
        self.env.assertEqual(self.reach(2), [1, 2])

        g.query("MATCH (:N {v: 1})-[r:E]->(:N {v: 2}) DELETE r")
        self.env.assertEqual(self.reach(), [1])

        self.env.dumpAndReload()
        self.env.assertEqual(self.reach(), [1])

        g.query("MATCH (a:N {v: 1}), (b:N {v: 3}) CREATE (a)-[:E]->(b)")
        self.env.assertEqual(self.reach(), [1, 3, 4])
        self.env.assertEqual(self.reach(2), [1, 3])

    def test02_push_pull_push_transitions(self):
        # A wide layer (W mid nodes) between a single source and a narrow
        # tail drives the direction-optimizing BFS push -> pull -> push, with
        # converging edges (every mid node reaches the hub), a cycle back to
        # the source and dense edges among the mids. Matrices keep a capacity
        # of 16384 rows, so the layer must hold more than 16384/14 nodes to
        # switch to pull and the graph needs
        # avg_degree^depth >= 16384/64 (an average degree above ~1.2 at depth
        # 30) for a depth-limited run to take the same path; the test is
        # coupled to that capacity and to the BFS heuristics. A reload leaves
        # the transpose clean so pull is available.
        g = self.db.select_graph("proc_bfs_push_pull")
        W, TAIL, FANOUT = 2000, 30, 8
        hub = W + 1
        last = hub + TAIL
        g.query("CREATE INDEX FOR (n:N) ON (n.v)")
        g.query(f"UNWIND range(0, {last}) AS i CREATE (:N {{v: i}})")
        g.query(f"""MATCH (s:N {{v: 0}}), (m:N) WHERE 1 <= m.v <= {W}
                   CREATE (s)-[:E]->(m)""")
        g.query(f"""MATCH (m:N), (h:N {{v: {hub}}}) WHERE 1 <= m.v <= {W}
                   CREATE (m)-[:E]->(h)""")
        g.query(f"""UNWIND range(1, {W}) AS i UNWIND range(1, {FANOUT}) AS d
                   MATCH (a:N {{v: i}}), (b:N {{v: 1 + (i - 1 + d) % {W}}})
                   CREATE (a)-[:E]->(b)""")
        g.query(f"""UNWIND range({hub}, {last - 1}) AS i
                   MATCH (a:N {{v: i}}), (b:N {{v: i + 1}})
                   CREATE (a)-[:E]->(b)""")
        g.query(f"MATCH (a:N {{v: {last}}}), (b:N {{v: 0}}) CREATE (a)-[:E]->(b)")
        self.env.dumpAndReload()

        def reach(depth, edges=False):
            y = "nodes, edges" if edges else "nodes"
            q = f"""MATCH (s:N {{v: 0}})
                    CALL algo.BFS(s, {depth}, 'E') YIELD {y}
                    RETURN [n IN nodes | n.v]"""
            return sorted(g.query(q).result_set[0][0])

        # level of tail node hub + j is 2 + j; the mids are level 1 and the
        # hub level 2
        def upto(depth):
            return list(range(1, W + 1)) + list(range(hub, min(last, hub + depth - 2) + 1))

        for edges in (False, True):
            self.env.assertEqual(reach(-1, edges), list(range(1, last + 1)))
            for depth in (1, 2, 4, 30, 40):
                self.env.assertEqual(reach(depth, edges), upto(depth))


class testBFSConcurrentWrites(FlowTestsBase):
    """algo.BFS lends the stored relationship matrix to GraphBLAS without a
    copy. That is only sound while the lent matrix is immutable: writers must
    copy-on-write a new version, never touch the committed one. These tests
    run BFS from several threads while a writer keeps changing the graph and
    check every result against the states the graph can be in. A torn or
    corrupted read shows up as a node set no snapshot can produce (or as a
    crash, under the ASan flow job)."""

    W = 2000      # width of the layer between the source and the hub
    TAIL = 20     # nodes hanging off the hub in a chain
    FANOUT = 8    # edges among the mids, so the layer is dense enough to pull
    READERS = 4

    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True, moduleArgs="THREAD_COUNT 8")

    def build(self, name):
        W, TAIL, FANOUT = self.W, self.TAIL, self.FANOUT
        hub = W + 1
        last = hub + TAIL
        g = self.db.select_graph(name)
        g.query("CREATE INDEX FOR (n:N) ON (n.v)")
        g.query(f"UNWIND range(0, {last}) AS i CREATE (:N {{v: i}})")
        g.query(f"""MATCH (s:N {{v: 0}}), (m:N) WHERE 1 <= m.v <= {W}
                   CREATE (s)-[:E]->(m)""")
        g.query(f"""MATCH (m:N), (h:N {{v: {hub}}}) WHERE 1 <= m.v <= {W}
                   CREATE (m)-[:E]->(h)""")
        g.query(f"""UNWIND range(1, {W}) AS i UNWIND range(1, {FANOUT}) AS d
                   MATCH (a:N {{v: i}}), (b:N {{v: 1 + (i - 1 + d) % {W}}})
                   CREATE (a)-[:E]->(b)""")
        g.query(f"""UNWIND range({hub}, {last - 1}) AS i
                   MATCH (a:N {{v: i}}), (b:N {{v: i + 1}})
                   CREATE (a)-[:E]->(b)""")
        # start from a reloaded graph: clean bases, so BFS lends them
        self.env.dumpAndReload()
        return g, hub, last

    def reach(self, g, depth, edges):
        y = "nodes, edges" if edges else "nodes"
        q = f"""MATCH (s:N {{v: 0}})
                CALL algo.BFS(s, {depth}, 'E') YIELD {y}
                RETURN [n IN nodes | n.v]"""
        return g.query(q).result_set[0][0]

    def run_readers(self, name, writer, valid, depths):
        """Run `writer(g)` while READERS threads run BFS in a loop; every
        result must satisfy valid(depth, sorted_nodes). The writer starts once
        every reader has completed a BFS, and the reads that began after it
        started and ended before it finished are counted: the test fails if
        there were none, so it cannot pass without reader-writer overlap."""
        ready = threading.Barrier(self.READERS + 1)
        writing = threading.Event()
        done = threading.Event()
        errors = []
        overlapped = [0] * self.READERS

        def reader(i):
            g = self.db.select_graph(name)
            n = 0
            try:
                while not done.is_set() or n < 5:
                    depth = depths[n % len(depths)]
                    edges = (n // len(depths)) % 2 == 1
                    began_during = writing.is_set() and not done.is_set()
                    got = sorted(self.reach(g, depth, edges))
                    if not valid(depth, got):
                        errors.append((depth, edges, got[:5], len(got)))
                        break
                    if began_during and not done.is_set():
                        overlapped[i] += 1
                    n += 1
                    if n == 1:
                        ready.wait(timeout=60)
            except Exception as e:
                errors.append(repr(e))
                ready.abort()

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(self.READERS)]
        for t in threads:
            t.start()
        try:
            ready.wait(timeout=60)
            writing.set()
            writer(self.db.select_graph(name))
        finally:
            done.set()
            for t in threads:
                t.join()
        self.env.assertEqual(errors, [])
        self.env.assertGreaterEqual(min(overlapped), 1,
                                    message="a reader never ran while the writer did")
        self.env.assertGreaterEqual(sum(overlapped), 4 * self.READERS,
                                    message=f"too little overlap: {overlapped}")

    def test01_tail_and_bulk_rewrites(self):
        # The writer shrinks and regrows the hub's tail one edge per
        # transaction and, each round, deletes and recreates all W mid->hub
        # edges (large deltas, so the forward base is folded into a new
        # matrix). At every instant the reach set is the mids plus a prefix of
        # the tail, or just the mids while the hub edges are gone.
        g0, hub, last = self.build("bfs_conc_a")
        W, TAIL = self.W, self.TAIL
        mids = set(range(1, W + 1))

        def valid(depth, got):
            s = set(got)
            if len(got) != len(s) or not mids <= s:
                return False
            extra = sorted(s - mids)
            if not extra:
                return True
            if extra != list(range(hub, hub + len(extra))):
                return False
            limit = TAIL + 1 if depth < 0 else max(depth - 1, 0)
            return len(extra) <= limit

        def writer(g):
            for _ in range(3):
                for j in range(TAIL - 1, -1, -1):
                    g.query(f"MATCH (:N {{v: {hub + j}}})-[r:E]->(:N {{v: {hub + j + 1}}}) DELETE r")
                g.query(f"MATCH (:N)-[r:E]->(:N {{v: {hub}}}) DELETE r")
                g.query(f"""MATCH (m:N), (h:N {{v: {hub}}}) WHERE 1 <= m.v <= {W}
                           CREATE (m)-[:E]->(h)""")
                for j in range(TAIL):
                    g.query(f"""MATCH (a:N {{v: {hub + j}}}), (b:N {{v: {hub + j + 1}}})
                               CREATE (a)-[:E]->(b)""")

        self.run_readers("bfs_conc_a", writer, valid, [-1, 2, 5, 30])

        # quiesced: exactly the final snapshot, on the clean and reloaded base
        for _ in range(2):
            want = list(range(1, last + 1))
            for edges in (False, True):
                self.env.assertEqual(sorted(self.reach(g0, -1, edges)), want)
            self.env.dumpAndReload()

    def test02_other_type_churn_and_node_growth(self):
        # E never changes, so every BFS must return exactly the static
        # answer, while the writer churns another relationship type and
        # creates and deletes nodes, so every version resizes (and copies)
        # matrices that share E's committed base.
        g0, hub, last = self.build("bfs_conc_b")
        W, TAIL = self.W, self.TAIL
        expect = {d: sorted(list(range(1, W + 1)) + list(range(hub, min(last, hub + d - 2) + 1)))
                  if d >= 2 else (sorted(range(1, W + 1)) if d == 1 else [])
                  for d in (2, 5, 30)}
        expect[-1] = list(range(1, last + 1))

        def valid(depth, got):
            return got == expect[depth]

        def writer(g):
            for r in range(60):
                g.query(f"""UNWIND range(1, 300) AS i
                           CREATE (:X {{v: i}})-[:F]->(:X {{v: i + {r}}})""")
                g.query("MATCH (x:X)-[f:F]->() WHERE x.v % 2 = 0 DELETE f")
                g.query("MATCH (x:X) WHERE x.v % 3 = 0 DELETE x")
                g.query("""MATCH (a:N), (b:N) WHERE a.v = 1 AND b.v = 2
                           CREATE (a)-[:G]->(b)""")
                g.query("MATCH ()-[g:G]->() DELETE g")

        self.run_readers("bfs_conc_b", writer, valid, [-1, 2, 5, 30])
        self.env.assertEqual(sorted(self.reach(g0, -1, False)), expect[-1])
