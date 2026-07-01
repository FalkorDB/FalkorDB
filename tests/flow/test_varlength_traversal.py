"""
Flow tests for variable-length traversal correctness.

Covers the "convergent diamond" bug where a bounded range pattern like
[:F*1..2] returns a STRICT SUBSET of the exact-length pattern [:F*2..2],
dropping legitimately-reachable endpoints.  The result set of [lo..hi]
MUST be a superset of every exact-length [k..k] for lo <= k <= hi.

Also verifies the split-union invariant (with multiplicity):
  [1..n]  ==  [1..m]  UNION  [m+1..n]   for every 1 <= m < n

where UNION is the multiset sum -- if id 3 appears 5 times across the two
sub-ranges it must appear exactly 5 times in the full range result.
"""

from collections import Counter

from common import *

GRAPH_ID = "varlength_traversal"


class testVarLengthTraversal(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def setUp(self):
        self.conn.delete(GRAPH_ID)

    # ------------------------------------------------------------------ #
    #  Helper: build a graph from an edge-list with explicit numeric ids   #
    # ------------------------------------------------------------------ #

    def build_graph_from_edgelist(self, edges, label="U", rel="F"):
        """Create a fresh graph from a list of (src_id, dst_id) integer pairs.

        Every distinct integer becomes a node with property ``id`` equal to
        that integer.  All edges receive the same relationship type ``rel``.
        The graph is created in a single UNWIND query so that internal node
        ids follow the order the integers first appear in the edge list,
        which is intentional for id-layout-sensitive bug reproduction.
        """
        pairs = ",".join(f"[{a},{b}]" for a, b in edges)
        self.graph.query(
            f"UNWIND [{pairs}] AS e "
            f"MERGE (x:{label} {{id:e[0]}}) "
            f"MERGE (y:{label} {{id:e[1]}}) "
            f"MERGE (x)-[:{rel}]->(y)"
        )

    # ------------------------------------------------------------------ #
    #  Helper: build a multigraph (parallel edges allowed)               #
    # ------------------------------------------------------------------ #

    def build_multigraph_from_edgelist(self, edges, label="U", rel="F"):
        """Like build_graph_from_edgelist but allows multiple edges between
        the same pair of nodes.  Nodes are MERGEd (unique by id); edges are
        CREATEd so that repeated (src, dst) pairs produce parallel edges.
        """
        unique_ids = sorted({n for e in edges for n in e})
        ids_str = ",".join(str(i) for i in unique_ids)
        self.graph.query(f"UNWIND [{ids_str}] AS i MERGE (x:{label} {{id:i}})")
        pairs = ",".join(f"[{a},{b}]" for a, b in edges)
        self.graph.query(
            f"UNWIND [{pairs}] AS e "
            f"MATCH (x:{label} {{id:e[0]}}), (y:{label} {{id:e[1]}}) "
            f"CREATE (x)-[:{rel}]->(y)"
        )

    # ------------------------------------------------------------------ #
    #  Helper: query endpoints reachable via a range traversal           #
    # ------------------------------------------------------------------ #

    def endpoints(self, start_id, rng=None, label="U", rel="F"):
        """Return a sorted list of b.id values (with multiplicity) reachable
        from start_id via ``-[:rel*rng]->`` where rng is a string like "1..2".

        Pass rng=None for an unbounded traversal (equivalent to ``*``).
        Multiplicity is preserved so that callers can perform multiset
        comparisons.  Use ``set(self.endpoints(...))`` when only distinct
        reachability is needed.
        """
        rng_spec = f"*{rng}" if rng is not None else "*"
        q = (
            f"MATCH (a:{label} {{id:{start_id}}})-[:{rel}{rng_spec}]->(b:{label}) "
            f"RETURN b.id"
        )
        return sorted(row[0] for row in self.graph.query(q).result_set)

    # ------------------------------------------------------------------ #
    #  Helper: assert the split-union invariant for range [1..n]         #
    # ------------------------------------------------------------------ #

    def assert_range_split_invariant(self, start_id, n, label="U", rel="F"):
        """For every split point m in [1, n-1] verify that

            Counter(endpoints([1..n])) == Counter(endpoints([1..m])) +
                                          Counter(endpoints([m+1..n]))

        Multiplicity is intentional: if id 3 is returned 5 times across the
        two sub-ranges it must appear exactly 5 times in the full range.
        A failure means the range traversal is either over- or
        under-counting relative to the union of its constituent sub-ranges.
        """
        full = Counter(self.endpoints(start_id, f"1..{n}", label=label, rel=rel))
        for m in range(1, n):
            low = Counter(self.endpoints(start_id, f"1..{m}", label=label, rel=rel))
            high = Counter(
                self.endpoints(start_id, f"{m + 1}..{n}", label=label, rel=rel)
            )
            union = low + high
            msg = (
                f"Split-union invariant violated for start={start_id} n={n} m={m}: "
                f"[1..{n}]={dict(full)}  !=  "
                f"[1..{m}]={dict(low)} + [{m + 1}..{n}]={dict(high)}"
            )
            self.env.assertEqual(full, union, depth=1, message=msg)

    # ------------------------------------------------------------------ #
    #  Helper: assert the constant traversal invariant for range [1..n]  #
    # ------------------------------------------------------------------ #

    def assert_traversal_union_invariant(self, start_id, n, label="U", rel="F"):
        """For every split point m in [1, n-1] verify that

        Counter(endpoints([1..n])) == Counter(endpoints([1..1])) +
                                      Counter(endpoints([2..2])) +
                                      . . .                      +
                                      Counter(endpoints([n..n])) +

        """
        full = Counter(self.endpoints(start_id, f"1..{n}", label=label, rel=rel))

        union = Counter()
        for m in range(1, n + 1):
            union += Counter(
                self.endpoints(start_id, f"{m}..{m}", label=label, rel=rel)
            )

        msg = (
            f"union invariant violated for start={start_id} n={n}: "
            f"[1..{n}]={dict(full)}  !=  "
            f"[1..1] + [2..2] + ... + [{n}..{n}] = {dict(union)}"
        )
        self.env.assertEqual(full, union, depth=1, message=msg)

    # ------------------------------------------------------------------ #
    #  Tests                                                             #
    # ------------------------------------------------------------------ #

    def test01_diamond_range_superset(self):
        """Convergent-diamond: [1..2] must be a superset of [2..2].

        Graph (start = 1):
            1 -> 2
            1 -> 3
            3 -> 2   (diamond: 2 reachable at depth 1 AND depth 2)
            2 -> 4
            2 -> 5

        Expected behaviour
          *1..1 -> {2, 3}
          *2..2 -> {2, 4, 5}
          *1..2 -> superset of *2..2, i.e. {2, 3, 4, 5}
        """
        edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 2)]
        self.build_graph_from_edgelist(edges)

        start = 1
        one = set(self.endpoints(start, "1..1"))
        two = set(self.endpoints(start, "2..2"))
        rng = set(self.endpoints(start, "1..2"))

        # Every exact-depth-2 endpoint must appear in the range result.
        missing = two - rng
        self.env.assertEqual(
            missing,
            set(),
            message=(
                f"*1..2 is NOT a superset of *2..2 -- drops endpoints {sorted(missing)}"
            ),
        )

        # Every exact-depth-1 endpoint must also be in the range result.
        missing1 = one - rng
        self.env.assertEqual(
            missing1,
            set(),
            message=(
                f"*1..2 is NOT a superset of *1..1 -- drops endpoints {sorted(missing1)}"
            ),
        )

        self.assert_range_split_invariant(1, n=3)
        self.assert_traversal_union_invariant(1, n=5)

    def test01b_edge_creation_order(self):
        """Convergent-diamond: [1..2] must be a superset of [2..2].

        Graph (start = 1):
            1 -> 2
            1 -> 3
            3 -> 2   (diamond: 2 reachable at depth 1 AND depth 2)
            2 -> 4
            2 -> 5

        Expected behaviour
          *1..2 -> does not care about the order in which edges are declared
        """
        edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 2)]
        self.build_graph_from_edgelist(edges)

        start = 1
        rng0 = set(self.endpoints(start, "1..2"))

        self.conn.delete(GRAPH_ID)
        edges = [(1, 2), (1, 3), (3, 2), (2, 4), (2, 5)]
        self.build_graph_from_edgelist(edges)
        rng1 = set(self.endpoints(start, "1..2"))

        # Every exact-depth-2 endpoint must appear in the range result.
        self.env.assertEqual(
            rng0,
            rng1,
            message=(f"*1..2 changes depending on order of edge declaration"),
        )

    def test02_cycle_counting(self):
        """
        Graph (start = 1):
            1 -> 2
            2 -> 1

        Expected behaviour
          *1..1 -> {2}
          *2..2 -> {1}
          *3..3 -> {}
          *1..10 -> {1,2}
        """
        edges = [(1, 2), (2, 1)]
        self.build_graph_from_edgelist(edges)

        self.assert_range_split_invariant(1, n=3)
        self.assert_traversal_union_invariant(1, n=10)

    def test03_paths_with_cycles(self):
        """
        Graph (start = 1):
            1 -> 2
            2 -> 3
            3 -> 1
            1 -> 4

        Expected behaviour
          *1..4  -> [1,2,3,4,4] (There are two paths to 4, direct, and after
                                 navigating the cycle)
          *1..10 -> [1,2,3,4,4] (There are only two paths to 4, none more may
                                 be traversed because they would require edges
                                 to be crossed multiple times)
        """

        edges = [(1, 2), (2, 3), (3, 1), (1, 4)]
        self.build_graph_from_edgelist(edges)

        self.assert_range_split_invariant(1, n=4)
        self.assert_traversal_union_invariant(1, n=10)

    def test04_multigraph_cycle(self):
        """Triangle cycle with 3 parallel edges at each hop.

        Graph:
            0 -[x3]-> 1 -[x3]-> 2 -[x3]-> 0

        There are 9 edges total: 3 each in the ab, bc, ca classes.
        Paths from node 0 back to node 0 (edge repetition not allowed):

          *3..3   = 3^3          =  27   (one full cycle, all 3^3 combos)
          *6..6   = P(3,2)^3     = 216   (two cycles, choose 2 of 3 per class)
          *9..9   = 3! ^3        = 216   (three cycles, all edges used once each)
          *12..12 = 0            (would require repeating an edge)

        Cumulative count of paths from 0 ending at 0:
          [1..3]  =  27
          [1..6]  =  27 + 216 = 243
          [1..9]  =  27 + 216 + 216 = 459
          [1..12] =  459  (no new paths beyond depth 9)
          [1..]   =  459  (unbounded, same as above)
        """
        edges = [(0, 1)] * 3 + [(1, 2)] * 3 + [(2, 0)] * 3
        self.build_multigraph_from_edgelist(edges)

        # -- exact-depth counts (paths from 0 to 0 only) --
        for depth, expected in [(3, 27), (6, 216), (9, 216), (12, 0)]:
            got = Counter(self.endpoints(0, f"{depth}..{depth}"))[0]
            self.env.assertEqual(
                got,
                expected,
                message=f"*{depth}..{depth} paths from node 0 to node 0: expected {expected}, got {got}",
            )

        # -- cumulative range counts (paths from 0 to 0 only) --
        for rng, expected in [
            ("1..3", 27),
            ("1..6", 243),
            ("1..9", 459),
            ("1..12", 459),
        ]:
            got = Counter(self.endpoints(0, rng))[0]
            self.env.assertEqual(
                got,
                expected,
                message=f"[{rng}] paths from node 0 to node 0: expected {expected}, got {got}",
            )

        # -- unbounded should equal [1..9] since no new paths exist beyond depth 9 --
        got_unbounded = Counter(self.endpoints(0))[0]
        self.env.assertEqual(
            got_unbounded,
            459,
            message=f"[1..] paths from node 0 to node 0: expected 459, got {got_unbounded}",
        )

    def test05_chain_exact_depths(self):
        """Linear chain: *k..k returns exactly the node at hop k.

        Graph (start = 0):
            0 -> 1 -> 2 -> 3 -> 4 -> 5

        Expected exact-depth results:
          *1..1 = {1}
          *2..2 = {2}
          *3..3 = {3}
          *4..4 = {4}
          *5..5 = {5}

        Also verifies cumulative ranges and the split-union / traversal-union
        invariants, which together guarantee *n..n contributes the right slice
        to every wider range.
        """
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        self.build_graph_from_edgelist(edges)

        for k in range(1, 6):
            got = self.endpoints(0, f"{k}..{k}")
            self.env.assertEqual(
                got,
                [k],
                message=f"*{k}..{k} from node 0: expected [{k}], got {got}",
            )

        got_range = self.endpoints(0, "1..3")
        self.env.assertEqual(
            got_range,
            [1, 2, 3],
            message=f"*1..3 from node 0: expected [1,2,3], got {got_range}",
        )

        self.assert_range_split_invariant(0, n=5)
        self.assert_traversal_union_invariant(0, n=5)

    def test06_minlen_semantics(self):
        """minLen boundary: nodes shallower than minLen must not appear.

        Graph (start = 0):
            0 -> 1 -> 2 -> 3 -> 4

        *2..4 must return {2, 3, 4} and must NOT include node 1 (depth 1).
        """
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        self.build_graph_from_edgelist(edges)

        got = self.endpoints(0, "2..4")
        self.env.assertEqual(
            got,
            [2, 3, 4],
            message=f"*2..4 from node 0: expected [2,3,4], got {got}",
        )

        # Node 1 (depth 1) must be absent.
        self.env.assertEqual(
            1 in got,
            False,
            message=f"*2..4 incorrectly includes node 1 (depth 1)",
        )

        got_33 = self.endpoints(0, "3..3")
        self.env.assertEqual(
            got_33,
            [3],
            message=f"*3..3 from node 0: expected [3], got {got_33}",
        )

        self.assert_range_split_invariant(0, n=4)
        self.assert_traversal_union_invariant(0, n=4)

    def test07_edge_based_semantics(self):
        """Edge-based cycle detection: a node may be visited twice if the
        two traversals use different edges.

        Graph (start = 0):
            0 -[e1]-> 1 -[e2]-> 2 -[e3]-> 1 -[e4]-> 3

        Under node-based semantics (old behaviour) the path 0->1->2->1->3
        would be rejected because node 1 appears twice.  Under edge-based
        semantics it is valid because all four edges are distinct.

        Expected exact-depth results:
          *1..1 = [1]            (0->1 via e1)
          *2..2 = [2, 3]         (0->1->2, 0->1->3)
          *3..3 = [1]            (0->1->2->1 via e1,e2,e3 -- all distinct)
          *4..4 = [3]            (0->1->2->1->3 via e1,e2,e3,e4 -- node 1 twice, OK!)
          *5..5 = []             (node 3 is a dead-end; e2 would be repeated)
          *1..4 = Counter({1:2, 2:1, 3:2})
        """
        edges = [(0, 1), (1, 2), (2, 1), (1, 3)]
        self.build_graph_from_edgelist(edges)

        expected_exact = [
            ("1..1", [1]),
            ("2..2", [2, 3]),
            ("3..3", [1]),
            ("4..4", [3]),
            ("5..5", []),
        ]
        for rng, expected in expected_exact:
            got = self.endpoints(0, rng)
            self.env.assertEqual(
                got,
                expected,
                message=f"*{rng} from node 0: expected {expected}, got {got}",
            )

        got_14 = Counter(self.endpoints(0, "1..4"))
        self.env.assertEqual(
            got_14,
            Counter({1: 2, 2: 1, 3: 2}),
            message=f"*1..4 from node 0: expected {{1:2, 2:1, 3:2}}, got {dict(got_14)}",
        )

        self.assert_range_split_invariant(0, n=4)
        self.assert_traversal_union_invariant(0, n=5)

    def test08_selfloop_contributes_path(self):
        """Self-loop on the start node must contribute to bounded traversals.

        When the start node has a self-loop, a path that uses that self-loop
        as an intermediate hop must be counted.  FalkorDB previously dropped
        these paths, returning fewer results than expected.

        Minimised repro (nodes identified by the 'id' property):
            0 -[e1]-> 1          (direct edge to destination)
            0 -[e2]-> 0          (self-loop on start node)

        Valid paths from 0 to 1 with *1..2:
          1. 0 -[e1]-> 1                    (length 1)
          2. 0 -[e2]-> 0 -[e1]-> 1          (length 2, self-loop first)
        Expected count: 2

        Control checks (match the bug report):
          *1..2 from 0 to 0: path  0 -[e2]-> 0  gives count 1
          *1..2 from 0 to 0 (no self-loop): count 0
          *1..2 from 0 to 1 (no self-loop): count 1  (only the direct edge)
        """
        # -- minimised repro: self-loop + outgoing edge --
        edges = [(0, 0), (0, 1)]
        self.build_graph_from_edgelist(edges)

        # Direct count via MATCH path = ... RETURN count(path)
        q = (
            "MATCH path = (a:U {id:0})-[:F*1..2]->(b:U {id:1}) "
            "RETURN count(path) AS pathCount"
        )
        count = self.graph.query(q).result_set[0][0]
        self.env.assertEqual(
            count, 2,
            message=f"*1..2 paths from 0 to 1 with self-loop: expected 2, got {count}",
        )

        # The self-loop itself must appear once in *1..2 from 0 to 0.
        q_loop = (
            "MATCH path = (a:U {id:0})-[:F*1..2]->(b:U {id:0}) "
            "RETURN count(path) AS pathCount"
        )
        count_loop = self.graph.query(q_loop).result_set[0][0]
        self.env.assertEqual(
            count_loop, 1,
            message=f"*1..2 paths from 0 to 0 (self-loop): expected 1, got {count_loop}",
        )

        # -- control: no self-loop, only direct edge --
        self.conn.delete(GRAPH_ID)
        self.build_graph_from_edgelist([(0, 1)])

        count_direct = self.graph.query(q).result_set[0][0]
        self.env.assertEqual(
            count_direct, 1,
            message=f"*1..2 paths from 0 to 1 (no self-loop): expected 1, got {count_direct}",
        )

        # -- extended repro: Diana/Bob scenario from the bug report --
        self.conn.delete(GRAPH_ID)
        # nodes: 0=Diana, 1=Bob; self-loop on Diana; Diana->Bob
        edges_diana = [(0, 1), (0, 0)]
        self.build_graph_from_edgelist(edges_diana, label="Person", rel="KNOWS")

        q_diana = (
            "MATCH path = (d:Person {id:0})-[:KNOWS*1..2]->(b:Person {id:1}) "
            "RETURN count(path) AS pathCount"
        )
        count_diana = self.graph.query(q_diana).result_set[0][0]
        self.env.assertEqual(
            count_diana, 2,
            message=f"Diana->Bob *1..2 with self-loop: expected 2, got {count_diana}",
        )

        # The split-union and traversal-union invariants must hold.
        # (Use the original simple graph: 0->0 self-loop + 0->1)
        self.conn.delete(GRAPH_ID)
        self.build_graph_from_edgelist(edges)
        self.assert_range_split_invariant(0, n=3)
        self.assert_traversal_union_invariant(0, n=4)

