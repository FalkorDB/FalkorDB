from common import *
from index_utils import *
import re

GRAPH_ID = "TraversalConstruction"

class testTraversalConstruction():
    def __init__(self):
        self.env, self.db = Env()
        redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        # Create the graph
        self.graph.query("RETURN 1")

    # Test differing starting points for the same search pattern
    def test_starting_point(self):
        # Neither the source nor the destination are labeled
        # perform an AllNodeScan from the source node.
        query = """MATCH (a)-[]->(b) RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertContains("All Node Scan | (a)", plan)

        # Destination is labeled, perform a LabelScan from the destination node.
        query = """MATCH (a)-[]->(b:B) RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertContains("Node By Label Scan | (b:B)", plan)

        # Destination is filtered, perform an AllNodeScan from the destination node.
        query = """MATCH (a)-[]->(b) WHERE b.v = 2 RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertContains("All Node Scan | (b)", plan)

        # Destination is labeled but source is filtered, perform an AllNodeScan from the source node.
        query = """MATCH (a)-[]->(b:B) WHERE a.v = 1 OR a.v = 3 RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertContains("All Node Scan | (a)", plan)

        # Both are labeled and source is filtered, perform a LabelScan from the source node.
        query = """MATCH (a:A)-[]->(b:B) WHERE a.v = 3 RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertContains("Node By Label Scan | (a:A)", plan)

        # Both are labeled and dest is filtered, perform a LabelScan from the dest node.
        query = """MATCH (a:A)-[]->(b:B) WHERE b.v = 2 RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertContains("Node By Label Scan | (b:B)", plan)

    # make sure traversal begins with labeled entity
    def test_start_with_label(self):
        queries = ["MATCH (A:L)-->(B)-->(C) RETURN 1",
                   # "MATCH (A)-->(B:L)-->(C) RETURN 1", # improve on this case
                   "MATCH (A)-->(B)-->(C:L) RETURN 1"]

        for q in queries:
            plan = str(self.graph.explain(q))
            ops = plan.split(os.linesep)
            ops.reverse()
            self.env.assertTrue("Node By Label Scan" in ops[0])

    # make sure traversal begins with filtered entity
    def test_start_with_filter(self):
        # MATCH (A)-->(B)-->(C) WHERE A.val = 1 RETURN *
        # MATCH (A)-->(B)-->(C) WHERE B.val = 1 RETURN *
        # MATCH (A)-->(B)-->(C) WHERE C.val = 1 RETURN *
        entities = ['A', 'B', 'C']
        for e in entities:
            q = """MATCH (A)-->(B)-->(C) WHERE {}.val = 1 RETURN *""".format(e)
            plan = str(self.graph.explain(q))
            ops = plan.split(os.linesep)
            ops.reverse()

            self.env.assertTrue("All Node Scan | ({})".format(e) in ops[0])
            self.env.assertTrue("Filter" in ops[1])

    # make sure traversal begins with bound entity
    def test_start_with_bound(self):
        # MATCH (X) WITH X as A MATCH (A)-->(B)-->(C) RETURN *
        # MATCH (X) WITH X as B MATCH (A)-->(B)-->(C) RETURN *
        # MATCH (X) WITH X as C MATCH (A)-->(B)-->(C) RETURN *
        entities = ['A', 'B', 'C']
        for e in entities:
            q = "MATCH (X) WITH X as {} MATCH (A)-->(B)-->(C) RETURN *".format(e)
            plan = str(self.graph.explain(q))
            ops = plan.split(os.linesep)
            ops.reverse()
            self.env.assertTrue("Conditional Traverse | ({}".format(e) in ops[2])

    # make sure traversal begins with bound entity and follows with filter
    def test_start_with_bound_follows_with_filter(self):
        queries = ["MATCH (X) WITH X AS B MATCH (A {v:1})-->(B)-->(C) RETURN *",
                "MATCH (X) WITH X AS B MATCH (A)-->(B)-->(C {v:1}) RETURN *"]
        for q in queries:
            plan = str(self.graph.explain(q))
            ops = plan.split(os.linesep)
            ops.reverse()
            self.env.assertTrue("Filter" in ops[3])

    def test_filter_as_early_as_possible(self):
        # The branch point B is two hops from either filtered endpoint, so the
        # other filtered node is reached by two traverses, not one. This used to
        # assert it at ops[2] — a single hop away — which only held because the
        # chain-reversal emitted the far hop at the bottom of the plan, where
        # neither of its endpoints was bound and the runtime answered it by
        # enumerating every edge of the type. The property this test is named
        # for is unchanged: each filtered node is filtered immediately after the
        # operator that binds it.
        q = """MATCH (A:L {v: 1})-->(B)-->(C), (B)-->(D:L {v: 1}) RETURN 1"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        ops.reverse()
        self.env.assertTrue("Node By Label Scan" in ops[0]) # scan either A or D
        self.env.assertTrue("Filter" in ops[1]) # filter it immediately
        self.env.assertTrue("Conditional Traverse" in ops[2]) # traverse to the branch point
        self.env.assertTrue("Conditional Traverse" in ops[3]) # traverse to the other filtered node
        self.env.assertTrue("Filter" in ops[4]) # filter it as soon as it is bound
        self.env.assertTrue("Conditional Traverse" in ops[5]) # continue traversing

    # Endpoint scoring reads label cardinality and index presence, so the
    # reversal cases need a graph that holds both — an indexed anchor is the
    # shape the benchmark measures. A separate graph keeps that data from moving
    # the plans the tests above assert on an empty one.
    def _reversal_graph(self):
        g = self.db.select_graph(GRAPH_ID + "Reversal")
        # MERGE first: it creates the key, and listing indices on a key that
        # does not exist yet is an error rather than an empty answer.
        g.query("MERGE (a:M {v: 0})-[:E]->(b:M {v: 2})-[:E]->(c:M {v: 1})")
        if not list_indicies(g).result_set:
            create_node_range_index(g, "M", "v", sync=True)
        return g

    def _assert_traverses_start_bound(self, graph, q):
        # Every traverse prints its source alias first, whichever way its arrow
        # points, so that alias must already be bound by an operator below it. A
        # traverse whose source nothing binds is not wrong — the runtime answers
        # it by enumerating every edge of the type, once per input row — but on
        # the 10k-node benchmark graph that read as 34.4M instructions for
        # `MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c {id: 1})` against 419k for the
        # same query on the C engine.
        plan = str(graph.explain(q))
        ops = plan.split(os.linesep)
        ops.reverse()
        bound = set()
        for op in ops:
            body = op.split("|", 1)[1] if "|" in op else ""
            # one alias per parenthesised endpoint, label suffix dropped; a named
            # edge prints in brackets and is therefore skipped
            aliases = [a.split(":")[0].strip() for a in re.findall(r"\(([^)]*)\)", body)]
            if "Conditional Traverse" in op or "Expand Into" in op:
                if aliases and aliases[0] not in bound:
                    print(f"traverse from unbound {aliases[0]!r}:\n{plan}", flush=True)
                    self.env.assertTrue(False)
                bound.update(aliases)
            else:
                # scans, Argument, Project, Unwind … bind what they name
                bound.update(aliases)

    def test_reversed_chain_hop_order(self):
        # Reversing a chain to start from the better endpoint has to reorder the
        # hops as well, or the bottom of the plan ends up holding the far hop —
        # the one whose endpoints nothing below it binds.
        g = self._reversal_graph()
        for q in ["MATCH (a)-[:E]->(b)-[:E]->(c:M {v: 1}) RETURN count(a)",
                  "MATCH (a)-[:E]->(b)-[:E]->(c:M {v: 1}) WHERE b.v > 0 RETURN count(a)",
                  "MATCH (a)-[:E]->(b)-[:E]->(d)-[:E]->(c:M {v: 1}) RETURN count(a)",
                  "MATCH (c:M {v: 1})-[:E]->(b)-[:E]->(a) RETURN count(a)",
                  # branching: only the path to the branch point is reversed, so
                  # the hop leaving it stays in storage direction
                  "MATCH (A:M {v: 1})-->(B)-->(C), (B)-->(D:M {v: 1}) RETURN 1",
                  "MATCH (A:M)-->(B)-->(C), (B)-->(D:M {v: 1}) RETURN 1"]:
            self._assert_traverses_start_bound(g, q)

    def test_selective_hop_runs_first(self):
        # Among the hops that *can* run next, the one that constrains the most
        # goes first, so a selective side prunes before an unselective side fans
        # out. Pattern order decides ties only. This is what the C engine gets
        # from the label-cardinality tiebreak in its scored search; taking the
        # pattern's order instead measured 1,712,085 instructions against
        # 780,735 on a 200-node branch whose two sides reach 50 nodes and 1.
        g = self.db.select_graph(GRAPH_ID + "Selective")
        g.query("UNWIND range(0, 19) AS i CREATE (:S {v: i})")
        g.query("UNWIND range(0, 999) AS i CREATE (:Wide {v: i})")
        g.query("MATCH (s:S), (w:Wide) WHERE w.v / 50 = s.v CREATE (s)-[:E]->(w)")
        g.query("CREATE (:Narrow {v: 0})")
        g.query("MATCH (s:S {v: 0}), (n:Narrow) CREATE (s)-[:E]->(n)")

        # the pattern mentions the wide side first; the plan must not
        q = "MATCH (s:S {v: 0})-[:E]->(w:Wide), (s)-[:E]->(n:Narrow) RETURN count(*)"
        ops = str(g.explain(q)).split(os.linesep)
        ops.reverse()
        narrow_at = next(i for i, o in enumerate(ops) if "(:Narrow)" in o or "(n:Narrow)" in o)
        wide_at = next(i for i, o in enumerate(ops) if "(w:Wide)" in o)
        self.env.assertTrue(narrow_at < wide_at)

        # and every traverse still starts from something bound
        self._assert_traverses_start_bound(g, q)

    def test_reversed_chain_mid_filter_placement(self):
        # A filter between the hops of a reversed chain belongs above the hop
        # that binds the alias it reads, which after reordering is a different
        # hop than before.
        q = """MATCH (a)-[:E]->(b)-[:E]->(c:M {v: 1}) WHERE b.v > 0 RETURN count(a)"""
        ops = str(self._reversal_graph().explain(q)).split(os.linesep)
        ops.reverse()
        self.env.assertTrue("Node By Index Scan | (c:M)" in ops[0])
        self.env.assertTrue("Conditional Traverse | (c)<-(b)" in ops[1]) # binds b
        self.env.assertTrue("Filter" in ops[2])                          # then filters b
        self.env.assertTrue("Conditional Traverse | (b)<-(a)" in ops[3])

    def test_reversed_chain_results_match(self):
        # The broken order produced correct rows by brute force, so the guard
        # against regressing it has to check rows as well as plan shape.
        g = self._reversal_graph()
        res = g.query("""MATCH (a)-[:E]->(b)-[:E]->(c:M {v: 1}) RETURN a.v, b.v""")
        self.env.assertEqual(res.result_set, [[0, 2]])
        res = g.query(
            """MATCH (a)-[:E]->(b)-[:E]->(c:M {v: 1}) WHERE b.v > 0 RETURN a.v, b.v"""
        )
        self.env.assertEqual(res.result_set, [[0, 2]])
        res = g.query(
            """MATCH (a)-[:E]->(b)-[:E]->(c:M {v: 1}) WHERE b.v > 5 RETURN a.v, b.v"""
        )
        self.env.assertEqual(res.result_set, [])

    def test_long_pattern(self):
        q = """match (a)--(b)--(c)--(d)--(e)--(f)--(g)--(h)--(i)--(j)--(k)--(l) return *"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        # 13, not upstream's 14: the plan is C-equivalent (Project + 11
        # Conditional Traverses + a scan) but this engine has no root
        # `Results` operator, which is C's 14th line.
        self.env.assertEqual(len(ops), 13)

    def test_start_with_index_filter(self):
        # TODO: enable this test, once we'll score higher filters that
        # have the potential turn into index scan
        return

        create_node_range_index(graph, 'L', 'v', sync=True)

        q = """MATCH (a:L {v:1})-[]-(b:L {x:1}) RETURN a, b"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        ops.reverse()
        self.env.assertTrue("Index Scan" in ops[0]) # start with index scan

        q = """MATCH (a:L {x:1})-[]-(b:L {v:1}) RETURN a, b"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        ops.reverse()
        self.env.assertTrue("Index Scan" in ops[0]) # start with index scan

    def test_variable_length_traversal_placement(self):
        # cyclic traversal followed by variable-length traversal
        q = """MATCH (b)<-[*]-(a:L {v: 5})<--(a) WHERE b.v = 10 RETURN a"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        ops.reverse()
        self.env.assertTrue("Node By Label Scan | (a:L)" in ops[0]) # scan A
        self.env.assertTrue("Filter" in ops[1]) # filter A
        self.env.assertTrue("Expand Into" in ops[2]) # traverse from A to itself
        self.env.assertTrue("Conditional Variable Length Traverse" in ops[3]) # var-len traverse from A to B

        # bidirectional variable-length traversal
        q = """MATCH (a:L {v: 5})-[*]-(b:L) WHERE a <> b RETURN b"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        ops.reverse()
        self.env.assertTrue("Node By Label Scan | (a:L)" in ops[0]) # scan A
        self.env.assertTrue("Filter" in ops[1]) # filter A
        self.env.assertTrue("Conditional Variable Length Traverse" in ops[2]) # bidirectional var-len traverse from A to B

    def test_traverse_zero_length_edge(self):
        # populate graph
        self.graph.query("CREATE (:A{v:1})-[:R{x:1}]->(:B{v:2})-[:R{x:2}]->(:C{v:3})")

        # traverse from 'a' to itself
        q1 = """MATCH (a)-[*0]->(b) RETURN a, b"""
        q2 = """MATCH (a:A)-[*0]->(b) RETURN a, b"""
        q3 = """MATCH (a)-[*0]->(b:A) RETURN a, b"""

        queries = [q1, q2, q3]

        for q in queries:
            plan = self.graph.explain(q)
            root = plan.structured_plan
            self.env.assertTrue(root.name == "Project")

            child = root.children[0]
            self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

            child = child.children[0]
            self.env.assertTrue("All Node Scan" in child.name or
                                "Node By Label Scan" in child.name)

            # validate that 'a' == 'b'
            result = self.graph.query(q).result_set
            for row in result:
                self.env.assertTrue(row[0] == row[1])

        #-----------------------------------------------------------------------

        # traverse from 'a' back to itself via a 0 length edge
        q1 = """MATCH (a)-[*0]->(a) RETURN a"""
        q2 = """MATCH (a:A)-[*0]->(a) RETURN a"""
        q3 = """MATCH (a)-[*0]->(a:A) RETURN a"""
        q4 = """MATCH (a:A)-[*0]->(a:A) RETURN a"""

        queries = [q1, q2, q3, q4]

        for q in queries:
            plan = self.graph.explain(q)
            root = plan.structured_plan
            self.env.assertTrue(root.name == "Project")

            child = root.children[0]
            self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

            child = child.children[0]
            self.env.assertTrue("All Node Scan" in child.name or
                                "Node By Label Scan" in child.name)

            # validate 'a' was found
            result = self.graph.query(q).result_set
            self.env.assertTrue(len(result) > 0)

        #-----------------------------------------------------------------------

        # traverse from 'a' to 'b' using 0 length edge
        q = """MATCH (a:A)-[*0]->(b:B) RETURN a, b"""
        plan = self.graph.explain(q)

        root = plan.structured_plan
        self.env.assertTrue(root.name == "Project")

        child = root.children[0]
        self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

        child = child.children[0]
        self.env.assertTrue("Node By Label Scan" in child.name)

        # make sure 'b' isn't reachable
        result = self.graph.query(q).result_set
        self.env.assertTrue(len(result) == 0)

        #-----------------------------------------------------------------------

        # create a multi label node
        q = "CREATE (:X:Y)"
        result = self.graph.query(q)
        self.env.assertEqual(result.nodes_created, 1)

        # traverse from a multi label node 'a' to itself using a 0 length edge
        q1 = """MATCH (a:X)-[*0]->(b:Y) RETURN a, b"""
        q2 = """MATCH (a:Y)-[*0]->(b:X) RETURN a, b"""
        q3 = """MATCH (a:X:Y)-[*0]->(b:X) RETURN a, b"""
        q4 = """MATCH (a:X:Y)-[*0]->(b:Y) RETURN a, b"""
        q5 = """MATCH (a:X:Y)-[*0]->(b:Y:X) RETURN a, b"""
        queries = [q1, q2, q3, q4, q5]

        for q in queries:
            plan = self.graph.explain(q)

            root = plan.structured_plan
            self.env.assertTrue(root.name == "Project")

            child = root.children[0]
            self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

            child = child.children[0]
            self.env.assertTrue("Node By Label Scan" in child.name)

            # make sure 'a' == 'b'
            result = self.graph.query(q).result_set
            self.env.assertTrue(len(result) == 1)
            self.env.assertTrue(result[0][0] == result[0][1])

        #-----------------------------------------------------------------------

        # traverse from 'a' to itself
        q1 = """MATCH (a)-[*0]->(b{v:1}) RETURN a, b"""
        q2 = """MATCH (a{v:1})-[*0]->(b) RETURN a, b"""

        queries = [q1, q2]

        for q in queries:
            plan = self.graph.explain(q)

            root = plan.structured_plan
            self.env.assertTrue(root.name == "Project")

            child = root.children[0]
            self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

            child = child.children[0]
            self.env.assertTrue(child.name == "Filter")

            child = child.children[0]
            self.env.assertTrue("All Node Scan" in child.name)

            # validate that 'a' == 'b'
            result = self.graph.query(q).result_set
            self.env.assertTrue(len(result) == 1)
            for row in result:
                self.env.assertTrue(row[0] == row[1])

        #-----------------------------------------------------------------------

        # traverse from 'a' to itself
        q = """MATCH (a{v:1})-[*0]->(b{v:1}) RETURN a, b"""
        plan = self.graph.explain(q)

        root = plan.structured_plan
        self.env.assertTrue(root.name == "Project")

        child = root.children[0]
        self.env.assertTrue(child.name == "Filter")

        child = child.children[0]
        self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

        child = child.children[0]
        self.env.assertTrue(child.name == "Filter")

        child = child.children[0]
        self.env.assertTrue("All Node Scan" in child.name)

        # validate that 'a' == 'b'
        result = self.graph.query(q).result_set
        self.env.assertTrue(len(result) == 1)
        for row in result:
            self.env.assertTrue(row[0] == row[1])

        #-----------------------------------------------------------------------

        # traverse from 'a' to a none existing node
        q = """MATCH (a{v:1})-[*0]->(b{v:2}) RETURN a, b"""
        plan = self.graph.explain(q)

        root = plan.structured_plan
        self.env.assertTrue(root.name == "Project")

        child = root.children[0]
        self.env.assertTrue(child.name == "Filter")

        child = child.children[0]
        self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

        child = child.children[0]
        self.env.assertTrue(child.name == "Filter")

        child = child.children[0]
        self.env.assertTrue("All Node Scan" in child.name)

        # validate 'b' wasn't reached
        result = self.graph.query(q).result_set
        self.env.assertTrue(len(result) == 0)

        #-----------------------------------------------------------------------

        # build expected ordered result-set
        q = """MATCH (a) RETURN a ORDER BY a"""
        expected = self.graph.query(q).result_set

        # return named path with a 0 length edge
        q = """MATCH p = ()-[*0]->() RETURN p ORDER BY p"""
        result = self.graph.query(q).result_set

        # validate result sets
        self.env.assertTrue(len(result) == len(expected))
        for i in range(0, len(result)):
            a = expected[i][0]
            path = result[i][0]
            b = path.nodes()[0]
            self.env.assertTrue(a == b)

        #-----------------------------------------------------------------------

        q = """MATCH (a) RETURN a ORDER BY a"""
        expected = self.graph.query(q).result_set

        # get 0 length edge
        # returning a variable length edge returns a path
        # similar to MATCH ()-[e*0..2]->() return e
        q = """MATCH (a)-[e*0]->(b) RETURN e ORDER BY e"""
        result = self.graph.query(q).result_set
        for i in range(0, len(result)):
            a = expected[i][0]
            p = result[i][0]
            b = p.nodes()[0]
            self.env.assertTrue(len(p.edges()) == 0)
            self.env.assertTrue(len(p.nodes()) == 1)
            self.env.assertTrue(a == b)

        # graph: (:A{v:1})-[:R{x:1}]->(:B{v:2})-[:R{x:2}]->(:C{v:3})
        q = """MATCH (c:C) RETURN c"""
        expected = self.graph.query(q).result_set

        # make sure 'c' is reachable
        q = """MATCH (a)-[*0]->(b)-[]->(c:C) RETURN c"""
        result = self.graph.query(q).result_set
        self.env.assertTrue(result == expected)


    # A variable-length traverse whose bound endpoint is supplied by an operator
    # other than a directly selectable scan — a WITH barrier or an UNWIND —
    # must still anchor on that bound endpoint. Before this was handled, the
    # planner inserted a scan on the *free* endpoint and the runtime seeded its
    # DFS with every node carrying that endpoint's label, once per input row.
    def test_var_len_anchor_with_non_leaf_child(self):
        g = self.db.select_graph("VarLenAnchorNonLeaf")
        g.query("""CREATE (a:P {id:0})-[:E]->(b:P {id:1})-[:E]->(c:P {id:2})
                          -[:E]->(d:P {id:3})-[:E]->(e:P {id:4})""")

        # Baseline: the leaf form already anchors on the bound `u`.
        leaf = """MATCH path=(p:P)-[:E*0..]->(u:P) WHERE u.id=3 RETURN count(path)"""
        leaf_plan = str(g.explain(leaf))
        self.env.assertNotContains("Node By Label Scan | (p:P)", leaf_plan)
        expected = g.query(leaf).result_set

        # A WITH/LIMIT barrier between the scan and the traverse.
        barrier = """MATCH (u:P) WHERE u.id=3 WITH u LIMIT 1
                     MATCH path=(p:P)-[:E*0..]->(u) RETURN count(path)"""
        barrier_plan = str(g.explain(barrier))
        self.env.assertNotContains("Node By Label Scan | (p:P)", barrier_plan)
        self.env.assertContains("Conditional Variable Length Traverse", barrier_plan)
        self.env.assertEqual(g.query(barrier).result_set, expected)

        # `u` arriving through an UNWIND chain.
        unwound = """UNWIND [3] AS wanted
                     MATCH (u:P) WHERE u.id = wanted
                     MATCH path=(p:P)-[:E*0..]->(u) RETURN count(path)"""
        unwound_plan = str(g.explain(unwound))
        self.env.assertNotContains("Node By Label Scan | (p:P)", unwound_plan)
        self.env.assertEqual(g.query(unwound).result_set, expected)

        # The free endpoint's label is still enforced — by the traverse itself,
        # as its destination filter — so an unlabeled `p` must match strictly
        # more rows once a non-P node can reach the same `u`. Attach to the
        # existing id=3 node so `WITH u LIMIT 1` stays deterministic.
        g.query("MATCH (d:P {id:3}) CREATE (:Q {id:99})-[:E]->(d)")
        labeled = g.query("""MATCH (u:P) WHERE u.id=3 WITH u LIMIT 1
                             MATCH path=(p:P)-[:E*0..]->(u) RETURN count(path)""").result_set
        unlabeled = g.query("""MATCH (u:P) WHERE u.id=3 WITH u LIMIT 1
                               MATCH path=(p)-[:E*0..]->(u) RETURN count(path)""").result_set
        self.env.assertGreater(unlabeled[0][0], labeled[0][0])

        # An inline attribute on the free endpoint puts a Filter between the
        # traverse and the planner's scan. The scan must still go — and the
        # Filter must not go with it, or `p` loses its predicate.
        attrs = """MATCH (u:P) WHERE u.id=3 WITH u LIMIT 1
                    MATCH path=(p:P {id:1})-[:E*0..]->(u) RETURN count(path)"""
        attrs_plan = str(g.explain(attrs))
        self.env.assertNotContains("Node By Label Scan | (p:P)", attrs_plan)
        self.env.assertContains("Filter", attrs_plan)
        self.env.assertEqual(g.query(attrs).result_set, [[1]])
        g.delete()

    # `Argument(None)` is opaque about which variables it carries, so a scan
    # with one anywhere below it cannot be proven redundant and the plan must
    # be left alone — even though the outer `u` is in fact bound.
    def test_var_len_anchor_opaque_argument(self):
        g = self.db.select_graph("VarLenAnchorOpaqueArgument")
        g.query("CREATE (a:P {id:0})-[:E]->(b:P {id:1})-[:E]->(c:P {id:2})")

        for q in ["MATCH (u:P) WHERE u.id=2 OPTIONAL MATCH path=(p:P)-[:E*0..]->(u) RETURN count(path)",
                  """MATCH (u:P) WHERE u.id=2
                     CALL { WITH u MATCH path=(p:P)-[:E*0..]->(u) RETURN count(path) AS c }
                     RETURN c"""]:
            plan = str(g.explain(q))
            self.env.assertContains("Argument", plan)
            self.env.assertContains("Node By Label Scan | (p:P)", plan)
            self.env.assertEqual(g.query(q).result_set, [[3]])
        g.delete()
