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

    # A variable-length hop anchored on an indexed endpoint must start the plan
    # even when the same MATCH has a second, fixed-length pattern on the other
    # endpoint (#2558). Otherwise the planner starts from the unrelated label
    # scan and expands the var-length hop from every row it finds.
    def test_var_len_anchor_with_second_pattern(self):
        g = self.db.select_graph("VarLenAnchorSecondPattern")
        g.query("CREATE INDEX FOR (p:Person) ON (p.id)")
        wait_for_indices_to_sync(g)
        g.query("UNWIND range(0, 199) AS i CREATE (:Person {id: i})")
        g.query("UNWIND range(0, 4) AS i CREATE (:City {id: i})")
        g.query("""UNWIND range(0, 199) AS i
                   MATCH (p:Person {id: i}), (c:City {id: i % 5}) CREATE (p)-[:LIVES]->(c)""")
        g.query("""UNWIND range(0, 199) AS i UNWIND [1, 7] AS d
                   MATCH (a:Person {id: i}), (b:Person {id: (i * d + d) % 200})
                   CREATE (a)-[:KNOWS]->(b)""")

        single = "MATCH (person:Person {id: 5})-[:KNOWS*2..2]-(friend) RETURN count(*)"
        for q in ["""MATCH (person:Person {id: 5})-[:KNOWS*2..2]-(friend),
                           (friend)-[:LIVES]->(city:City) RETURN count(*)""",
                  """MATCH (friend)-[:LIVES]->(city:City),
                           (person:Person {id: 5})-[:KNOWS*2..2]-(friend) RETURN count(*)""",
                  """MATCH (person:Person)-[:KNOWS*2..2]-(friend), (friend)-[:LIVES]->(city:City)
                     WHERE person.id = 5 RETURN count(*)"""]:
            plan = str(g.explain(q))
            # the index scan on `person` is the only scan: it leads the plan
            scans = [l.strip() for l in plan.splitlines() if "Scan" in l]
            self.env.assertEqual(scans, ["Node By Index Scan | (person:Person)"], message=plan)
            # every friend lives in exactly one city, so the counts agree
            self.env.assertEqual(g.query(q).result_set, g.query(single).result_set, message=q)

        # A fixed-length hop that is itself anchored keeps leading.
        anchored = """MATCH (friend:Person {id: 9})-[:LIVES]->(city:City),
                            (person:Person)-[:KNOWS*2..2]-(friend) RETURN count(*)"""
        self.env.assertContains("Node By Index Scan | (friend:Person)", str(g.explain(anchored)))
        g.delete()

    # The walk that leads a component must be one that can start the plan, and
    # must not be moved before the hops that bind variables it depends on.
    def test_var_len_anchor_not_promoted_when_unsafe(self):
        g = self.db.select_graph("VarLenAnchorUnsafe")
        g.query("CREATE INDEX FOR (p:Person) ON (p.id)")
        wait_for_indices_to_sync(g)
        g.query("UNWIND range(0, 199) AS i CREATE (:Person {id: i})")
        g.query("UNWIND range(0, 4) AS i CREATE (:City {id: i})")
        g.query("""UNWIND range(0, 199) AS i
                   MATCH (p:Person {id: i}), (c:City {id: i % 5}) CREATE (p)-[:LIVES]->(c)""")
        g.query("""UNWIND range(0, 199) AS i UNWIND [1, 7] AS d
                   MATCH (a:Person {id: i}), (b:Person {id: (i * d + d) % 200})
                   CREATE (a)-[:KNOWS]->(b)""")

        # A walk anchored on its `to` endpoint leads too, bidirectional or
        # directed: the runtime runs it backwards from the bound end, so the
        # index scan on `person` starts the plan and no City scan precedes it.
        single = "MATCH (person:Person {id: 5})-[:KNOWS*2..2]-(friend) RETURN count(*)"
        for q in ["""MATCH (friend)-[:KNOWS*2..2]-(person:Person {id: 5}),
                           (friend)-[:LIVES]->(city:City) RETURN count(*)""",
                  """MATCH (friend)-[:KNOWS*2..2]-(person:Person), (friend)-[:LIVES]->(city:City)
                     WHERE person.id = 5 RETURN count(*)"""]:
            plan = str(g.explain(q))
            scans = [l.strip() for l in plan.splitlines() if "Scan" in l]
            self.env.assertEqual(scans, ["Node By Index Scan | (person:Person)"], message=plan)
            self.env.assertEqual(g.query(q).result_set, g.query(single).result_set, message=q)
        q = """MATCH (friend)-[:LIVES]->(city:City),
                     (friend)-[:KNOWS*2..2]->(person:Person {id: 5}) RETURN count(*)"""
        single = "MATCH (friend)-[:KNOWS*2..2]->(person:Person {id: 5}) RETURN count(*)"
        plan = str(g.explain(q))
        scans = [l.strip() for l in plan.splitlines() if "Scan" in l]
        self.env.assertEqual(scans, ["Node By Index Scan | (person:Person)"], message=plan)
        self.env.assertEqual(g.query(q).result_set, g.query(single).result_set)
        g.delete()

        # A variable an earlier clause bound (UNWIND's x) is not lost when the
        # walk is reversed to start from its anchored endpoint.
        g.query("""CREATE (b:B)-[:S]->(:C), (b)-[:R]->(:A {id: 1})""")
        q = """UNWIND [1, 2] AS x MATCH (b)-[:S]->(c:C), (b)-[:R*1..2]->(a:A {id: 1})
               RETURN count(*)"""
        self.env.assertEqual(g.query(q).result_set, [[2]], message=q)
        g.delete()

        # A WHERE conjunct that also reads the walk's path cannot run at the
        # starting endpoint, so it does not anchor the walk.
        g = self.db.select_graph("VarLenAnchorPathWhere")
        g.query("CREATE INDEX FOR (p:Person) ON (p.id)")
        wait_for_indices_to_sync(g)
        g.query("UNWIND range(0, 199) AS i CREATE (:Person {id: i})")
        g.query("UNWIND range(0, 4) AS i CREATE (:City {id: i})")
        g.query("""UNWIND range(0, 3) AS i
                   MATCH (p:Person {id: i}), (c:City {id: i}) CREATE (p)-[:LIVES]->(c)""")
        g.query("""UNWIND range(0, 199) AS i UNWIND [1, 7] AS d
                   MATCH (a:Person {id: i}), (b:Person {id: (i * d + d) % 200})
                   CREATE (a)-[:KNOWS]->(b)""")
        q = """MATCH (p:Person)-[r:KNOWS*2..2]->(f), (f)-[:LIVES]->(c:City)
               WHERE p.id = length(r) RETURN count(*)"""
        self.env.assertContains("Node By Label Scan | (c:City)", str(g.explain(q)))
        g.delete()

        # Attribute expressions that read a variable keep their binding hop first.
        g = self.db.select_graph("VarLenAnchorDeps")
        g.query("""CREATE (a:A {id: 1})-[:R {weight: 5}]->(b:B)-[:S]->(c:C {limit: 5, aid: 1})""")
        for q in ["MATCH (b)-[:S]->(c), (a:A {id: 1})-[:R*1..2 {weight: c.limit}]->(b) RETURN count(*)",
                  "MATCH (b)-[:S]->(c), (a:A {id: c.aid})-[:R*1..2]->(b) RETURN count(*)"]:
            self.env.assertEqual(g.query(q).result_set, [[1]], message=q)
        g.delete()

        # A component with a second walk promotes none (the walks would run
        # before the hop that binds one of their endpoints) and keeps its result.
        g = self.db.select_graph("VarLenAnchorChain")
        g.query("""CREATE (a:A {id: 1})-[:R]->(b:B)-[:S]->(c:C)-[:T]->(d:D {id: 2})""")
        g.query("UNWIND range(0, 99) AS i CREATE (:C)-[:T]->(:D {id: 3})")
        q = "MATCH (a:A {id: 1})-[:R*1..2]->(b)-[:S]->(c)-[:T*1..2]->(d:D {id: 2}) RETURN count(*)"
        self.env.assertEqual(g.query(q).result_set, [[1]])
        g.delete()

        # Predicates the starting endpoint cannot evaluate alone do not anchor
        # the walk: one over two variables, one holding a pattern.
        g = self.db.select_graph("VarLenAnchorMixed")
        g.query("CREATE INDEX FOR (p:Person) ON (p.id)")
        wait_for_indices_to_sync(g)
        g.query("UNWIND range(0, 199) AS i CREATE (:Person {id: i})")
        g.query("UNWIND range(0, 199) AS i CREATE (:A {x: i})-[:R]->(:B {id: i})-[:T]->(:D {id: 2})")
        g.query("UNWIND range(0, 4) AS i CREATE (:City {id: i})")
        g.query("MATCH (a:A {x: 0}), (c:City {id: 0}) CREATE (a)-[:S]->(c)")
        for q in [
                ("""MATCH (a:A)-[:R*1..2]->(b:B {id: 2}), (a)-[:S]->(c:City)
                    WHERE a.x = c.id RETURN count(*)"""),
                ("""MATCH (p:Person)-[:KNOWS*2..2]->(f), (f)-[:LIVES]->(c:City)
                    WHERE p.id = size([(c)-[:X]->() | 1]) RETURN count(*)""")]:
            self.env.assertContains("Node By Label Scan | (c:City)", str(g.explain(q)), message=q)
        # adjacent walks: the undirected one stays bound by the hop before it
        q = "MATCH (a:A)-[:R*1..1]-(b:B)-[:T*1..1]->(d:D {id: 2}) RETURN count(*)"
        self.env.assertContains("Node By Label Scan | (d:D)", str(g.explain(q)))
        self.env.assertEqual(g.query(q).result_set, [[200]])
        g.delete()

    # An undirected walk reverses like a directed one: with only its `to`
    # endpoint bound it starts there. Emitted endpoints and the path keep the
    # pattern's from -> to orientation.
    def test_undirected_var_len_runs_from_bound_to(self):
        g = self.db.select_graph("UndirectedVarLenReversed")
        g.query("CREATE (:A {id: 1})-[:R {w: 1}]->(:M {id: 2})<-[:R {w: 2}]-(:B {id: 3})")
        g.query("CREATE (:A {id: 10})-[:R]->(:A {id: 11}), (:A {id: 11})-[:R]->(:A {id: 12})")
        # the bound end is the pattern's `to`, bound by an earlier clause
        q = """MATCH (b:B) WITH b MATCH p = (a:A)-[:R*2..2]-(b)
               RETURN a.id, b.id, [e IN relationships(p) | e.w]"""
        res = g.query(q).result_set
        self.env.assertEqual(res, [[1, 3, [1, 2]]])
        # the path reads from `a` to `b`, whichever end the walk started at
        q = """MATCH (b:B) WITH b MATCH p = (a:A)-[:R*2..2]-(b) RETURN [n IN nodes(p) | n.id]"""
        self.env.assertEqual(g.query(q).result_set, [[[1, 2, 3]]])
        # zero-length and unconstrained ends
        q = "MATCH (b:B) WITH b MATCH (a)-[:R*0..1]-(b) RETURN a.id ORDER BY a.id"
        self.env.assertEqual(g.query(q).result_set, [[2], [3]])
        g.delete()

    # A walk whose `to` endpoint is the more selective one is scanned from
    # there. Rebuilding the scan must keep whatever an earlier clause put under
    # the walk: its WHERE, its rows' multiplicity and the variables it binds.
    def test_var_len_reversal_keeps_earlier_clause(self):
        g = self.db.select_graph("VarLenReversalKeepsEarlier")
        g.query("CREATE INDEX FOR (n:B) ON (n.id)")
        wait_for_indices_to_sync(g)
        g.query("""CREATE (:A {id: 1})-[:R]->(:B {id: 1}), (:A {id: 99})-[:R]->(:B {id: 1}),
                          (:A {id: 5})-[:R]->(:C {id: 1})""")
        for q, expected in [
                ("MATCH (a) WHERE a.id = 99 MATCH (a)-[:R*1..2]->(b:B {id: 1}) RETURN count(*)", 1),
                ("MATCH (a) WHERE a.id = 99 MATCH (a)-[:R*1..2]-(b:B {id: 1}) RETURN count(*)", 1),
                ("MATCH (a:A) WHERE a.id = 99 MATCH (a)<-[:R*1..2]-(b:B {id: 1}) RETURN count(*)", 0),
                ("UNWIND [1, 2] AS x MATCH (a)-[:R*1..2]->(b:B {id: 1}) RETURN count(*)", 4),
                ("UNWIND [1, 2] AS x MATCH (a)-[:R*1..2]-(b:B {id: 1}) RETURN count(*)", 4),
                ("MATCH (x:A) WHERE x.id = 5 MATCH (a)-[:R*1..2]->(b:B {id: 1}) RETURN count(*)", 2)]:
            self.env.assertEqual(g.query(q).result_set, [[expected]], message=q)
        res = g.query("""UNWIND [1, 2] AS x MATCH (a)-[:R*1..2]->(b:B {id: 1})
                         RETURN x, a.id ORDER BY x, a.id""").result_set
        self.env.assertEqual(res, [[1, 1], [1, 99], [2, 1], [2, 99]])
        g.delete()

    # Several walks in one component: the anchored walk leads and the others
    # follow outward from what it binds, each one with a bound endpoint. A DAG
    # keeps edge uniqueness out of the count, so it can be computed directly.
    def test_var_len_anchor_with_several_walks(self):
        g = self.db.select_graph("VarLenAnchorSeveralWalks")
        g.query("CREATE INDEX FOR (p:Person) ON (p.id)")
        wait_for_indices_to_sync(g)
        n = 120
        g.query(f"UNWIND range(0, {n - 1}) AS i CREATE (:Person {{id: i}})")
        g.query("UNWIND range(0, 4) AS i CREATE (:City {id: i})")
        g.query(f"""UNWIND range(0, {n - 1}) AS i
                    MATCH (p:Person {{id: i}}), (c:City {{id: i % 5}}) CREATE (p)-[:LIVES]->(c)""")
        g.query(f"""UNWIND range(0, {n - 1}) AS i UNWIND [1, 3] AS d
                    MATCH (a:Person {{id: i}}), (b:Person {{id: i + d}}) CREATE (a)-[:KNOWS]->(b)""")
        out = {i: [j for j in (i + 1, i + 3) if j < n] for i in range(n)}

        def reach(src):
            level, res = {src: 1}, {}
            for _ in range(2):
                nxt = {}
                for u, c in level.items():
                    for v in out[u]:
                        nxt[v] = nxt.get(v, 0) + c
                for v, c in nxt.items():
                    res[v] = res.get(v, 0) + c
                level = nxt
            return res
        # (p)-[*1..2]->(f)-[*1..2]->(g)-[:LIVES]->(city), one city per person
        expected = sum(c1 * c2 for f, c1 in reach(5).items() for g_, c2 in reach(f).items())

        for q in ["""MATCH (p:Person {id: 5})-[:KNOWS*1..2]->(f), (f)-[:KNOWS*1..2]->(g),
                           (g)-[:LIVES]->(c:City) RETURN count(*)""",
                  """MATCH (c:City)<-[:LIVES]-(g)<-[:KNOWS*1..2]-(f)<-[:KNOWS*1..2]-(p:Person {id: 5})
                     RETURN count(*)""",
                  """MATCH (g)-[:LIVES]->(c:City), (f)-[:KNOWS*1..2]->(g), (p:Person)-[:KNOWS*1..2]->(f)
                     WHERE p.id = 5 RETURN count(*)"""]:
            plan = str(g.explain(q))
            scans = [l.strip() for l in plan.splitlines() if "Scan" in l]
            self.env.assertEqual(scans, ["Node By Index Scan | (p:Person)"], message=plan)
            self.env.assertEqual(g.query(q).result_set, [[expected]], message=q)
        g.delete()

    # Hops that read a variable another hop binds keep their order: moving the
    # walk first must not run a fixed hop's edge filter before the hop that
    # binds what it reads, and reversing a walk must not unbind an endpoint its
    # own edge attributes read.
    def test_var_len_anchor_keeps_attribute_dependencies(self):
        g = self.db.select_graph("VarLenAnchorAttrDeps")
        g.query("""CREATE (a:A {id: 1})-[:R]->(b)-[:S {weight: 5}]->(e),
                          (b)-[:U]->(c)-[:T]->(d {limit: 5})""")
        q = """MATCH (c)-[:T]->(d), (b)-[:S {weight: d.limit}]->(e), (a:A)-[:R*1..2]->(b)-[:U*1..2]->(c)
               WHERE a.id = 1 RETURN count(*)"""
        self.env.assertEqual(g.query(q).result_set, [[1]], message=q)
        g.delete()

        g = self.db.select_graph("VarLenAnchorEdgeAttrEndpoint")
        g.query("CREATE INDEX FOR (b:B) ON (b.id)")
        wait_for_indices_to_sync(g)
        g.query("CREATE (:A {id: 1})-[:R {w: 1}]->(:B {id: 1}), (:A {id: 2})-[:R {w: 7}]->(:B {id: 2})")
        for q in ["MATCH (a:A)-[:R*1..2 {w: a.id}]-(b:B {id: 1}) RETURN count(*)",
                  "MATCH (a:A)-[:R*1..2 {w: a.id}]->(b:B {id: 1}) RETURN count(*)"]:
            self.env.assertEqual(g.query(q).result_set, [[1]], message=q)
        g.delete()

    # The WHERE sits above the whole clause: with another component (a
    # CartesianProduct) or a cycle (an ExpandInto) between, scan selection does
    # not see it, so it cannot make an endpoint the walk's starting scan and
    # the walk keeps its place instead of expanding from every `a`.
    def test_var_len_anchor_where_hidden_by_barrier(self):
        g = self.db.select_graph("VarLenAnchorWhereBarrier")
        g.query("CREATE INDEX FOR (b:B) ON (b.id)")
        wait_for_indices_to_sync(g)
        g.query("UNWIND range(1, 100) AS i CREATE (:A {id: i})-[:R]->(:B {id: i})")
        g.query("UNWIND range(1, 100) AS i MATCH (a:A {id: i}) CREATE (a)-[:S]->(:C {id: i})")
        g.query("CREATE (:X)")
        for q in ["""MATCH (x:X), (a:A)-[:R*1..2]->(b:B), (a)-[:S]->(c:C)
                     WHERE b.id = 1 RETURN count(*)""",
                  """MATCH (a:A)-[:R*1..2]->(b:B), (a)-[:S]->(c:C), (c)-[:S]->(c)
                     WHERE b.id = 1 RETURN count(*)""",
                  # a pattern in the WHERE plans an Apply between the clause and it
                  """MATCH (a:A)-[:R*1..2]->(b:B), (a)-[:S]->(c:C)
                     WHERE b.id = 1 AND size([(c)-[:X]->() | 1]) > 0 RETURN count(*)"""]:
            plan = str(g.explain(q))
            # the fixed hop keeps leading: its traverse sits below the walk
            walk = plan.index("Variable Length Traverse")
            self.env.assertTrue(plan.index("Conditional Traverse") > walk, message=plan)
            self.env.assertEqual(g.query(q).result_set, [[1]] if "(x:X)" in q else [[0]], message=q)
        g.delete()

    # An input that already binds the walk's `to` endpoint lets the planner's
    # scan of `from` go even when the edge attributes read `to`: the walk runs
    # backwards from the bound node instead of from every `a`.
    def test_var_len_walk_from_bound_endpoint_with_attr_dependency(self):
        g = self.db.select_graph("VarLenBoundEndpointAttrs")
        g.query("UNWIND range(1, 100) AS i CREATE (:A {id: i})-[:R {w: 1}]->(:M {id: i})")
        g.query("CREATE (:A {id: 0})-[:R {w: 1}]->(:B {id: 1})")
        q = """MATCH (b:B {id: 1}) WITH b MATCH (a:A)-[:R*1..2 {w: b.id}]->(b)
               RETURN count(*)"""
        plan = str(g.explain(q))
        self.env.assertNotContains("Node By Label Scan | (a:A)", plan)
        self.env.assertEqual(g.query(q).result_set, [[1]], message=q)
        g.delete()
