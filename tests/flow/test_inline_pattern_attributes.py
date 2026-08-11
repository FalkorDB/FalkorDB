from common import *

GRAPH_ID = "inline_pattern_attributes"


class testInlinePatternAttributes(FlowTestsBase):
    """Inline attributes on a MATCH pattern (`MATCH (a {x: 1})`) are a
    predicate, and the planner is the only thing that enforces them for most
    operators. These cover the pattern positions where the lowering used to be
    skipped, so the predicate was silently dropped and the query returned rows
    it should not have."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        self.graph.query(
            """CREATE (n1:N {x: 1}), (n2:N {x: 2}), (n3:N {x: 3}), (n4:N {x: 4}),
                      (n1)-[:R]->(n2), (n2)-[:R]->(n3),
                      (n1)-[:S]->(n3), (n2)-[:S]->(n4)"""
        )

    # A variable-length traverse whose source is already bound: the endpoint's
    # inline attrs were not lowered (the alias is already visited) and
    # CondVarLenTraverse only evaluates the *edge*'s attrs at runtime, so
    # nothing enforced `x: 1`.
    def test01_var_len_bound_source(self):
        query = """MATCH (a:N) WITH a
                   MATCH (a {x: 1})-[:R*1..2]->(b)
                   RETURN b.x ORDER BY b.x"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, [[2], [3]])

    # Same hole on the destination endpoint.
    def test02_var_len_bound_destination(self):
        query = """MATCH (b:N) WITH b
                   MATCH (a:N)-[:R*1..2]->(b {x: 3})
                   RETURN a.x ORDER BY a.x"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, [[1], [2]])

    # allShortestPaths reads the edge's attrs only, exactly like the var-length
    # traverse, so a bound endpoint's inline attrs went unenforced.
    def test03_all_shortest_paths_bound_endpoints(self):
        query = """MATCH (a:N), (b:N) WITH a, b
                   MATCH p = allShortestPaths((a {x: 1})-[:R*]->(b {x: 3}))
                   RETURN length(p)"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, [[2]])

    # A fixed-length traverse whose source is bound by an earlier clause. This
    # one happened to work, because CondTraverse re-checks the endpoints'
    # attrs per row — it is here so the runtime check can be removed later
    # without losing the coverage.
    def test04_traverse_bound_source(self):
        query = """MATCH (a:N) WITH a
                   MATCH (a {x: 1})-[:R]->(b)
                   RETURN b.x"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, [[2]])

    # One alias, two occurrences, each writing its own inline attrs. The binder
    # builds a separate enriched QueryNode per occurrence, so both maps have to
    # be lowered; these two contradict, so nothing can match.
    def test05_two_occurrences_of_one_alias(self):
        query = """MATCH (a {x: 1})-[:R]->(c), (a {x: 2})-[:S]->(d)
                   RETURN count(*)"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, [[0]])

    # ...and the same shape where the two occurrences agree, so it does match.
    def test06_two_occurrences_agreeing(self):
        query = """MATCH (a {x: 1})-[:R]->(c), (a {x: 1})-[:S]->(d)
                   RETURN c.x, d.x"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, [[2, 3]])

    # A third hop whose `from` endpoint is already bound by the first two. The
    # chained-relationship loop lowered only the `to` endpoint, so this
    # predicate had no representation at all.
    def test07_chained_hop_bound_source(self):
        query = """MATCH (a)-[:R]->(b)-[:R]->(c), (b {x: 9})-[:S]->(d)
                   RETURN count(*)"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, [[0]])

    def test08_chained_hop_bound_source_matching(self):
        query = """MATCH (a)-[:R]->(b)-[:R]->(c), (b {x: 2})-[:S]->(d)
                   RETURN a.x, b.x, c.x, d.x"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, [[1, 2, 3, 4]])

    # The predicate must survive whichever endpoint `select_scan_node` picks as
    # the scan root, including when it reverses the chain.
    def test09_predicate_survives_chain_reversal(self):
        query = """MATCH (a:N)-[:R]->(b:N {x: 3})
                   RETURN a.x"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set, [[2]])

    # `utilize_index` reaches a MERGE branch's scan through `IncludePending`.
    # The Filter above it must survive that pushdown: `IncludePending` unions
    # in nodes created earlier in the same query, which are not in the index,
    # so the index scan cannot have filtered them. Without the Filter the
    # second MERGE matches the first's pending node and creates nothing.
    def test11_merge_index_pushdown_still_filters_pending(self):
        g = self.db.select_graph(GRAPH_ID + "_pending")
        g.query("CREATE INDEX FOR (p:P) ON (p.age)")
        result = g.query("MERGE (a:P {age: 40}) MERGE (b:P {age: 41})")
        self.env.assertEqual(result.nodes_created, 2)

        plan = str(g.explain("MERGE (a:P {age: 40}) MERGE (b:P {age: 41})"))
        self.env.assertEqual(plan.count("Node By Index Scan"), 2)

        # Re-running must match both, not create more.
        result = g.query("MERGE (a:P {age: 40}) MERGE (b:P {age: 41})")
        self.env.assertEqual(result.nodes_created, 0)

        # Three distinct values in one query: every pending row has to be
        # re-checked, not just the first.
        result = g.query("MERGE (c:P {age: 1}) MERGE (d:P {age: 2}) MERGE (e:P {age: 3})")
        self.env.assertEqual(result.nodes_created, 3)
        actual = g.query("MATCH (p:P) RETURN p.age ORDER BY p.age")
        self.env.assertEqual(actual.result_set, [[1], [2], [3], [40], [41]])
        g.delete()

    # Inline attrs on an *edge*. Each operator enforces them differently:
    # CondTraverse and ExpandInto via a Filter above themselves, with the
    # collapse of parallel edges disabled; CondVarLenTraverse and
    # AllShortestPaths via their edge_filter, applied per edge during the walk.
    def test12_edge_attrs_parallel_edges(self):
        g = self.db.select_graph(GRAPH_ID + "_edges")
        # Two parallel edges between the same pair, distinguished only by `k`.
        g.query(
            "CREATE (a:E {n: 'a'}), (b:E {n: 'b'}), "
            "(a)-[:R {k: 1}]->(b), (a)-[:R {k: 2}]->(b)"
        )

        # An anonymous edge with a predicate must not collapse to one
        # representative edge: whichever were picked, the other is the match.
        for k in (1, 2):
            actual = g.query("MATCH (a:E)-[{k: %d}]->(b:E) RETURN count(*)" % k)
            self.env.assertEqual(actual.result_set, [[1]])

        # Contrast: with no predicate on the edge, an anonymous traverse still
        # collapses the pair to a single representative edge, so this counts 1
        # rather than 2. That is long-standing behaviour, independent of this
        # file — it is here to pin the boundary the predicate cases rely on.
        actual = g.query("MATCH (a:E)-[]->(b:E) RETURN count(*)")
        self.env.assertEqual(actual.result_set, [[1]])

        # ExpandInto: both endpoints already bound.
        actual = g.query(
            "MATCH (a:E {n: 'a'}), (b:E {n: 'b'}) WITH a, b "
            "MATCH (a)-[{k: 2}]->(b) RETURN count(*)"
        )
        self.env.assertEqual(actual.result_set, [[1]])
        g.delete()

    def test13_edge_attrs_var_len_and_shortest_paths(self):
        g = self.db.select_graph(GRAPH_ID + "_edgewalk")
        g.query(
            "CREATE (a:W {n: 1}), (b:W {n: 2}), (c:W {n: 3}), "
            "(a)-[:R {k: 1}]->(b), (b)-[:R {k: 1}]->(c), (a)-[:R {k: 9}]->(c)"
        )

        # Var-length: the k=9 shortcut is pruned, so 3 is reached in two hops.
        actual = g.query(
            "MATCH (a:W {n: 1})-[:R*1..2 {k: 1}]->(x:W) RETURN x.n ORDER BY x.n"
        )
        self.env.assertEqual(actual.result_set, [[2], [3]])

        # Without the predicate the one-hop shortcut is available.
        actual = g.query("MATCH (a:W {n: 1})-[:R*1..1]->(x:W) RETURN x.n ORDER BY x.n")
        self.env.assertEqual(actual.result_set, [[2], [3]])

        # allShortestPaths: pruning the shortcut makes the shortest path 2 hops.
        actual = g.query(
            "MATCH (a:W {n: 1}), (c:W {n: 3}) WITH a, c "
            "MATCH p = allShortestPaths((a)-[:R* {k: 1}]->(c)) RETURN length(p)"
        )
        self.env.assertEqual(actual.result_set, [[2]])

        actual = g.query(
            "MATCH (a:W {n: 1}), (c:W {n: 3}) WITH a, c "
            "MATCH p = allShortestPaths((a)-[:R*]->(c)) RETURN length(p)"
        )
        self.env.assertEqual(actual.result_set, [[1]])
        g.delete()

    # Inline attrs on a MERGE pattern are a constructor, not a predicate: the
    # created node must carry them.
    def test10_merge_pattern_attrs_are_constructed(self):
        g = self.db.select_graph(GRAPH_ID + "_merge")
        g.query("MERGE (n:M {a: 1, b: 2})")
        actual_result = g.query("MATCH (n:M) RETURN n.a, n.b")
        self.env.assertEqual(actual_result.result_set, [[1, 2]])
        # Second MERGE of the same pattern must match, not create a duplicate.
        g.query("MERGE (n:M {a: 1, b: 2})")
        actual_result = g.query("MATCH (n:M) RETURN count(*)")
        self.env.assertEqual(actual_result.result_set, [[1]])
        g.delete()
