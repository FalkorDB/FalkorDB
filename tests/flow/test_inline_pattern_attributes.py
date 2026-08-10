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
