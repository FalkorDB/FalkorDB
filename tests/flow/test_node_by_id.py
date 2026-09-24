from common import *

GRAPH_ID = "node_by_id"

class testNodeByIDFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        # Create entities
        self.graph.query("UNWIND range(0, 9) AS i CREATE (n:person {id:i})")

        # Make sure node id attribute matches node's internal ID.
        query = """MATCH (n) SET n.id = ID(n)"""
        self.graph.query(query)

    # Expect an error when trying to use a function which does not exists.
    def test_get_nodes(self):
        # All nodes, not including first node.
        query = """MATCH (n) WHERE ID(n) > 0 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id > 0 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 0 < ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 0 < n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # All nodes.
        query = """MATCH (n) WHERE ID(n) >= 0 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id >= 0 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 0 <= ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 0 <= n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # A single node.
        query = """MATCH (n) WHERE ID(n) = 0 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id = 0 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # 4 nodes (6,7,8,9)
        query = """MATCH (n) WHERE ID(n) > 5 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id > 5 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 5 < ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 5 < n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # 5 nodes (5, 6,7,8,9)
        query = """MATCH (n) WHERE ID(n) >= 5 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id >= 5 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 5 <= ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 5 <= n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # 5 nodes (0,1,2,3,4)
        query = """MATCH (n) WHERE ID(n) < 5 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id < 5 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 5 < ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 5 < n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # 6 nodes (0,1,2,3,4,5)
        query = """MATCH (n) WHERE ID(n) <= 5 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id <= 5 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 5 >= ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 5 >= n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # All nodes except last one.
        query = """MATCH (n) WHERE ID(n) < 9 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id < 9 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 9 > ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 9 > n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # All nodes.
        query = """MATCH (n) WHERE ID(n) <= 9 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id <= 9 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 9 >= ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 9 >= n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # All nodes.
        query = """MATCH (n) WHERE ID(n) < 100 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id < 100 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 100 > ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 100 > n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # All nodes.
        query = """MATCH (n) WHERE ID(n) <= 100 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id <= 100 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 100 >= ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 100 >= n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # cartesian product, tests reset works as expected.
        query = """MATCH (a), (b) WHERE ID(a) > 5 AND ID(b) <= 5 RETURN a,b ORDER BY a.id, b.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (a), (b) WHERE a.id > 5 AND b.id <= 5 RETURN a,b ORDER BY a.id, b.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (a), (b) WHERE 5 < ID(a) AND 5 >= ID(b) RETURN a,b ORDER BY a.id, b.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (a), (b) WHERE 5 < a.id AND 5 >= b.id RETURN a,b ORDER BY a.id, b.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # runtime optimization
        query = """UNWIND range(1, 5) AS x MATCH (n) WHERE ID(n) = x RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """UNWIND range(1, 5) AS x MATCH (n) WHERE n.id = x RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """UNWIND range(1, 5) AS x MATCH (n:person) WHERE ID(n) = x RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("Node By Label and ID Scan", str(self.graph.explain(query)))
        query = """UNWIND range(1, 5) AS x MATCH (n:person) WHERE n.id = x RETURN n ORDER BY n.id"""
        self.env.assertNotIn("Node By Label and ID Scan", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

    # the seek-by-id optimization also applies when the value ID() is
    # compared against is a compound expression (e.g. x + 1) rather than a
    # bare constant / parameter / variable, as long as that expression does
    # not reference the node being scanned
    def test_seek_by_id_expression(self):
        # ID(n) = <expression over an UNWIND variable>, nodes 1,2,3,4,5
        query = """UNWIND range(0, 4) AS x MATCH (n) WHERE ID(n) = x + 1 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """UNWIND range(0, 4) AS x MATCH (n) WHERE n.id = x + 1 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # reversed operands: <expression> = ID(n)
        query = """UNWIND range(0, 4) AS x MATCH (n) WHERE x + 1 = ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """UNWIND range(0, 4) AS x MATCH (n) WHERE x + 1 = n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # inequality against an expression: ID(n) > x + 1
        query = """UNWIND [1] AS x MATCH (n) WHERE ID(n) > x + 1 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """UNWIND [1] AS x MATCH (n) WHERE n.id > x + 1 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # two expression bounds folded into a single range
        query = """UNWIND [1] AS x MATCH (n) WHERE ID(n) >= x AND ID(n) <= x + 2 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """UNWIND [1] AS x MATCH (n) WHERE n.id >= x AND n.id <= x + 2 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # value expression referencing a previously resolved node: ID(b) = ID(a) + 1
        query = """MATCH (a {id:1}) WITH a MATCH (b) WHERE ID(b) = ID(a) + 1 RETURN b ORDER BY b.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (a {id:1}) WITH a MATCH (b) WHERE b.id = ID(a) + 1 RETURN b ORDER BY b.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # label scan retains the ID range when the bound is an expression
        query = """UNWIND range(0, 4) AS x MATCH (n:person) WHERE ID(n) = x + 1 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertIn("Node By Label and ID Scan", str(self.graph.explain(query)))
        query = """UNWIND range(0, 4) AS x MATCH (n:person) WHERE n.id = x + 1 RETURN n ORDER BY n.id"""
        self.env.assertNotIn("Node By Label and ID Scan", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # the value expression must not reference the scanned node itself:
        # ID(n) = n.id references 'n' on both sides, so it must NOT be optimized
        query = """MATCH (n) WHERE ID(n) = n.id RETURN n ORDER BY n.id"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        # n.id == ID(n) for every node, so every node is returned
        resultsetA = self.graph.query(query).result_set
        resultsetB = self.graph.query("MATCH (n) RETURN n ORDER BY n.id").result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # the ID() side must be ID(<scanned node>) itself, not id() of an
        # attribute or other expression. the optimization discards the ID(...)
        # expression, so optimizing ID(n.id) = x + 1 would silently execute it
        # as ID(n) IN {x+1} instead of evaluating id() on the attribute. it must
        # NOT be optimized, and then errors at runtime like the plain filter
        query = """UNWIND range(0, 4) AS x MATCH (n) WHERE ID(n.id) = x + 1 RETURN n"""
        self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(query)))
        raised = False
        try:
            self.graph.query(query)
        except Exception as e:
            raised = True
            self.env.assertContains("Type mismatch", str(e))
        self.env.assertTrue(raised)

    # Try to fetch none existing entities by ID(s).
    def test_for_none_existing_entity_ids(self):
        # Try to fetch an entity with a none existing ID.
        queries = ["""MATCH (a:person) WHERE ID(a) = 999 RETURN a""",
                    """MATCH (a:person) WHERE ID(a) > 999 RETURN a""",
                    """MATCH (a:person) WHERE ID(a) > 800 AND ID(a) < 900 RETURN a"""]

        for query in queries:
            resultset = self.graph.query(query).result_set        
            self.env.assertEquals(len(resultset), 0)    # Expecting no results.
            self.env.assertIn("Node By Label and ID Scan", str(self.graph.explain(query)))

    def test_node_by_id_scan_reset(self):
        # the following query used to crash due to wrong reset handeling by
        # the op_node_by_label_scan operation

        q = """UNWIND $pairs AS pair
               UNWIND pair.feature_ids AS feature_id
               MATCH (f:Feature), (n)
               WHERE id(f) = feature_id AND id(n) = pair.node_id
               RETURN 1"""

        try:
            res = self.graph.query(q, {'pairs': [{'node_id':1,'feature_ids':[2]}]}).result_set
            self.env.assertEquals(len(res), 0)
        except Exception as e:
            self.env.assertFalse("query crashed")

    # a per-record ID bound that resolves to an EMPTY range - a non-integer
    # value such as 1.1, or an out-of-range integer - must cause that single
    # record to be skipped, NOT terminate the seek and silently drop every
    # remaining record (regression for the NodeByIdSeekConsumeFromChild fix in
    # op_node_by_id_seek.c). The un-optimized 'n.id = x' form is used as the
    # oracle: 'n.id' equals ID(n) for every node but does not trigger the seek.
    def test_seek_by_id_skips_empty_id_ranges(self):
        # the optimized (ID(n)) query must use the seek AND return the same
        # rows as the un-optimized (n.id) equivalent
        def assert_matches(id_q, prop_q, op="NodeByIdSeek"):
            self.env.assertIn(op, str(self.graph.explain(id_q)))
            self.env.assertNotIn(op, str(self.graph.explain(prop_q)))
            self.env.assertEqual(self.graph.query(id_q).result_set,
                                 self.graph.query(prop_q).result_set)

        # non-integer leads the list - used to drop nodes 2 and 4 entirely
        assert_matches(
            "UNWIND [1.1, 2, 3.5, 4] AS x MATCH (n) WHERE ID(n) = x RETURN n ORDER BY n.id",
            "UNWIND [1.1, 2, 3.5, 4] AS x MATCH (n) WHERE n.id  = x RETURN n ORDER BY n.id")

        # non-integer in the middle - used to drop node 4
        assert_matches(
            "UNWIND [2, 1.1, 4] AS x MATCH (n) WHERE ID(n) = x RETURN n ORDER BY n.id",
            "UNWIND [2, 1.1, 4] AS x MATCH (n) WHERE n.id  = x RETURN n ORDER BY n.id")

        # out-of-range integer leads the list - same premature-termination bug
        assert_matches(
            "UNWIND [100, 2, 4] AS x MATCH (n) WHERE ID(n) = x RETURN n ORDER BY n.id",
            "UNWIND [100, 2, 4] AS x MATCH (n) WHERE n.id  = x RETURN n ORDER BY n.id")

        # every value is a non-integer - result is correctly empty (no crash,
        # no leftover state from a previous record's range)
        assert_matches(
            "UNWIND [1.1, 3.5] AS x MATCH (n) WHERE ID(n) = x RETURN n ORDER BY n.id",
            "UNWIND [1.1, 3.5] AS x MATCH (n) WHERE n.id  = x RETURN n ORDER BY n.id")

        # same skip behaviour on the label-scan path (Node By Label and ID Scan)
        assert_matches(
            "UNWIND [1.1, 2, 3.5, 4] AS x MATCH (n:person) WHERE ID(n) = x RETURN n ORDER BY n.id",
            "UNWIND [1.1, 2, 3.5, 4] AS x MATCH (n:person) WHERE n.id  = x RETURN n ORDER BY n.id",
            op="Node By Label and ID Scan")

        # a single non-integer constant (tap, no child record) is correctly empty
        assert_matches(
            "MATCH (n) WHERE ID(n) = 2.1 RETURN n",
            "MATCH (n) WHERE n.id  = 2.1 RETURN n")

        # a heterogeneous UNWIND mixing assorted non-integer TYPES (string,
        # list, map, bool, fractional) with valid integers: every non-integer
        # value yields an empty range and is skipped, the integers (2, 4) match
        assert_matches(
            "UNWIND ['abc', 2, [1,2], {c:1}, true, 3.5, 4] AS x MATCH (n) WHERE ID(n) = x RETURN n ORDER BY n.id",
            "UNWIND ['abc', 2, [1,2], {c:1}, true, 3.5, 4] AS x MATCH (n) WHERE n.id  = x RETURN n ORDER BY n.id")

        # a single non-integer constant of assorted types (tap) is empty and
        # matches the equivalent property filter (which also yields no rows)
        for v in ["'abc'", "[1, 2]", "{c: 1}", "true", "1.1"]:
            assert_matches(
                "MATCH (n) WHERE ID(n) = " + v + " RETURN n",
                "MATCH (n) WHERE n.id  = " + v + " RETURN n")

    # the value the seek is compared against may itself be an ID() call, e.g.
    # ID(n) = ID(m). Because the optimization evaluates that value expression
    # against the incoming record at runtime, 'm' can bind to a non-graph-entity
    # value (an integer, string, list, map, ...). id() of a non-node/edge is a
    # type error, and that error must surface (not be silently swallowed by the
    # seek treating the failed evaluation as an empty range)
    def test_seek_by_id_of_non_entity_raises(self):
        queries = [
            # constant argument
            "MATCH (n) WHERE ID(n) = ID('abc') RETURN n",
            # value carried on the record (WITH / UNWIND) - the runtime path
            "WITH 5 AS m MATCH (n) WHERE ID(n) = ID(m) RETURN n",
            "WITH {k:1} AS m MATCH (n) WHERE ID(n) = ID(m) RETURN n",
            "WITH [1,2] AS m MATCH (n) WHERE ID(n) = ID(m) RETURN n",
            "UNWIND [0, 2, 4] AS m MATCH (n) WHERE ID(n) = ID(m) RETURN n",
        ]
        for q in queries:
            raised = False
            try:
                self.graph.query(q)
            except Exception as e:
                raised = True
                self.env.assertContains("Type mismatch", str(e))
            self.env.assertTrue(raised)

    # a non-integer numeric bound must be resolved to the correct integer range
    # per the operator - ID(n) > 2.0 means id >= 3, ID(n) <= 2.5 means id <= 2,
    # ID(n) = 2.5 matches no id - instead of collapsing to an empty range and
    # returning nothing. covers compile-time doubles and runtime doubles
    # (parameters, UNWIND).
    def test_seek_by_id_double_bounds(self):
        # optimized (ID(n)) query must use the seek and match the un-optimized
        # (n.id) equivalent, which keeps full numeric-comparison semantics
        def assert_seek_eq(id_q, prop_q):
            self.env.assertIn("NodeByIdSeek", str(self.graph.explain(id_q)))
            self.env.assertNotIn("NodeByIdSeek", str(self.graph.explain(prop_q)))
            self.env.assertEqual(self.graph.query(id_q).result_set,
                                 self.graph.query(prop_q).result_set)

        # every comparison operator, against an integral (2.0) and a fractional
        # (2.5) double bound
        for op in [">", ">=", "<", "<=", "="]:
            for bound in ["2.0", "2.5"]:
                assert_seek_eq(
                    "MATCH (n) WHERE ID(n) %s %s RETURN n ORDER BY n.id" % (op, bound),
                    "MATCH (n) WHERE n.id  %s %s RETURN n ORDER BY n.id" % (op, bound))

        # integral doubles from an UNWIND match; fractional ones are skipped
        assert_seek_eq(
            "UNWIND [2.0, 3.5, 4.0] AS x MATCH (n) WHERE ID(n) = x RETURN n ORDER BY n.id",
            "UNWIND [2.0, 3.5, 4.0] AS x MATCH (n) WHERE n.id  = x RETURN n ORDER BY n.id")

        # a runtime double parameter is resolved the same way
        self.env.assertEqual(
            self.graph.query("MATCH (n) WHERE ID(n) > $p RETURN n ORDER BY n.id", {'p': 2.0}).result_set,
            self.graph.query("MATCH (n) WHERE n.id  > $p RETURN n ORDER BY n.id", {'p': 2.0}).result_set)

