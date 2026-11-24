from common import *
from index_utils import wait_for_indices_to_sync

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
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id > 0 RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 0 < ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 0 < n.id RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # All nodes.
        query = """MATCH (n) WHERE ID(n) >= 0 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id >= 0 RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 0 <= ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 0 <= n.id RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # A single node.
        query = """MATCH (n) WHERE ID(n) = 0 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id = 0 RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # 4 nodes (6,7,8,9)
        query = """MATCH (n) WHERE ID(n) > 5 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id > 5 RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 5 < ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 5 < n.id RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # 5 nodes (5, 6,7,8,9)
        query = """MATCH (n) WHERE ID(n) >= 5 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id >= 5 RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 5 <= ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 5 <= n.id RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # 5 nodes (0,1,2,3,4)
        query = """MATCH (n) WHERE ID(n) < 5 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id < 5 RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 5 < ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 5 < n.id RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # 6 nodes (0,1,2,3,4,5)
        query = """MATCH (n) WHERE ID(n) <= 5 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id <= 5 RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 5 >= ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 5 >= n.id RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # All nodes except last one.
        query = """MATCH (n) WHERE ID(n) < 9 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id < 9 RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 9 > ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 9 > n.id RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # All nodes.
        query = """MATCH (n) WHERE ID(n) <= 9 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id <= 9 RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 9 >= ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 9 >= n.id RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # All nodes.
        query = """MATCH (n) WHERE ID(n) < 100 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id < 100 RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 100 > ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 100 > n.id RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # All nodes.
        query = """MATCH (n) WHERE ID(n) <= 100 RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE n.id <= 100 RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (n) WHERE 100 >= ID(n) RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (n) WHERE 100 >= n.id RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # cartesian product, tests reset works as expected.
        query = """MATCH (a), (b) WHERE ID(a) > 5 AND ID(b) <= 5 RETURN a,b ORDER BY a.id, b.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (a), (b) WHERE a.id > 5 AND b.id <= 5 RETURN a,b ORDER BY a.id, b.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """MATCH (a), (b) WHERE 5 < ID(a) AND 5 >= ID(b) RETURN a,b ORDER BY a.id, b.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """MATCH (a), (b) WHERE 5 < a.id AND 5 >= b.id RETURN a,b ORDER BY a.id, b.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        # runtime optimization
        query = """UNWIND range(1, 5) AS x MATCH (n) WHERE ID(n) = x RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("NodeByIdSeek", str(self.graph.explain(query)))
        query = """UNWIND range(1, 5) AS x MATCH (n) WHERE n.id = x RETURN n ORDER BY n.id"""
        self.env.assertNotContains("NodeByIdSeek", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

        query = """UNWIND range(1, 5) AS x MATCH (n:person) WHERE ID(n) = x RETURN n ORDER BY n.id"""
        resultsetA = self.graph.query(query).result_set
        self.env.assertContains("Node By Label and ID Scan", str(self.graph.explain(query)))
        query = """UNWIND range(1, 5) AS x MATCH (n:person) WHERE n.id = x RETURN n ORDER BY n.id"""
        self.env.assertNotContains("Node By Label and ID Scan", str(self.graph.explain(query)))
        resultsetB = self.graph.query(query).result_set
        self.env.assertEqual(resultsetA, resultsetB)

    # Try to fetch none existing entities by ID(s).
    def test_for_none_existing_entity_ids(self):
        # Try to fetch an entity with a none existing ID.
        queries = ["""MATCH (a:person) WHERE ID(a) = 999 RETURN a""",
                    """MATCH (a:person) WHERE ID(a) > 999 RETURN a""",
                    """MATCH (a:person) WHERE ID(a) > 800 AND ID(a) < 900 RETURN a"""]

        for query in queries:
            resultset = self.graph.query(query).result_set        
            self.env.assertEqual(len(resultset), 0)    # Expecting no results.
            self.env.assertContains("Node By Label and ID Scan", str(self.graph.explain(query)))

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
            self.env.assertEqual(len(res), 0)
        except Exception as e:
            self.env.assertFalse("query crashed")


    def test_non_integer_id_bounds(self):
        # A seek must keep exactly the rows the `id(n) <op> x` filter it
        # replaces keeps. A float compares numerically; null, NaN and
        # non-numeric bounds never compare true, so they match nothing
        # (they used to raise "Node ID must be an integer").
        queries = [
            ("MATCH (n) WHERE ID(n) = null RETURN count(n)", 0),
            ("MATCH (n) WHERE ID(n) > null RETURN count(n)", 0),
            ("MATCH (n) WHERE ID(n) = '1' RETURN count(n)", 0),
            ("MATCH (n) WHERE ID(n) = true RETURN count(n)", 0),
            ("MATCH (n) WHERE ID(n) = 0.0 / 0.0 RETURN count(n)", 0),
            ("MATCH (n) WHERE ID(n) = 1.0 RETURN count(n)", 1),
            ("MATCH (n) WHERE ID(n) = 1.5 RETURN count(n)", 0),
            ("MATCH (n) WHERE ID(n) > 1.5 RETURN count(n)", 8),
            ("MATCH (n) WHERE ID(n) >= 1.5 RETURN count(n)", 8),
            ("MATCH (n) WHERE ID(n) < 2.5 RETURN count(n)", 3),
            ("MATCH (n) WHERE ID(n) <= 2.5 RETURN count(n)", 3),
            ("MATCH (n) WHERE ID(n) > 2.0 AND ID(n) < 5.0 RETURN count(n)", 2),
            ("MATCH (n) WHERE ID(n) > 1.0 / 0.0 RETURN count(n)", 0),
            ("MATCH (n) WHERE ID(n) < 1.0 / 0.0 RETURN count(n)", 10),
            ("MATCH (n) WHERE ID(n) > -1.0 / 0.0 RETURN count(n)", 10),
            ("MATCH (n) WHERE ID(n) > -1 RETURN count(n)", 10),
            ("MATCH (n) WHERE ID(n) >= -5 RETURN count(n)", 10),
            ("MATCH (n) WHERE ID(n) <= -1 RETURN count(n)", 0),
            ("MATCH (n) WHERE ID(n) < -1.5 RETURN count(n)", 0),
            ("MATCH (n) WHERE ID(n) > 9223372036854775807 RETURN count(n)", 0),
            ("MATCH (n:person) WHERE ID(n) = 3.0 RETURN count(n)", 1),
            ("MATCH (n:person) WHERE ID(n) = null RETURN count(n)", 0),
            ("MATCH (n:person) WHERE ID(n) > -1 RETURN count(n)", 10),
        ]
        for query, expected in queries:
            plan = str(self.graph.explain(query))
            self.env.assertTrue("NodeByIdSeek" in plan or "Node By Label and ID Scan" in plan)
            res = self.graph.query(query).result_set
            self.env.assertEqual(res, [[expected]], message=query)

    def test_id_seek_on_a_graph_that_never_held_a_node(self):
        # An empty graph reported its highest node id as 0, which the id seek
        # read as "id 0 exists", so it returned a phantom node 0. Writes to it
        # survived: SET landed on the node the next CREATE made, and DELETE
        # failed with "0 was never allocated here".
        # https://github.com/FalkorDB/FalkorDB/issues/2965
        g = self.db.select_graph("node_by_id_empty")

        q = "MATCH (n) WHERE id(n) = 0 RETURN id(n)"
        self.env.assertEqual(g.query(q).result_set, [])
        self.env.assertContains("NodeByIdSeek", str(g.explain(q)))
        res = g.query("MATCH (n) WHERE id(n) >= 0 RETURN count(n)")
        self.env.assertEqual(res.result_set, [[0]])

        res = g.query("MATCH (n) WHERE id(n) = 0 DELETE n")
        self.env.assertEqual(res.nodes_deleted, 0)

        res = g.query("MATCH (n) WHERE id(n) = 0 SET n.x = 1, n:Z RETURN id(n)")
        self.env.assertEqual(res.result_set, [])
        self.env.assertEqual(res.properties_set, 0)

        g.query("CREATE (:Q)")
        res = g.query("MATCH (n) RETURN labels(n), properties(n)")
        self.env.assertEqual(res.result_set, [[["Q"], {}]])
        g.delete()

        # Nodes created earlier in the same query are still found by an id seek.
        g = self.db.select_graph("node_by_id_same_query")
        res = g.query("UNWIND range(0, 2) AS i CREATE (:A) WITH i MATCH (n) WHERE id(n) = i RETURN i, id(n)")
        self.env.assertEqual(res.result_set, [[0, 0], [1, 1], [2, 2]])
        g.delete()

        # A non-integer bound matches nothing, as the filter it replaces would.
        g = self.db.select_graph("node_by_id_empty_type")
        res = g.query("MATCH (n) WHERE id(n) = '0' RETURN n")
        self.env.assertEqual(res.result_set, [])
        g.delete()

    def test_parenthesised_id_predicates(self):
        # #3187: parentheses around an id() predicate still seek by id.
        queries = [
            ("MATCH (n) WHERE (ID(n) = 3) RETURN n.id", [[3]], 1),
            ("MATCH (n) WHERE ((ID(n) = 3)) RETURN n.id", [[3]], 1),
            ("MATCH (n) WHERE (ID(n) > 7) RETURN n.id ORDER BY n.id", [[8], [9]], 1),
            ("MATCH (a), (b) WHERE ((ID(a) = 1) AND (ID(b) = 2)) RETURN a.id, b.id",
             [[1, 2]], 2),
            ("MATCH (a), (b) WHERE (ID(a) = $a AND ID(b) = $b) RETURN a.id, b.id",
             [[1, 2]], 2),
        ]
        for q, expected, seeks in queries:
            params = {'a': 1, 'b': 2}
            plan = str(self.graph.explain(q, params))
            self.env.assertEqual(plan.count("NodeByIdSeek"), seeks, message=q)
            self.env.assertNotContains("All Node Scan", plan, message=q)
            self.env.assertNotContains("Filter", plan, message=q)
            self.env.assertEqual(self.graph.query(q, params).result_set, expected, message=q)

    def test_seek_on_a_graph_without_nodes(self):
        # #3353: id 0 was sought on a graph that never had a node.
        g = self.db.select_graph("node_by_id_empty")
        queries = [
            "MATCH (n) WHERE id(n) = 0 RETURN n",
            "MATCH (n) WHERE id(n) >= 0 RETURN n",
            "MATCH (n) WHERE id(n) < 3 RETURN n",
            "MATCH (n) WHERE id(n) = 0 DELETE n",
        ]
        g.query("RETURN 1")
        for q in queries:
            self.env.assertContains("NodeByIdSeek", str(g.explain(q)), message=q)
            self.env.assertEqual(g.query(q).result_set, [], message=q)

        # Nor once every node it had is deleted.
        g.query("CREATE (), ()")
        g.query("MATCH (n) DELETE n")
        for q in queries:
            self.env.assertEqual(g.query(q).result_set, [], message=q)

        # A node created in the same query is still found.
        res = g.query("CREATE (a) WITH a MATCH (n) WHERE id(n) = id(a) RETURN count(n)")
        self.env.assertEqual(res.result_set, [[1]])
        g.delete()

    def test_unwind_then_match_by_id_seeks_per_row(self):
        # `UNWIND .. MATCH (a) WHERE id(a) = e.s MATCH (b) WHERE id(b) = e.d`:
        # the second MATCH is joined to the first by `id(b)` only. It used to
        # become a Value Hash Join over a scan of every node (or of b's label),
        # whose cost grows with the graph however few rows the UNWIND produces.
        # Each row now looks b up by id, as it already looks a up.
        g = self.db.select_graph("unwind_match_by_id")
        g.query("UNWIND range(0, 99) AS i CREATE (:A {v: i})")
        g.query("UNWIND range(0, 599) AS i CREATE (:B {v: i})")

        # no node has id 999999
        q = ("UNWIND [{s: 0, d: 100}, {s: 1, d: 650}, {s: 2, d: 999999}] AS e "
             "MATCH (a) WHERE id(a) = e.s MATCH (b) WHERE id(b) = e.d "
             "RETURN id(a), id(b), b.v ORDER BY id(a)")
        plan = str(g.explain(q))
        self.env.assertEqual(plan.count("NodeByIdSeek"), 2)
        self.env.assertNotContains("Value Hash Join", plan)
        self.env.assertNotContains("All Node Scan", plan)
        self.env.assertEqual(g.query(q).result_set, [[0, 100, 0], [1, 650, 550]])

        # with labels: node 5 is an A, not a B
        q = ("UNWIND [{s: 0, d: 100}, {s: 1, d: 5}] AS e "
             "MATCH (a:A) WHERE id(a) = e.s MATCH (b:B) WHERE id(b) = e.d "
             "RETURN id(a), id(b) ORDER BY id(a)")
        plan = str(g.explain(q))
        self.env.assertEqual(plan.count("Node By Label and ID Scan"), 2)
        self.env.assertNotContains("Value Hash Join", plan)
        self.env.assertNotContains("Node By Label Scan", plan)
        self.env.assertEqual(g.query(q).result_set, [[0, 100]])

        # the reported shape: a batch of edges between nodes found by id
        q = ("UNWIND [{s: 0, d: 100}, {s: 1, d: 101}] AS e "
             "MATCH (a) WHERE id(a) = e.s MATCH (b) WHERE id(b) = e.d CREATE (a)-[:R]->(b)")
        self.env.assertNotContains("Value Hash Join", str(g.explain(q)))
        self.env.assertEqual(g.query(q).relationships_created, 2)
        res = g.query("MATCH (a)-[:R]->(b) RETURN id(a), id(b) ORDER BY id(a)")
        self.env.assertEqual(res.result_set, [[0, 100], [1, 101]])

        # b's own filter and a conjunct on the outer row still apply
        q = ("UNWIND [{s: 0, d: 100, v: 0}, {s: 1, d: 101, v: 7}, {s: 2, d: 102, v: 2}] AS e "
             "MATCH (a) WHERE id(a) = e.s MATCH (b:B) WHERE id(b) = e.d AND b.v > 0 AND b.v = e.v "
             "RETURN id(a), id(b) ORDER BY id(a)")
        plan = str(g.explain(q))
        self.env.assertContains("Node By Label and ID Scan", plan)
        self.env.assertNotContains("Value Hash Join", plan)
        self.env.assertEqual(g.query(q).result_set, [[2, 102]])

        # the MATCH clauses are separate isomorphism scopes (#2923): b may be a
        q = "UNWIND [3] AS x MATCH (a) WHERE id(a) = x MATCH (b) WHERE id(b) = x RETURN id(a), id(b)"
        self.env.assertEqual(g.query(q).result_set, [[3, 3]])

        # A key that can fail stays behind b's filter, which rejects every B,
        # so neither `e.q / e.z` nor `e.w * 1` runs.
        for q in [
            ("UNWIND [{s: 0, q: 5, z: 0}] AS e MATCH (a) WHERE id(a) = e.s "
             "MATCH (b:B) WHERE b.v > 1000 AND id(b) = e.q / e.z RETURN count(*)"),
            ("UNWIND [{s: 0, w: 'x'}] AS e MATCH (a) WHERE id(a) = e.s "
             "MATCH (b:B) WHERE b.v > 1000 AND id(b) = e.w * 1 RETURN count(*)"),
        ]:
            self.env.assertEqual(g.query(q).result_set, [[0]], message=q)

        # A key that is not deterministic would be evaluated twice, and a key
        # that reads b cannot be evaluated before b is found: only a is sought.
        for q in [
            ("UNWIND [{s: 0, d: 100}, {s: 1, d: 101}] AS e MATCH (a) WHERE id(a) = e.s "
             "MATCH (b) WHERE id(b) = e.d + toInteger(rand() * 0) RETURN id(a), id(b) ORDER BY id(a)"),
            ("UNWIND [{s: 0, d: 100}, {s: 1, d: 101}] AS e MATCH (a) WHERE id(a) = e.s "
             "MATCH (b) WHERE id(b) = e.d + b.v * 0 RETURN id(a), id(b) ORDER BY id(a)"),
        ]:
            self.env.assertEqual(str(g.explain(q)).count("NodeByIdSeek"), 1, message=q)
            self.env.assertEqual(g.query(q).result_set, [[0, 100], [1, 101]], message=q)

        # more rows than the graph has nodes: one scan beats a lookup per row
        q = ("UNWIND range(0, 999) AS i MATCH (a) WHERE id(a) = i % 100 "
             "MATCH (b) WHERE id(b) = 100 + i % 600 RETURN count(*)")
        self.env.assertContains("Value Hash Join", str(g.explain(q)))
        self.env.assertEqual(g.query(q).result_set, [[1000]])

        # A lookup on two labels intersects both label matrices for every row,
        # so one scan for the hash join is cheaper.
        g.query("MATCH (b:B) WHERE b.v < 10 SET b:C")
        q = ("UNWIND [{s: 0, d: 101}, {s: 1, d: 200}] AS e MATCH (a) WHERE id(a) = e.s "
             "MATCH (b:B:C) WHERE id(b) = e.d RETURN id(a), id(b)")
        self.env.assertContains("Value Hash Join", str(g.explain(q)))
        self.env.assertEqual(g.query(q).result_set, [[0, 101]])

        # An indexed later MATCH after an id seek: the id seek yields one row
        # per outer row, so the index is sought per row too.
        g.create_node_range_index("B", "v")
        wait_for_indices_to_sync(g)
        q = ("UNWIND [{s: 0, v: 7}, {s: 1, v: 9}] AS e MATCH (a) WHERE id(a) = e.s "
             "MATCH (b:B {v: e.v}) RETURN id(a), b.v ORDER BY id(a)")
        plan = str(g.explain(q))
        self.env.assertContains("Node By Index Scan", plan)
        self.env.assertNotContains("Value Hash Join", plan)
        self.env.assertEqual(g.query(q).result_set, [[0, 7], [1, 9]])

        # A third MATCH joins a product that holds the first two, which are
        # lookups by then: it is sought per row too.
        q = ("UNWIND [{s: 0, d: 101, x: 102}] AS e MATCH (a) WHERE id(a) = e.s "
             "MATCH (b) WHERE id(b) = e.d MATCH (c) WHERE id(c) = e.x RETURN id(a), id(b), id(c)")
        plan = str(g.explain(q))
        self.env.assertEqual(plan.count("NodeByIdSeek"), 3)
        self.env.assertNotContains("Value Hash Join", plan)
        self.env.assertEqual(g.query(q).result_set, [[0, 101, 102]])

        q = ("UNWIND [{s: 0, d: 101, v: 7}] AS e MATCH (a) WHERE id(a) = e.s "
             "MATCH (b) WHERE id(b) = e.d MATCH (c:B {v: e.v}) RETURN id(a), id(b), c.v")
        plan = str(g.explain(q))
        self.env.assertEqual(plan.count("NodeByIdSeek"), 2)
        self.env.assertContains("Node By Index Scan", plan)
        self.env.assertNotContains("Value Hash Join", plan)
        self.env.assertEqual(g.query(q).result_set, [[0, 101, 7]])

        g.delete()
