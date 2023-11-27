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

