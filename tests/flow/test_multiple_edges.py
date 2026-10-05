from common import *

GRAPH_ID = "multi_edge"

class testGraphMultipleEdgeFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    # Connect a single node to all other nodes.
    def test_multiple_edges(self):
        # Create graph with no edges.
        query = """CREATE (a {v:1}), (b {v:2})"""
        actual_result = self.graph.query(query)

        # Expecting no connections.
        query = """MATCH (a {v:1})-[e]->(b {v:2}) RETURN count(e)"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(len(actual_result.result_set), 1)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEqual(edge_count, 0)

        # Connect a to b with a single edge of type R.
        query = """MATCH (a {v:1}), (b {v:2}) CREATE (a)-[:R {v:1}]->(b)"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.relationships_created, 1)

        # Expecting single connections.
        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN count(e)"""
        actual_result = self.graph.query(query)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEqual(edge_count, 1)

        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN ID(e)"""
        actual_result = self.graph.query(query)
        edge_id = actual_result.result_set[0][0]
        self.env.assertEqual(edge_id, 0)

        # Connect a to b with additional edge of type R.
        query = """MATCH (a {v:1}), (b {v:2}) CREATE (a)-[:R {v:2}]->(b)"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.relationships_created, 1)

        # Expecting two connections.
        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN count(e)"""
        actual_result = self.graph.query(query)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEqual(edge_count, 2)

        # Variable length path.
        query = """MATCH (a {v:1})-[:R*]->(b {v:2}) RETURN count(b)"""
        actual_result = self.graph.query(query)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEqual(edge_count, 2)

        # Remove first connection.
        query = """MATCH (a {v:1})-[e:R {v:1}]->(b {v:2}) DELETE e"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.relationships_deleted, 1)

        # Expecting single connections.
        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN e.v"""
        actual_result = self.graph.query(query)

        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN ID(e)"""
        actual_result = self.graph.query(query)
        edge_id = actual_result.result_set[0][0]
        self.env.assertEqual(edge_id, 1)

        # Remove second connection.
        query = """MATCH (a {v:1})-[e:R {v:2}]->(b {v:2}) DELETE e"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.relationships_deleted, 1)

        # Expecting no connections.
        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN count(e)"""
        actual_result = self.graph.query(query)        
        self.env.assertEqual(len(actual_result.result_set), 1)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEqual(edge_count, 0)

        # Remove none existing connection.
        query = """MATCH (a {v:1})-[e]->(b {v:2}) DELETE e"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.relationships_deleted, 0)

        # Make sure we can reform connections.
        query = """MATCH (a {v:1}), (b {v:2}) CREATE (a)-[:R {v:3}]->(b)"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.relationships_created, 1)

        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN count(e)"""
        actual_result = self.graph.query(query)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEqual(edge_count, 1)


    # Each of these reads `e` only inside an operator's own expression: the list
    # UNWIND or FOREACH iterates, or MERGE's pattern. When that read was missed,
    # nothing seemed to need the individual edges, the parallel pair collapsed to
    # one representative, and each saw one edge instead of two.
    def test_parallel_edges_read_inside_unwind_foreach_merge(self):
        g = self.db.select_graph(GRAPH_ID + "_readers")
        g.query("CREATE (a:A)-[:R {w: 1}]->(b:B), (a)-[:R {w: 2}]->(b)")

        actual = g.query("MATCH (a:A)-[e:R]->(b) UNWIND [e] AS x RETURN x.w ORDER BY x.w")
        self.env.assertEqual(actual.result_set, [[1], [2]])

        g.query("MATCH (a:A)-[e:R]->(b) FOREACH (x IN [e] | SET x.seen = 1)")
        actual = g.query("MATCH ()-[e:R]->() WHERE e.seen = 1 RETURN count(e)")
        self.env.assertEqual(actual.result_set, [[2]])

        g.query("MATCH (a:A)-[e:R]->(b) MERGE (:X {w: e.w})")
        actual = g.query("MATCH (x:X) RETURN x.w ORDER BY x.w")
        self.env.assertEqual(actual.result_set, [[1], [2]])
        g.delete()
