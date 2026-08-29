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
        self.env.assertEquals(len(actual_result.result_set), 1)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEquals(edge_count, 0)

        # Connect a to b with a single edge of type R.
        query = """MATCH (a {v:1}), (b {v:2}) CREATE (a)-[:R {v:1}]->(b)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.relationships_created, 1)

        # Expecting single connections.
        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN count(e)"""
        actual_result = self.graph.query(query)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEquals(edge_count, 1)

        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN ID(e)"""
        actual_result = self.graph.query(query)
        edge_id = actual_result.result_set[0][0]
        self.env.assertEquals(edge_id, 0)

        # Connect a to b with additional edge of type R.
        query = """MATCH (a {v:1}), (b {v:2}) CREATE (a)-[:R {v:2}]->(b)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.relationships_created, 1)

        # Expecting two connections.
        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN count(e)"""
        actual_result = self.graph.query(query)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEquals(edge_count, 2)

        # Variable length path.
        query = """MATCH (a {v:1})-[:R*]->(b {v:2}) RETURN count(b)"""
        actual_result = self.graph.query(query)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEquals(edge_count, 2)

        # Remove first connection.
        query = """MATCH (a {v:1})-[e:R {v:1}]->(b {v:2}) DELETE e"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.relationships_deleted, 1)

        # Expecting single connections.
        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN e.v"""
        actual_result = self.graph.query(query)

        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN ID(e)"""
        actual_result = self.graph.query(query)
        edge_id = actual_result.result_set[0][0]
        self.env.assertEquals(edge_id, 1)

        # Remove second connection.
        query = """MATCH (a {v:1})-[e:R {v:2}]->(b {v:2}) DELETE e"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.relationships_deleted, 1)

        # Expecting no connections.
        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN count(e)"""
        actual_result = self.graph.query(query)        
        self.env.assertEquals(len(actual_result.result_set), 1)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEquals(edge_count, 0)

        # Remove none existing connection.
        query = """MATCH (a {v:1})-[e]->(b {v:2}) DELETE e"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.relationships_deleted, 0)

        # Make sure we can reform connections.
        query = """MATCH (a {v:1}), (b {v:2}) CREATE (a)-[:R {v:3}]->(b)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.relationships_created, 1)

        query = """MATCH (a {v:1})-[e:R]->(b {v:2}) RETURN count(e)"""
        actual_result = self.graph.query(query)
        edge_count = actual_result.result_set[0][0]
        self.env.assertEquals(edge_count, 1)

    # a star aggregation, e.g. count(*), doesn't reference the edge explicitly
    # its result must still account for the multiplicity of parallel edges
    # rather than collapsing them into a single record
    def test_star_aggregation_over_parallel_edges(self):
        g = self.db.select_graph("multi_edge_star")

        # connect a to b with 3 parallel edges of type R
        g.query("""CREATE (a:N {v:1}), (b:N {v:2}),
                          (a)-[:R]->(b), (a)-[:R]->(b), (a)-[:R]->(b)""")

        # every form of count(*) over a named (referenced) edge must account
        # for each of the 3 parallel edges
        queries = [
            "MATCH (a)-[e]->(b) RETURN count(*)",
            "MATCH (a:N)-[e]->(b) RETURN count(*)",      # labeled src, no reduceCount
            "MATCH (a)-[e]->(b) WITH count(*) AS c RETURN c",
            "MATCH (a)-[e]->(b) RETURN count(*) + 0",    # nested star aggregation
        ]
        for query in queries:
            actual_result = g.query(query)
            self.env.assertEquals(actual_result.result_set[0][0], 3)

        # count(e), where the edge is referenced, was already correct
        actual_result = g.query("MATCH (a)-[e]->(b) RETURN count(e)")
        self.env.assertEquals(actual_result.result_set[0][0], 3)

