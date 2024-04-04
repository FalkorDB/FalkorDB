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

