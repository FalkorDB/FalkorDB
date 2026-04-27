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

    # validate count(*) over parallel edges of different relationship types
    # between the same pair of endpoints
    def test_count_parallel_edges_of_different_types(self):
        graph = self.db.select_graph("count_parallel_edges_of_different_types")

        # build a small graph:
        # (Alice)-[:FRIENDS_WITH]->(Bob)
        # (Alice)-[:WORKS_WITH]->(Bob)
        graph.query("""CREATE (a:Person {name: 'Alice'}),
                              (b:Person {name: 'Bob'}),
                              (a)-[:FRIENDS_WITH]->(b),
                              (a)-[:WORKS_WITH]->(b)""")

        # count(*) without an edge alias must count both matching edges
        # for multiple relationship types between the same endpoints
        query = """MATCH (p:Person {name: 'Alice'})
                   MATCH (p)-[:FRIENDS_WITH|WORKS_WITH]->(friend)
                   WITH friend, count(*) AS connectionCount
                   RETURN friend.name AS friendName, connectionCount
                   ORDER BY friendName"""
        actual_result = graph.query(query)
        self.env.assertEquals(actual_result.result_set, [['Bob', 2]])

        # the same query, but counting rows directly
        query = """MATCH (:Person {name: 'Alice'})-[:FRIENDS_WITH|WORKS_WITH]->(b)
                   RETURN count(*)"""
        actual_result = graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[2]])

        # add a second WORKS_WITH edge, exercising the multi-edge path
        # within the multi-reltype pattern
        graph.query("""MATCH (a:Person {name: 'Alice'}),
                            (b:Person {name: 'Bob'})
                       CREATE (a)-[:WORKS_WITH]->(b)""")

        # we now expect 3 matching edges: 1 FRIENDS_WITH + 2 WORKS_WITH
        query = """MATCH (:Person {name: 'Alice'})-[:FRIENDS_WITH|WORKS_WITH]->(b)
                   RETURN count(*)"""
        actual_result = graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[3]])

        # bidirectional variant exercises the undirected branch of the
        # algebraic expression construction; with Alice anchored as one
        # endpoint and 3 outgoing edges (and 0 incoming), every matching
        # edge must be counted (3), not collapsed by the boolean ADD
        query = """MATCH (:Person {name: 'Alice'})-[:FRIENDS_WITH|WORKS_WITH]-(b)
                   RETURN count(*)"""
        actual_result = graph.query(query)
        self.env.assertEquals(actual_result.result_set, [[3]])

