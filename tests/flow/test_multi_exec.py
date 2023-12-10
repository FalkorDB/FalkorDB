from common import *

GRAPH_ID = "multiexec-graph"

# Fully connected 3 nodes graph,
CREATE_QUERY = """CREATE (al:person {name:'Al'}), (betty:person {name:'Betty'}), (michael:person {name:'Michael'}),
                    (al)-[:knows]->(betty), (al)-[:knows]->(michael),
                    (betty)-[:knows]->(al), (betty)-[:knows]->(michael),
                    (michael)-[:knows]->(al), (michael)-[:knows]->(betty)"""
CREATE_QUERY = CREATE_QUERY.replace("\n", "")

# Count outgoing connections from Al.
MATCH_QUERY = """MATCH (al:person {name:'Al'})-[]->(b:person) RETURN al.name, count(b)"""

# Disconnect edge connecting Al to Betty.
DEL_QUERY = """MATCH (al:person {name:'Al'})-[e:knows]->(b:person {name:'Betty'}) DELETE e"""

# Change Al name from Al to Steve.
UPDATE_QUERY = "MATCH (al:person {name:'Al'}) SET al.name = 'Steve'"


class testMultiExecFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()

    def test_graph_entities(self):
        # Delete previous graph if exists.
        self.redis_con.execute_command("DEL", GRAPH_ID)

        # Start a multi exec transaction.
        self.redis_con.execute_command("MULTI")

        # Create graph.
        self.redis_con.execute_command("GRAPH.QUERY", GRAPH_ID, CREATE_QUERY)

        # Count outgoing connections from Al, expecting 2 edges.
        # (Al)-[e]->() count (e)
        self.redis_con.execute_command("GRAPH.QUERY", GRAPH_ID, MATCH_QUERY)

        # Disconnect edge connecting Al to Betty.
        # (Al)-[e]->(Betty) delete (e)
        self.redis_con.execute_command("GRAPH.QUERY", GRAPH_ID, DEL_QUERY)

        # Count outgoing connections from Al, expecting 1 edges.
        # (Al)-[e]->() count (e)
        self.redis_con.execute_command("GRAPH.QUERY", GRAPH_ID, MATCH_QUERY)

        # Change Al name from Al to Steve.
        # (Al) set Al.name = Steve
        self.redis_con.execute_command("GRAPH.QUERY", GRAPH_ID, UPDATE_QUERY)

        # Count outgoing connections from Al, expecting 0 edges.
        # (Al)-[e]->() count (e)
        self.redis_con.execute_command("GRAPH.QUERY", GRAPH_ID, MATCH_QUERY)

        # Commit transaction.
        results = self.redis_con.execute_command("EXEC")

        # [
        #   [
        #       ['al.name', 'count(b)'],
        #       ['Al', '2']
        #   ],
        #       ['Query internal execution time: 0.143000 milliseconds']
        # ]

        two_edges = results[1]
        two_edges = two_edges[1][0][1]        
        self.env.assertEquals(two_edges, 2)

        one_edge = results[3]
        one_edge = one_edge[1][0][1]
        self.env.assertEquals(one_edge, 1)

        no_edges = results[5]
        no_edges = no_edges[1]
        self.env.assertEquals(len(no_edges), 0)

    def test_transaction_failure(self):
        redis_con_a = self.env.getConnection()
        redis_con_b = self.env.getConnection()
        results = redis_con_b.execute_command("INFO", "CLIENTS")
        self.env.assertGreaterEqual(results['connected_clients'], 2)

        # Delete previous graph if exists.
        redis_con_a.execute_command("DEL", GRAPH_ID)
        redis_con_a.execute_command("GRAPH.QUERY", GRAPH_ID, CREATE_QUERY)

        redis_con_a.execute_command("WATCH", GRAPH_ID)
        redis_con_a.execute_command("MULTI")
        redis_con_a.execute_command("GRAPH.QUERY", GRAPH_ID, MATCH_QUERY)
        results = redis_con_a.execute_command("EXEC")

        self.env.assertNotEqual(results, None)

        # read only query from client B - transaction OK
        redis_con_a.execute_command("WATCH", GRAPH_ID)
        redis_con_a.execute_command("MULTI")
        redis_con_b.execute_command("GRAPH.QUERY", GRAPH_ID, MATCH_QUERY)
        redis_con_a.execute_command("GRAPH.QUERY", GRAPH_ID, MATCH_QUERY)
        results = redis_con_a.execute_command("EXEC")

        self.env.assertNotEqual(results, None)

        # write query from client B - transaction fails
        redis_con_a.execute_command("WATCH", GRAPH_ID)
        redis_con_a.execute_command("MULTI")
        redis_con_b.execute_command("GRAPH.QUERY", GRAPH_ID, UPDATE_QUERY)
        redis_con_a.execute_command("GRAPH.QUERY", GRAPH_ID, MATCH_QUERY)
        results = redis_con_a.execute_command("EXEC")

        self.env.assertEqual(results, None)

        # GRAPH.EXPLAIN is read only - no data change - transaction OK
        redis_con_a.execute_command("WATCH", GRAPH_ID)
        redis_con_a.execute_command("MULTI")
        redis_con_b.execute_command("GRAPH.EXPLAIN", GRAPH_ID, UPDATE_QUERY)
        redis_con_a.execute_command("GRAPH.QUERY", GRAPH_ID, MATCH_QUERY)
        results = redis_con_a.execute_command("EXEC")

        self.env.assertNotEqual(results, None)
