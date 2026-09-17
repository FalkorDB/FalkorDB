from common import *

GRAPH_ID = "LEIDEN"

class testLeiden(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def test_invalid_invocation(self):
        invalid_queries = [
            """CALL algo.leiden({nodeLabels: 'Person'})""",
            """CALL algo.leiden({relationshipTypes: 'KNOWS'})""",
            """CALL algo.leiden({weightAttribute: 4})""",
            """CALL algo.leiden({weightProperty: 4})""",
            """CALL algo.leiden({weightAttribute: 'cost', weightProperty: 'cost'})""",
            """CALL algo.leiden({weightAttribute: 'fake'})""",
            """CALL algo.leiden({invalidParam: 'value'})""",
            """CALL algo.leiden('invalid')""",
            """CALL algo.leiden({nodeLabels: [1, 2, 3]})""",
            """CALL algo.leiden({relationshipTypes: [1, 2, 3]})""",
            """CALL algo.leiden({relationshipTypes: ['FAKE']})""",
            """CALL algo.leiden(null) YIELD node, invalidField""",
            """CALL algo.leiden('arg1', 'arg2') YIELD node""",
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except redis.exceptions.ResponseError:
                pass

    def test_leiden_on_empty_graph(self):
        result = self.graph.query("CALL algo.leiden() YIELD node, communityId")
        self.env.assertEqual(len(result.result_set), 0)

    def test_leiden_basic_with_filters_and_weights(self):
        self.graph.query("""
            CREATE
            (a:Person {name: 'A'}),
            (b:Person {name: 'B'}),
            (c:Person {name: 'C'}),
            (x:Company {name: 'X'}),
            (y:Company {name: 'Y'}),
            (a)-[:KNOWS {cost: 5.0}]->(b),
            (b)-[:KNOWS {cost: 4.0}]->(c),
            (c)-[:KNOWS {cost: 3.0}]->(a),
            (x)-[:WORKS_WITH {cost: 1.0}]->(y)
        """)

        result = self.graph.query("""
            CALL algo.leiden({
                nodeLabels: ['Person'],
                relationshipTypes: ['KNOWS'],
                weightAttribute: 'cost'
            })
            YIELD node, communityId
            RETURN node.name, communityId
            ORDER BY node.name
        """)

        for node_name, community_id in result.result_set:
            self.env.assertContains(node_name, ['A', 'B', 'C'])
            self.env.assertNotEqual(community_id, None)

    def test_leiden_weight_property_alias(self):
        self.graph.query("""
            CREATE
            (a:Person {name: 'A'}),
            (b:Person {name: 'B'}),
            (a)-[:KNOWS {score: 10.0}]->(b)
        """)

        result = self.graph.query("""
            CALL algo.leiden({
                relationshipTypes: ['KNOWS'],
                weightProperty: 'score'
            })
            YIELD node, communityId
            RETURN node.name, communityId
            ORDER BY node.name
        """)

        for node_name, community_id in result.result_set:
            self.env.assertContains(node_name, ['A', 'B'])
            self.env.assertNotEqual(community_id, None)
