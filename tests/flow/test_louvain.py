from common import *

GRAPH_ID = "LOUVAIN"

class testLouvain(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def test_invalid_invocation(self):
        invalid_queries = [
            """CALL algo.louvain({nodeLabels: 'Person'})""",
            """CALL algo.louvain({relationshipTypes: 'KNOWS'})""",
            """CALL algo.louvain({invalidParam: 'value'})""",
            """CALL algo.louvain('invalid')""",
            """CALL algo.louvain({nodeLabels: [1, 2, 3]})""",
            """CALL algo.louvain({relationshipTypes: [1, 2, 3]})""",
            """CALL algo.louvain({relationshipTypes: ['FAKE']})""",
            """CALL algo.louvain(null) YIELD node, invalidField""",
            """CALL algo.louvain('arg1', 'arg2') YIELD node""",
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except redis.exceptions.ResponseError:
                pass

    def test_louvain_on_empty_graph(self):
        result = self.graph.query("CALL algo.louvain() YIELD node, communityId")
        self.env.assertEqual(len(result.result_set), 0)

    def test_louvain_basic_with_filters(self):
        self.graph.query("""
            CREATE
            (a:Person {name: 'A'}),
            (b:Person {name: 'B'}),
            (c:Person {name: 'C'}),
            (x:Company {name: 'X'}),
            (y:Company {name: 'Y'}),
            (a)-[:KNOWS]->(b),
            (b)-[:KNOWS]->(c),
            (c)-[:KNOWS]->(a),
            (x)-[:WORKS_WITH]->(y)
        """)

        result = self.graph.query("""
            CALL algo.louvain({
                nodeLabels: ['Person'],
                relationshipTypes: ['KNOWS']
            })
            YIELD node, communityId
            RETURN node.name, communityId
            ORDER BY node.name
        """)

        # only the 'Person' nodes matching 'KNOWS' should be returned,
        # each with some community id; the clustering itself is validated
        # by LAGraph's own test suite, not here
        names = [row[0] for row in result.result_set]
        self.env.assertEqual(sorted(names), ['A', 'B', 'C'])
        for node_name, community_id in result.result_set:
            self.env.assertNotEqual(community_id, None)

    def test_louvain_no_filters(self):
        self.graph.query("""
            CREATE
            (a:Person {name: 'A'}),
            (b:Person {name: 'B'}),
            (a)-[:KNOWS]->(b)
        """)

        result = self.graph.query("""
            CALL algo.louvain()
            YIELD node, communityId
            RETURN node.name, communityId
            ORDER BY node.name
        """)

        names = [row[0] for row in result.result_set]
        self.env.assertEqual(sorted(names), ['A', 'B'])
        for node_name, community_id in result.result_set:
            self.env.assertNotEqual(community_id, None)

    def test_louvain_yield_subset(self):
        self.graph.query("""
            CREATE
            (a:Person {name: 'A'}),
            (b:Person {name: 'B'}),
            (a)-[:KNOWS]->(b)
        """)

        # yield only communityId
        result = self.graph.query("""
            CALL algo.louvain() YIELD communityId
            RETURN communityId
        """)
        self.env.assertEqual(len(result.result_set), 2)
        for row in result.result_set:
            self.env.assertEqual(len(row), 1)

        # yield only node
        result = self.graph.query("""
            CALL algo.louvain() YIELD node
            RETURN node.name
            ORDER BY node.name
        """)
        names = [row[0] for row in result.result_set]
        self.env.assertEqual(sorted(names), ['A', 'B'])
