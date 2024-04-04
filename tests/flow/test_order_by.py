from common import *

GRAPH_ID = "order_by_test"

class testOrderBy(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        self.graph.query("""CREATE (:Person {id: 622, name: 'Mo'}),
                                   (:Person {id: 819, name: 'Bing'}),
                                   (:Person {id: 819, name: 'Qiu'})""")

    def test01_multiple_order_by(self):
        # Query with multiple order by operation
        q = """MATCH (n:Person) RETURN n.id, n.name ORDER BY n.id DESC, n.name ASC"""
        expected = [[819, "Bing"], [819, "Qiu"], [622, "Mo"]]
        actual_result = self.graph.query(q)
        self.env.assertEquals(actual_result.result_set, expected)

        # Same query with limit, force use heap sort
        q = """MATCH (n:Person) RETURN n.id, n.name ORDER BY n.id DESC, n.name ASC LIMIT 10"""
        actual_result = self.graph.query(q)
        self.env.assertEquals(actual_result.result_set, expected)

    def test02_foreach(self):
        """Tests that ORDER BY works properly with FOREACH before it"""

        res = self.graph.query("CREATE (:N {v: 1}), (:N {v: 2})")
        self.env.assertEquals(res.nodes_created, 2)

        res = self.graph.query(
            """
            MATCH (n:N)
            FOREACH(node in [n] |
                SET n.v = n.v
            )
            RETURN n
            ORDER BY n.v DESC
            """
        )

        # assert the order of the results
        self.env.assertEquals(res.result_set[0][0], Node(labels='N', properties={'v': 2}))
        self.env.assertEquals(res.result_set[1][0], Node(labels='N', properties={'v': 1}))

        res = self.graph.query(
            """
            MATCH (n:N)
            FOREACH(node in [n] |
                SET n.v = n.v
            )
            RETURN n
            ORDER BY n.v ASC
            """
        )

        # assert the order of the results
        self.env.assertEquals(res.result_set[0][0], Node(labels='N', properties={'v': 1}))
        self.env.assertEquals(res.result_set[1][0], Node(labels='N', properties={'v': 2}))
