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

    def test03_order_by_projected_exp(self):
        """order-by accesses projected alias by its original form"""

        q = """UNWIND [{v:3}, {v:1}, {v:2}] AS element
               WITH element.v AS X
               ORDER BY element.v
               RETURN X"""

        expected = [[1], [2], [3]]

        actual = self.graph.query(q).result_set
        self.env.assertEquals(actual, expected)

    def test04_order_by_alias_prop(self):
        """order-by access projected alias by its original name"""

        q = """UNWIND [{v:3}, {v:1}, {v:2}] AS element
               WITH element AS X
               ORDER BY element.v
               RETURN X"""

        expected = [[{'v':1}], [{'v':2}], [{'v':3}]]

        actual = self.graph.query(q).result_set
        self.env.assertEquals(actual, expected)

        # nest replaced expression within a larger expression
        q = """UNWIND [{v:3}, {v:1}, {v:2}] AS element
               WITH element AS X
               ORDER BY element.v + 12 + element.v
               RETURN X"""

        actual = self.graph.query(q).result_set
        self.env.assertEquals(actual, expected)

        # combine the two types of order-by rewritting
        q = """UNWIND [{v:3, y:3}, {v:1, y:1}, {v:2, y:2}] AS element
               WITH element AS X, element.y AS Y
               ORDER BY toInteger(element.v) + 12 + element.y, 1 + toInteger(element.y) + 2
               RETURN X"""

        expected = [[{'v':1, 'y':1}], [{'v':2, 'y':2}], [{'v':3, 'y':3}]]
        actual = self.graph.query(q).result_set
        self.env.assertEquals(actual, expected)

    def test05_order_by_nonprojected(self):
        """order-by reference non-projected variables"""

        # `Y` is implicitly added to the projected clause
        q = """UNWIND [3, 1, 2] AS X
               UNWIND [3, 1, 2] AS Y
               WITH X
               ORDER BY Y, X
               RETURN X"""

        expected = [[1], [2], [3], [1], [2], [3], [1], [2], [3]]
        actual = self.graph.query(q).result_set
        self.env.assertEquals(actual, expected)

        q = """UNWIND [1, 2] AS X
               UNWIND [1, 2] AS Y
               WITH X
               ORDER BY Y + X + 1, X
               RETURN X"""

        expected = [[1], [1], [2], [2]]
        actual = self.graph.query(q).result_set
        self.env.assertEquals(actual, expected)

    def test06_order_by_unallowed(self):
        """order-by can not refer to non-projected variabels in aggregation scope"""

        q = """UNWIND [1, 2, 3] AS X
               WITH count(X) AS cnt
               ORDER BY X
               RETURN cnt"""

        try:
            self.graph.query(q).result_set
            self.env.assertTrue(False and "should fail")
        except Exception as e:
            self.env.assertIn("ORDER BY cannot reference variables not projected", str(e))

