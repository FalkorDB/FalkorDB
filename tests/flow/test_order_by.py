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
        self.env.assertEqual(actual_result.result_set, expected)

        # Same query with limit, force use heap sort
        q = """MATCH (n:Person) RETURN n.id, n.name ORDER BY n.id DESC, n.name ASC LIMIT 10"""
        actual_result = self.graph.query(q)
        self.env.assertEqual(actual_result.result_set, expected)

    def test02_foreach(self):
        """Tests that ORDER BY works properly with FOREACH before it"""

        res = self.graph.query("CREATE (:N {v: 1}), (:N {v: 2})")
        self.env.assertEqual(res.nodes_created, 2)

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
        self.env.assertEqual(res.result_set[0][0], Node(labels='N', properties={'v': 2}))
        self.env.assertEqual(res.result_set[1][0], Node(labels='N', properties={'v': 1}))

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
        self.env.assertEqual(res.result_set[0][0], Node(labels='N', properties={'v': 1}))
        self.env.assertEqual(res.result_set[1][0], Node(labels='N', properties={'v': 2}))

    def test03_order_by_projected_exp(self):
        """order-by accesses projected alias by its original form"""

        q = """UNWIND [{v:3}, {v:1}, {v:2}] AS element
               WITH element.v AS X
               ORDER BY element.v
               RETURN X"""

        expected = [[1], [2], [3]]

        actual = self.graph.query(q).result_set
        self.env.assertEqual(actual, expected)

    def test04_order_by_alias_prop(self):
        """order-by access projected alias by its original name"""

        q = """UNWIND [{v:3}, {v:1}, {v:2}] AS element
               WITH element AS X
               ORDER BY element.v
               RETURN X"""

        expected = [[{'v':1}], [{'v':2}], [{'v':3}]]

        actual = self.graph.query(q).result_set
        self.env.assertEqual(actual, expected)

        # nest replaced expression within a larger expression
        q = """UNWIND [{v:3}, {v:1}, {v:2}] AS element
               WITH element AS X
               ORDER BY element.v + 12 + element.v
               RETURN X"""

        actual = self.graph.query(q).result_set
        self.env.assertEqual(actual, expected)

        # nest replaced expression within a larger expression
        q = """UNWIND [{v:3}, {v:1}, {v:2}] AS element
               WITH element AS X
               ORDER BY X.v + 12 + element.v
               RETURN X"""

        actual = self.graph.query(q).result_set
        self.env.assertEqual(actual, expected)

        # nest replaced expression within a larger expression
        q = """UNWIND [{v:3}, {v:1}, {v:2}] AS element
               WITH element AS X
               ORDER BY element.v + 12 + X.v
               RETURN X"""

        actual = self.graph.query(q).result_set
        self.env.assertEqual(actual, expected)

        # nest replaced expression within a larger expression
        q = """UNWIND [{v:3}, {v:1}, {v:2}] AS element
               WITH element AS X
               ORDER BY X.v + 12 + X.v
               RETURN X"""

        actual = self.graph.query(q).result_set
        self.env.assertEqual(actual, expected)

        # combine the two types of order-by rewritting
        q = """UNWIND [{v:3, y:3}, {v:1, y:1}, {v:2, y:2}] AS element
               WITH element AS X, element.y AS Y
               ORDER BY toInteger(element.v) + 12 + element.y, 1 + toInteger(element.y) + 2
               RETURN X"""

        expected = [[{'v':1, 'y':1}], [{'v':2, 'y':2}], [{'v':3, 'y':3}]]
        actual = self.graph.query(q).result_set
        self.env.assertEqual(actual, expected)

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
        self.env.assertEqual(actual, expected)

        q = """UNWIND [1, 2] AS X
               UNWIND [1, 2] AS Y
               WITH X
               ORDER BY Y + X + 1, X
               RETURN X"""

        expected = [[1], [1], [2], [2]]
        actual = self.graph.query(q).result_set
        self.env.assertEqual(actual, expected)

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
            self.env.assertContains("ORDER BY cannot reference variables not projected", str(e))

    def test07_order_aggregation(self):
        """computing an aggregation expression within the ORDER-BY clause
           should fail"""

        q = "UNWIND range (0, 9) AS x RETURN x ORDER BY MAX(x)"

        try:
            res = self.graph.query(q).result_set
            self.env.assertTrue(False and "expecting a failure")
        except Exception as e:
            self.env.assertContains("failed to map aggregation expression", str(e))

    def test08_order_variable_length_paths_by_length(self):
        """ordering variable length paths by a non-projected length(p)"""

        # See https://github.com/FalkorDB/FalkorDB/issues/303 - paths were
        # emitted out of order whenever length(p) was only a sort key and not
        # part of the projection.
        g = self.db.select_graph("order_by_path_length")

        g.query("""CREATE (a:City {name: 'A'}), (b:City {name: 'B'}),
                          (c:City {name: 'C'}), (d:City {name: 'D'}),
                          (e:City {name: 'E'}), (f:City {name: 'F'}),
                          (g:City {name: 'G'}),
                          (a)-[:Road]->(b), (a)-[:Road]->(c), (a)-[:Road]->(d),
                          (b)-[:Road]->(e), (b)-[:Road]->(d),
                          (d)-[:Road]->(e), (c)-[:Road]->(f),
                          (d)-[:Road]->(c), (d)-[:Road]->(f),
                          (e)-[:Road]->(g), (f)-[:Road]->(g)""")

        # 8 paths lead from A to G, of lengths 3, 3, 3, 3, 4, 4, 4 and 5
        q = """MATCH p = (:City {name: 'A'})-[*]->(:City {name: 'G'})
               RETURN [n IN nodes(p) | n.name] AS names
               ORDER BY length(p)"""

        # length(p) is one less than the number of nodes on the path, so the
        # projected names let us recompute the sort key the server used
        lengths = [len(row[0]) - 1 for row in g.query(q).result_set]
        self.env.assertEqual(lengths, [3, 3, 3, 3, 4, 4, 4, 5])

        res = g.query(q + " DESC").result_set
        lengths = [len(row[0]) - 1 for row in res]
        self.env.assertEqual(lengths, [5, 4, 4, 4, 3, 3, 3, 3])

        # every path must still start at A and end at G
        for row in res:
            self.env.assertEqual(row[0][0], 'A')
            self.env.assertEqual(row[0][-1], 'G')

