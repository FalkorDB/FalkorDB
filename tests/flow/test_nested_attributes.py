from common import *

GRAPH_ID = "nested_attributes"

class testNestedAttributes():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()
        self.graph = self.db.select_graph(GRAPH_ID)

    # test creation of a nested attribute
    def test01_create_nested_attribute(self):
        # create a node with nested attributes
        # n['a']['b']['c'] = 1
        q = "CREATE (n {a: {b: {c: 1}}}) RETURN n.a"
        res = self.graph.query(q)
        actual = res.result_set[0][0]
        expected = {'b': {'c': 1}}
        self.env.assertEquals(actual, expected)

    def test01_delete_nested_attribute(self):
        # create a node with nested attributes
        # n['a']['b']['c'] = 1
        q = "CREATE (n {a: {b: {c: 1, d: 2}}}) RETURN n.a"
        self.graph.query(q)

        # delete n['a']['b']['c']
        q = "MATCH (n) SET n.a.b.c = NULL"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_removed, 1)

        q = "MATCH (n) RETURN n.a"
        res = self.graph.query(q)
        actual = res.result_set[0][0]
        expected = {'b': { 'd': 2 }}
        self.env.assertEquals(actual, expected)

        # delete n['a']['b']
        q = "MATCH (n) SET n.a.b = NULL"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_removed, 1)

        q = "MATCH (n) RETURN n.a"
        res = self.graph.query(q)
        actual = res.result_set[0][0]
        expected = { }
        self.env.assertEquals(actual, expected)

        # try to delete a none existing nested attribute
        q = "MATCH (n) SET n.a.x.y.z = NULL"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_removed, 0)

    def test02_update_nested_attribute(self):
        # create an empty node and add a nested attribute
        q = "CREATE (n) SET n.a.b.c.d = 4 RETURN n.a"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_set, 1)

        actual = res.result_set[0][0]
        expected = {'b': {'c': {'d': 4}}}
        self.env.assertEquals(actual, expected)

        # extend an existing attribute
        q = "MATCH (n) SET n.a.b.c.e = 5 RETURN n.a"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_set, 1)

        actual = res.result_set[0][0]
        expected = {'b': {'c': {'d': 4, 'e': 5}}}
        self.env.assertEquals(actual, expected)

        # this one is tricky
        # should the result be a node with sub attributes
        # or should it be a node with a single attribute 'a' with a map value?
        q = "MATCH (n) SET n = {a: {b: {c: {d: 6, e: 7}}}} RETURN n.a"
        res = self.graph.query(q)
        actual = res.result_set[0][0]
        expected = {'b': {'c': {'d': 6, 'e': 7}}}
        self.env.assertEquals(actual, expected)

        # set an attribute to its own value
        # no update should be reported
        q = "MATCH (n) SET n.a = n.a RETURN n.a"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_set, 0)

        # try to extend a scalar
        q = "MATCH (n) SET n.a.b.c.d.f = 8 RETURN n"
        try:
            res = self.graph.query(q)
            # TODO: determine if this should fail silantly
            self.env.assertFalse("should not be able to extend a scalar")
        except Exception as e:
            print(e)

    def test03_ownership(self):
        # assign one entity's nested attribute to another
        q = "CREATE ( {a: {b: {c: 'nested'}}})"
        self.graph.query(q)

        q = """MATCH (n)
               CREATE (m), (o)
               SET m.a = n.a, o = n
               DELETE n
               RETURN n.a, m.a, o.a"""

        res = self.graph.query(q)
        n = res.result_set[0][0]
        m = res.result_set[0][1]
        o = res.result_set[0][2]

        self.env.assertEquals(n, m)
        self.env.assertEquals(m, o)

