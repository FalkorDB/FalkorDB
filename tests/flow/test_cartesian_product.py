from common import *

GRAPH_ID = "cartesian_product"

class testCartesianProduct():
    def __init__(self):
        self.env = Env(decodeResponses=True)
        self.con = self.env.getConnection()
        self.g = Graph(self.con, GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        # create a graph with 3 A nodes, 2 B nodes and a single C node
        q = """CREATE (:A {v:'a1'}), (:A {v:'a2'}), (:A {v:'a3'}),
                      (:B {v:'b1'}), (:B {v:'b2'}),
                      (:C {v:'c1'})"""

        self.g.query(q)

    def test01_permute_two_branches(self):
        # permute all A nodes with all B nodes
        q = "MATCH (a:A), (b:B) RETURN a.v, b.v ORDER BY a.v, b.v"
        res = self.g.query(q).result_set

        expected = [['a1', 'b1'], ['a1', 'b2'],
                    ['a2', 'b1'], ['a2', 'b2'],
                    ['a3', 'b1'], ['a3', 'b2']]

        self.env.assertEquals(res, expected)

    def test02_permute_three_branches(self):
        # permute all A, B and C nodes
        q = """MATCH (a:A), (b:B), (c:C)
               RETURN a.v, b.v, c.v
               ORDER BY a.v, b.v"""
        res = self.g.query(q).result_set

        expected = [['a1', 'b1', 'c1'], ['a1', 'b2', 'c1'],
                    ['a2', 'b1', 'c1'], ['a2', 'b2', 'c1'],
                    ['a3', 'b1', 'c1'], ['a3', 'b2', 'c1']]

        self.env.assertEquals(res, expected)

    def _test03_reset_bound_branch(self):
        # at the moment the bound branch get implented as the first branch
        # in the cartesian product operation, this needs to change!
        # the bound branch should be the left hand side branch in an Apply op
        # and the cartesian product should be the right hand side branch of Apply
        #
        # this test make sure resetting the bound branch, in this case a
        # CREATE and DELETE operation, does not affect the cartesian product

        q = """CREATE (d:D)
               DELETE d
               WITH *
               MATCH (a:A), (b:B)
               RETURN a.v, b.v
               ORDER BY a.v, b.v"""

        res = self.g.query(q).result_set

        expected = [['a1', 'b1'], ['a1', 'b2'],
                    ['a2', 'b1'], ['a2', 'b2'],
                    ['a3', 'b1'], ['a3', 'b2']]

        self.env.assertEquals(res, expected)

