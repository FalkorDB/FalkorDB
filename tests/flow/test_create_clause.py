from common import *
from index_utils import *

GRAPH_ID = "create-clause"

class testCreateClause():
    def __init__(self):
        self.env, self.db = Env()
        self.g = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.g.delete()
    
    def test01_create_dependency(self):
        # create clauses where one entity depends on another
        # e.g. CREATE (a)-[e:R {v:1}]->(b), (z {v:e.v+2})
        # are not allowed
        # the solution to the above requires introduction of an additional
        # create clause:
        # CREATE (a)-[e:R {v:1}]->(b) CREATE (z {v:e.v+2})

        # make sure an error is raised when there's dependency between
        # new entities within the same clause

        queries = [
                "CREATE (a {v:1}), (z {v:a.v+2})",
                "CREATE (z {v:a.v+2}), (a {v:1})",
                "CREATE (z {v:a.v}), (a {v:z.v})",
                "CREATE (a)-[e:R {v:1}]->(b), (z {v:e.v+2})",
                "CREATE (z {v:e.v+2}), (a)-[e:R {v:1}]->(b)",
                "CREATE (a)-[e:R {v:z.v+1}]->(b), (z {v:2})",
                "CREATE (z {v:2}), (a)-[e:R {v:z.v+1}]->(b)",
                "CREATE ()-[e:R{v:1}]->()-[z:R{v:e.v+1}]->()",
                "CREATE ()-[e:R{v:z.v+1}]->()-[z:R{v:1}]->()",
                "CREATE ()-[e:R{v:z.v}]->()-[z:R{v:e.v}]->()"]

        for q in queries:
            try:
                self.g.query(q)
                # should not reach this point
                self.env.assertTrue(False)
            except Exception as e:
                self.env.assertTrue("not defined" in str(e))

        # following query is going to produce a run-time error
        q = "CREATE (a {v:a.x})"
        try:
            self.g.query(q)
            # should not reach this point
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertTrue("Attempted to access undefined attribute" in str(e))

    def test02_edge_reuse(self):
        # bound edges can not be used in a CREATE clause
        q = "CREATE ()-[e:R]->()-[e:R]->()"
        try:
            self.g.query(q)
            # should not reach this point
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertTrue("The bound edge 'e' can't be redeclared in a CREATE clause" in str(e))

        queries = ["MATCH ()-[e:R]->() CREATE ()-[e:R]->()",
                   "CREATE ()-[e:R]->() CREATE ()-[e:R]->()"]

        for q in queries:
            try:
                self.g.query(q)
                # should not reach this point
                self.env.assertTrue(False)
            except Exception as e:
                self.env.assertTrue("The bound edge 'e' can't be redeclared in a CREATE clause" in str(e))

    def test_03_edge_attributes(self):
        """make sure attribute-sets do not get swapped"""

        # low ids: 0, 1
        q = "CREATE (:A), (:B)"
        self.g.query(q)

        # high ids: 2, 3
        q = "CREATE (:C), (:D)"
        self.g.query(q)

        # create (:A)-[{v:2}]->(:B), (:C)-[{v:1}]->(:D)
        q = """MATCH (a:A), (b:B), (c:C), (d:D)
               WITH [[c,d,1],[a,b,2]] AS tuples
               UNWIND tuples AS tuple
               WITH tuple[0] as src, tuple[1] as dest, tuple[2] as val
               CREATE (src)-[:R{v:val}]->(dest)"""
        self.g.query(q)

        q = """MATCH (src)-[e]->(dest)
               RETURN labels(src)[0], e.v, labels(dest)[0]
               ORDER BY e.v"""
        res = self.g.query(q).result_set

        self.env.assertEqual(len(res), 2)

        s = res[0][0]
        v = res[0][1]
        d = res[0][2]
        self.env.assertEqual(s, "C")
        self.env.assertEqual(v, 1)
        self.env.assertEqual(d, "D")

        s = res[1][0]
        v = res[1][1]
        d = res[1][2]
        self.env.assertEqual(s, "A")
        self.env.assertEqual(v, 2)
        self.env.assertEqual(d, "B")

