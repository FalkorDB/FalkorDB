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
                "CREATE (a {v:a.x})",
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

    def test02_edge_reuse(self):
        queries = ["CREATE ()-[e:R]->()-[e:R]->()",
                   "MATCH ()-[e:R]->() CREATE ()-[e:R]->()",
                   "CREATE ()-[e:R]->() CREATE ()-[e:R]->()"]

        for q in queries:
            try:
                self.g.query(q)
                # should not reach this point
                self.env.assertTrue(False)
            except Exception as e:
                self.env.assertTrue("The bound variable 'e' can't be redeclared in a CREATE clause" in str(e))

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


    def _fresh(self):
        # The class graph, emptied first. The tests below assert on absolute
        # ids, so they only mean anything starting from an empty id space —
        # relying on tearDown for that made one of them pass against a build it
        # was written to catch, because the ids started somewhere else.
        try:
            self.g.delete()
        except Exception:
            pass
        return self.g

    def test16_create_after_deleting_a_pending_node(self):
        # Deleting a node this same query created — before it commits — hands
        # its id back to the recycle bin. A later CREATE in the same query must
        # not be pushed on top of a node an earlier clause already made.
        #
        # It used to be: the allocator tracked how many ids were outstanding
        # rather than which, and used that count as a position in the recycle
        # bin. After the delete the count described neither, so the next id
        # landed on the previous clause's last node. This query asked for four
        # nodes and left three, with no error raised.
        g = self._fresh()
        g.query("CREATE (a), (b), (c) DELETE b CREATE (d), (e)")

        # a, c, d, e — b was created and deleted inside the one query.
        res = g.query("MATCH (n) RETURN count(n)").result_set
        self.env.assertEqual(res[0][0], 4)

        # b's id is *not* reused inside the same query, so the live ids skip it
        # and the two later nodes are allocated fresh. Reusing it would put that
        # id in two records of one effects buffer — the cancelled node's own
        # create/delete pair, and the create that reused it — and a replica
        # refuses the whole buffer as "already live".
        res = g.query("MATCH (n) RETURN id(n) ORDER BY id(n)").result_set
        ids = [row[0] for row in res]
        self.env.assertEqual(ids, [0, 2, 3, 4])

    def test17_create_after_deleting_a_pending_node_with_an_edge(self):
        # The same, with an unrelated edge in the pattern, so the relationship
        # allocator runs alongside the node one.
        g = self._fresh()
        g.query("CREATE (a), (b), (c)-[:R]->(z) DELETE b CREATE (d), (e)")

        res = g.query("MATCH (n) RETURN count(n)").result_set
        self.env.assertEqual(res[0][0], 5)

        res = g.query("MATCH (n) RETURN id(n) ORDER BY id(n)").result_set
        self.env.assertEqual([row[0] for row in res], [0, 2, 3, 4, 5])

        res = g.query("MATCH ()-[r]->() RETURN id(r) ORDER BY id(r)").result_set
        self.env.assertEqual([row[0] for row in res], [0])

    def test18_create_after_deleting_a_pending_node_with_a_pending_edge(self):
        # Deleting a pending node cascades to the edges it holds, so this hands
        # back a node id and a relationship id in one query and then asks for
        # both again. Deleting the edge on its own would not: only the cascade
        # out of a node cancels a relationship reservation.
        #
        # The relationship allocator carried the same bug as the node one and
        # showed it more plainly — the two surviving edges came back under one
        # id, so the query returned the same relationship twice.
        g = self._fresh()
        g.query("CREATE (a)-[:R]->(b), (c)-[:R]->(d) DELETE a CREATE (x)-[:R]->(y)")

        # b, c, d, x, y — a was created and deleted inside the one query, and
        # its id is left in the bin rather than reused.
        res = g.query("MATCH (n) RETURN id(n) ORDER BY id(n)").result_set
        self.env.assertEqual([row[0] for row in res], [1, 2, 3, 4, 5])

        # Two edges survive: (c)->(d) and (x)->(y), on two distinct ids. Before
        # the fix both came back as id 1.
        res = g.query("MATCH ()-[r]->() RETURN id(r) ORDER BY id(r)").result_set
        self.env.assertEqual([row[0] for row in res], [1, 2])

        # And they really are two edges, between the endpoints asked for.
        res = g.query(
                "MATCH (s)-[r]->(t) RETURN id(s), id(r), id(t) ORDER BY id(r)"
        ).result_set
        self.env.assertEqual(len(res), 2)
        self.env.assertEqual(len(set(row[1] for row in res)), 2)
