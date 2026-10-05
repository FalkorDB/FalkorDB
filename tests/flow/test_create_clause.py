import random
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
        except ResponseError as e:
            # The graph not existing yet is the ordinary case and the one being
            # arranged for. Anything else — a delete that failed against a graph
            # that IS there — would leave the id space non-empty and make the
            # absolute ids below assert against the wrong starting point, so it
            # has to surface rather than be swallowed.
            if "empty key" not in str(e):
                raise
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

    def test19_id_pool_never_reissues_a_live_id(self):
        # The property behind the three cases above, over sequences nobody
        # wrote by hand: after any run of creates and deletes, no two live
        # entities share an id.
        #
        # Seeded, so a failure is reproducible and CI cannot go green or red
        # for reasons nobody can reproduce. It earns its place — the hand-
        # written cases above are all node-shaped, and this is what surfaced
        # that the collisions are mostly on *edges*: against the allocator
        # before the fix it found one in roughly half of these trials within a
        # dozen steps, most of them shapes I would not have thought to write.
        random.seed(1234)

        ops = [
            "CREATE (a), (b)",
            "CREATE (a), (b), (c)",
            "CREATE (a)-[:R]->(b)",
            "CREATE (a)-[:R]->(a)",
            "CREATE (a), (b), (c) DELETE b",
            "CREATE (a), (b), (c) DELETE b CREATE (x), (y)",
            "CREATE (a)-[:R]->(b), (c)-[:R]->(d) DELETE a",
            "CREATE (a)-[:R]->(b), (c)-[:R]->(d) DELETE a CREATE (x)-[:R]->(y)",
            "CREATE (a)-[:R]->(a), (b)-[:R]->(c) DELETE a CREATE (x)-[:R]->(y)",
        ]

        for trial in range(25):
            g = self._fresh()
            history = []
            for _ in range(8):
                q = random.choice(ops)
                history.append(q)
                g.query(q)

                # Delete a live node outright every so often, so committed
                # deletes and pending cancellations interleave.
                if random.random() < 0.25:
                    live = [r[0] for r in
                            g.query("MATCH (n) RETURN id(n)").result_set]
                    if live:
                        victim = random.choice(live)
                        q = f"MATCH (n) WHERE id(n) = {victim} DELETE n"
                        history.append(q)
                        g.query(q)

                trace = "\n  ".join(history)
                nodes = [r[0] for r in
                         g.query("MATCH (n) RETURN id(n)").result_set]
                self.env.assertEqual(
                        len(nodes), len(set(nodes)),
                        message=f"trial {trial}: two live nodes share an id\n"
                                f"  ids: {sorted(nodes)}\n  {trace}")

                edges = [r[0] for r in
                         g.query("MATCH ()-[e]->() RETURN id(e)").result_set]
                self.env.assertEqual(
                        len(edges), len(set(edges)),
                        message=f"trial {trial}: two live edges share an id\n"
                                f"  ids: {sorted(edges)}\n  {trace}")

                # A fused id leaves the graph wrong for every later step, which
                # would report the same collision once per step. Stop this trial
                # at the first one so the output names the sequence that caused
                # it and nothing else.
                if len(nodes) != len(set(nodes)) or len(edges) != len(set(edges)):
                    break
