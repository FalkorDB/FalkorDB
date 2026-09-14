"""Effects v3 -- the record shapes the writer partitions and the reader reassembles, including the indexes a query never named.

See `effects_v3_common.py` for the shared fixture and why these are split.
"""



from common import *
from graph_utils import graph_eq
from index_utils import (create_edge_range_index, create_node_fulltext_index, create_node_range_index, wait_for_indices_to_sync)

from effects_v3_common import MONITOR_MARK_KEY, _EffectsV3Base


class testEffectsV3_04_Shapes(_EffectsV3Base):
    """The record shapes the v3 writer has to partition and the reader has to
    reassemble: multiple labels, differing property shapes in one query,
    points, lists, large batches, deletes interleaved with creates, and ids
    recycled by a delete-then-recreate.
    """

    GRAPH_ID = "effects_v3_shapes"

    def __init__(self):
        self._setup()

    def test01_multi_label_nodes(self):
        # Three label sets in one CREATE is three record partitions: the writer
        # groups by label set, so this is the smallest query that exercises
        # partitioning at all.
        res = self.query_and_sync("CREATE (:A), (:B), (:A:B)")
        self.env.assertEqual(res.nodes_created, 3)
        self.env.assertEqual(res.labels_added, 2)

        self.assert_agree("MATCH (n) RETURN count(n)", [[3]])
        self.assert_agree("MATCH (n:A) RETURN count(n)", [[2]])
        self.assert_agree("MATCH (n:B) RETURN count(n)", [[2]])
        self.assert_agree("MATCH (n:A:B) RETURN count(n)", [[1]])
        self.assert_agree(
            "CALL db.labels() YIELD label RETURN label ORDER BY label",
            [['A'], ['B']])

        # a label added after the fact, and one removed
        self.query_and_sync("MATCH (n:A) WHERE NOT n:B SET n:C:D")
        self.assert_agree("MATCH (n:C:D) RETURN count(n)", [[1]])
        self.query_and_sync("MATCH (n:C) REMOVE n:C")
        self.assert_agree("MATCH (n:C) RETURN count(n)", [[0]])
        self.assert_agree("MATCH (n:D) RETURN count(n)", [[1]])

        self.assert_graph_eq()

    def test02_differing_property_shapes_in_one_query(self):
        # Same label, four different property shapes: another four partitions,
        # this time keyed on the attribute set rather than the label set.
        res = self.query_and_sync(
            "CREATE (:S {a: 1}), (:S {a: 1, b: 2}), (:S {b: 2, c: 3}), (:S)")
        self.env.assertEqual(res.nodes_created, 4)

        m, r = self.probe("MATCH (n:S) RETURN keys(n) ORDER BY keys(n)")
        self.env.assertEqual(m, [[[]], [['a']], [['a', 'b']], [['b', 'c']]])
        self.env.assertEqual(r, m)
        self.assert_agree("MATCH (n:S) RETURN sum(n.a), sum(n.b), sum(n.c)",
                          [[2, 4, 3]])

        # and the same for edges
        res = self.query_and_sync(
            "CREATE ()-[:E {x: 1}]->(), ()-[:E]->(), ()-[:F {y: 's', z: [1, 2]}]->()")
        self.env.assertEqual(res.relationships_created, 3)
        m, r = self.probe(
            "MATCH ()-[e]->() RETURN type(e), keys(e) ORDER BY type(e), keys(e)")
        self.env.assertEqual(m, [['E', []], ['E', ['x']], ['F', ['y', 'z']]])
        self.env.assertEqual(r, m)

        self.assert_graph_eq()

    def test03_points_and_lists(self):
        self.query_and_sync("""CREATE (:Geo {
                                    p: point({latitude: 32.07, longitude: 34.79}),
                                    l: [1, 'two', [3.5], true],
                                    empty: []
                               })""")
        self.query_and_sync("""CREATE ()-[:GEO {
                                    p: point({latitude: -12.5, longitude: 0.0}),
                                    l: ['a', [1, 2], 3]
                               }]->()""")

        # Nested lists and points are structured values; comparing the rendered
        # value on both sides is what pins that the wire carried the structure
        # rather than something that merely counts the same.
        m, r = self.probe("MATCH (n:Geo) RETURN n.p, n.l, n.empty")
        self.env.assertEqual(len(m), 1)
        self.env.assertEqual(r, m)
        m, r = self.probe("MATCH ()-[e:GEO]->() RETURN e.p, e.l")
        self.env.assertEqual(len(m), 1)
        self.env.assertEqual(r, m)

        self.assert_agree("MATCH (n:Geo) RETURN size(n.l), size(n.empty)", [[4, 0]])
        self.assert_agree(
            "MATCH (n:Geo) RETURN n.p.latitude > 32.0 AND n.p.longitude > 34.0",
            [[True]])

        self.assert_graph_eq()

    def test04_large_batch(self):
        # ~100k nodes in one query: one record with 100k ids, which is where
        # the id-list encodings and the block sizing actually get exercised.
        # Aggregate probes rather than graph_eq — comparing 100k whole nodes
        # twice measures the client, not the wire.
        N = 100_000
        res = self.query_and_sync(
            f"UNWIND range(1, {N}) AS x CREATE (:Bulk {{v: x, s: 's' + x}})")
        self.env.assertEqual(res.nodes_created, N)

        self.assert_agree("MATCH (n:Bulk) RETURN count(n)", [[N]])
        self.assert_agree(
            "MATCH (n:Bulk) RETURN sum(n.v), min(n.v), max(n.v)",
            [[N * (N + 1) // 2, 1, N]])
        # a spot check that the values landed against the right entities, not
        # merely that the right number of them arrived
        self.assert_agree(
            "MATCH (n:Bulk) WHERE n.s = 's50000' RETURN n.v", [[50000]])
        self.assert_agree(
            "MATCH (n:Bulk) WHERE n.s <> 's' + n.v RETURN count(n)", [[0]])

        # a large batch of edges over those nodes
        E = 20_000
        res = self.query_and_sync(
            f"""MATCH (a:Bulk), (b:Bulk)
                WHERE a.v <= {E} AND b.v = a.v + 1
                CREATE (a)-[:NEXT {{w: a.v}}]->(b)""")
        self.env.assertEqual(res.relationships_created, E)
        self.assert_agree("MATCH ()-[e:NEXT]->() RETURN count(e), sum(e.w)",
                          [[E, E * (E + 1) // 2]])

    def test05_deletes_interleaved_with_creates(self):
        # One query that deletes and creates, so a single effects buffer holds
        # both a delete record and a create record.
        res = self.query_and_sync(
            """MATCH (n:Bulk) WHERE n.v % 2 = 0 DELETE n
               WITH count(1) AS x
               UNWIND range(1, 100) AS i CREATE (:Fresh {i: i})""")
        self.env.assertEqual(res.nodes_deleted, 50_000)
        self.env.assertEqual(res.nodes_created, 100)

        self.assert_agree("MATCH (n:Bulk) RETURN count(n)", [[50_000]])
        self.assert_agree("MATCH (n:Bulk) WHERE n.v % 2 = 0 RETURN count(n)", [[0]])
        self.assert_agree("MATCH (n:Fresh) RETURN count(n), sum(n.i)", [[100, 5050]])
        # the edges hanging off the deleted nodes went with them, on both sides
        self.assert_agree("MATCH ()-[e:NEXT]->() RETURN count(e)", [[0]])

        # and an edge delete in the same shape
        self.query_and_sync(
            """MATCH (a:Fresh {i: 1}), (b:Fresh {i: 2})
               CREATE (a)-[:TMP]->(b), (b)-[:TMP]->(a)""")
        res = self.query_and_sync(
            """MATCH ()-[e:TMP]->() DELETE e
               WITH count(1) AS x
               CREATE (:AfterEdgeDelete)""")
        self.env.assertEqual(res.relationships_deleted, 2)
        self.assert_agree("MATCH ()-[e:TMP]->() RETURN count(e)", [[0]])
        self.assert_agree("MATCH (n:AfterEdgeDelete) RETURN count(n)", [[1]])

    def test06_delete_then_recreate_recycles_ids(self):
        # A create record after a delete hands out ids the delete freed. v3
        # refuses a node id the replica cannot legitimately hold, so a recycled
        # id is exactly the case where that check has to agree with the
        # primary's allocator rather than fight it.
        before = self.master_graph.ro_query(
            "MATCH (n:Fresh) RETURN min(ID(n)), max(ID(n))").result_set[0]

        self.query_and_sync("MATCH (n:Fresh) DELETE n")
        self.assert_agree("MATCH (n:Fresh) RETURN count(n)", [[0]])

        res = self.query_and_sync(
            "UNWIND range(1, 100) AS i CREATE (:Recycled {i: i})")
        self.env.assertEqual(res.nodes_created, 100)

        self.assert_agree("MATCH (n:Recycled) RETURN count(n), sum(n.i)",
                          [[100, 5050]])
        # the ids really were reused — the new nodes sit inside the range the
        # deleted ones occupied, so this is not just "100 nodes arrived"
        m, r = self.probe("MATCH (n:Recycled) RETURN min(ID(n)), max(ID(n))")
        self.env.assertEqual(r, m)
        self.env.assertLessEqual(m[0][0], before[1])

        # round-trip it again, this time deleting and recreating in one query
        res = self.query_and_sync(
            """MATCH (n:Recycled) DELETE n
               WITH count(1) AS x
               UNWIND range(1, 100) AS i CREATE (:Recycled2 {i: i})""")
        self.env.assertEqual(res.nodes_deleted, 100)
        self.env.assertEqual(res.nodes_created, 100)
        self.assert_agree("MATCH (n:Recycled) RETURN count(n)", [[0]])
        self.assert_agree("MATCH (n:Recycled2) RETURN count(n), sum(n.i)",
                          [[100, 5050]])

    def test06b_create_delete_and_recreate_one_id_in_one_buffer(self):
        # The case test06 does *not* reach. A query that commits three times
        # puts all three commits in ONE effects buffer, and the allocator
        # recycles a freed id across commits — so the buffer creates an id,
        # deletes it, and creates it again. The replica has to accept the third
        # record: the delete released the id.
        #
        # It needs a graph whose recycle bin is empty, because `reserve_node`
        # hands out the *smallest* freed id. On a graph with older free ids the
        # third commit gets one of those instead and the collision never
        # happens — which is exactly why the shared-graph tests miss this.
        fresh = "effects_v3_recycle_one_buffer"
        m = Graph(self.master, fresh)
        r = Graph(self.replica, fresh)

        res = m.query("CREATE (n:Doomed) WITH n DELETE n "
                      "WITH 1 AS z CREATE (:Reborn)")
        self.env.assertEqual(res.nodes_created, 2)
        self.env.assertEqual(res.nodes_deleted, 1)
        self.wait_for_replica_offset()

        # The primary kept exactly one node, on the recycled id 0.
        self.env.assertEqual(
            m.ro_query("MATCH (n) RETURN count(n), labels(n)[0], ID(n)").result_set,
            [[1, "Reborn", 0]])
        # And the replica must hold the same. Refusing the recreate discards the
        # whole payload, so the failure here is an empty graph, not a wrong one.
        self.env.assertEqual(
            r.ro_query("MATCH (n) RETURN count(n), labels(n)[0], ID(n)").result_set,
            m.ro_query("MATCH (n) RETURN count(n), labels(n)[0], ID(n)").result_set)
        self.env.assertTrue(graph_eq(m, r))

    def test07_everything_still_agrees(self):
        # A whole-graph comparison once the shapes above have all been applied.
        # Cheap now: test05 removed the bulk of the nodes.
        self.query_and_sync("MATCH (n:Bulk) DELETE n")
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)
        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# 4b. every value tag, over a live wire
#-----------------------------------------------------------------------------


class testEffectsV3_04c_HarderShapes(_EffectsV3Base):
    """Shapes with structure the writer's partitioning has to survive: an edge
    whose endpoints are one node, a very wide attribute set, and the operators
    that commit more than once so their records share a buffer.

    `testEffectsV3_04_Shapes` covers volume and interleaving. These are the
    shapes where the *blocks* are unusual — `IdSet` and `IdList` are distinct
    types precisely because edge endpoints repeat, and a self-loop is the
    smallest case where the same id appears in both endpoint lists.
    """

    GRAPH_ID = "effects_v3_harder_shapes"

    def __init__(self):
        self._setup()

    def test01_an_edge_whose_endpoints_are_the_same_node(self):
        # `IdList` exists because endpoints deduplicate neither by value nor by
        # position; a self-loop is where a set-shaped encoding would collapse
        # src and dst into one entry and misalign everything after it.
        self.set_effects_config()
        res = self.query_and_sync(
            "CREATE (a:Loop {v: 1}) WITH a CREATE (a)-[:SELF {w: 1}]->(a)")
        self.env.assertEqual(res.relationships_created, 1)
        self.assert_agree(
            "MATCH (a:Loop)-[e:SELF]->(b) RETURN ID(a) = ID(b), count(e)",
            [[True, 1]])

        # A batch of them, so the record carries many rows whose two endpoint
        # lists are identical.
        self.query_and_sync("UNWIND range(1, 500) AS i CREATE (:Ring {i: i})")
        res = self.query_and_sync(
            "MATCH (n:Ring) CREATE (n)-[:SELF2 {w: n.i}]->(n)")
        self.env.assertEqual(res.relationships_created, 500)
        self.assert_agree(
            """MATCH (a:Ring)-[e:SELF2]->(b) WHERE ID(a) = ID(b)
               RETURN count(e), sum(e.w)""",
            [[500, 125250]])

        # ... and they come off cleanly, endpoints and all
        res = self.query_and_sync("MATCH (n:Ring) WHERE n.i % 2 = 0 DETACH DELETE n")
        self.env.assertEqual(res.nodes_deleted, 250)
        self.env.assertEqual(res.relationships_deleted, 250)
        self.assert_agree("MATCH ()-[e:SELF2]->() RETURN count(e)", [[250]])
        self.assert_graph_eq()

    def test02_a_very_wide_attribute_set(self):
        # `AttrSet` is `u16 n` ids and then `count x n` values, so a wide shape
        # is the case where the reader's row/column arithmetic can go wrong
        # without the totals changing.
        self.set_effects_config()
        W = 300
        props = ", ".join(f"p{i}: {i}" for i in range(W))
        self.query_and_sync(f"CREATE (:Wide {{{props}}})")
        self.assert_agree("MATCH (n:Wide) RETURN size(keys(n))", [[W]])
        # every column against its own id, not just the count
        self.assert_agree(
            f"""MATCH (n:Wide)
                RETURN size([k IN keys(n) WHERE n[k] <> toInteger(substring(k, 1))])""",
            [[0]])

        # the same width across many entities in one record
        self.query_and_sync(
            f"UNWIND range(1, 50) AS i CREATE (:Wide2 {{{props}, i: i}})")
        self.assert_agree(
            "MATCH (n:Wide2) RETURN count(n), sum(n.p299), sum(n.i)",
            [[50, 50 * 299, 1275]])
        self.assert_graph_eq()

    def test03_operators_that_commit_more_than_once(self):
        # MERGE, FOREACH and a chain of CREATEs each drive `CommitOp` several
        # times in one query, and every commit appends to the *same* buffer.
        # That is the shape where a payload compressed or finished per commit
        # produces something unreadable, and where a record written against a
        # stale schema baseline names an id the earlier commit had not yet
        # announced.
        self.set_effects_config()
        res = self.query_and_sync(
            "UNWIND range(1, 100) AS i MERGE (:Merged {i: i})")
        self.env.assertEqual(res.nodes_created, 100)
        # a second run of the same MERGE creates nothing, so the buffer is
        # empty and nothing must be replicated for it either
        res = self.query_and_sync(
            "UNWIND range(1, 100) AS i MERGE (:Merged {i: i})")
        self.env.assertEqual(res.nodes_created, 0)
        self.assert_agree("MATCH (n:Merged) RETURN count(n), sum(n.i)",
                          [[100, 5050]])

        # MERGE that matches some and creates others, in one query
        res = self.query_and_sync(
            "UNWIND range(50, 150) AS i MERGE (:Merged {i: i})")
        self.env.assertEqual(res.nodes_created, 50)
        self.assert_agree("MATCH (n:Merged) RETURN count(n), sum(n.i)",
                          [[150, 11325]])

        # MERGE on a relationship, which commits both endpoints and the edge
        self.query_and_sync(
            """UNWIND range(1, 50) AS i
               MATCH (a:Merged {i: i}), (b:Merged {i: i + 1})
               MERGE (a)-[:MERGED {w: i}]->(b)""")
        self.assert_agree("MATCH ()-[e:MERGED]->() RETURN count(e), sum(e.w)",
                          [[50, 1275]])

        # FOREACH, which commits once per iteration
        res = self.query_and_sync(
            "FOREACH (i IN range(1, 100) | CREATE (:Foreach {i: i}))")
        self.env.assertEqual(res.nodes_created, 100)
        self.assert_agree("MATCH (n:Foreach) RETURN count(n), sum(n.i)",
                          [[100, 5050]])

        # a nested FOREACH, and one that mutates rather than creates
        self.query_and_sync(
            """FOREACH (i IN range(1, 10) |
                 FOREACH (j IN range(1, 10) |
                   CREATE (:Nested {v: i * 10 + j})))""")
        self.assert_agree("MATCH (n:Nested) RETURN count(n), sum(n.v)",
                          [[100, 6050]])
        self.query_and_sync(
            "MATCH (n:Foreach) WITH collect(n) AS ns "
            "FOREACH (n IN ns | SET n.touched = true, n.i = n.i + 1000)")
        self.assert_agree(
            "MATCH (n:Foreach) RETURN count(n.touched), sum(n.i)",
            [[100, 105050]])

        # several CREATE clauses in one query, each its own commit
        res = self.query_and_sync(
            """CREATE (:Chain {s: 1})
               WITH 1 AS x CREATE (:Chain {s: 2})
               WITH 1 AS x CREATE (:Chain {s: 3})
               WITH 1 AS x MATCH (c:Chain) SET c.seen = true""")
        self.env.assertEqual(res.nodes_created, 3)
        self.assert_agree("MATCH (n:Chain) RETURN count(n), count(n.seen)",
                          [[3, 3]])

        self.assert_graph_eq()

    def test04_a_delete_that_takes_its_own_edges_with_it(self):
        # A node and the edges that hang off it in one buffer: the delete
        # record for the edges has to be readable *after* the node's, since
        # applying them the other way round leaves an edge whose endpoint is
        # gone. Self-loops are in the mix on purpose — they appear in both
        # endpoint lists of the record being deleted.
        self.set_effects_config()
        self.query_and_sync(
            """UNWIND range(1, 200) AS i CREATE (:Hub {i: i})""")
        self.query_and_sync(
            """MATCH (a:Hub {i: 1}), (b:Hub) WHERE b.i > 1
               CREATE (a)-[:OUT {w: b.i}]->(b), (b)-[:IN {w: b.i}]->(a)""")
        self.query_and_sync("MATCH (a:Hub {i: 1}) CREATE (a)-[:OWN]->(a)")
        self.assert_agree(
            "MATCH (a:Hub {i: 1}) RETURN size((a)--())", [[399]])

        res = self.query_and_sync("MATCH (a:Hub {i: 1}) DETACH DELETE a")
        self.env.assertEqual(res.nodes_deleted, 1)
        self.env.assertEqual(res.relationships_deleted, 399)
        # scoped by type: the class shares one graph, and the earlier tests
        # left edges of their own in it
        self.assert_agree(
            "MATCH ()-[e:OUT|IN|OWN]->() RETURN count(e)", [[0]])
        self.assert_agree("MATCH (n:Hub) RETURN count(n)", [[199]])

        # and deleting a node and one of its own edges explicitly, in one query
        self.query_and_sync(
            """MATCH (a:Hub {i: 2}), (b:Hub {i: 3})
               CREATE (a)-[:E1]->(b), (a)-[:E2]->(b)""")
        res = self.query_and_sync(
            "MATCH (a:Hub {i: 2})-[e:E1]->() DELETE e, a")
        self.env.assertEqual(res.nodes_deleted, 1)
        self.env.assertEqual(res.relationships_deleted, 2)
        self.assert_agree("MATCH ()-[e:E1|E2]->() RETURN count(e)", [[0]])
        self.assert_graph_eq()


    def test05_an_edge_updated_and_deleted_in_one_query(self):
        # UPDATE_EDGE carries its relationship type, which the emitter reads off
        # the graph *after* commit has applied. `SET e.x = 1 DELETE e` leaves
        # the edge in `existing_relationships_attrs` while its row in the type
        # matrix is already gone, so a panicking type lookup takes the primary
        # down here — on a legitimate query, before anything reaches a replica.
        self.set_effects_config()
        self.query_and_sync(
            """CREATE (a:Doomed {i: 1})-[:GONE {w: 0}]->(b:Doomed {i: 2})""")

        res = self.query_and_sync(
            "MATCH (a:Doomed)-[e:GONE]->(b) SET e.w = 99 DELETE e")
        self.env.assertEqual(res.relationships_deleted, 1)
        self.assert_agree("MATCH ()-[e:GONE]->() RETURN count(e)", [[0]])
        # both endpoints survive: only the edge was named
        self.assert_agree("MATCH (n:Doomed) RETURN count(n)", [[2]])

        # and in bulk, where the update and the delete land in one buffer with
        # many rows rather than one
        self.query_and_sync(
            """UNWIND range(1, 200) AS i
               CREATE (:Src {i: i})-[:ALSOGONE {w: i}]->(:Dst {i: i})""")
        res = self.query_and_sync(
            "MATCH ()-[e:ALSOGONE]->() SET e.w = e.w * 2 DELETE e")
        self.env.assertEqual(res.relationships_deleted, 200)
        self.assert_agree("MATCH ()-[e:ALSOGONE]->() RETURN count(e)", [[0]])
        self.assert_graph_eq()

    def test06_edges_of_two_types_updated_in_one_query(self):
        # The type is stated once per record, so it is part of the group key:
        # two types sharing an attribute shape must split into two records
        # rather than land in one under whichever type was seen first.
        self.set_effects_config()
        self.query_and_sync(
            """UNWIND range(1, 50) AS i
               CREATE (a:Two {i: i}), (b:Two {i: -i}),
                      (a)-[:TA {w: i}]->(b), (a)-[:TB {w: i}]->(b)""")

        res = self.query_and_sync(
            "MATCH ()-[e:TA|TB]->() SET e.w = e.w + 1000")
        self.env.assertEqual(res.properties_set, 100)
        self.assert_agree(
            """MATCH ()-[e:TA]->() RETURN count(e), min(e.w), max(e.w)""",
            [[50, 1001, 1050]])
        self.assert_agree(
            """MATCH ()-[e:TB]->() RETURN count(e), min(e.w), max(e.w)""",
            [[50, 1001, 1050]])
        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# 4f. The replica's indexes, for the schemas the query never named
#-----------------------------------------------------------------------------


class testEffectsV3_04f_IndexesTheQueryNeverNamed(_EffectsV3Base):
    """An update touches every index the entity belongs to, not the one the
    pattern matched on.

    `MATCH (n:A) SET n.x = 2` on an `(:A:B)` node has to leave **`:B`'s** index
    on `x` correct too, and the query never says `B`. Same for edges: an
    untyped `MATCH ()-[r]->() SET r.x = 2` has to leave `:R`'s index correct.

    The interesting half is the replica. The primary can see the entity and
    walk its own matrices; the replica only has the record. So these assert
    through the index rather than through a scan — a stale index does not
    return nothing, it returns the value the entity used to have, which a
    `count(*)` over a full scan would never notice.
    """

    GRAPH_ID = "effects_v3_derived_indexes"

    def __init__(self):
        self._setup()

    def _assert_uses_index(self, q, op):
        # Otherwise the assertions below pass on a full scan and prove nothing
        # about index maintenance at all.
        #
        # **The primary only**, and that is a real limit rather than a
        # convenience: `GRAPH.EXPLAIN` is registered `write` (`src/lib.rs`, and
        # C registers it the same way), so a replica refuses it with "You can't
        # write against a read only replica" even though it plans rather than
        # executes. There is no way to read a replica's plan.
        #
        # So on the replica these tests assert the *answer*, and that the
        # answer matches the primary's — which is index-backed only to the
        # extent that the replica's planner makes the same choice, which is an
        # inference. `test04` closes that hole: a fulltext procedure reads the
        # index and nothing else, so there is no plan to infer about.
        self.env.assertContains(op, str(self.master_graph.explain(q)))

    def test01_a_second_label_index_the_query_never_mentioned(self):
        self.set_effects_config()
        create_node_range_index(self.master_graph, 'A', 'x', sync=True)
        create_node_range_index(self.master_graph, 'B', 'x', sync=True)
        self.wait_for_replica_offset()

        self.query_and_sync("CREATE (:A:B {x: 1}), (:A {x: 1}), (:B {x: 1})")

        by_b = "MATCH (n:B) WHERE n.x = $v RETURN count(n)"
        self._assert_uses_index(
            "MATCH (n:B) WHERE n.x = 1 RETURN count(n)", 'Node By Index Scan')
        self.assert_agree(by_b, [[2]], params={'v': 1})

        # only :A is named, and only the :A:B node and the :A node match
        res = self.query_and_sync("MATCH (n:A) SET n.x = 2")
        self.env.assertEqual(res.properties_set, 2)

        # B's index has to have followed the :A:B node to its new value. The
        # sharp assertion is the second one: a stale index still holds x = 1,
        # so it answers this with 2 rather than 1.
        self.assert_agree(by_b, [[1]], params={'v': 2})
        self.assert_agree(by_b, [[1]], params={'v': 1})

        # and the index agrees with an unindexed read of the same thing
        self.assert_agree(
            "MATCH (n:B) RETURN n.x ORDER BY n.x", [[1], [2]])
        self.assert_graph_eq()

    def test02_an_edge_type_index_the_query_never_mentioned(self):
        # The reason UPDATE_EDGE carries its RelType: an untyped pattern still
        # has to leave the type-scoped index correct on the replica.
        self.set_effects_config()
        create_edge_range_index(self.master_graph, 'R', 'x', sync=True)
        self.wait_for_replica_offset()

        self.query_and_sync(
            """CREATE (a:EN {i: 1})-[:R {x: 1}]->(b:EN {i: 2}),
                      (b)-[:R {x: 1}]->(a)""")

        by_r = "MATCH ()-[r:R]->() WHERE r.x = $v RETURN count(r)"
        self._assert_uses_index(
            "MATCH ()-[r:R]->() WHERE r.x = 1 RETURN count(r)",
            'Edge By Index Scan')
        self.assert_agree(by_r, [[2]], params={'v': 1})

        # untyped — the query never says R
        res = self.query_and_sync("MATCH ()-[r]->() SET r.x = 2")
        self.env.assertEqual(res.properties_set, 2)

        self.assert_agree(by_r, [[2]], params={'v': 2})
        self.assert_agree(by_r, [[0]], params={'v': 1})
        self.assert_agree(
            "MATCH ()-[r:R]->() RETURN r.x ORDER BY r.x", [[2], [2]])
        self.assert_graph_eq()

    def test03_two_edge_types_one_indexed(self):
        # The record splits by type, so the unindexed type must not drag the
        # indexed one's rows into its record — and the indexed type's index
        # must still see every row that belongs to it.
        self.set_effects_config()
        create_edge_range_index(self.master_graph, 'IX', 'x', sync=True)
        self.wait_for_replica_offset()

        self.query_and_sync(
            """UNWIND range(1, 20) AS i
               CREATE (a:TN {i: i})-[:IX {x: i}]->(b:TN {i: -i}),
                      (a)-[:NOIX {x: i}]->(b)""")
        self.assert_agree(
            "MATCH ()-[r:IX]->() WHERE r.x > 10 RETURN count(r)", [[10]])

        res = self.query_and_sync("MATCH ()-[r:IX|NOIX]->() SET r.x = r.x + 100")
        self.env.assertEqual(res.properties_set, 40)

        self.assert_agree(
            "MATCH ()-[r:IX]->() WHERE r.x > 110 RETURN count(r)", [[10]])
        self.assert_agree(
            "MATCH ()-[r:IX]->() WHERE r.x <= 100 RETURN count(r)", [[0]])
        self.assert_agree(
            "MATCH ()-[r:NOIX]->() RETURN count(r), min(r.x), max(r.x)",
            [[20, 101, 120]])
        self.assert_graph_eq()


    def test04_the_replica_index_itself_answers(self):
        # The other three assert what the replica *returns*, which goes through
        # its planner, and a replica's plan cannot be read — GRAPH.EXPLAIN is a
        # `write` command on both engines. So they establish the answer is
        # right without establishing the index produced it.
        #
        # `db.idx.fulltext.queryNodes` has no such gap: it reads the fulltext
        # index directly. If the replica never added the :FB document when the
        # :FA half of the pattern was updated, this returns nothing, whatever
        # the planner would have preferred.
        self.set_effects_config()
        create_node_fulltext_index(self.master_graph, 'FA', 'body', sync=True)
        create_node_fulltext_index(self.master_graph, 'FB', 'body', sync=True)
        self.wait_for_replica_offset()

        self.query_and_sync("CREATE (:FA:FB {body: 'alpha'})")
        probe = ("CALL db.idx.fulltext.queryNodes('FB', $t) "
                 "YIELD node RETURN count(node)")
        self.assert_agree(probe, [[1]], params={'t': 'alpha'})

        # only :FA is named; :FB's index has to follow the same node
        res = self.query_and_sync("MATCH (n:FA) SET n.body = 'omega'")
        self.env.assertEqual(res.properties_set, 1)

        self.assert_agree(probe, [[1]], params={'t': 'omega'})
        # and the old term is gone from it — a stale index still answers 'alpha'
        self.assert_agree(probe, [[0]], params={'t': 'alpha'})
        self.assert_graph_eq()

    def test05_deleting_a_node_clears_the_index_of_a_label_it_was_not_matched_by(self):
        """A deleted node leaves every label index it was in, not just the
        matched one.

        This is the half that cannot be re-derived. `delete_nodes` clears the
        label matrices, so by the time effects are built the node's labels are
        gone from the graph — they are captured during the delete as flat
        `(node, label)` pairs and regrouped per node by the emitter. If that
        capture or that regrouping dropped a label, the replica would keep
        serving a deleted node out of that label's index, and the node no
        longer exists to notice it with.

        Fulltext again, for the reason in `test04`: it reads the index directly,
        so the replica assertion does not depend on what its planner chose.
        """
        self.set_effects_config()
        create_node_fulltext_index(self.master_graph, 'DFA', 'body', sync=True)
        create_node_fulltext_index(self.master_graph, 'DFB', 'body', sync=True)
        self.wait_for_replica_offset()

        self.query_and_sync(
            "CREATE (:DFA:DFB {body: 'alpha'}), (:DFB {body: 'beta'})")
        probe = ("CALL db.idx.fulltext.queryNodes('DFB', $t) "
                 "YIELD node RETURN count(node)")
        self.assert_agree(probe, [[1]], params={'t': 'alpha'})

        # only :DFA is named
        res = self.query_and_sync("MATCH (n:DFA) DELETE n")
        self.env.assertEqual(res.nodes_deleted, 1)

        # gone from :DFB's index too, and the :DFB-only node is untouched
        self.assert_agree(probe, [[0]], params={'t': 'alpha'})
        self.assert_agree(probe, [[1]], params={'t': 'beta'})
        self.assert_agree("MATCH (n:DFB) RETURN count(n)", [[1]])
        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# 4d. GRAPH.RECORD is a real write
#-----------------------------------------------------------------------------
