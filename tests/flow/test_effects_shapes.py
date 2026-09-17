"""Effects v3 -- the record shapes the writer partitions and the reader reassembles, including the indexes a query never named.

See `effects_common.py` for the shared fixture and why these are split.
"""



from common import *
from graph_utils import graph_eq
from constraint_utils import create_unique_node_constraint
from index_utils import (create_edge_range_index, create_node_fulltext_index,
                         create_node_range_index, list_indicies,
                         wait_for_indices_to_sync)

from effects_common import _EffectsBase


class testEffects_04_Shapes(_EffectsBase):
    """The record shapes the v3 writer has to partition and the reader has to
    reassemble: multiple labels, differing property shapes in one query,
    points, lists, large batches, deletes interleaved with creates, and ids
    recycled by a delete-then-recreate.
    """

    GRAPH_ID = "effects_shapes"

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
        fresh = "effects_recycle_one_buffer"
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
# 4c. shapes the writer's partitioning has to survive
#-----------------------------------------------------------------------------


class testEffects_04c_HarderShapes(_EffectsBase):
    """Shapes with structure the writer's partitioning has to survive: an edge
    whose endpoints are one node, a very wide attribute set, and the operators
    that commit more than once so their records share a buffer.

    `testEffects_04_Shapes` covers volume and interleaving. These are the
    shapes where the *blocks* are unusual — `IdSet` and `IdList` are distinct
    types precisely because edge endpoints repeat, and a self-loop is the
    smallest case where the same id appears in both endpoint lists.
    """

    GRAPH_ID = "effects_harder_shapes"

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
# 4g. several kinds of record from one statement
#
# Moved here from the former test_effects_v3_compound.py: a compound statement
# is a shape question -- what one buffer carries and in what order -- so it
# belongs beside the other partitioning cases rather than in a file of its own.
# The module docstring there explained why DDL cannot share a COMMIT with data;
# that reasoning now lives on the class below.
#-----------------------------------------------------------------------------


class testEffects_04g_CompoundSequences(_EffectsBase):
    """Several record kinds from one statement, applied in an order that works.

    `test07`-`test09` used to live here and were removed: they were DDL and
    data in *separate* statements, which is a sequence rather than a
    compound, and each was a weaker form of a test in `..._ddl.py` --
    `05.test02` does rows-then-index and also compares the execution plan
    on both sides, `05.test01` covers the drops, and a violating write is
    already refused three times over in that file.
    """

    GRAPH_ID = "effects_compound"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT')

    def test01_schema_and_data_ride_one_buffer(self):
        # A label, a relationship type and three attributes that do not exist
        # yet, all introduced by the statement that first uses them. The replica
        # has to intern every name before it can resolve the ids the node and
        # edge records carry, so the schema records must lead.
        self.set_effects_config()
        self.monitor_mark()
        self.query_and_sync(
            "CREATE (:Fresh {alpha: 1, beta: 'two'})"
            "-[:FRESHLY {gamma: 3.5}]->(:Fresh {alpha: 2, beta: 'three'})")
        window = self.monitor_mark()

        # One commit, so one payload -- and it leads with schema, not with the
        # node it describes. That is the assertion: a buffer that led with
        # CREATE_NODE would name a label id the replica has not interned.
        leading = self.leading_opcodes(window, self.GRAPH_ID)
        self.env.assertTrue(
            leading and leading[0] in ('ADD_SCHEMA', 'ADD_ATTRIBUTE'),
            message=f"a buffer introducing new schema must lead with it, led with {leading}")

        self.assert_agree("MATCH (n:Fresh) RETURN count(n)", [[2]])
        self.assert_agree(
            "MATCH (:Fresh)-[e:FRESHLY]->(:Fresh) RETURN count(e), sum(e.gamma)",
            [[1, 3.5]])
        self.assert_agree(
            "MATCH (n:Fresh) RETURN count(n.alpha), count(n.beta)", [[2, 2]])
        self.assert_graph_eq()

    def test02_create_update_label_and_delete_in_one_statement(self):
        # Four mutation kinds from one MATCH: a node created, an existing one
        # updated, a label added to a third, a fourth deleted. Whatever order the
        # writer partitions them into, both sides must end up agreeing -- and the
        # deleted node must not come back as a side effect of the create.
        self.set_effects_config()
        self.query_and_sync(
            "UNWIND range(1, 4) AS i CREATE (:Multi {i: i, tag: 'start'})")

        self.query_and_sync("""
            MATCH (a:Multi {i: 1}), (b:Multi {i: 2}), (c:Multi {i: 3})
            CREATE (:Multi {i: 5, tag: 'added'})
            SET a.tag = 'updated'
            SET b:Extra
            DELETE c
        """)

        self.assert_agree("MATCH (n:Multi) RETURN count(n)", [[4]])
        self.assert_agree("MATCH (n:Multi {tag: 'updated'}) RETURN count(n)", [[1]])
        self.assert_agree("MATCH (n:Extra) RETURN count(n)", [[1]])
        self.assert_agree("MATCH (n:Multi {i: 3}) RETURN count(n)", [[0]])
        self.assert_agree("MATCH (n:Multi {i: 5}) RETURN count(n)", [[1]])
        self.assert_graph_eq()

    def test03_index_maintenance_rides_the_same_commit(self):
        # With an index already present, a statement that creates, updates and
        # deletes indexed rows carries all of that in one commit. The replica's
        # index must answer for the rows as they finally are, not as any
        # intermediate step left them.
        self.set_effects_config()
        create_node_range_index(self.master_graph, 'Indexed', 'k', sync=True)
        self.wait_for_replica_offset()
        self.query_and_sync(
            "UNWIND range(1, 6) AS i CREATE (:Indexed {k: i, keep: true})")

        self.query_and_sync("""
            MATCH (n:Indexed) WHERE n.k <= 3
            SET n.k = n.k + 100
        """)
        self.query_and_sync("MATCH (n:Indexed) WHERE n.k = 6 DELETE n")

        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)

        # Read through the index on both sides. A stale replica index answers
        # these with the pre-update values and is the failure this catches.
        self.assert_agree("MATCH (n:Indexed) WHERE n.k = 101 RETURN count(n)", [[1]])
        self.assert_agree("MATCH (n:Indexed) WHERE n.k = 1 RETURN count(n)", [[0]])
        self.assert_agree("MATCH (n:Indexed) WHERE n.k = 6 RETURN count(n)", [[0]])
        self.assert_agree(
            "MATCH (n:Indexed) WHERE n.k > 100 RETURN count(n)", [[3]])
        self.assert_agree("MATCH (n:Indexed) RETURN count(n)", [[5]])
        self.env.assertEqual(list_indicies(self.replica_graph).result_set,
                             list_indicies(self.master_graph).result_set)
        self.assert_graph_eq()

    def test04_a_violating_write_beside_a_valid_one_commits_neither(self):
        # A UNIQUE constraint, then one statement whose rows are individually
        # fine and collectively are not. The write is refused whole, so the
        # replica must not have been told about the half that would have
        # succeeded -- a partial buffer here is divergence, not a partial write.
        self.set_effects_config()
        self.query_and_sync("CREATE (:Uniq {u: 1})")
        create_unique_node_constraint(self.master_graph, 'Uniq', 'u', sync=True)
        self.wait_for_replica_offset()
        self.wait_for_constraint_settled(self.master_graph, 'Uniq')

        full_before = self.master.info()["sync_full"]
        failures_before = self.effect_failures()

        rejected = None
        try:
            self.master_graph.query(
                "CREATE (:Uniq {u: 2}), (:Uniq {u: 1})")
        except Exception as e:
            rejected = str(e).lower()
        self.env.assertTrue(
            rejected is not None and "unique constraint violation" in rejected,
            message=f"the duplicate must be refused, got {rejected!r}")

        self.wait_for_replica_offset()
        # Neither row survives: the valid one was in the same statement.
        self.assert_agree("MATCH (n:Uniq) RETURN count(n)", [[1]])
        self.assert_agree("MATCH (n:Uniq {u: 2}) RETURN count(n)", [[0]])
        # And nothing was refused on the replica -- a refused write should send
        # nothing, rather than sending something the replica then rejects.
        self.env.assertEqual(self.effect_failures(), failures_before,
            message="a refused write still put a buffer on the wire")
        self.env.assertEqual(self.master.info()["sync_full"], full_before,
            message="a refused write forced a resync")
        self.assert_graph_eq()

    def test05_new_schema_and_a_delete_of_the_same_label_in_one_statement(self):
        # The shape `digest_cancelled` exists for, one level up: a label
        # introduced and then emptied inside one statement. The schema addition
        # is real and must survive even though no row that used it does.
        self.set_effects_config()
        self.query_and_sync(
            "CREATE (:Doomed {only: 1}) WITH 1 AS x "
            "MATCH (d:Doomed) DELETE d")

        self.assert_agree("MATCH (n:Doomed) RETURN count(n)", [[0]])
        # The label is interned on both sides even with no rows carrying it, so
        # a later create resolves the same id rather than minting a second.
        self.query_and_sync("CREATE (:Doomed {only: 2})")
        self.assert_agree("MATCH (n:Doomed) RETURN count(n), sum(n.only)", [[1, 2]])
        self.assert_graph_eq()

    # ── DDL and data in consecutive commits ───────────────────────────────
    #
    # Not one buffer -- the parser forbids that -- but one *sequence*, which is
    # what a replica actually sees. Each of these is two or more commits whose
    # order is load-bearing on the far side.

    def test06_an_index_created_then_populated_answers_on_the_replica(self):
        # DDL first, data second. The replica applies CREATE_INDEX, then the
        # CREATE_NODE records for rows the index must contain. If the index
        # arrived but the rows were indexed against the pre-index state, the
        # replica's index answers short.
        self.set_effects_config()
        self.query_and_sync("CREATE INDEX FOR (n:Seq) ON (n.k)")
        self.query_and_sync(
            "UNWIND range(1, 20) AS i CREATE (:Seq {k: i, pad: 'x'})")
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)

        self.assert_agree("MATCH (n:Seq) WHERE n.k = 7 RETURN count(n)", [[1]])
        self.assert_agree("MATCH (n:Seq) WHERE n.k > 15 RETURN count(n)", [[5]])
        self.env.assertEqual(list_indicies(self.replica_graph).result_set,
                             list_indicies(self.master_graph).result_set)
        self.assert_graph_eq()

    def test10_an_index_procedure_inside_a_write_query(self):
        """DDL and data in ONE statement — and what actually happens today.

        This is the shape the class is named for and the one it could not
        reach: a single query that creates an index *and* mutates nodes. It is
        written as a pin on current behaviour rather than on the intended
        behaviour, because the procedure does not work, and the way it does not
        work is worth catching.

        `db.idx.fulltext.createNodeIndex` is registered as a `write procedure`
        (`graph/src/runtime/functions/procedures.rs:328`) but its body is
        `Ok(empty_procedure_batch())` — a stub. Two consequences, both pinned
        below:

        * it creates no index, so there is no index DDL in this buffer at all;
        * it yields ZERO rows, so every clause after it runs zero times and the
          trailing `CREATE` is silently dropped. Not an error — the query
          succeeds and reports fewer nodes than it names.

        When the stub is implemented this test goes red on both counts, which
        is the point: whoever implements it has to come here and decide what
        the replica should see.
        """
        self.set_effects_config()
        self.monitor_mark()

        # The arity, first. The C-style call takes label and field as separate
        # arguments; this build takes exactly one map, so the C form is an
        # error rather than a call that quietly does something else.
        try:
            self.master_graph.query(
                "CALL db.idx.fulltext.createNodeIndex('Doc', 'body')")
            self.env.assertTrue(False, 1)
        except ResponseError as e:
            self.env.assertContains("expected at most 1", str(e))

        self.query_and_sync(
            "CREATE (:Pre {id: 100})-[:REL {w: 1}]->(:Pre {id: 101}) "
            "WITH 1 AS one "
            "CALL db.idx.fulltext.createNodeIndex({label: 'Doc'}) "
            "CREATE (:Post {id: 200}) "
            "RETURN one")

        # No index was created, on either side. Scoped to the label the call
        # named, not a global count: this class shares one graph and the tests
        # before it leave indexes behind.
        self.assert_agree(
            "CALL db.indexes() YIELD label WHERE label = 'Doc' "
            "RETURN count(label)", [[0]])

        # The two `:Pre` nodes and the relationship are there...
        self.assert_agree("MATCH (n:Pre) RETURN count(n)", [[2]])
        self.assert_agree("MATCH ()-[r:REL]->() RETURN count(r)", [[1]])

        # ...and `:Post` is not, on either side. The write after the `CALL`
        # never ran, because the stub ended the pipeline. The replica agreeing
        # is the part that matters here: the master's buffer describes what the
        # master actually did, so a write the master skipped is a write the
        # replica must also not have.
        self.assert_agree("MATCH (n:Post) RETURN count(n)", [[0]])

        # And it all rode effects, with no verbatim GRAPH.QUERY fallback.
        window = self.monitor_mark()
        self.env.assertEqual(self.count_in(window, 'GRAPH.EFFECT'), 1)
        self.env.assertEqual(self.count_in(window, 'GRAPH.QUERY'), 0)
        self.assert_graph_eq()

    def test11_a_non_deterministic_write_replicates_its_outcome(self):
        """A write whose size and values the query text does not determine.

        This is the one shape a replica cannot get right by re-running the
        statement: `rand()` decides how many rows survive and what gets stored,
        so re-execution reaches a different graph every time. It can only match
        by applying what the primary actually did, which is the whole claim v3
        makes. Nothing else in these files varies run to run — every other
        write has an outcome fixed by its text.

        So the assertions cannot hardcode a count. They compare the two sides.
        """
        self.set_effects_config()
        self.monitor_mark()

        self.query_and_sync(
            "UNWIND range(0, 40) AS i "
            "WITH i WHERE i = 0 OR rand() < 0.5 "
            "CREATE (:Rnd {id: i, r: rand()}) "
            "RETURN count(*)")

        # The id SET, not just its size: two graphs can hold the same number of
        # nodes and disagree about which ones, and a count alone cannot see it.
        m, r = self.probe("MATCH (n:Rnd) RETURN n.id ORDER BY n.id")
        self.env.assertEqual(r, m)

        # `i = 0` is unconditional, so the write is never empty — without it a
        # run where every `rand()` fell the wrong way would compare two empty
        # graphs and pass without replicating anything.
        self.env.assertTrue(len(m) >= 1)
        self.env.assertTrue(len(m) <= 41)

        # The stored `rand()` values too. These are the tightest pin in the
        # file: they are not derivable from the query, not derivable from the
        # ids, and a replica that re-executed would have its own.
        m, r = self.probe("MATCH (n:Rnd) RETURN n.id, n.r ORDER BY n.id")
        self.env.assertEqual(r, m)

        # And the values have to be real randoms, or the comparison above is
        # two columns of the same constant agreeing with each other. `rand()`
        # returning a fixed value would leave every assertion here green while
        # the test stopped meaning anything.
        distinct = self.master_graph.ro_query(
            "MATCH (n:Rnd) RETURN count(DISTINCT n.r), count(n)").result_set
        self.env.assertEqual(distinct[0][0], distinct[0][1])
        self.env.assertTrue(distinct[0][0] >= 1)

        # One buffer, and no verbatim fallback -- a `rand()` write replicated
        # verbatim is exactly the bug this test exists to catch.
        window = self.monitor_mark()
        self.env.assertEqual(self.count_in(window, 'GRAPH.EFFECT'), 1)
        self.env.assertEqual(self.count_in(window, 'GRAPH.QUERY'), 0)
        self.assert_graph_eq()

