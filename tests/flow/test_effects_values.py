"""Effects -- does every value survive the wire?

Two halves of one question. A null must arrive as a *removal* rather than as a
stored value, and every other value type must arrive unchanged. The `SIValue`
tags themselves round-trip in `graph/src/effects/v3/value.rs`, which is where a
codec claim belongs; this file is about the replica-side apply path.

See `effects_common.py` for the shared fixture.
"""

from common import *

from effects_common import _EffectsBase


class testEffects_01_NullIsRemove(_EffectsBase):
    """`SET x = NULL` removes the property, and the replica must remove it too.

    Regression for a real v3 divergence: the apply side filtered nulls out of a
    record's value rows before merging them, so `MATCH (n) SET n.x = NULL`
    removed the property on the primary and did nothing at all on the replica.
    Nothing surfaced it -- the link stayed healthy, no error anywhere, and the
    two only re-converged on the next full resync.

    FalkorDB never stores a null property value, which is what makes a null on
    the wire unambiguous: it can only mean "remove this attribute".
    """

    GRAPH_ID = "effects_null"

    def __init__(self):
        self._setup()

    def test01_node_property_set_to_null(self):
        self.query_and_sync("CREATE (:N {id: 1, x: 'gone', y: 'kept'})")
        # baseline — both sides can see the property before it is removed
        self.assert_agree("MATCH (n:N) RETURN count(n.x)", [[1]])

        res = self.query_and_sync("MATCH (n:N) SET n.x = NULL")
        self.env.assertEqual(res.properties_removed, 1)

        # count(expr) skips nulls, so a value surviving on either side is a 1
        self.assert_agree("MATCH (n:N) RETURN count(n.x)", [[0]])
        # ... and the property is *absent*, not merely reading as null
        self.assert_agree("MATCH (n:N) RETURN 'x' IN keys(n)", [[False]])
        # the untouched sibling is still there on both sides
        self.assert_agree("MATCH (n:N) RETURN n.y", [['kept']])
        # keys() enumerates registered attributes in a deterministic order, so
        # the two sides' whole property shape is comparable
        m, r = self.probe("MATCH (n:N) RETURN keys(n)")
        self.env.assertEqual(m, [[['id', 'y']]])
        self.env.assertEqual(r, m)

        self.assert_graph_eq()

    def test02_edge_property_set_to_null(self):
        self.query_and_sync("CREATE ()-[:R {id: 1, x: 'gone', y: 'kept'}]->()")
        self.assert_agree("MATCH ()-[e:R]->() RETURN count(e.x)", [[1]])

        res = self.query_and_sync("MATCH ()-[e:R]->() SET e.x = NULL")
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_agree("MATCH ()-[e:R]->() RETURN count(e.x)", [[0]])
        self.assert_agree("MATCH ()-[e:R]->() RETURN 'x' IN keys(e)", [[False]])
        self.assert_agree("MATCH ()-[e:R]->() RETURN e.y", [['kept']])

        self.assert_graph_eq()

    def test03_one_of_several_properties_set_to_null(self):
        # A null in the middle of a multi-assignment SET: the record carries
        # three attribute columns, one of which is a removal.
        self.query_and_sync("CREATE (:M {a: 1, b: 2, c: 3, d: 4})")

        res = self.query_and_sync("MATCH (n:M) SET n.a = 10, n.b = NULL, n.c = 30")
        # `a` and `c` are overwrites, which FalkorDB accounts for as a removal
        # plus a set; `b` is the only pure removal. So two sets, three removals.
        self.env.assertEqual(res.properties_set, 2)
        self.env.assertEqual(res.properties_removed, 3)

        self.assert_agree("MATCH (n:M) RETURN n.a, n.c, n.d", [[10, 30, 4]])
        self.assert_agree("MATCH (n:M) RETURN 'b' IN keys(n)", [[False]])
        self.assert_agree("MATCH (n:M) RETURN count(n.b)", [[0]])

        # ... and the same via map addition, which reaches the same record
        res = self.query_and_sync("MATCH (n:M) SET n += {a: NULL, e: 5}")
        self.env.assertEqual(res.properties_set, 1)
        self.env.assertEqual(res.properties_removed, 1)
        self.assert_agree("MATCH (n:M) RETURN n.c, n.d, n.e", [[30, 4, 5]])
        self.assert_agree("MATCH (n:M) RETURN 'a' IN keys(n)", [[False]])

        self.assert_graph_eq()

    def test04_whole_shape_null(self):
        # Every column of the record is a removal.
        self.query_and_sync("CREATE (:W {a: 1, b: 2, c: 3})")
        self.query_and_sync("CREATE ()-[:WR {a: 1, b: 2}]->()")

        res = self.query_and_sync("MATCH (n:W) SET n = {a: NULL, b: NULL, c: NULL}")
        self.env.assertEqual(res.properties_removed, 3)
        self.assert_agree("MATCH (n:W) RETURN keys(n)", [[[]]])

        res = self.query_and_sync("MATCH ()-[e:WR]->() SET e = {a: NULL, b: NULL}")
        self.env.assertEqual(res.properties_removed, 2)
        self.assert_agree("MATCH ()-[e:WR]->() RETURN keys(e)", [[[]]])

        # the empty-map form of the same thing
        self.query_and_sync("MATCH (n:W) SET n.z = 1")
        res = self.query_and_sync("MATCH (n:W) SET n = {}")
        self.env.assertEqual(res.properties_removed, 1)
        self.assert_agree("MATCH (n:W) RETURN keys(n)", [[[]]])

        self.assert_graph_eq()

    def test05_null_arriving_as_a_query_parameter(self):
        # The literal NULL and a null-valued parameter take different routes
        # through the planner; both must reach the wire as a removal.
        self.query_and_sync("CREATE (:Param {id: 7, v: 'here'})")
        res = self.master_graph.query("MATCH (n:Param) SET n.v = $new",
                                      {'new': None})
        self.wait_for_replica_offset()
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_agree("MATCH (n:Param) RETURN count(n.v)", [[0]])
        self.assert_agree("MATCH (n:Param) RETURN 'v' IN keys(n)", [[False]])
        self.assert_agree("MATCH (n:Param) RETURN n.id", [[7]])

        self.assert_graph_eq()

    def test06_null_in_only_some_rows_of_one_record(self):
        # The shape most likely to break: one record, many entities, and the
        # column is a real value for some rows and a removal for others. A
        # filtering apply path either drops the whole column or misaligns the
        # values against the ids.
        self.query_and_sync(
            "UNWIND range(1, 1000) AS i CREATE (:P {id: i, x: 'v'})")
        self.assert_agree("MATCH (p:P) RETURN count(p.x)", [[1000]])

        res = self.query_and_sync(
            "MATCH (p:P) SET p.x = CASE WHEN p.id % 2 = 0 THEN NULL ELSE 'v2' END")
        # 500 nulls are pure removals; the 500 overwrites are each accounted
        # for as a removal plus a set.
        self.env.assertEqual(res.properties_removed, 1000)
        self.env.assertEqual(res.properties_set, 500)

        self.assert_agree("MATCH (p:P) RETURN count(p)", [[1000]])
        self.assert_agree("MATCH (p:P) RETURN count(p.x)", [[500]])
        # the survivors are exactly the odd ids, on both sides — this is what
        # catches a column that landed against the wrong entities
        self.assert_agree(
            "MATCH (p:P) WHERE p.x IS NOT NULL RETURN count(p), sum(p.id % 2)",
            [[500, 500]])
        self.assert_agree(
            "MATCH (p:P) WHERE p.x IS NULL RETURN count(p), sum(p.id % 2)",
            [[500, 0]])
        self.assert_agree("MATCH (p:P) WHERE p.x IS NOT NULL RETURN DISTINCT p.x",
                          [['v2']])

        self.assert_graph_eq()

    def test07_null_removes_every_property_of_many_entities(self):
        self.query_and_sync(
            "UNWIND range(1, 500) AS i CREATE (:Q {id: i, x: i, y: i})")
        res = self.query_and_sync("MATCH (q:Q) SET q.x = NULL, q.y = NULL")
        self.env.assertEqual(res.properties_removed, 1000)

        self.assert_agree("MATCH (q:Q) RETURN count(q), count(q.x), count(q.y)",
                          [[500, 0, 0]])
        self.assert_agree("MATCH (q:Q) RETURN DISTINCT keys(q)", [[['id']]])

        self.assert_graph_eq()


class testEffects_02_ValueTypes(_EffectsBase):
    """Value shapes whose encoding a whole-graph comparison cannot check.

    `graph_eq` compares the two sides against each other, so it is blind to
    both being wrong in the same way -- which is exactly the failure mode for a
    value whose *representation* is lossy rather than absent.
    """

    GRAPH_ID = "effects_value_types"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT')
        # Discard anything setup put on the wire so the first test's window
        # starts empty and `assert_effect_emitted` counts only what it wrote.
        self.monitor_mark()

    def test01_empty_vector(self):
        # test creation of an empty vector

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = "CREATE ({v:vecf32([])})"
        res = self.query_and_sync(q)
        self.env.assertEqual(res.nodes_created, 1)
        self.env.assertEqual(res.properties_set, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test02_value_edges_survive_the_round_trip(self):
        """Value shapes that otherwise only `graph/src/effects/v3/value.rs`
        covers.

        The `04b_ValueTypes` class was cut when this suite was split and its
        unique cases moved down into unit tests. Those pin the codec and say
        nothing about a real replica -- and two of them are cases a unit test
        structurally *cannot* fail. `Value`'s `PartialEq` compares floats with
        `compare_floats`, so `-0.0 == 0.0`: a round trip that drops the sign
        bit round-trips "successfully". Reading the sign back out through
        `toString`, on both sides, is what turns that into a claim.

        The string cases are here for the other reason. The wire carries a
        string's length as a *byte* count while `size()` reports characters --
        27 bytes against 11 characters below -- so a multi-byte string is
        exactly the shape where a length confusion shows up, and an ASCII one
        never would.
        """
        self.assert_effect_emitted(0)

        self.query_and_sync("""CREATE (:VEdge {
                                   nz:    -0.0,
                                   pz:     0.0,
                                   pinf:   1.0/0.0,
                                   ninf:  -1.0/0.0,
                                   empty:  '',
                                   multi:  'שלום 世界 🐦\u200d🔥'
                              })""")
        self.assert_effect_emitted()

        # Signed zero. `n.nz = n.pz` is true on both sides no matter what
        # happened, so the assertion has to go through the rendering.
        self.assert_agree(
            "MATCH (n:VEdge) RETURN toString(n.nz), toString(n.pz)",
            [['-0', '0']])

        # Infinities, which have no decimal representation to round-trip
        # through and so exercise the float encoding directly.
        self.assert_agree(
            "MATCH (n:VEdge) RETURN toString(n.pinf), toString(n.ninf)",
            [['inf', '-inf']])

        # Byte count against character count, and the empty string -- whose
        # encoded length is the case a "length includes the terminator" rule
        # gets wrong first.
        self.assert_agree("MATCH (n:VEdge) RETURN size(n.empty)", [[0]])
        self.assert_agree("MATCH (n:VEdge) RETURN size(n.multi)", [[11]])
        self.assert_agree("MATCH (n:VEdge) RETURN n.multi",
                          [['שלום 世界 🐦\u200d🔥']])

        # Temporal types, which the old suite only ever created and compared
        # whole-graph. Their values are non-deterministic, so the claim is that
        # the two sides hold the same one rather than any particular one.
        self.query_and_sync("""CREATE (:TEdge {
                                   d:  date(),
                                   t:  localtime(),
                                   dt: localdatetime(),
                                   ts: timestamp()
                              })""")
        self.assert_effect_emitted()
        m, r = self.probe("MATCH (n:TEdge) RETURN n.d, n.t, n.dt, n.ts")
        self.env.assertEqual(len(m), 1)
        self.env.assertEqual(len(m[0]), 4)
        self.env.assertEqual(r, m)

        self.assert_graph_eq()
