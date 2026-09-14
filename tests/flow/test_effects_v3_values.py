"""Effects v3 -- values surviving the round trip: a null that must remove rather than store, and every `SIValue` tag a query can produce.

See `effects_v3_common.py` for the shared fixture and why these are split.
"""



from common import *

from effects_v3_common import MONITOR_MARK_KEY, _EffectsV3Base


class testEffectsV3_01_NullIsRemove(_EffectsV3Base):
    """`SET x = NULL` removes the property, and the replica must remove it too.

    Regression for a real v3 divergence: the apply side filtered nulls out of a
    record's value rows before merging them, so `MATCH (n) SET n.x = NULL`
    removed the property on the primary and did nothing at all on the replica.
    Nothing surfaced it — the link stayed healthy, no error anywhere, and the
    two only re-converged on the next full resync.

    FalkorDB never stores a null property value, which is what makes a null on
    the wire unambiguous: it can only mean "remove this attribute".
    """

    GRAPH_ID = "effects_v3_null"

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


#-----------------------------------------------------------------------------
# 2. constraints replicate as GRAPH.EFFECT, not verbatim GRAPH.CONSTRAINT
#-----------------------------------------------------------------------------


class testEffectsV3_04b_ValueTypes(_EffectsV3Base):
    """Every `SIValue` tag the v3 codec writes, replicated for real.

    `graph/src/effects/v3/value.rs` pins these byte for byte, and that is a
    different claim: the pins say the encoder and the decoder agree about the
    bytes, not that a value a query produced survives being written, framed,
    replicated, applied, folded into the replica's attribute set and read back.
    The two have come apart before — the null-is-remove divergence
    `testEffectsV3_01_NullIsRemove` exists for was a value the codec carried
    correctly and the apply path dropped.

    Points and lists are already covered by
    `testEffectsV3_04_Shapes.test03_points_and_lists`; this is the rest of the
    tag list, plus the numeric and string edges that only a live round trip
    reaches. Maps are deliberately absent: `T_MAP` exists in the codec for
    index `OPTIONS`, and a map is not a legal property value ("Property values
    can only be of primitive types or arrays of primitive types"), so there is
    no query that puts one on this path.
    """

    GRAPH_ID = "effects_v3_value_types"

    def __init__(self):
        self._setup()

    def _round_trips(self, label, props, probe, expected):
        """Store `props` on a node and on an edge, and require both sides to
        read back `expected` through `probe`."""
        assignments = ", ".join(f"{k}: {v}" for k, v in props.items())
        self.query_and_sync(f"CREATE (:{label} {{{assignments}}})")
        self.query_and_sync(f"CREATE ()-[:{label}R {{{assignments}}}]->()")
        self.assert_agree(f"MATCH (n:{label}) RETURN {probe('n')}", expected)
        self.assert_agree(f"MATCH ()-[e:{label}R]->() RETURN {probe('e')}", expected)

    def test01_temporal_values(self):
        # T_DATE, T_TIME, T_DATETIME and T_DURATION are four separate tags over
        # the same i64 payload, so a swapped pair round-trips as a number and
        # reads back as the wrong type. Comparing the *rendered* value is what
        # distinguishes them.
        self.set_effects_config()
        self._round_trips(
            'Temporal',
            {'d':  "date('2024-01-15')",
             't':  "localtime('10:30:00')",
             'dt': "localdatetime('2024-01-15T10:30:00')",
             'du': "duration({days: 3, hours: 4})"},
            lambda v: (f"toString({v}.d), toString({v}.t), "
                       f"toString({v}.dt), toString({v}.du)"),
            [['2024-01-15', '10:30:00', '2024-01-15T10:30:00', 'P3DT4H']])
        self.assert_graph_eq()

    def test02_vecf32(self):
        # T_VECTOR_F32 is `u32 dim · f32 x dim`, the one value whose payload is
        # length-prefixed rather than fixed or NUL-terminated.
        self.set_effects_config()
        # A vector is not a stringifiable type, so the content is pinned by
        # distance to a literal built the same way: zero to itself, non-zero to
        # a vector one component away.
        self._round_trips(
            'Vec',
            {'v': "vecf32([1.5, -2.25, 0.0, 3.0e10])", 'one': "vecf32([0.5])"},
            lambda x: (f"vec.euclideanDistance({x}.v, vecf32([1.5, -2.25, 0.0, 3.0e10])), "
                       f"vec.euclideanDistance({x}.one, vecf32([0.5])), "
                       f"vec.euclideanDistance({x}.one, vecf32([1.5]))"),
            [[0.0, 0.0, 1.0]])
        self.assert_graph_eq()

    def test03_numeric_edges(self):
        # i64 at both ends, and the float values whose bit patterns a
        # round-trip through anything but `to_le_bytes` mangles: signed zero
        # (which compares equal to 0.0, so only the *sign* separates them) and
        # the infinities.
        self.set_effects_config()
        self._round_trips(
            'Num',
            {'imax': "9223372036854775807",
             'imin': "-9223372036854775808",
             'zero': "0.0",
             'negzero': "-0.0",
             'inf': "1.0/0.0",
             'neginf': "-1.0/0.0",
             # not f64::MAX: the parser folds 1.797...e308 to infinity, and
             # refuses a subnormal like 5.0e-324 outright ("unhandled type in
             # inlined properties"). Both sides agree on those, so they are a
             # parser limit rather than anything this file is about.
             'huge': "1.0e300",
             'small': "-1.0e300"},
            lambda v: (f"{v}.imax, {v}.imin, {v}.inf, {v}.neginf, "
                       f"{v}.huge, {v}.small, "
                       # `1/x` is the only expression that tells -0.0 from 0.0
                       f"1.0/{v}.negzero, 1.0/{v}.zero"),
            [[9223372036854775807, -9223372036854775808,
              float('inf'), float('-inf'),
              1.0e300, -1.0e300,
              float('-inf'), float('inf')]])
        self.assert_graph_eq()

    def test04_booleans_are_not_integers(self):
        # T_BOOL is a tag plus one byte, T_INT64 a tag plus eight. Losing the
        # distinction reads back as 0/1, which most probes cannot see.
        self.set_effects_config()
        self._round_trips(
            'Bool', {'t': "true", 'f': "false", 'i': "1", 'z': "0"},
            lambda v: (f"{v}.t, {v}.f, toString({v}.t), toString({v}.i), "
                       f"{v}.t = {v}.i, {v}.f = {v}.z"),
            [[True, False, 'true', '1', False, False]])
        self.assert_graph_eq()

    def test05_string_edges(self):
        # `write_string` is a *C* string: a length that includes the
        # terminator, then the bytes, then a NUL. So the edges are the empty
        # string (length 1, no bytes), multi-byte UTF-8 (a length in bytes, not
        # characters), and bytes that the RESP framing around the payload would
        # be sensitive to if the payload were not length-prefixed.
        self.set_effects_config()
        params = {
            'empty':   '',
            'space':   ' ',
            'crlf':    'a\r\nb',
            'quote':   'he said "hi" and \\ then',
            'unicode': 'שלום · 世界 · 🐦‍🔥',
            'long':    'x' * 1_000_000,
        }
        self.master_graph.query(
            """CREATE (:Str {empty: $empty, space: $space, crlf: $crlf,
                             quote: $quote, unicode: $unicode, long: $long})""",
            params)
        self.master_graph.query(
            """CREATE ()-[:StrR {empty: $empty, space: $space, crlf: $crlf,
                                 quote: $quote, unicode: $unicode, long: $long}]->()""",
            params)
        self.wait_for_replica_offset()

        for entity, pattern in (('n', "MATCH (n:Str)"),
                                ('e', "MATCH ()-[e:StrR]->()")):
            q = (f"{pattern} RETURN {entity}.empty, {entity}.space, "
                 f"{entity}.crlf, {entity}.quote, {entity}.unicode, "
                 f"size({entity}.long), {entity}.long = $long")
            m, r = self.probe(q, {'long': params['long']})
            self.env.assertEqual(
                m, [['', ' ', 'a\r\nb', params['quote'], params['unicode'],
                     1_000_000, True]])
            self.env.assertEqual(r, m)

        # An interior NUL is the one string shape the writer calls out — it
        # `debug_assert`s against one, because C reads these as C strings and
        # would truncate at it. There is no case for it here: the parser
        # refuses a NUL in a parameter ("Failed to parse the value of parameter
        # 's'") and truncates a query string at one, so no query can put such a
        # value on this path in the first place.

        self.assert_graph_eq()

    def test06_list_edges(self):
        # T_ARRAY is a count and then values, so nesting is the interesting
        # axis rather than length: depth costs a tag per level and a decoder
        # that recursed without a bound is what the codec's depth cap is for.
        self.set_effects_config()
        self.query_and_sync("""CREATE (:List {
            empty:  [],
            nested: [[1, [2, [3, [4, [5]]]]]],
            mixed:  [1, 'two', 3.5, true, [6]],
            long:   range(1, 10000)
        })""")
        self.assert_agree(
            """MATCH (n:List) RETURN size(n.empty), size(n.nested),
                      size(n.mixed), size(n.long),
                      n.nested[0][1][1][1][1][0],
                      n.mixed[1], n.mixed[4][0],
                      reduce(s = 0, x IN n.long | s + x)""",
            [[0, 1, 5, 10000, 5, 'two', 6, 50005000]])
        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# 4c. shapes the earlier classes do not reach
#-----------------------------------------------------------------------------
