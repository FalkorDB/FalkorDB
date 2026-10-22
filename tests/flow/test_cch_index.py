"""CCH indexes, end to end.

The CCH index exists only in a module built with `--features index-falkordb`,
so every class here skips unless `FALKORDB_FEATURES` declares it (see
`common.py`), and the file is not in `flow_tests_done.txt`, the default build's
CI matrix: CI runs it from `flow_tests_index_falkordb.txt`, against a module
built with the feature. All CCH flow classes share this one file.

Effects -- CCH index DDL replicates as C's v3 does it: a `CREATE_INDEX` /
`DROP_INDEX` record carrying `INDEX_FLD_CCH` (0x20), whose schemas are the
relationship types and whose one field is the weight attribute, after the
schema a create registers. See `effects_common.py` for the shared fixture.

Weights -- an indexed weight is a constraint: a write that would give an edge
of an indexed type a negative or NaN weight is refused.
"""

import struct

from common import *

from effects_common import _EffectsBase


#-----------------------------------------------------------------------------
# 1. CCH DDL replicates, schema first
#-----------------------------------------------------------------------------


NOT_HELD = "no CCH index over these relationship types and weight attribute"
ALREADY_HELD = ("a CCH index already exists over these relationship types and "
                "weight attribute")


class testEffects_01_CchDdlReplicates(_EffectsBase):
    """CREATE/DROP CCH INDEX reach the replica as effects.

    A create registers a missing relationship type or weight attribute, as C
    does, and announces it ahead of the index record, so the replica resolves
    every id the record names. The replica builds its own hierarchy; the record
    carries the definition only, in C's v3 byte layout.

    Neither side lists a CCH index in `db.indexes()` yet, so the replica's
    index is observed the one way a client can: with writes allowed on the
    replica, a CREATE of the same index is refused as already held, and after
    the replicated drop, a DROP is refused as not held. Both probes are refused
    queries, so they change nothing on the replica.
    """

    GRAPH_ID = "effects_cch_ddl"

    # C's v3 payload header: the version byte, then the flags byte, clear
    # (`_EffectsBuffer_WriteHeader`, src/effects/effects.c).
    HEADER = b"\x03\x00"

    def __init__(self):
        if not INDEX_FALKORDB:
            Environment.skip(None)
        self._setup()

    @staticmethod
    def _schema_of(graph):
        """(relationship types, property keys), in id order."""
        return tuple([r[0] for r in graph.ro_query(q).result_set]
                     for q in ("CALL db.relationshipTypes()",
                               "CALL db.propertyKeys()"))

    def _schema(self):
        """The primary's schema, asserted equal (same ids) on the replica."""
        m = self._schema_of(self.master_graph)
        self.env.assertEqual(self._schema_of(self.replica_graph), m)
        return m

    def _refused(self, q, *needles, read_only=False, graph=None):
        """`q` fails (on the primary by default) with an error containing
        every needle."""
        graph = graph or self.master_graph
        run = graph.ro_query if read_only else graph.query
        try:
            run(q)
        except ResponseError as e:
            for needle in needles:
                self.env.assertContains(needle, str(e))
            return
        self.env.assertTrue(False, message=f"{q} was not refused")

    def _on_replica(self, q, *needles, key=None):
        """`q` is refused on the replica with `needles`: a write probe, run with
        writes briefly allowed there."""
        graph = Graph(self.replica, key) if key else self.replica_graph
        self.replica.config_set("replica-read-only", "no")
        try:
            self._refused(q, *needles, graph=graph)
        finally:
            self.replica.config_set("replica-read-only", "yes")

    def _healthy(self, full_before, failures_before):
        self.env.assertEqual(self.master.info()["sync_full"], full_before,
            message="the replica was resynced")
        self.env.assertEqual(self.effect_failures(), failures_before,
            message="the replica refused a GRAPH.EFFECT")

    @staticmethod
    def _c_string(s):
        """C's `EffectsBuffer_WriteString`: `size_t` length counting the NUL,
        then the bytes and the NUL. The length is in bytes, not characters."""
        b = s.encode()
        return struct.pack("<Q", len(b) + 1) + b + b"\0"

    @classmethod
    def _c_add_type(cls, type_id, name):
        """C's v3 `ADD_SCHEMA` (9) for a relationship type: opcode, `SchemaType`
        `SCHEMA_EDGE`, the id as an `int`, the name."""
        return struct.pack("<IIi", 9, 1, type_id) + cls._c_string(name)

    @classmethod
    def _c_add_attribute(cls, attr_id, name):
        """C's v3 `ADD_ATTRIBUTE` (10): opcode, `AttributeID` (u16), name."""
        return struct.pack("<IH", 10, attr_id) + cls._c_string(name)

    @classmethod
    def _c_cch_record(cls, create, weight, types):
        """The record C's v3 encoder writes for a CCH statement
        (`_encode_ddl_record` over `EffectsV3Grouping_AddCCHIndex`'s
        announcement): `CREATE_INDEX` (11) or `DROP_INDEX` (12), `SCHEMA_EDGE`,
        a u16 count and each type as (u32 id, name) ascending by id with
        repeats kept, field type `INDEX_FLD_CCH` (0x20), one field -- the
        weight, (u16 id, name) -- and on a create five clear option presence
        bytes."""
        weight_id, weight_name = weight
        out = struct.pack("<IIH", 11 if create else 12, 1, len(types))
        for type_id, type_name in sorted(types, key=lambda t: t[0]):
            out += struct.pack("<I", type_id) + cls._c_string(type_name)
        out += struct.pack("<IHH", 0x20, 1, weight_id) + cls._c_string(weight_name)
        if create:
            out += b"\0" * 5
        return out

    def test01_create_registers_schema_and_replicates_the_index(self):
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]
        failures_before = self.effect_failures()
        self.start_monitor('GRAPH.EFFECT')
        self.assert_effect_emitted(count=0)

        # Nothing exists yet: the create registers both types and the
        # attribute, like C, and the replica gets them with the same ids.
        res = self.query_and_sync(
            "CREATE CCH INDEX FOR ()-[e:ROAD|FERRY|ROAD]->() ON (e.time)")
        self.env.assertEqual(res.indices_created, 1)
        self.env.assertEqual(self._schema(), (['ROAD', 'FERRY'], ['time']))

        # One payload, every record of it: ADD_SCHEMA x2, ADD_ATTRIBUTE, then
        # the CCH index record naming ROAD twice, as the statement did.
        window = self.assert_effect_emitted(count=1)
        [payload] = self.effect_payloads(window, self.GRAPH_ID)
        types = [(0, 'ROAD'), (1, 'FERRY'), (0, 'ROAD')]
        self.env.assertEqual(self.payload_bytes(payload),
            self.HEADER
            + self._c_add_type(0, 'ROAD') + self._c_add_type(1, 'FERRY')
            + self._c_add_attribute(0, 'time')
            + self._c_cch_record(True, (0, 'time'), types))

        # The replica holds the index.
        self._on_replica("CREATE CCH INDEX FOR ()-[e:FERRY|ROAD]->() ON (e.time)",
                         ALREADY_HELD)

        # Later writes over the indexed types still apply.
        self.query_and_sync("CREATE (:N)-[:ROAD {time: 2.0}]->(:N)")
        self.query_and_sync("CREATE (:N)-[:FERRY {time: 4.0, length: 5}]->(:N)")
        self.env.assertEqual(self._schema(),
                             (['ROAD', 'FERRY'], ['time', 'length']))
        self.assert_graph_eq()
        self.assert_effect_emitted(count=2)

        # A drop is the index record alone: it registers nothing.
        res = self.query_and_sync(
            "DROP CCH INDEX FOR ()-[e:ROAD|FERRY]->() ON (e.time)")
        self.env.assertEqual(res.indices_deleted, 1)
        window = self.assert_effect_emitted(count=1)
        [payload] = self.effect_payloads(window, self.GRAPH_ID)
        self.env.assertEqual(self.payload_bytes(payload),
            self.HEADER + self._c_cch_record(False, (0, 'time'),
                                             [(0, 'ROAD'), (1, 'FERRY')]))
        self._on_replica("DROP CCH INDEX FOR ()-[e:ROAD|FERRY]->() ON (e.time)",
                         NOT_HELD)

        # A DROP naming a type or an attribute that does not exist matches no
        # index, is refused like the drop of an index that is not held, and
        # creates nothing.
        for q in ("DROP CCH INDEX FOR ()-[e:TRAIN]->() ON (e.time)",
                  "DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.speed)"):
            self._refused(q, NOT_HELD)
        self.env.assertEqual(self._schema(),
                             (['ROAD', 'FERRY'], ['time', 'length']))
        self.assert_effect_emitted(count=0)

        self._healthy(full_before, failures_before)

    def test02_c_bytes_apply_as_they_are(self):
        """A payload built from C's v3 layout -- the two-byte header, then the
        CCH index record -- sent as `GRAPH.EFFECT`, builds and drops the index
        on the primary and, replicated, on the replica.

        Over schema that already exists, this is the whole payload a C primary
        emitting v3 sends for the statement (its v3 header and framing are
        this engine's), so it is what a Rust replica of that primary receives.
        C emits v2 by default today (`EFFECTS_VERSION_EMIT`); v3 is what it
        sends once that switch moves."""
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]
        failures_before = self.effect_failures()
        key = "effects_cch_from_c"
        g = Graph(self.master, key)
        # Names whose byte length is not their character count: `道路` is
        # type 1 and `время` attribute 1.
        g.query("CREATE (:N)-[:A {w: 1.0}]->(:N)")
        g.query("CREATE (:N)-[:`道路` {`время`: 2.0}]->(:N)")
        self.wait_for_replica_offset()

        def effect(record):
            self.env.assertEqual(
                self.master.execute_command("GRAPH.EFFECT", key,
                                            self.HEADER + record), "OK")

        def refused(record, needle):
            try:
                self.master.execute_command("GRAPH.EFFECT", key,
                                            self.HEADER + record)
                self.env.assertTrue(False, message="a diverged record applied")
            except ResponseError as e:
                self.env.assertContains(needle, str(e))

        # `[:道路|A|道路]`, repeats included, as C sends them.
        types = [(1, '道路'), (0, 'A'), (1, '道路')]
        weight = (1, 'время')
        spelled = "()-[e:A|`道路`]->() ON (e.`время`)"
        effect(self._c_cch_record(True, weight, types))
        self._refused(f"CREATE CCH INDEX FOR {spelled}", ALREADY_HELD, graph=g)
        self.wait_for_replica_offset()
        self._on_replica(f"CREATE CCH INDEX FOR {spelled}", ALREADY_HELD,
                         key=key)

        # Idempotent, as C's v3 apply is.
        effect(self._c_cch_record(True, weight, types))
        effect(self._c_cch_record(False, weight, types))
        effect(self._c_cch_record(False, weight, types))
        self._refused(f"DROP CCH INDEX FOR {spelled}", NOT_HELD, graph=g)
        self.wait_for_replica_offset()
        self._on_replica(f"DROP CCH INDEX FOR {spelled}", NOT_HELD, key=key)

        # An id that names another type or attribute here is divergence,
        # refused by name -- on the primary only: a refused GRAPH.EFFECT is
        # not replicated.
        refused(self._c_cch_record(True, weight, [(0, '道路')]),
                "'道路' (id 0), which is 'A' here")
        refused(self._c_cch_record(True, (0, 'время'), types),
                "'время' (id 0), which is 'w' here")

        self.wait_for_replica_offset()
        self._healthy(full_before, failures_before)

    def test03_create_drop_and_refusals(self):
        self.set_effects_config()
        self.query_and_sync("CREATE (:N)-[:A {w: 1.0}]->(:N)-[:B {w: 2.0}]->(:N)")

        # The types are a set: repeats and order do not make a new index.
        res = self.query_and_sync("CREATE CCH INDEX FOR ()-[e:A]->() ON (e.w)")
        self.env.assertEqual(res.indices_created, 1)
        self._refused("CREATE CCH INDEX FOR ()-[e:A|A]->() ON (e.w)",
                      ALREADY_HELD)
        res = self.query_and_sync(
            "CREATE CCH INDEX FOR ()-[e:A|B]->() ON (e.w)")
        self.env.assertEqual(res.indices_created, 1)
        self._refused("CREATE CCH INDEX FOR ()-[e:B|A]->() ON (e.w)",
                      ALREADY_HELD)

        # CCH DDL is a write: GRAPH.RO_QUERY refuses it.
        for verb in ("CREATE", "DROP"):
            self._refused(f"{verb} CCH INDEX FOR ()-[e:A]->() ON (e.w)",
                          "graph.RO_QUERY is to be executed only on read-only "
                          "queries", read_only=True)

        # A drop names the same set in any spelling, and frees it for a create.
        res = self.query_and_sync(
            "DROP CCH INDEX FOR ()-[e:B|A|B]->() ON (e.w)")
        self.env.assertEqual(res.indices_deleted, 1)
        self._refused("DROP CCH INDEX FOR ()-[e:A|B]->() ON (e.w)", NOT_HELD)
        res = self.query_and_sync(
            "CREATE CCH INDEX FOR ()-[e:A|B]->() ON (e.w)")
        self.env.assertEqual(res.indices_created, 1)

        # The replica followed every step: it holds A and A|B, and not B.
        # (Every probe must be refused: one that ran would diverge the replica.)
        for types in ("A", "B|A"):
            self._on_replica(f"CREATE CCH INDEX FOR ()-[e:{types}]->() ON (e.w)",
                             ALREADY_HELD)
        self._on_replica("DROP CCH INDEX FOR ()-[e:B]->() ON (e.w)", NOT_HELD)
        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# 2. An indexed weight is a constraint: refused at the write
#-----------------------------------------------------------------------------


class testCchWeightConstraint():
    """A negative (-inf included) or NaN weight on an edge of an indexed type fails
    the write that sets it, through Cypher and GRAPH.BULK alike, and leaves the graph
    as it was. Other types and attributes are not checked. +inf is legal: a closed
    edge."""

    def __init__(self):
        if not INDEX_FALKORDB:
            Environment.skip(None)
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.graph = self.db.select_graph("cch_weight_constraint")
        self.graph.query(
            "CREATE (a:N {id: 0}), (b:N {id: 1}), (c:N {id: 2}), "
            "(a)-[:ROAD {time: 1.0}]->(b), (b)-[:ROAD {time: 2.0}]->(c), "
            "(a)-[:FERRY {time: 9.0}]->(c)")
        self.graph.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")

    def _refused(self, run):
        try:
            run()
        except ResponseError as e:
            self.env.assertContains("CCH weight constraint violation", str(e))
            self.env.assertContains("(:ROAD {time})", str(e))
            self.env.assertContains(
                "needs weights that are >= 0 (+inf marks a closed edge)", str(e))
            return
        self.env.assertTrue(False, message="the write was not refused")

    def _unchanged(self):
        self.env.assertEquals(self.graph.ro_query(
            "MATCH ()-[r]->() RETURN type(r), r.time ORDER BY type(r), r.time").result_set,
            [["FERRY", 9.0], ["ROAD", 1.0], ["ROAD", 2.0]])

    def test01_cypher(self):
        # One write shape reaching the client; maint.rs's unit tests cover every shape.
        self._refused(lambda: self.graph.query("MATCH ()-[r:ROAD]->() SET r.time = -1"))
        self._unchanged()
        # Not indexed: another type. Then put it back.
        self.graph.query("MATCH ()-[r:FERRY]->() SET r.time = -1")
        self.graph.query("MATCH ()-[r:FERRY]->() SET r.time = 9.0")
        self._unchanged()

    def test02_bulk(self):
        # Append one edge 0 -> 2 to the existing graph: type, one double property `time`.
        def edge(rel, w):
            return (rel.encode() + b"\0" + struct.pack("=I", 1) + b"time\0" +
                    struct.pack("=QQ", 0, 2) + struct.pack("=Bd", 2, w))
        bulk = lambda rel, w: self.con.execute_command(
            "GRAPH.BULK", "cch_weight_constraint", 0, 1, 0, 1, edge(rel, w))
        self._refused(lambda: bulk("ROAD", -1.0))
        self._unchanged()
        bulk("FERRY", -1.0)
        self.env.assertEquals(self.graph.ro_query(
            "MATCH ()-[r:FERRY]->() RETURN count(r)").result_set, [[2]])

    def test03_closed_edge(self):
        # +inf (1.0/0.0 in Cypher) is a closed edge: legal through SET, CREATE and
        # GRAPH.BULK, and an index can be created over it. -inf is refused.
        g = self.db.select_graph("cch_closed_edge")
        g.query("CREATE (a:N {id: 0}), (b:N {id: 1}), (c:N {id: 2}), "
                "(a)-[:ROAD {time: 1.0}]->(b), (b)-[:ROAD {time: 1.0 / 0.0}]->(c)")
        g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        g.query("MATCH (:N {id: 0})-[r:ROAD]->() SET r.time = 1.0 / 0.0")
        g.query("MATCH (a:N {id: 0}), (c:N {id: 2}) CREATE (a)-[:ROAD {time: 1.0 / 0.0}]->(c)")
        edge = (b"ROAD\0" + struct.pack("=I", 1) + b"time\0" + struct.pack("=QQ", 2, 0) +
                struct.pack("=Bd", 2, float("inf")))
        self.con.execute_command("GRAPH.BULK", "cch_closed_edge", 0, 1, 0, 1, edge)
        self.env.assertEquals(g.ro_query(
            "MATCH ()-[r:ROAD]->() RETURN toString(r.time), count(r)").result_set,
            [["inf", 4]])
        try:
            g.query("MATCH (:N {id: 0})-[r:ROAD]->(:N {id: 1}) SET r.time = -1.0 / 0.0")
            self.env.assertTrue(False, message="-inf was not refused")
        except ResponseError as e:
            self.env.assertContains("CCH weight constraint violation", str(e))

        # CREATE CCH INDEX over data: +inf is fine, a negative weight names the rule.
        g.query("MATCH ()-[r:ROAD]->() SET r.len = 1.0 / 0.0")
        g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.len)")
        g.query("MATCH ()-[r:ROAD]->() SET r.cost = -2")
        try:
            g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.cost)")
            self.env.assertTrue(False, message="the index was created over -2")
        except ResponseError as e:
            self.env.assertContains("edge weights must be >= 0 (+inf marks a closed edge)",
                                    str(e))

    def test04_a_closed_edge_is_routed_around(self):
        # 0 -> 1 -> 2 costs 2, the direct 0 -> 2 costs 10.
        g = self.db.select_graph("cch_closed_route")
        g.query("CREATE (a:N {id: 0}), (b:N {id: 1}), (c:N {id: 2}), "
                "(a)-[:ROAD {time: 1.0}]->(b), (b)-[:ROAD {time: 1.0}]->(c), "
                "(a)-[:ROAD {time: 10.0}]->(c)")
        g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        q = ("MATCH (a:N {id: $s}), (b:N {id: $t}) "
             "CALL db.idx.cch.query({sourceNode: a, targetNode: b, relTypes: ['ROAD'], "
             "weightProp: 'time'}) YIELD pathWeight, path "
             "RETURN pathWeight, [n IN nodes(path) | n.id]")
        route = lambda: g.ro_query(q, {"s": 0, "t": 2}).result_set
        self.env.assertEquals(route(), [[2.0, [0, 1, 2]]])
        # Close 1 -> 2: the detour is taken.
        g.query("MATCH (:N {id: 1})-[r:ROAD]->(:N {id: 2}) SET r.time = 1.0 / 0.0")
        self.env.assertEquals(route(), [[10.0, [0, 2]]])
        # A finite edge parallel to the closed one wins the pair.
        g.query("MATCH (b:N {id: 1}), (c:N {id: 2}) CREATE (b)-[:ROAD {time: 3.0}]->(c)")
        self.env.assertEquals(route(), [[4.0, [0, 1, 2]]])
        # Every way into 2 closed: no row.
        g.query("MATCH ()-[r:ROAD]->(:N {id: 2}) SET r.time = 1.0 / 0.0")
        self.env.assertEquals(route(), [])
        # Reopen every edge into 2 at 1.0: the direct edge now wins.
        g.query("MATCH ()-[r:ROAD]->(:N {id: 2}) SET r.time = 1.0")
        self.env.assertEquals(route(), [[1.0, [0, 2]]])


#-----------------------------------------------------------------------------
# 3. The hierarchy is built in the background
#-----------------------------------------------------------------------------


class testCchBackgroundBuild():
    """CREATE CCH INDEX checks the weights and returns; the index builder builds
    the hierarchy afterwards, off the GIL, as it builds a column. The build is
    observed through GRAPH.MEMORY USAGE, which counts a hierarchy once it is
    adopted, and through db.indexes, which lists its state."""

    GRAPH = "cch_background_build"

    def __init__(self):
        if not INDEX_FALKORDB:
            Environment.skip(None)
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.graph = self.db.select_graph(self.GRAPH)
        # A 100 x 100 grid: its hierarchy takes a few MiB, enough to show in whole MiB.
        side = 100
        params = {'n': side * side, 'side': side}
        self.graph.query("UNWIND range(0, $n - 1) AS x CREATE (:J)", params)
        self.graph.query("""UNWIND range(0, $n - 1) AS x
                            MATCH (a) WHERE id(a) = x
                            MATCH (b) WHERE id(b) = x + 1 AND x % $side <> $side - 1
                            CREATE (a)-[:ROAD {time: 1.0}]->(b)""", params)
        self.graph.query("""UNWIND range(0, $n - $side - 1) AS x
                            MATCH (a) WHERE id(a) = x
                            MATCH (b) WHERE id(b) = x + $side
                            CREATE (a)-[:ROAD {time: 1.0}]->(b)""", params)

    def _indices_mb(self):
        res = self.con.execute_command("GRAPH.MEMORY", "USAGE", self.GRAPH)
        return dict(zip(res[0::2], res[1::2]))['indices_sz_mb']

    def _wait_for_memory(self, done, timeout=30):
        deadline = time.time() + timeout
        while not done(self._indices_mb()):
            if time.time() > deadline:
                raise AssertionError("the CCH build did not finish in time")
            time.sleep(0.05)

    def test01_built_after_create_returns_and_released_by_drop(self):
        self.env.assertEqual(self._indices_mb(), 0)
        res = self.graph.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        self.env.assertEqual(res.indices_created, 1)
        # The builder adopts the hierarchy shortly after.
        self._wait_for_memory(lambda mb: mb > 0)
        # Held: a second CREATE is refused whatever the build's state.
        try:
            self.graph.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
            self.env.assertTrue(False, message="a second CREATE was not refused")
        except ResponseError as e:
            self.env.assertContains("already exists", str(e))
        self.graph.query("DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        self.env.assertEqual(self._indices_mb(), 0)

    def test02_drop_and_create_again_right_away(self):
        # A drop, and a create again, while the first build may still run: the
        # first build is discarded and the second one is the index adopted.
        self.graph.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        self.graph.query("DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        self.graph.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        self._wait_for_memory(lambda mb: mb > 0)
        self.graph.query("DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        self.env.assertEqual(self._indices_mb(), 0)
        self.env.assertTrue(self.con.ping())

    def _status(self):
        rows = self.graph.ro_query("CALL db.indexes()").result_set
        return [r[7] for r in rows
                if r[2] and any(v == ["CCH"] for v in r[2].values())]

    def test03_listed_by_state_and_queried_in_any(self):
        # Whatever the build's state, the query answers: from the hierarchy once
        # OPERATIONAL, by searching the graph while UNDER CONSTRUCTION.
        q = ("MATCH (a), (b) WHERE id(a) = 0 AND id(b) = $t "
             "CALL db.idx.cch.query({sourceNode: a, targetNode: b, relTypes: ['ROAD'], "
             "weightProp: 'time'}) YIELD pathWeight RETURN pathWeight")
        self.graph.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        self.env.assertContains(self._status()[0], ["UNDER CONSTRUCTION", "OPERATIONAL"])
        corner = 100 * 100 - 1
        self.env.assertEquals(self.graph.ro_query(q, {"t": corner}).result_set, [[198.0]])
        deadline = time.time() + 30
        while self._status() != ["OPERATIONAL"]:
            self.env.assertLess(time.time(), deadline)
            time.sleep(0.05)
        self.env.assertEquals(self.graph.ro_query(q, {"t": corner}).result_set, [[198.0]])
        self.graph.query("DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")


#-----------------------------------------------------------------------------
# 4. Writes maintain the index through its change set
#-----------------------------------------------------------------------------


QUERY = ("MATCH (a:N {{id: $s}}), (b:N {{id: $t}}) "
         "CALL db.idx.cch.query({{sourceNode: a, targetNode: b, relTypes: {types}, "
         "weightProp: '{weight}'}}) YIELD pathWeight RETURN pathWeight")


def _dijkstra(n, edges, s):
    """Cheapest weights from `s` over `edges` [(u, v, w)]; +inf weights are closed."""
    import heapq
    adj = {}
    for u, v, w in edges:
        if w != float("inf"):
            adj.setdefault(u, []).append((v, w))
    dist = {s: 0.0}
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            if d + w < dist.get(v, float("inf")):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return dist


class testCchChangeSet():
    """GRAPH.BULK appends of edges and of nodes with edges keep the index answering
    like Dijkstra on the graph as it stands: the bulk command reaches the change set as
    Cypher writes do (those are the unit tests' random writes). Most appends are absorbed
    by a scoped recustomization of the held hierarchy; an edge between nodes it does not
    join, an edge onto a node past its capacity included, sends it back to the builder,
    and the query searches the graph until the rebuilt hierarchy is adopted. A node alone
    changes nothing."""

    def __init__(self):
        if not INDEX_FALKORDB:
            Environment.skip(None)
        self.env, self.db = Env()
        self.con = self.env.getConnection()

    def _edges(self, g):
        return [(u, v, w) for u, v, w in g.ro_query(
            "MATCH (a:N)-[r:ROAD]->(b:N) RETURN a.id, b.id, "
            "CASE WHEN r.time IS NULL THEN 1.0 ELSE toFloat(r.time) END").result_set]

    def _check(self, g, ids, rng, what):
        edges = self._edges(g)
        q = QUERY.format(types="['ROAD']", weight="time")
        for s in rng.sample(ids, 6):
            dist = _dijkstra(len(ids), edges, s)
            for t in rng.sample(ids, 8):
                got = g.ro_query(q, {"s": s, "t": t}).result_set
                want = dist.get(t)
                if want is None:
                    self.env.assertEquals(got, [], message=f"{what}: {s}->{t}")
                else:
                    self.env.assertEquals(len(got), 1, message=f"{what}: {s}->{t}")
                    if got:
                        self.env.assertAlmostEqual(got[0][0], want, 1e-9,
                                                   message=f"{what}: {s}->{t}")

    def test01_bulk_appends(self):
        import random
        rng = random.Random(11)
        g = self.db.select_graph("cch_change_set_bulk")
        n = 40
        g.query(f"UNWIND range(0, {n - 1}) AS i CREATE (:N {{id: i}})")
        g.query("UNWIND range(0, 38) AS i MATCH (a:N {id: i}), (b:N {id: i + 1}) "
                "CREATE (a)-[:ROAD {time: 2.0}]->(b), (b)-[:ROAD {time: 2.0}]->(a)")
        g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        ids = list(range(n))
        self._check(g, ids, rng, "after create")

        def edges_token(pairs):
            out = b"ROAD\0" + struct.pack("=I", 1) + b"time\0"
            for u, v, w in pairs:
                out += struct.pack("=QQ", u, v) + struct.pack("=Bd", 2, w)
            return out

        # Edges between joined pairs (the reverse of a path edge): absorbed.
        self.con.execute_command("GRAPH.BULK", "cch_change_set_bulk", 0, 3, 0, 1,
                                 edges_token([(5, 4, 0.5), (9, 10, 0.25), (20, 19, float("inf"))]))
        self._check(g, ids, rng, "bulk joined")
        # Edges between unjoined pairs: back to the builder.
        self.con.execute_command("GRAPH.BULK", "cch_change_set_bulk", 0, 2, 0, 1,
                                 edges_token([(0, 39, 1.0), (39, 0, 1.0)]))
        self._check(g, ids, rng, "bulk unjoined")
        # New nodes (ids 40.., `N` with an `id`) and edges onto them, in one insert.
        k = 30
        nodes = b"N\0" + struct.pack("=I", 1) + b"id\0"
        for i in range(k):
            nodes += struct.pack("=Bq", 4, n + i)
        rels = edges_token([(n + i, (n + i * 7) % (n + k), 1.0) for i in range(k)
                            if n + i != (n + i * 7) % (n + k)] + [(0, n, 3.0)])
        cnt = sum(1 for i in range(k) if n + i != (n + i * 7) % (n + k)) + 1
        self.con.execute_command("GRAPH.BULK", "cch_change_set_bulk", k, cnt, 1, 1, nodes, rels)
        ids += list(range(n, n + k))
        self._check(g, ids, rng, "bulk new nodes")
