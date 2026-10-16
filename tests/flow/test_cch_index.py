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

import os
import struct
import time

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
# 2. RDB v20: the CCH definitions payload
#-----------------------------------------------------------------------------

GRAPH_ID = "rdb_cch_layout"

# BufferedWriter frames every unsigned word as a 1-byte type tag plus eight
# little-endian bytes.
TYPE_UNSIGNED = 4
WORD_BYTES = 9

# EncodeState discriminators, as written to the payload directory of a key.
NODES, EDGES, LABELS_MATRICES, RELATION_MATRICES, ADJ_MATRIX, LBLS_MATRIX, \
    CCH_INDICES, FINAL = 1, 3, 6, 7, 8, 9, 10, 11

# Redis RDB opcodes and module-value opcodes.
MODULE_OPCODE_EOF = 0
MODULE_OPCODE_STRING = 5

# Relationship types are created in this order, so their ids are 0, 1, 2.
RAIL, ROAD, FERRY = 0, 1, 2
ENCODING_VERSION = 20

# Redis encodes a module type id as nine 6-bit characters, then the 10-bit encver.
_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def word(v):
    return bytes([TYPE_UNSIGNED]) + v.to_bytes(8, "little")


def words(*vs):
    return b"".join(word(v) for v in vs)


def module_type_id(name, encver):
    v = 0
    for c in name:
        v = (v << 6) | _B64.index(c)
    return (v << 10) | encver


def _module_head(name, encver):
    """A module type id as RDB saves it: a 64-bit length. It follows the key name,
    so the id alone is what can be searched for."""
    return b"\x81" + module_type_id(name, encver).to_bytes(8, "big")


def read_len(blob, i):
    """Decode a Redis RDB length at `blob[i]`; returns (length, bytes consumed)."""
    kind = blob[i] >> 6
    if kind == 0:
        return blob[i] & 0x3F, 1
    if kind == 1:
        return ((blob[i] & 0x3F) << 8) | blob[i + 1], 2
    if blob[i] == 0x80:
        return int.from_bytes(blob[i + 1:i + 5], "big"), 5
    if blob[i] == 0x81:
        return int.from_bytes(blob[i + 1:i + 9], "big"), 9
    raise AssertionError(f"not a plain RDB length: {blob[i]:#x}")


def cch_payload(*indexes):
    """What C's `RdbSaveCCH_v20` writes: the index count, then per index its
    relationship-type count, the type ids and the weight attribute id."""
    out = words(len(indexes))
    for types, weight_attr in indexes:
        out += words(len(types), *types, weight_attr)
    return out


class testRdbCchLayout():
    """Guard the CCH definitions payload of the RDB against C's v20 encoder.

    Like `test_rdb_c_layout.py`, these assert the bytes of a real saved RDB. A
    Rust-to-Rust reload cannot see a wrong layout: our decoder reads what our
    encoder writes, so swapping the type count with the ids, or writing the
    weight attribute as a signed word, round-trips cleanly while a C server
    reads a different index (or garbage).

    Only the definition is persisted -- relationship types and the weight
    attribute; the hierarchy is rebuilt from the graph on load. C's comment on
    `RdbSaveCCH_v20` still says "full hierarchy", but its code (and
    `CCHIndex_RdbSave`) writes nothing else.

    Reference: `RdbSaveCCH_v20` in `src/serializers/encoder/v20/encode_schema.c`,
    `CCHIndex_RdbSave` in `src/index/cch_index.c`, and the
    `ENCODE_STATE_CCH_INDICES` cases of `encode_graph.c`, on the `master` branch.
    """

    def __init__(self):
        if not INDEX_FALKORDB:
            Environment.skip(None)
        self.env, self.db = Env()
        # These read the RDB off disk, which the shared services container does not
        # expose to the test process.
        if os.getenv("FALKORDB_USE_SERVICE"):
            Environment.skip(None)
        self.con = self.env.getConnection()
        # LZF would defeat every assertion here.
        self.con.execute_command("CONFIG", "SET", "rdbcompression", "no")

    def _reset(self):
        self.con.execute_command("FLUSHALL")

    def _build(self, name):
        """4 nodes, 5 edges over three relationship types. Types are created in
        the order RAIL, ROAD, FERRY and attributes in the order time, dist, so
        the ids the payload must carry are known."""
        graph = self.db.select_graph(name)
        graph.query(
            "CREATE (a:N), (b:N), (c:N), (d:N), "
            "(a)-[:RAIL {time: 1.0, dist: 2.0}]->(b), "
            "(b)-[:ROAD {time: 2.0, dist: 1.0}]->(c), "
            "(c)-[:ROAD {time: 1.5, dist: 1.5}]->(d), "
            "(a)-[:FERRY {time: 4.0, dist: 3.0}]->(d), "
            "(b)-[:FERRY {time: 3.0, dist: 1.0}]->(d)")
        types = [r[0] for r in graph.ro_query("CALL db.relationshipTypes()").result_set]
        keys = [r[0] for r in graph.ro_query("CALL db.propertyKeys()").result_set]
        self.env.assertEquals(types, ["RAIL", "ROAD", "FERRY"])
        self.env.assertEquals(keys[:2], ["time", "dist"])
        return graph

    def _save(self):
        """SAVE and return the RDB as raw bytes, having checked that exactly one
        user graph is in it (the engine's own `telemetry{...}` graph is not)."""
        keys = [k for k in self.con.execute_command("KEYS", "*")
                if not str(k).startswith("telemetry")]
        self.env.assertEquals(len(keys), 1,
            message=f"exactly one user graph should exist while dumping; found {keys}")
        self.con.execute_command("SAVE")
        directory = self.con.execute_command("CONFIG", "GET", "dir")[1]
        filename = self.con.execute_command("CONFIG", "GET", "dbfilename")[1]
        with open(os.path.join(directory, filename), "rb") as f:
            return f.read()

    def _key_chunk(self, blob):
        """The one buffer a small graph's key is written as: (start, end) of its
        bytes in `blob`. The key is `RDB_TYPE_MODULE_2`, its name, the `graphdata`
        type id (a 64-bit length), a STRING opcode and the buffer, then the module
        EOF opcode."""
        head = _module_head("graphdata", ENCODING_VERSION)
        # The type also saves an AUX field at the top of the file, with the same
        # id; the key's value is the occurrence that opens with a STRING opcode.
        found = []
        at = blob.find(head)
        while at >= 0:
            if blob[at + len(head)] == MODULE_OPCODE_STRING:
                found.append(at + len(head))
            at = blob.find(head, at + 1)
        self.env.assertEquals(len(found), 1,
            message="expected exactly one `graphdata` key with encoding version "
                    f"{ENCODING_VERSION} in the RDB")
        at = found[0]
        n, used = read_len(blob, at + 1)
        start = at + 1 + used
        self.env.assertEquals(blob[start + n], MODULE_OPCODE_EOF,
            message="the graph key must be exactly one buffer")
        return start, start + n

    def _directory_end(self, chunk):
        """Offset just past the (adjacency matrix, labels matrix) entries, which
        every graph key has at the end of its payload directory (or just before
        the CchIndices entry)."""
        tail = words(ADJ_MATRIX, 1, LBLS_MATRIX, 1)
        ends = []
        i = chunk.find(tail)
        while i >= 0:
            ends.append(i + len(tail))
            i = chunk.find(tail, i + 1)
        self.env.assertEquals(len(ends), 1,
            message="the adjacency/labels pair must appear exactly once in the key")
        return ends[0]

    def _states_before(self, chunk, end, pairs):
        """Decode the `pairs` (state, count) entries ending at `end` and the count
        word before them: (offset of that count word, its value, [(state, count)])."""
        start = end - pairs * 2 * WORD_BYTES - WORD_BYTES
        ws = [int.from_bytes(chunk[start + 1 + k * WORD_BYTES:start + WORD_BYTES * (k + 1)], "little")
              for k in range(2 * pairs + 1)]
        return start, ws[0], list(zip(ws[1::2], ws[2::2]))

    def _drop_all(self, graph, ddl):
        for stmt in ddl:
            graph.query(stmt.replace("CREATE", "DROP", 1))

    DDL_ROAD_TIME = "CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)"
    # Written FERRY|RAIL on purpose: the ids in the payload must come out sorted.
    DDL_FERRY_RAIL_DIST = "CREATE CCH INDEX FOR ()-[e:FERRY|RAIL]->() ON (e.dist)"

    def test01_payload_follows_the_last_payload_as_c_writes_it(self):
        self._reset()
        graph = self._build(GRAPH_ID + "_two")
        ddl = [self.DDL_ROAD_TIME, self.DDL_FERRY_RAIL_DIST]
        for stmt in ddl:
            graph.query(stmt)

        blob = self._save()
        start, end = self._key_chunk(blob)
        chunk = blob[start:end]

        # time is attribute 0, dist is 1; types are sorted whatever the DDL order,
        # and the indexes are saved in definition order (sorted type ids, then the
        # weight attribute), not in creation order: RAIL|FERRY (0, 2) before ROAD (1).
        expected = cch_payload(([RAIL, FERRY], 1), ([ROAD], 0))
        self.env.assertTrue(chunk.endswith(expected),
            message="the key must end with the CCH payload: index count, then "
                    "per index the type count, the sorted type ids and the weight "
                    f"attribute id, each a type tag and 8 LE bytes. Wanted {expected.hex()}, "
                    f"key ends {chunk[-len(expected):].hex()}")

        # The directory announces the payload last, as state 10 (never 11), after
        # every payload a graph without CCH has.
        dir_end = self._directory_end(chunk)
        self.env.assertEquals(chunk[dir_end:dir_end + 2 * WORD_BYTES], words(CCH_INDICES, 1),
            message="the directory must end with (CchIndices=10, 1), right after the "
                    "labels-matrix entry")
        pairs = 7
        _, n, entries = self._states_before(chunk, dir_end + 2 * WORD_BYTES, pairs)
        self.env.assertEquals(n, pairs)
        self.env.assertEquals([s for s, _ in entries],
                              [NODES, EDGES, LABELS_MATRICES, RELATION_MATRICES,
                               ADJ_MATRIX, LBLS_MATRIX, CCH_INDICES])
        self.env.assertEquals(entries[-1], (CCH_INDICES, 1))
        self.env.assertFalse(any(s == FINAL for s, _ in entries),
            message="Final (11) is never written")

        # Position: payloads are written in directory order and the directory ends
        # with CchIndices, so the payload is the last thing in the key, after the
        # labels-matrix data. With both indexes dropped the same graph saves to a
        # key shorter by exactly one directory entry and the payload. (Not equal
        # bytes otherwise: the serialized GraphBLAS matrices differ from save to
        # save even with nothing changed.)
        self._drop_all(graph, ddl)
        plain = self._chunk_of_save()
        self.env.assertEquals(len(chunk) - len(plain), 2 * WORD_BYTES + len(expected),
            message="the CCH payload and its directory entry must be the only "
                    "difference in size from the same graph without a CCH index")
        graph.delete()

    def test03_no_index_writes_no_cch_bytes(self):
        # A graph without CCH indexes keeps exactly the pre-v20 directory: no
        # CchIndices entry, so no count word either. C gates the payload on the
        # same condition (`GraphContext_CCHIndexCount (gc) > 0`).
        self._reset()
        graph = self._build(GRAPH_ID + "_none")
        chunk = self._chunk_of_save()
        dir_end = self._directory_end(chunk)
        _, n, entries = self._states_before(chunk, dir_end, 6)
        self.env.assertEquals(n, 6)
        self.env.assertEquals([s for s, _ in entries],
                              [NODES, EDGES, LABELS_MATRICES, RELATION_MATRICES,
                               ADJ_MATRIX, LBLS_MATRIX])
        # An index that was created and dropped leaves nothing behind either.
        graph.query(self.DDL_ROAD_TIME)
        graph.query(self.DDL_ROAD_TIME.replace("CREATE", "DROP", 1))
        again = self._chunk_of_save()
        self.env.assertEquals(len(again), len(chunk),
            message="a dropped CCH index must not change what the graph saves")
        _, n, entries = self._states_before(again, self._directory_end(again), 6)
        self.env.assertEquals((n, entries[-1]), (6, (LBLS_MATRIX, 1)))
        graph.delete()

    def _chunk_of_save(self):
        blob = self._save()
        start, end = self._key_chunk(blob)
        return blob[start:end]


QUERY = ("MATCH (a:N {{id: $s}}), (b:N {{id: $t}}) "
         "CALL db.idx.cch.query({{sourceNode: a, targetNode: b, relTypes: {types}, "
         "weightProp: '{weight}'}}) YIELD pathWeight, path "
         "RETURN pathWeight, [n IN nodes(path) | n.id]")


class testRdbCchReload():
    """CCH definitions survive a reload and the hierarchy is rebuilt from them.

    Only definitions are saved, so a reload that answers `db.idx.cch.query` as
    before proves the rebuild ran, not just that the definition came back. The
    second test saves a definition this engine cannot rebuild and checks the
    graph still loads, with the rest of its indexes.
    """

    def __init__(self):
        if not INDEX_FALKORDB:
            Environment.skip(None)
        # DEBUG RELOAD is refused by redis unless enabled.
        self.env, self.db = Env(enableDebugCommand=True)
        if os.getenv("FALKORDB_USE_SERVICE"):
            Environment.skip(None)
        self.con = self.env.getConnection()
        self.con.execute_command("CONFIG", "SET", "rdbcompression", "no")
        # The corrupted-definition test edits a saved RDB, so its footer checksum
        # no longer matches.
        self.con.execute_command("DEBUG", "SET-SKIP-CHECKSUM-VALIDATION", 1)

    def _build(self, name):
        self.con.execute_command("FLUSHALL")
        graph = self.db.select_graph(name)
        graph.query(
            "CREATE (a:N {id: 0}), (b:N {id: 1}), (c:N {id: 2}), (d:N {id: 3}), "
            "(a)-[:ROAD {time: 1.0, dist: 5.0}]->(b), "
            "(b)-[:ROAD {time: 2.0, dist: 5.0}]->(c), "
            "(c)-[:ROAD {time: 1.5, dist: 5.0}]->(d), "
            "(a)-[:FERRY {time: 9.0, dist: 1.0}]->(d), "
            "(b)-[:FERRY {time: 9.0, dist: 1.0}]->(d)")
        graph.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        graph.query("CREATE CCH INDEX FOR ()-[e:ROAD|FERRY]->() ON (e.dist)")
        # Built in the background: wait until both are adopted, so what is listed
        # before the reload is what the reload's synchronous build lists.
        deadline = time.time() + 30
        while any(r[7] != "OPERATIONAL" for r in graph.ro_query("CALL db.indexes()").result_set
                  if r[2] and any(v == ["CCH"] for v in r[2].values())):
            self.env.assertLess(time.time(), deadline)
            time.sleep(0.05)
        return graph

    def _answers(self, graph):
        out = []
        for types, weight in (("['ROAD']", "time"), ("['FERRY','ROAD']", "dist")):
            for s, t in ((0, 3), (0, 2), (1, 3)):
                q = QUERY.format(types=types, weight=weight)
                out.append(graph.ro_query(q, {"s": s, "t": t}).result_set)
        return out

    def _indexes(self, graph):
        return sorted(map(str, graph.ro_query("CALL db.indexes()").result_set))

    def test01_reload_lists_the_indexes_and_answers_the_same(self):
        graph = self._build("rdb_cch_reload")
        before_idx = self._indexes(graph)
        before = self._answers(graph)
        self.env.assertEquals(len(before_idx), 2)
        # 0 -> 3 over ROAD by time is 1.0 + 2.0 + 1.5; by dist over both types the
        # ferry is one hop of 1.0.
        self.env.assertEquals(before[0], [[4.5, [0, 1, 2, 3]]])
        self.env.assertEquals(before[3], [[1.0, [0, 3]]])

        self.con.execute_command("DEBUG", "RELOAD")

        graph = self.db.select_graph("rdb_cch_reload")
        self.env.assertEquals(self._indexes(graph), before_idx)
        self.env.assertEquals(self._answers(graph), before)
        graph.delete()

    def test02_a_definition_that_cannot_be_rebuilt_is_dropped_with_a_warning(self):
        self._build("rdb_cch_dropped")
        # Type ids: ROAD 0, FERRY 1. Attribute ids: id 0, time 1, dist 2.
        keep, drop = ([0], 1), ([0, 1], 2)
        expected = cch_payload(keep, drop)
        self.con.execute_command("SAVE")
        path = os.path.join(self.con.execute_command("CONFIG", "GET", "dir")[1],
                            self.con.execute_command("CONFIG", "GET", "dbfilename")[1])
        with open(path, "rb") as f:
            blob = f.read()
        at = blob.find(expected)
        self.env.assertNotEqual(at, -1, message="the saved payload was not found")
        # Point the multi-type index at relationship type 99, which the graph does
        # not have. Same-size edit, so every length in the file stays valid.
        bad = cch_payload(keep, ([0, 99], 2))
        self.env.assertEquals(len(bad), len(expected))
        with open(path, "wb") as f:
            f.write(blob[:at] + bad + blob[at + len(expected):])

        # NOSAVE: load the edited file rather than overwrite it with a fresh save.
        self.con.execute_command("DEBUG", "RELOAD", "NOSAVE")

        graph = self.db.select_graph("rdb_cch_dropped")
        self.env.assertEquals(
            graph.ro_query("MATCH (n:N) RETURN count(n)").result_set, [[4]])
        idx = graph.ro_query("CALL db.indexes()").result_set
        self.env.assertEquals([(r[0], r[1]) for r in idx], [("ROAD", ["time"])],
            message="the intact index is rebuilt and the unbuildable one is dropped")
        self.env.assertEquals(
            graph.ro_query(QUERY.format(types="['ROAD']", weight="time"),
                           {"s": 0, "t": 3}).result_set,
            [[4.5, [0, 1, 2, 3]]])
        if self.env.log_path:
            with open(self.env.log_path, "rb") as f:
                self.env.assertContains(b"was not rebuilt and is dropped", f.read(),
                    message="the dropped definition must be logged as a warning")
        graph.delete()

    def test03_a_closed_edge_survives_a_reload(self):
        # +inf (a closed edge) is saved as the attribute it is, and the hierarchy
        # rebuilt on load routes around it as before.
        graph = self._build("rdb_cch_closed")
        graph.query("MATCH (:N {id: 1})-[r:ROAD]->(:N {id: 2}) SET r.time = 1.0 / 0.0")
        graph.query("MATCH (:N {id: 0})-[r:FERRY]->() SET r.dist = 1.0 / 0.0")
        before = self._answers(graph)
        # ROAD by time: 1 -> 2 is closed, so nothing reaches 3 from 0 or 1. By dist:
        # the ferry 0 -> 3 is closed, so 0 -> 1 -> 3 takes the other ferry.
        self.env.assertEquals(before[0], [])
        self.env.assertEquals(before[2], [])
        self.env.assertEquals(before[3], [[6.0, [0, 1, 3]]])

        self.con.execute_command("DEBUG", "RELOAD")

        graph = self.db.select_graph("rdb_cch_closed")
        self.env.assertEquals(self._answers(graph), before)
        self.env.assertEquals(graph.ro_query(
            "MATCH ()-[r:FERRY]->() WHERE r.dist > 1.0e300 RETURN count(r)").result_set,
            [[1]])
        graph.delete()


class testCchReplayedWeight(_EffectsBase):
    """A replayed write is never refused by a CCH weight rule.

    CCH DDL replicates (section 1), so a replica holds the indexes its primary
    holds. A buffer can still carry a weight a replica's index cannot take, when
    its primary checks no weights (C checks none) or the two have diverged.
    Here the replica alone holds the index (created on it with writes briefly
    allowed), and the primary, holding none, accepts a negative weight and
    replicates it. Refusing it on the replica would fail the buffer, a full
    resync; the replica applies it and marks the index FAILED, with a warning;
    its queries search the graph from then on.
    """

    GRAPH_ID = "effects_cch_replayed_weight"

    def __init__(self):
        if not INDEX_FALKORDB:
            Environment.skip(None)
        self._setup()

    @staticmethod
    def _cch(graph):
        return [r for r in graph.ro_query("CALL db.indexes()").result_set
                if r[2] and any(v == ["CCH"] for v in r[2].values())]

    def test01_a_weight_the_primary_accepted_replays_on_the_replica(self):
        self.query_and_sync(
            "CREATE (a:N {id: 0}), (b:N {id: 1}), (c:N {id: 2}), "
            "(a)-[:ROAD {time: 1.0}]->(b), (b)-[:ROAD {time: 2.0}]->(c)")
        self.replica.config_set("replica-read-only", "no")
        try:
            self.replica_graph.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        finally:
            self.replica.config_set("replica-read-only", "yes")
        # Built by the replica's own builder before the primary writes.
        deadline = time.time() + 30
        while [r[7] for r in self._cch(self.replica_graph)] != ["OPERATIONAL"]:
            self.env.assertLess(time.time(), deadline)
            time.sleep(0.05)
        self.env.assertEqual(self._cch(self.master_graph), [])

        full_before = self.master.info()["sync_full"]
        failures_before = self.effect_failures()
        warned_before = self.replica_log_count(
            b"a replicated write set a weight it cannot take")
        # No index on the primary: accepted, and replicated as an effect.
        self.query_and_sync("MATCH (:N {id: 0})-[r:ROAD]->() SET r.time = -1.0")

        # The replica applied the write and marked the index it cannot take FAILED.
        self.env.assertEqual(self.replica_graph.ro_query(
            "MATCH (:N {id: 0})-[r:ROAD]->() RETURN r.time").result_set, [[-1.0]])
        self.env.assertEqual([r[7] for r in self._cch(self.replica_graph)], ["FAILED"])
        self.env.assertEqual(self.master.info()["sync_full"], full_before,
            message="the replica was resynced")
        self.env.assertEqual(self.effect_failures(), failures_before,
            message="the replica refused the buffer")
        if warned_before is not None:
            self.env.assertGreater(self.replica_log_count(
                b"a replicated write set a weight it cannot take"), warned_before)


#-----------------------------------------------------------------------------
# 3. An indexed weight is a constraint: refused at the write
#-----------------------------------------------------------------------------


class testCchWeightConstraint():
    """A negative (-inf included) or NaN weight on an edge of an indexed type fails
    the write that sets it, through Cypher and GRAPH.BULK alike, and leaves the graph
    and the index as they were. Other types and attributes are not checked. +inf is
    legal: a closed edge, which no route uses."""

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
        self.env.assertEquals(self.graph.ro_query(
            QUERY.format(types="['ROAD']", weight="time"), {"s": 0, "t": 2}).result_set,
            [[3.0, [0, 1, 2]]])

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
        q = QUERY.format(types="['ROAD']", weight="time")
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
# 4. The hierarchy is built in the background
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
# 5. Writes maintain the index through its change set
#-----------------------------------------------------------------------------



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
