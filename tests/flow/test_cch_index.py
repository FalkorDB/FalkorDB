"""CCH indexes, end to end: the flow classes that need `CCH_INDEX yes`.

`CCH_INDEX` is an immutable load-time argument, so the literals below route this
file to CI's spawn bucket. All CCH flow classes share this one file so the rest
of the flow suite stays in the shared-services bucket and the CI matrix grows by
a single cell.

Effects -- CCH index DDL, which is not replicated yet. See `effects_common.py`
for the shared fixture.
"""

import os

from common import *

from effects_common import _EffectsBase


#-----------------------------------------------------------------------------
# 1. CCH DDL is not replicated, so it must never create schema
#-----------------------------------------------------------------------------


class testEffects_01_CchDdlCreatesNoSchema(_EffectsBase):
    """CREATE CCH INDEX emits no effects yet, so it must not create schema.

    A relationship type or attribute it created would exist on the primary
    only. The next write naming it carries an id no earlier effect announced,
    the replica refuses it ("relationship type id 0 out of range") and
    resyncs. So the statement refuses to run until the type and the weight
    attribute exist, and a replica that never sees CCH DDL still resolves every
    id the primary uses afterwards.

    `CCH_INDEX` is immutable, so it is a load-time argument, on both sides: the
    literal below is also what routes this file to CI's spawn bucket.
    """

    GRAPH_ID = "effects_cch_ddl"

    def __init__(self):
        self._setup(moduleArgs="CCH_INDEX yes")

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

    def _refused(self, q, needle):
        try:
            self.master_graph.query(q)
        except ResponseError as e:
            self.env.assertContains(needle, str(e))
            self.env.assertContains("first", str(e))
            return
        self.env.assertTrue(False, message=f"{q} was not refused")

    def test01_ddl_over_missing_schema_is_refused_and_writes_still_apply(self):
        self.set_effects_config()
        for con in (self.master, self.replica):
            self.env.assertEqual(
                con.execute_command("GRAPH.CONFIG", "GET", "CCH_INDEX"),
                ["CCH_INDEX", 1])
        full_before = self.master.info()["sync_full"]
        failures_before = self.effect_failures()
        out_of_range_before = self.replica_log_count(b"out of range")

        # Nothing exists yet: refused, and nothing was created. The query left
        # an empty graph on the primary, which no effect announces, so the
        # replica has no key to read until the first write below.
        self._refused("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)",
                      "relationship type 'ROAD' does not exist")
        self.env.assertEqual(self._schema_of(self.master_graph), ([], []))

        # A normal write creates the type and the attribute, as an effect.
        self.query_and_sync("CREATE (:N)-[:ROAD {time: 1.0}]->(:N)")
        self._refused("CREATE CCH INDEX FOR ()-[e:ROAD|FERRY]->() ON (e.time)",
                      "relationship type 'FERRY' does not exist")
        self._refused("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.length)",
                      "attribute 'length' does not exist")
        self.wait_for_replica_offset()
        self.env.assertEqual(self._schema(), (['ROAD'], ['time']))

        # Over existing schema the index is created.
        res = self.query_and_sync(
            "CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        self.env.assertEqual(res.indices_created, 1)

        # Later writes apply on the replica: the indexed type and its weight,
        # then a new type and a new attribute.
        self.query_and_sync("CREATE (:N)-[:ROAD {time: 2.0}]->(:N)")
        self.query_and_sync("MATCH ()-[r:ROAD]->() SET r.time = 3.0")
        self.query_and_sync("CREATE (:N)-[:FERRY {time: 4.0, length: 5}]->(:N)")
        self.env.assertEqual(self._schema(),
                             (['ROAD', 'FERRY'], ['time', 'length']))
        self.assert_agree(
            "MATCH ()-[r]->() RETURN type(r), r.time, r.length "
            "ORDER BY type(r), r.time",
            [['FERRY', 4.0, 5], ['ROAD', 3.0, None], ['ROAD', 3.0, None]])
        # Not `assert_graph_eq`: it compares index lists, and the CCH index is
        # on the primary only until CCH DDL is replicated.

        # And none of it was refused: no resync, no refused GRAPH.EFFECT.
        self.env.assertEqual(self.master.info()["sync_full"], full_before,
            message="the replica was resynced")
        self.env.assertEqual(self.effect_failures(), failures_before,
            message="the replica refused a GRAPH.EFFECT")
        out_of_range = self.replica_log_count(b"out of range")
        if out_of_range is not None:
            self.env.assertEqual(out_of_range, out_of_range_before,
                message="the replica could not resolve an id")


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

    `CCH_INDEX` is immutable, so the literal below also routes this file to CI's
    spawn bucket.
    """

    def __init__(self):
        self.env, self.db = Env(moduleArgs="CCH_INDEX yes")
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

        # time is attribute 0, dist is 1; types are sorted whatever the DDL order.
        expected = cch_payload(([ROAD], 0), ([RAIL, FERRY], 1))
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

    def test02_multi_type_index_alone(self):
        self._reset()
        graph = self._build(GRAPH_ID + "_multi")
        graph.query("CREATE CCH INDEX FOR ()-[e:FERRY|ROAD|RAIL]->() ON (e.time)")
        chunk = self._chunk_of_save()
        expected = cch_payload(([RAIL, ROAD, FERRY], 0))
        self.env.assertTrue(chunk.endswith(expected),
            message=f"wanted {expected.hex()}, key ends {chunk[-len(expected):].hex()}")
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

    def test04_encoding_version_is_20(self):
        self._reset()
        graph = self._build(GRAPH_ID + "_ver")
        graph.query(self.DDL_ROAD_TIME)
        blob = self._save()
        # `_key_chunk` already requires the graphdata type id with encver 20.
        self._key_chunk(blob)
        for stale in (19,):
            old = _module_head("graphdata", stale)
            self.env.assertEquals(blob.find(old), -1,
                message=f"a graphdata key written with encoding version {stale}")
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
        # DEBUG RELOAD is refused by redis unless enabled.
        self.env, self.db = Env(moduleArgs="CCH_INDEX yes", enableDebugCommand=True)
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
        graph = self._build("rdb_cch_dropped")
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
                self.env.assertTrue(b"was not rebuilt and is dropped" in f.read(),
                    message="the dropped definition must be logged as a warning")
        graph.delete()
