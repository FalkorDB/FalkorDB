from common import *
import redis
import socket
import struct
import time


# The deepest a value may nest (`MAX_VALUE_NESTING` in graph/src/runtime/value.rs).
MAX_VALUE_NESTING = 4096


def _read_line(f):
    return f.readline()[:-2]


def _read_resp(f):
    t = f.read(1)
    if t == b"*":
        return [_read_resp(f) for _ in range(int(_read_line(f)))]
    if t == b"$":
        data = f.read(int(_read_line(f)))
        f.read(2)
        return data
    if t in (b"+", b"-", b":"):
        return _read_line(f)
    raise RuntimeError(f"unexpected RESP type {t!r}")


def capture_restore_payload(conn, src, dst):
    """The `GRAPH.RESTORE` payload `GRAPH.COPY src dst` replicates, read by
    posing as a replica on a raw socket."""
    # The server's address as the test connection sees it: a local server, or
    # the CI service container.
    kw = conn.connection_pool.connection_kwargs
    s = socket.create_connection((kw["host"], kw["port"]))
    try:
        f = s.makefile("rb")
        s.sendall(b"*3\r\n$8\r\nREPLCONF\r\n$4\r\ncapa\r\n$6\r\npsync2\r\n")
        _read_line(f)
        s.sendall(b"*3\r\n$5\r\nPSYNC\r\n$1\r\n?\r\n$2\r\n-1\r\n")
        line = _read_line(f)
        while not line.startswith(b"$"):
            line = _read_line(f)
        f.read(int(line[1:]))  # the RDB snapshot, which has no trailing CRLF
        conn.execute_command("GRAPH.COPY", src, dst)
        while True:
            cmd = _read_resp(f)
            if isinstance(cmd, list) and cmd and cmd[0].upper() == b"GRAPH.RESTORE":
                return cmd[2]
    finally:
        s.close()


def _unsigned(v):
    # `VecWriter::write_unsigned`: a TYPE_UNSIGNED byte, then 8 bytes LE.
    return b"\x04" + struct.pack("<Q", v)


T_ARRAY = 1 << 3
T_STRING = 1 << 11
T_NULL = 1 << 15
MARKER = b"DEEPMARKER"
# The encoded string property `MARKER`: its tag, then a TYPE_BYTES buffer
# holding the NUL-terminated bytes.
ENCODED_MARKER = (
    _unsigned(T_STRING) + b"\x00" + struct.pack("<Q", len(MARKER) + 1) + MARKER + b"\x00"
)


class testRdbLoad():
    def __init__(self):
        self.env, self.db = Env(moduleArgs='VKEY_MAX_ENTITY_COUNT 10',
                                enableDebugCommand=True)
        self.conn = self.env.getConnection()

    def tearDown(self):
        self.conn.flushall()

    # assert that |keyspace| == `n`
    def validate_key_count(self, n):
        keys = self.conn.keys('*')
        self.env.assertEqual(len(keys), n)

    # The telemetry stream is written by a background thread on its own
    # schedule — batched, so it lands a few milliseconds after the query that
    # produced it. Counting the keyspace right after a query and expecting the
    # stream to be in it was a race; wait for it instead.
    def _wait_for_telemetry(self, timeout=30):
        deadline = time.monotonic() + timeout
        while self.conn.type('telemetry{x}') == 'none':
            if time.monotonic() >= deadline:
                raise AssertionError("telemetry stream never appeared")
            time.sleep(0.01)

    # validate that the imported data exists
    def _test_data(self):
        expected = [[i] for i in range(1, 31)]
        q = "MATCH (n:N) RETURN n.v ORDER BY n.v"
        result = self.conn.execute_command("GRAPH.RO_QUERY", "x", q)
        self.env.assertEqual(result[1], expected)

    def test_rdb_load(self):
        # Create a graph with 30 nodes so virtual keys are generated
        graph = self.db.select_graph("x")
        graph.query("UNWIND range(1, 30) AS v CREATE (:N {v: v})")

        # Verify data before save
        self._test_data()

        # Use GRAPH.DEBUG AUX START to create virtual keys
        aux = self.conn.execute_command("GRAPH.DEBUG", "AUX", "START")
        self.env.assertEqual(aux, 1)

        # Dump all keys (graphdata + graphmeta virtual keys + telemetry stream)
        self._wait_for_telemetry()
        all_keys = self.conn.keys('*')
        self.env.assertEqual(len(all_keys), 4)  # 1 graphdata key + 2 graphmeta keys + 1 telemetry stream
        dumps = {}
        for key in all_keys:
            dumps[key] = self.conn.dump(key)

        # Separate graphdata key from graphmeta keys (skip telemetry stream)
        graphdata_key = None
        graphmeta_keys = []
        for key in all_keys:
            # The graphdata key is just the graph name 'x'
            key_str = key.decode() if isinstance(key, bytes) else key
            if key_str == 'x':
                graphdata_key = key
            elif key_str.startswith('telemetry'):
                continue  # skip telemetry stream
            else:
                graphmeta_keys.append(key)

        self.env.assertIsNotNone(graphdata_key)

        # Virtual keys must be typed graphmeta, as C types its own: C frees a
        # graphdata key by dropping its graph from GRAPH.LIST, so a virtual
        # key saved as graphdata emptied GRAPH.LIST once C finished loading
        # the RDB and deleted it (#3160).
        for key in graphmeta_keys:
            key_type = self.conn.type(key)
            if isinstance(key_type, bytes):
                key_type = key_type.decode()
            self.env.assertEqual(key_type, 'graphmeta')

        # Flush and verify empty
        self.conn.flushall()
        self.validate_key_count(0)

        # Start AUX load simulation
        aux = self.conn.execute_command("GRAPH.DEBUG", "AUX", "START")
        self.env.assertEqual(aux, 1)

        # Restore graphmeta keys first, then the graphdata key
        for key in graphmeta_keys:
            self.conn.restore(key, '0', dumps[key])

        self.conn.restore(graphdata_key, '0', dumps[graphdata_key])

        # Finalize
        aux = self.conn.execute_command("GRAPH.DEBUG", "AUX", "END")
        self.env.assertEqual(aux, 0)

        # Verify only the graphdata key remains (graphmeta keys cleaned up)
        self.validate_key_count(1)
        self._test_data()

        # Verify save works after load
        self.conn.save()

    def test_restore_under_a_different_key(self):
        # Restoring a graph DUMP under a key name other than the one it was
        # dumped from used to alias the two keys: only the original showed up
        # in GRAPH.LIST, writes to the restored key mutated the original, and
        # the server segfaulted while loading the resulting RDB.
        # See https://github.com/FalkorDB/FalkorDB/issues/2048
        self.conn.flushall()

        src = self.db.select_graph("src")
        src.query("CREATE (:A {v: 1})-[:R {w: 2}]->(:B {v: 3})")

        self.conn.restore("dst", 0, self.conn.dump("src"))

        # both graphs are listed, and both hold the same data
        graphs = [g.decode() if isinstance(g, bytes) else g
                  for g in self.conn.execute_command("GRAPH.LIST")]
        self.env.assertIn("src", graphs)
        self.env.assertIn("dst", graphs)

        dst = self.db.select_graph("dst")
        q = "MATCH (a:A)-[e:R]->(b:B) RETURN a.v, e.w, b.v"
        self.env.assertEqual(src.query(q).result_set, [[1, 2, 3]])
        self.env.assertEqual(dst.query(q).result_set, [[1, 2, 3]])

        # the two graphs are independent: a write to one is not visible in the
        # other, in either direction
        dst.query("CREATE (:ONLY_IN_DST)")
        src.query("CREATE (:ONLY_IN_SRC)")

        count = "MATCH (n:%s) RETURN count(n)"
        self.env.assertEqual(src.query(count % "ONLY_IN_DST").result_set, [[0]])
        self.env.assertEqual(dst.query(count % "ONLY_IN_SRC").result_set, [[0]])
        self.env.assertEqual(src.query(count % "ONLY_IN_SRC").result_set, [[1]])
        self.env.assertEqual(dst.query(count % "ONLY_IN_DST").result_set, [[1]])

        # the restored graph must survive an RDB round trip
        self.conn.execute_command("DEBUG", "RELOAD")

        self.env.assertEqual(src.query(q).result_set, [[1, 2, 3]])
        self.env.assertEqual(dst.query(q).result_set, [[1, 2, 3]])

        # each graph kept its own write, and only its own
        self.env.assertEqual(src.query(count % "ONLY_IN_SRC").result_set, [[1]])
        self.env.assertEqual(dst.query(count % "ONLY_IN_DST").result_set, [[1]])
        self.env.assertEqual(src.query(count % "ONLY_IN_DST").result_set, [[0]])
        self.env.assertEqual(dst.query(count % "ONLY_IN_SRC").result_set, [[0]])

        # a write issued after the reload must still not leak across keys: a
        # decoder that re-created the alias would only show up on mutation
        dst.query("CREATE (:AFTER_RELOAD_DST)")
        src.query("CREATE (:AFTER_RELOAD_SRC)")

        self.env.assertEqual(src.query(count % "AFTER_RELOAD_SRC").result_set, [[1]])
        self.env.assertEqual(dst.query(count % "AFTER_RELOAD_DST").result_set, [[1]])
        self.env.assertEqual(src.query(count % "AFTER_RELOAD_DST").result_set, [[0]])
        self.env.assertEqual(dst.query(count % "AFTER_RELOAD_SRC").result_set, [[0]])

    def nested_payload(self, depth):
        """A real `GRAPH.RESTORE` payload for one node whose property is
        `depth` nested single-element lists around NULL."""
        self.conn.flushall()
        self.conn.execute_command(
            "GRAPH.QUERY", "src", f"CREATE ({{v: '{MARKER.decode()}'}})"
        )
        payload = capture_restore_payload(self.conn, "src", "dst")
        self.conn.flushall()
        self.env.assertEqual(payload.count(ENCODED_MARKER), 1)
        nested = (_unsigned(T_ARRAY) + _unsigned(1)) * depth + _unsigned(T_NULL)
        return payload.replace(ENCODED_MARKER, nested)

    def test_restore_nested_payload(self):
        # The crafted payload is well-formed: shallow nesting restores intact.
        payload = self.nested_payload(10)
        self.env.assertEqual(self.conn.execute_command("GRAPH.RESTORE", "x", payload), "OK")
        res = self.conn.execute_command("GRAPH.QUERY", "x", "MATCH (n) RETURN n.v")
        self.env.assertEqual(res[1], [["[" * 10 + "NULL" + "]" * 10]])

        # So does one nested exactly at the cap.
        payload = self.nested_payload(MAX_VALUE_NESTING)
        self.env.assertEqual(self.conn.execute_command("GRAPH.RESTORE", "x", payload), "OK")
        res = self.conn.execute_command("GRAPH.QUERY", "x", "MATCH (n) RETURN size(n.v)")
        self.env.assertEqual(res[1], [[1]])

    def test_restore_refuses_nesting_past_the_cap(self):
        # One past the cap, and 60,000 levels: deep enough to exhaust the
        # decoder's stack before the depth was bounded.
        for depth in (MAX_VALUE_NESTING + 1, 60_000):
            payload = self.nested_payload(depth)
            try:
                self.conn.execute_command("GRAPH.RESTORE", "x", payload)
                self.env.assertTrue(False, message=f"a {depth}-deep payload restored")
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("nesting exceeds the maximum depth", str(e))
            self.env.assertEqual(self.conn.exists("x"), 0)
            self.env.assertTrue(self.conn.ping())

    def test_nested_list_properties_survive_reload(self):
        self.conn.flushall()
        q = "CREATE ({v: [1, [2, [3, ['a', [true, [4.5]]]]], []]})"
        self.conn.execute_command("GRAPH.QUERY", "x", q)
        before = self.conn.execute_command("GRAPH.QUERY", "x", "MATCH (n) RETURN n.v")[1]
        self.conn.execute_command("DEBUG", "RELOAD")
        after = self.conn.execute_command("GRAPH.QUERY", "x", "MATCH (n) RETURN n.v")[1]
        self.env.assertEqual(before, after)
        self.env.assertEqual(after, [["[1, [2, [3, [a, [true, [4.500000]]]]], []]"]])

    def test_runtime_built_nesting_is_bounded_on_write(self):
        # Expressions can build a list of any depth at runtime, so the bound
        # is enforced where a value is stored, not when the query is parsed.
        self.conn.flushall()
        deep = "reduce(a = 1, i IN range(1, {}) | [a])"

        # At the cap: stored, and it loads back from the RDB.
        for q in (f"CREATE ({{v: {deep.format(MAX_VALUE_NESTING)}}})",
                  f"CREATE ()-[:R {{v: {deep.format(MAX_VALUE_NESTING)}}}]->()"):
            self.conn.execute_command("GRAPH.QUERY", "x", q)
        self.conn.execute_command("DEBUG", "RELOAD")
        res = self.conn.execute_command(
            "GRAPH.QUERY", "x",
            "MATCH (n) WHERE n.v IS NOT NULL RETURN size(n.v) AS s "
            "UNION ALL MATCH ()-[e]->() RETURN size(e.v) AS s")
        self.env.assertEqual(res[1], [[1], [1]])

        # One past the cap: refused by CREATE and SET, on nodes and edges.
        past = deep.format(MAX_VALUE_NESTING + 1)
        for q in (f"CREATE ({{v: {past}}})",
                  f"CREATE ()-[:R {{v: {past}}}]->()",
                  f"MATCH (n) SET n.w = {past}",
                  f"MATCH ()-[e]->() SET e.w = {past}"):
            try:
                self.conn.execute_command("GRAPH.QUERY", "x", q)
                self.env.assertTrue(False, message=f"stored a too-deep value: {q}")
            except redis.exceptions.ResponseError as e:
                self.env.assertContains(
                    f"nested up to {MAX_VALUE_NESTING} levels deep", str(e))
        self.env.assertTrue(self.conn.ping())


# `GRAPH.RESTORE` takes bytes from the client, so nothing in them can be
# trusted: an id, a count or a length that the payload's own header does not
# account for has to produce an error, never a crash or an allocation sized by
# the lie. These tests corrupt a real payload, field by field and byte by byte,
# and check that the server answers every one of them and is still serving.
#
# TYPE_UNSIGNED, the tag the payload writes before every u64 (see
# `VecWriter::write_unsigned`): overwriting the eight bytes after one replaces
# an id, a count or a dimension.
TYPE_UNSIGNED = 4

HUGE_VALUES = (2**64 - 1, 2**40, 2**31)


class testRestoreCorrupt():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

    def tearDown(self):
        self.conn.flushall()

    def payload(self):
        """A `GRAPH.RESTORE` payload for a small graph that has every kind of
        structure the decoder reads: labels, attributes of several types, a
        multi-edge pair, deleted nodes and edges, and an index."""
        self.conn.flushall()
        g = self.db.select_graph("src")
        g.query("UNWIND range(1, 8) AS i "
                "CREATE (:A:B {id: i, name: 'n' + toString(i), tags: [i, 'x'], "
                "v: vecf32([1.0, 2.0])})")
        g.query("MATCH (a:A), (b:A) WHERE b.id = a.id + 1 "
                "CREATE (a)-[:R {w: a.id}]->(b)")
        # Three edges between one pair: the pair is promoted to a multi-edge
        g.query("MATCH (a:A {id: 1}), (b:A {id: 2}) CREATE (a)-[:R]->(b), (a)-[:R]->(b)")
        g.query("MATCH (a:A {id: 8}) DELETE a")
        g.query("MATCH ()-[e:R {w: 3}]->() DELETE e")
        g.query("CREATE INDEX FOR (n:A) ON (n.id)")
        payload = capture_restore_payload(self.conn, "src", "dst")
        self.conn.flushall()
        return payload

    def restore(self, payload):
        """Restore `payload`; True if it was accepted. A refusal must be a
        clean error: anything else (a dropped connection) fails the test."""
        try:
            self.conn.execute_command("GRAPH.RESTORE", "x", payload)
        except redis.exceptions.ResponseError:
            self.env.assertEqual(self.conn.exists("x"), 0)
            return False
        self.conn.delete("x")
        return True

    def unsigned_offsets(self, payload):
        return [i for i in range(len(payload) - 8) if payload[i] == TYPE_UNSIGNED]

    def test_intact_payload_restores(self):
        # The baseline every corruption below departs from.
        payload = self.payload()
        self.env.assertTrue(self.restore(payload))

        self.conn.execute_command("GRAPH.RESTORE", "x", payload)
        q = "MATCH (a:A)-[e:R]->(b:A) RETURN count(e)"
        src = self.db.select_graph("x")
        # 7 along the chain, minus the one into the deleted node and the one
        # deleted by `w`, plus the two parallel to 1 -> 2.
        self.env.assertEqual(src.query(q).result_set, [[7]])

    def test_restore_refuses_a_huge_id_or_count(self):
        # Every u64 field in turn becomes a huge value. This reaches the
        # entity id of an attribute span (which sized the attribute directory:
        # "Out Of Memory allocating 36028797018963968 bytes"), the header's
        # node and edge counts (#3588), a matrix dimension, and every count
        # and length in between.
        payload = self.payload()
        offsets = self.unsigned_offsets(payload)
        self.env.assertGreater(len(offsets), 100)

        refused = 0
        for off in offsets:
            for value in HUGE_VALUES:
                bad = (payload[:off + 1] + value.to_bytes(8, "little")
                       + payload[off + 9:])
                refused += not self.restore(bad)
        self.env.assertGreater(refused, len(offsets))
        self.env.assertTrue(self.conn.ping())

    def test_restore_refuses_a_corrupted_matrix(self):
        # Every byte of the payload overwritten in turn, which takes in each
        # GraphBLAS matrix and blob it carries. A flipped index, pointer or
        # size made the decoder walk off the matrix (`GxB_Iterator_get_UINT64`
        # in `Tensor::decode`, or GraphBLAS's own consistency check) and crash
        # the server.
        payload = self.payload()
        for off in range(len(payload)):
            bad = payload[:off] + b"\x94" + payload[off + 1:]
            self.restore(bad)
        self.env.assertTrue(self.conn.ping())

    def test_restore_refuses_a_payload_cut_short(self):
        payload = self.payload()
        step = max(1, len(payload) // 300)
        for end in range(0, len(payload), step):
            self.env.assertFalse(self.restore(payload[:end]))
        self.env.assertTrue(self.conn.ping())

    def test_server_still_restores_after_corruption(self):
        # Refusing the bad payloads must leave nothing behind: the key is not
        # created, and the next good payload loads and is intact.
        payload = self.payload()
        for off in self.unsigned_offsets(payload)[::7]:
            bad = payload[:off + 1] + (2**64 - 1).to_bytes(8, "little") + payload[off + 9:]
            self.restore(bad)
        self.env.assertEqual(self.conn.keys("*"), [])

        self.conn.execute_command("GRAPH.RESTORE", "x", payload)
        res = self.db.select_graph("x").query("MATCH (n:A) RETURN count(n)")
        self.env.assertEqual(res.result_set, [[7]])

    def test_zero_dimension_vector_index_round_trips(self):
        # `OPTIONS {dimension:0}` is accepted at creation and kept by the
        # index, so a payload carrying it is valid and must restore.
        self.conn.flushall()
        g = self.db.select_graph("src")
        g.query("CREATE (:V {v: vecf32([1.0, 2.0])})")
        g.query("CREATE VECTOR INDEX FOR (n:V) ON (n.v) "
                "OPTIONS {dimension:0, similarityFunction:'euclidean'}")
        payload = capture_restore_payload(self.conn, "src", "dst")
        self.conn.flushall()
        self.env.assertTrue(self.restore(payload))

        # The index is on the restored graph, not just tolerated: it keeps its
        # zero dimension and its metric.
        self.conn.execute_command("GRAPH.RESTORE", "x", payload)
        res = self.db.select_graph("x").query(
            "CALL db.indexes() YIELD label, properties, types, options "
            "RETURN label, properties, types, options")
        self.env.assertEqual(len(res.result_set), 1)
        label, props, types, options = res.result_set[0]
        self.env.assertEqual(label, "V")
        self.env.assertEqual(props, ["v"])
        self.env.assertContains("VECTOR", str(types))
        self.env.assertEqual(options["v"]["dimension"], 0)
        self.env.assertEqual(options["v"]["similarityFunction"], "euclidean")

    def test_vector_index_options_accepted_at_creation_survive_restore(self):
        # Creation and load hold to the same limits: an option too large for the
        # vector library is refused when the index is created, so no saved graph
        # carries one that RDB load, COPY and RESTORE would refuse.
        self.conn.flushall()
        g = self.db.select_graph("src")
        g.query("CREATE (:V {v: vecf32([1.0, 2.0])})")
        refused = [
            ("dimension:2, similarityFunction:'cosine', %s" % opt, "must be at most 1048576")
            for opt in ("efRuntime:1048577", "M:1048577", "efConstruction:1048577")
        ]
        refused.append(("dimension:16777217, similarityFunction:'cosine'",
                        "dimension must be at most 16777216"))
        for options, message in refused:
            try:
                g.query("CREATE VECTOR INDEX FOR (n:V) ON (n.v) OPTIONS {%s}" % options)
                self.env.assertTrue(False, message="accepted: " + options)
            except redis.ResponseError as e:
                self.env.assertContains(message, str(e))
        # at the limit it is accepted and round-trips
        g.query("CREATE VECTOR INDEX FOR (n:V) ON (n.v) "
                "OPTIONS {dimension:0, similarityFunction:'cosine', efRuntime:1048576}")
        payload = capture_restore_payload(self.conn, "src", "dst")
        self.conn.flushall()
        self.env.assertTrue(self.restore(payload))

    def test_restore_refuses_an_unknown_vector_similarity_function(self):
        # The word after the three graph parameters of a vector index entry
        # picks the metric: 0, 1, 2 are the ones C writes. Anything else used
        # to become euclidean silently.
        self.conn.flushall()
        g = self.db.select_graph("src")
        g.query("CREATE (:V {v: vecf32([1.0, 2.0])})")
        g.query("CREATE VECTOR INDEX FOR (n:V) ON (n.v) "
                "OPTIONS {dimension:2, similarityFunction:'cosine'}")
        payload = capture_restore_payload(self.conn, "src", "dst")
        self.conn.flushall()
        self.env.assertTrue(self.restore(payload))
        # dimension, M, EF_CONSTRUCTION, EF_RUNTIME, then sim_func (cosine = 2)
        marker = bytes([TYPE_UNSIGNED]) + (2).to_bytes(8, "little")
        offsets = [i for i in range(len(payload) - 8)
                   if payload[i:i + 9] == marker]
        refused = 0
        for off in offsets:
            bad = payload[:off + 1] + (7).to_bytes(8, "little") + payload[off + 9:]
            refused += not self.restore(bad)
        # at least the sim_func word (and the dimension) refuse
        self.env.assertGreaterEqual(refused, 1)
        self.env.assertTrue(self.conn.ping())
