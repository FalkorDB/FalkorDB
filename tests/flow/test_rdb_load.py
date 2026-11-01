from common import *
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


# A `DUMP` of a graph key written by the engine on `main` before the encoding
# version moved to 21, i.e. at encoding version 19, where a duration is C's one
# epoch offset. Made with:
#   CREATE (:D {p0: duration({years: 1, months: 2, days: 3, hours: 4, minutes: 5, seconds: 6}),
#               p1: duration({days: 1}),  p2: duration({hours: 25}), p3: duration({months: 13}),
#               p4: duration({days: 40}), p5: duration({seconds: 59})})
#         -[:R {d: duration({days: 2, hours: 3})}]->(:E)
# whose durations `main` printed as LEGACY_DURATIONS below.
LEGACY_DUMP = bytes.fromhex(
    "078182b6a985d6ad681305c345257862020007008000086c6567616379000402800e0200040120044000000440044000"
    "e00108e01223e00035c05701000320034000017030e0020b0031e0020b0032e0020b0033e0020b0034e0020b02350000"
    "c0740064e0027f40576000c01c0044e00113e0001ce00093e000250045e0021c20166000e00025e00011e0002e0052e0"
    "0113e0001c01040620204000e00137c0830004c0f1e0011ae0002ce00023c11ee001230008404d2000e001110009200d"
    "4000e00111400d4000e00150400d40000004200160000303f229336065e00235c01a018051819ee00186200160000203"
    "905fa01ae000bc20016000030300120a8050a0102018a009020300bc21a4200001040520044000e0011a003b400e2000"
    "e00286a000e0010880722011e1001f20016000020330cd6073e003aac000006080190200004020034000c007a00000ff"
    "a000a00ee05000c0bde0ff00e0d100c388074772425f424f4f4cc2a5229e20036000c336010000c3c7096d696e757465"
    "7300000a20144000403d04494e543332e00135e300f4e0093ee00000000ba009403e0055a03fc27ae00a3fe00000e013"
    "76c024e00176e00000e003eb206d0038e0012ce00a35e30f8ac09fa38900ffc00fe04f00e0575fe0d700e1983fc354e0"
    "00a8e203a68392c01ae30a13c03de00000e30a1be01e3e6026e30b92c694e00a7ee300384000e0137ac028e34f96c000"
    "23916000c00fe04f00612ce0ff00e0d400e3098ec0eee00008e201d4c38e077965617273696f6ee30b13e0003ec02ce0"
    "093ee00000e30b8ec27ae00a3fe00000e30a8ac01be00008e00176e00000e003ebe71721e00087e7102ac00063902000"
    "c00fe04f00c118e0ff00e0d100e203a78722c0ebe00008e201ddc366e00000e3131cc02ce0123ee39293e3108ac31a23"
    "856000c00fe04f00e0575fe0d700e1983fc327e000a8e30992c01ae3125340223399e30c92401b4000e0123e40006026"
    "e30b92ea1ab5603be40a0d60172000eaffb5e0ff00e0a200e3098ec0bce00008e612e2e30a13e0003ec3f9e0123ee30b"
    "8ee30053e0093fe00000e01376c024e00176e00000e003ebea18b5e00035f1ffe0e0ff00e05000c2f0e00000e3071e01"
    "3634e3141ee0083fe30a26e3125de0033e6026e0077e55b5e7122cc03f603b4000e0257ee00000e323a5e3109cc0006a"
    "c42000c00fe04f00e000f4e0ff00e0d000e307154394c0ece00008e6017cc315e00000e30a15c01be60af2e0083ee007"
    "7e4394c27ce00a3fe00000e01376c024e001b5e00000e3ff8ce0ff00e08600e3070de3ff8c240be40014200c6000e000"
    "08e7ff34e0ff00e05a00e60346f5ff876000eeff5ce0b700c56de09000ca48e000a8e30992c01aea0a46c03de00000ea"
    "0a4ee01e3e6026e707b74ac5ee125ac6ba603be0137ac028eaffc9e0ff00e09900e3098ec0b3e00008e201d4e31b8ee0"
    "003ec02ce0123ee30b8e636b2000e00a3fe00000e30a8ac01be00008e00176e00000e003ebf51787e70821c426c00031"
    "dc6000c00fe04f00c10fe0ff00e0c800c327e00000e203a6ea06b4c02be201dcc011e00000e3131bc02ce0123ee32692"
    "c051e00000e0257ee00000e003f3e3309ac3a263972000c00fe04f00e3ff9ae02f00e1983fc2dde000a8e203a6839ac0"
    "1ae3165b6026e30a1bc2a2e00a3e000ce10256403e602ae30b9ec3cd0004c031c010e001c040004043e02682e00000e0"
    "03f7e3309ec00023996000c00fe04f0060f0e0ff00e0d400e2039e8396c0eee00008e30d136000e30a9660172000e30a"
    "96c413e00000e30b92c27ae00a3fe00000e01376c024e001b5e00000e003ebe3158a010000000d007f23a60f6cb88e15"
)
LEGACY_DURATIONS = ["P1Y2M3DT4H5M6S", "P1D", "P1DT1H", "P1Y1M", "P1M9D", "PT59S"]
LEGACY_EDGE_DURATION = "P2DT3H"

# The encoding version this build writes (`ENCODING_VERSION`), and the newest
# the C engine reads (`GRAPH_ENCODING_LATEST_V`), which is why it is not 20.
ENCODING_VERSION = 21
C_LATEST_ENCODING_VERSION = 20


def _crc64(data):
    """Redis's CRC-64/Jones, which ends every `DUMP` payload."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x95AC9329AC4BC9B5 if crc & 1 else crc >> 1
    return crc


def payload_encver(payload):
    """The encoding version a `DUMP` of a module key was written at: the low ten
    bits of the module id that follows the type byte."""
    assert payload[0] == 7 and payload[1] == 0x81, payload[:2]
    return int.from_bytes(payload[2:10], "big") & 1023


def with_encver(payload, encver):
    """`payload` rewritten to claim `encver`, its checksum recomputed."""
    mod_id = int.from_bytes(payload[2:10], "big")
    body = payload[:2] + ((mod_id & ~1023) | encver).to_bytes(8, "big") + payload[10:-8]
    return body + _crc64(body).to_bytes(8, "little")


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

    def _durations(self, key):
        res = self.conn.execute_command(
            "GRAPH.QUERY", key,
            "MATCH (n:D) RETURN " + ", ".join(f"toString(n.p{i})" for i in range(6)))
        edge = self.conn.execute_command(
            "GRAPH.QUERY", key, "MATCH ()-[e:R]->() RETURN toString(e.d)")
        return [[v for v in res[1][0]], edge[1][0][0]]

    # Encoding version 21 is where `T_DURATION` becomes three components. What
    # an earlier build wrote -- the C engine's, and `main`'s -- must still load.
    def test_restore_a_dump_from_before_duration_components(self):
        self.conn.flushall()
        self.env.assertEqual(payload_encver(LEGACY_DUMP), 19)
        self.conn.restore("legacy", 0, LEGACY_DUMP)
        self.env.assertEqual(self._durations("legacy"),
                             [LEGACY_DURATIONS, LEGACY_EDGE_DURATION])

        # Saved again it is written at the current version, and still reads the same.
        saved = self.conn.dump("legacy")
        self.env.assertEqual(payload_encver(saved), ENCODING_VERSION)
        self.conn.execute_command("DEBUG", "RELOAD")
        self.env.assertEqual(self._durations("legacy"),
                             [LEGACY_DURATIONS, LEGACY_EDGE_DURATION])

    def test_encoding_version_and_signed_durations(self):
        self.conn.flushall()
        exprs = ["duration({days: -1})", "duration({months: 1, days: -1})",
                 "duration({hours: -1, minutes: -30})", "duration({days: 40})",
                 "duration({months: -10000, days: 200000, seconds: -2000000000})",
                 "duration({seconds: 9000000000000000})"]
        self.conn.execute_command(
            "GRAPH.QUERY", "signed",
            "CREATE (:S {" + ", ".join(f"p{i}: {e}" for i, e in enumerate(exprs)) + "})")
        q = "MATCH (n:S) RETURN " + ", ".join(f"toString(n.p{i})" for i in range(len(exprs)))
        before = self.conn.execute_command("GRAPH.QUERY", "signed", q)[1]
        self.env.assertEqual(before[0][:2], ["P-1D", "P1M-1D"])

        dumped = self.conn.dump("signed")
        self.env.assertEqual(payload_encver(dumped), ENCODING_VERSION)
        self.env.assertGreater(ENCODING_VERSION, C_LATEST_ENCODING_VERSION)

        self.conn.execute_command("DEBUG", "RELOAD")
        self.env.assertEqual(self.conn.execute_command("GRAPH.QUERY", "signed", q)[1], before)
        self.conn.restore("signed2", 0, dumped)
        self.env.assertEqual(self.conn.execute_command("GRAPH.QUERY", "signed2", q)[1], before)

    def test_restore_refuses_a_newer_encoding_version(self):
        self.conn.flushall()
        self.conn.execute_command("GRAPH.QUERY", "future", "CREATE (:S {d: duration({days: -1})})")
        dumped = self.conn.dump("future")
        self.conn.flushall()
        for encver in (ENCODING_VERSION + 1, 100):
            try:
                self.conn.restore("future", 0, with_encver(dumped, encver))
                self.env.assertTrue(False, message=f"an encoding version {encver} payload restored")
            except redis.exceptions.ResponseError:
                pass  # expected: a newer encoding version is refused, never read as this layout
            self.env.assertEqual(self.conn.exists("future"), 0)
            self.env.assertTrue(self.conn.ping())
        # Control: the same bytes at the version they were written at restore.
        self.conn.restore("future", 0, with_encver(dumped, ENCODING_VERSION))
        self.env.assertEqual(
            self.conn.execute_command("GRAPH.QUERY", "future", "MATCH (n:S) RETURN toString(n.d)")[1],
            [["P-1D"]])

    # `GRAPH.COPY` replicates a serialized graph as a `GRAPH.RESTORE` argument,
    # with no version beside it, so the blob states its own.
    def test_restore_blob_names_its_encoding_version(self):
        self.conn.flushall()
        self.conn.execute_command(
            "GRAPH.QUERY", "src",
            "CREATE (:S {d: duration({months: 1, days: -1}), s: 'x'})-[:R {d: duration({days: -2})}]->(:S)")
        payload = capture_restore_payload(self.conn, "src", "dst")
        self.env.assertEqual(payload[:9], _unsigned(ENCODING_VERSION))
        self.conn.flushall()

        q = "MATCH (n:S) WHERE n.s = 'x' RETURN toString(n.d)"
        self.env.assertEqual(self.conn.execute_command("GRAPH.RESTORE", "a", payload), "OK")
        self.env.assertEqual(self.conn.execute_command("GRAPH.QUERY", "a", q)[1], [["P1M-1D"]])
        self.env.assertEqual(
            self.conn.execute_command("GRAPH.QUERY", "a", "MATCH ()-[e:R]->() RETURN toString(e.d)")[1],
            [["P-2D"]])

        # A newer build's blob is refused rather than read as this layout.
        try:
            self.conn.execute_command("GRAPH.RESTORE", "b", _unsigned(ENCODING_VERSION + 1) + payload[9:])
            self.env.assertTrue(False, message="a newer blob restored")
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("encoding version", str(e))
        self.env.assertEqual(self.conn.exists("b"), 0)

    def test_restore_blob_from_before_the_version_prefix(self):
        self.conn.flushall()
        self.conn.execute_command("GRAPH.QUERY", "src", "CREATE (:S {s: 'x', n: 3})-[:R {w: 1}]->(:S {s: 'y'})")
        payload = capture_restore_payload(self.conn, "src", "dst")
        self.conn.flushall()
        # What a build without the prefix sent: the graph's header comes first.
        # (No durations in it, so it is the same bytes at either layout.)
        legacy = payload[9:]
        self.env.assertEqual(legacy[:1], b"\x00")
        self.env.assertEqual(self.conn.execute_command("GRAPH.RESTORE", "old", legacy), "OK")
        self.env.assertEqual(
            self.conn.execute_command("GRAPH.QUERY", "old", "MATCH (a)-[e:R]->(b) RETURN a.n, e.w, b.s")[1],
            [[3, 1, "y"]])

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
