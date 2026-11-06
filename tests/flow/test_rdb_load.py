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


# `DUMP cch_v20` from a module built with `--features index-falkordb`, encoding
# version 20: `(:N {id})` nodes 0..3, ROAD edges 0->1 (time 1.0), 1->2 (2.0),
# 2->3 (1.5) and 0->3 (9.0), and `CREATE CCH INDEX FOR ()-[e:ROAD]->() ON
# (e.time)`, so the value carries one CCH index definition.
CCH_V20_DUMP = bytes.fromhex(
    "078182b6a985d6ad681405c3446c6e88020008008000096363685f763230000404800fe0"
    "0208200c6000e001080001600f2011e00008200c6000e001110002600f20000003200340"
    "00046964000005400820000374696d65e0022b20118000c034004ee00113e0001ce0015a"
    "802b2000c04e03524f4144e00116e0001f01040720234000e0013ac01b0004c095e00111"
    "000640292000e0012ce0093ec12de00123000920314000e00111000a400e2000e0011120"
    "0c6000e00a110100206019e10024e00223e1095be0002ce0001ac12fe0132ce00023e000"
    "f2e0003e206f6000e00159e0002360172000e0012ce00135004020172000a05801f03fe0"
    "011ae00023e00008e0062c010040e00ab3e00f2c01f83fe00ab3e00f2c012240e0021ac0"
    "000060a0a3c085e00007a00000ffa000a00ee05000c0b4e0ff00e0d100c40f074772425f"
    "424f4f4c2301a000e00008c363010000c44ee00000c43b403d04494e543332e00135e400"
    "7be0123e000ba048403e0055a03fc27ae00a3fe00000e01376c024e00176e00000e003eb"
    "206d0038e0012ce00a35e30f8ac02aa38900ffc00fe04f00c10fe0ff00e02700e1983fc3"
    "54e000a8e203a68392c01ae30a130010e1020f4026464062fae30a23c2aa0004c0346028"
    "6000001460058000e003466032e30ba6c8ca0004c039e0014bc05d4000e0034be02692e0"
    "0000e1030fe330b6c00023b16000c00fe04f0040f8e0ff00e0d500e2039e83aec0efe000"
    "08e3011bc3ece00000e30aa6c01be30aede0083ee30b9ac27ae00a3fe00000e01376c024"
    "e001b5e00000e003ebe3178ae00047e7104ac06063902000e00131e04d00c118e0ff00e0"
    "c800ae9ce00000c260c1f8471c6000e30736013634e7021dc045c03d0200000c2003c000"
    "404b6047e30a42c0310004c030c010e70514403e401962efe0078a4b4360182000e70aa8"
    "c75c40864043c04be00b8ee01246e00000e323d1e310c8c00023c36000c00fe04f0040fc"
    "e0ff00e0d500e3071d43a8c0f1e0000843078000e7081fe30aa48029203ec02ce0123ee0"
    "077e439cc27ce00a3fe00000e01376c024e00176e00000e3ff8ce0ff00e08600e3070de3"
    "ff8c240bc3d5c007e00108e7ff34e0ff00e05a00e60346eeff916000f50fd3c78a6abc20"
    "00e300c5e04e00c842e0ff00e02700e1983fca5ae000a8e30992c01aea0a46ce3d400040"
    "264de3ea0b52c2b6eeff6ee04f004183e0ff00e0d500e309a6c0efe00008c374e301e4e0"
    "0000ed13f5c411e0123eeb075eee936ce3071227346000e1000be04e00e00777e0ff00e0"
    "c000c327e00000e31292c02be31253e36592c090e00000e0257ee00000e003f3f91766ea"
    "08d4c322c4b163972000c00fe04f00c077e0ff00e02700e1983fc2dde000a8e203a68e67"
    "c01ae30a9ad8b8400040264ad463b8e30a23c2aaf80aff0014403fa000e003466032e70b"
    "410005a02c0004c03960746000e00392e00400e02592e00000e1030fe330b6c00023b160"
    "00c00fe04f004144e0ff00e0cc00c3b6e00000e203a683b6c01ae30ab6e0083de30a1be0"
    "233ee30ba2c282e0497ee00000e003f3e3179ae0018fe00098c047a011010000000d0018"
    "19c0581af91ccd"
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

    def test_restore_v20_cch_definitions_with_or_without_the_feature(self):
        # A graph saved with a CCH index loads in every build. With the feature
        # the index is rebuilt from its definition; without it (no CCH index
        # exists) the definition is read past and dropped with a warning, and
        # the graph itself loads whole.
        self.conn.flushall()
        self.conn.restore("cch_v20", 0, CCH_V20_DUMP)
        g = self.db.select_graph("cch_v20")
        self.env.assertEqual(
            g.ro_query("MATCH (n:N) RETURN count(n)").result_set, [[4]])
        self.env.assertEqual(
            g.ro_query("MATCH ()-[e:ROAD]->() RETURN count(e), sum(e.time)").result_set,
            [[4, 13.5]])
        rows = [(r[0], r[1]) for r in g.ro_query("CALL db.indexes()").result_set]
        if INDEX_FALKORDB:
            self.env.assertEqual(rows, [("ROAD", ["time"])])
            q = ("MATCH (s:N {id: 0}), (t:N {id: 3}) "
                 "CALL db.idx.cch.query({sourceNode: s, targetNode: t, "
                 "relTypes: ['ROAD'], weightProp: 'time'}) "
                 "YIELD pathWeight RETURN pathWeight")
            self.env.assertEqual(g.ro_query(q).result_set, [[4.5]])
        else:
            self.env.assertEqual(rows, [],
                message="a build without index-falkordb holds no CCH index; "
                        "FALKORDB_FEATURES may be wrong")
            if self.env.log_path:
                with open(self.env.log_path, "rb") as f:
                    self.env.assertTrue(
                        b"1 CCH index definition(s) dropped" in f.read(),
                        message="the dropped definition must be logged")
