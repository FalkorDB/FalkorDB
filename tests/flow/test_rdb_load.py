from common import *

# TODO: when introducing new encoder/decoder this needs to be updated consider
# using GRAPH.DEBUG command to be able to get this data
keys = {
    b'x': b'\x07\x81\x82\xb6\xa9\x85\xd6\xadh\n\x05\x02x\x00\x02\x1e\x02\x00\x02\x01\x02\x00\x02\x03\x02\x01\x05\x02v\x00\x02\x01\x02\x00\x05\x02N\x00\x02\x01\x02\x01\x05\x02v\x00\x02\x00\x02\x01\x02\x01\x02\n\x02\x00\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x01\x02\x01\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x02\x02\x02\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x03\x02\x03\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x04\x02\x04\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x05\x02\x05\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x06\x02\x06\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x07\x02\x07\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x08\x02\x08\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\t\x02\t\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\n\x00\t\x00\x84\xf96Z\xd1\x98\xec\xc0',
    b'{x}x_a244836f-fe81-4f8d-8ee2-83fc3fbcf102': b'\x07\x81\x82\xb6\xa9\x86g\xadh\n\x05\x02x\x00\x02\x1e\x02\x00\x02\x01\x02\x00\x02\x03\x02\x01\x05\x02v\x00\x02\x01\x02\x00\x05\x02N\x00\x02\x01\x02\x01\x05\x02v\x00\x02\x00\x02\x01\x02\x01\x02\n\x02\n\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x0b\x02\x0b\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x0c\x02\x0c\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\r\x02\r\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x0e\x02\x0e\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x0f\x02\x0f\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x10\x02\x10\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x11\x02\x11\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x12\x02\x12\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x13\x02\x13\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x14\x00\t\x00\x13H\x11\xb8\x15\xd3\xdc~',
    b'{x}x_53ab30bb-1dbb-47b2-a41d-cac3acd68b8c': b'\x07\x81\x82\xb6\xa9\x86g\xadh\n\x05\x02x\x00\x02\x1e\x02\x00\x02\x01\x02\x00\x02\x03\x02\x01\x05\x02v\x00\x02\x01\x02\x00\x05\x02N\x00\x02\x01\x02\x01\x05\x02v\x00\x02\x00\x02\x05\x02\x01\x02\n\x02\x02\x02\x00\x02\x03\x02\x00\x02\x04\x02\x00\x02\x05\x02\x01\x02\x14\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x15\x02\x15\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x16\x02\x16\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x17\x02\x17\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x18\x02\x18\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x19\x02\x19\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x1a\x02\x1a\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x1b\x02\x1b\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x1c\x02\x1c\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x1d\x02\x1d\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x1e\x00\t\x00\x1b\xa64\xd6\xf5\x0bk\xa6'
}


class testRdbLoad():
    def __init__(self):
        self.env, self.db = Env(moduleArgs='VKEY_MAX_ENTITY_COUNT 10')
        self.conn = self.env.getConnection()

    # assert that |keyspace| == `n`
    def validate_key_count(self, n):
        keys = self.conn.keys('*')
        self.env.assertEqual(len(keys), n)

    # restore the key data
    def restore_key(self, key):
        self.conn.restore(key, '0', keys[key])

    # validate that the imported data exists
    def _test_data(self):
        expected = [[i] for i in range(1, 31)]
        q = "MATCH (n:N) RETURN n.v"
        result = self.conn.execute_command("GRAPH.RO_QUERY", "x", q)
        self.env.assertEqual(result[1], expected)
    
    def test_rdb_load(self):
        aux = self.conn.execute_command("GRAPH.DEBUG", "AUX", "START")
        self.env.assertEqual(aux, 1)

        self.restore_key(b'{x}x_a244836f-fe81-4f8d-8ee2-83fc3fbcf102')
        self.restore_key(b'{x}x_53ab30bb-1dbb-47b2-a41d-cac3acd68b8c')

        self.conn.flushall()

        self.validate_key_count(0)

        aux = self.conn.execute_command("GRAPH.DEBUG", "AUX", "START")
        self.env.assertEqual(aux, 1)

        self.restore_key(b'{x}x_a244836f-fe81-4f8d-8ee2-83fc3fbcf102')
        self.restore_key(b'{x}x_53ab30bb-1dbb-47b2-a41d-cac3acd68b8c')
        self.restore_key(b'x')

        aux = self.conn.execute_command("GRAPH.DEBUG", "AUX", "END")
        self.env.assertEqual(aux, 0)

        self.validate_key_count(1)
        self._test_data()

        self.conn.save()

    # CRC-64 (Jones variant) - the checksum Redis stamps into DUMP/RESTORE
    # payload footers. reflected CRC, so the right-shift form uses the reflected
    # Jones polynomial (reflect(0xad93d23594c935a9))
    # table-driven, so multi-megabyte payloads checksum in well under a second
    _CRC64_TABLE = None

    @classmethod
    def _crc64(cls, data):
        POLY = 0x95ac9329ac4bc9b5
        if cls._CRC64_TABLE is None:
            table = []
            for i in range(256):
                crc = i
                for _ in range(8):
                    crc = (crc >> 1) ^ POLY if (crc & 1) else (crc >> 1)
                table.append(crc)
            cls._CRC64_TABLE = table
        table = cls._CRC64_TABLE
        crc = 0
        for byte in data:
            crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
        return crc & 0xFFFFFFFFFFFFFFFF

    # rebuild a valid DUMP payload from a (possibly truncated) module body:
    # <body><2-byte RDB version><8-byte CRC64>. without a correct footer Redis
    # rejects the payload before the module decoder ever runs
    @classmethod
    def _reframe(cls, body, version_bytes):
        payload = body + version_bytes
        return payload + cls._crc64(payload).to_bytes(8, 'little')

    # a truncated RDB payload must fail the load gracefully - the module must
    # detect the short read, tear down the partial graph and error out, without
    # crashing the server or leaking a graph
    def test_short_read(self):
        self.conn.flushall()

        # build a small graph with the current encoder and DUMP it to obtain a
        # valid, self-contained module payload (the v10 payloads baked into this
        # file exercise the legacy decoders, which are out of scope). a short
        # read on the current encoding is the diskless-replication case that
        # matters, since master and replica run the same version
        self.conn.execute_command("GRAPH.QUERY", "src",
            "CREATE (a:N {v:1})-[:R {w:2.5}]->(b:N {v:3}), (:M {s:'hello'})")
        self.conn.execute_command("GRAPH.QUERY", "src",
            "CREATE INDEX FOR (n:N) ON (n.v)")

        # DUMP returns a binary payload; use a raw (non-decoding) connection so
        # the bytes are not mangled by UTF-8 decoding
        kw  = self.conn.connection_pool.connection_kwargs
        raw = redis.Redis(host=kw.get('host', 'localhost'), port=kw['port'],
                          decode_responses=False)
        full = raw.execute_command("DUMP", "src")

        version_bytes = full[-10:-8]   # 2-byte RDB version
        body          = full[:-10]     # module payload without the footer

        # sanity: our CRC64 reproduces the footer of the known-good payload,
        # which proves the reframed truncated payloads below are accepted by
        # Redis and actually reach the module decoder
        self.env.assertEqual(self._crc64(full[:-8]),
                             int.from_bytes(full[-8:], 'little'))

        self.conn.flushall()

        # cut the payload at a range of offsets past the ~10 byte module header
        n       = len(body)
        offsets = [15, 25, 40, n // 3, n // 2, n - 40, n - 12]
        for off in offsets:
            if off <= 12 or off >= n:
                continue

            truncated = self._reframe(body[:off], version_bytes)

            self.conn.flushall()

            failed = False
            try:
                self.conn.restore('trunc', 0, truncated)
            except ResponseError:
                failed = True

            # truncation must be rejected ...
            self.env.assertTrue(failed)

            # ... the server must stay alive ...
            self.env.assertTrue(self.conn.ping())

            # ... and no partial graph may be left behind
            self.env.assertEqual(self.conn.keys('*'), [])

        # the server still works normally after all the failed restores
        self.conn.flushall()
        self.conn.execute_command("GRAPH.QUERY", "sane", "CREATE (:N {v: 1})")
        result = self.conn.execute_command("GRAPH.RO_QUERY", "sane",
                                           "MATCH (n:N) RETURN n.v")
        self.env.assertEqual(result[1], [[1]])

    # DUMP 'key' over a raw connection, returns (module body, version bytes)
    def _dump_body(self, key):
        kw  = self.conn.connection_pool.connection_kwargs
        raw = redis.Redis(host=kw.get('host', 'localhost'), port=kw['port'],
                          decode_responses=False)
        full = raw.execute_command("DUMP", key)
        self.env.assertEqual(self._crc64(full[:-8]),
                             int.from_bytes(full[-8:], 'little'))
        return full[:-10], full[-10:-8]

    # RESTORE of a truncated payload must fail, keep the server up and leave
    # no key behind
    def _assert_restore_rejected(self, payload):
        self.conn.flushall()
        failed = False
        try:
            self.conn.restore('trunc', 0, payload)
        except ResponseError:
            failed = True
        self.env.assertTrue(failed)
        self.env.assertTrue(self.conn.ping())
        self.env.assertEqual(self.conn.keys('*'), [])

    # Redis module type id: 9 chars of 6 bits each, then a 10-bit encver
    @staticmethod
    def _module_type_id(name, encver):
        charset = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                   "0123456789-_")
        mid = 0
        for ch in name:
            mid = (mid << 6) | charset.index(ch)
        return (mid << 10) | encver

    # the encoder writes the graph in 256000-byte chunks, one RDB string each;
    # a short read can only surface where a chunk fails to load. cutting
    # inside the 2nd+ chunk leaves the decoder mid-graph, typically inside an
    # entity's property list
    def test_short_read_mid_decode(self):
        self.conn.flushall()

        props = ', '.join(f"p{k}: 'value_{k}_' + toString(i)" for k in range(8))
        self.conn.execute_command("GRAPH.QUERY", "big",
            "CREATE INDEX FOR (n:N) ON (n.v)")
        self.conn.execute_command("GRAPH.QUERY", "big",
            f"UNWIND range(1, 4000) AS i CREATE (:N {{v: i, {props}}})")
        self.conn.execute_command("GRAPH.QUERY", "big",
            "MATCH (a:N) MATCH (b:N {v: a.v + 1}) CREATE (a)-[:R {w: a.v}]->(b)")

        # uncompressed, so each chunk occupies ~256000 bytes of the payload and
        # the offsets below land one per chunk
        self.conn.config_set('rdbcompression', 'no')
        try:
            body, version_bytes = self._dump_body("big")
        finally:
            self.conn.config_set('rdbcompression', 'yes')

        # the graph must span several chunks for this test to mean anything
        chunk = 256000
        self.env.assertGreater(len(body), 4 * chunk)

        # one cut in the middle of every chunk after the first
        for off in range(chunk + chunk // 2, len(body) - 64, chunk):
            self._assert_restore_rejected(self._reframe(body[:off],
                                                        version_bytes))

        # the untruncated payload still restores
        self.conn.flushall()
        self.conn.restore('big', 0, self._reframe(body, version_bytes))
        result = self.conn.execute_command("GRAPH.RO_QUERY", "big",
                                           "MATCH (n:N) RETURN count(n)")
        self.env.assertEqual(result[1], [[4000]])

    # graphs above VKEY_MAX_ENTITY_COUNT are split into a graphdata key plus
    # graphmeta virtual keys (RDB files, replication streams). both types share
    # the decoder, so a DUMP retagged as graphmeta exercises the graphmeta
    # loader; a short read there must fail cleanly as well
    def test_short_read_graphmeta(self):
        self.conn.flushall()

        self.conn.execute_command("GRAPH.QUERY", "src",
            "CREATE (a:N {v:1})-[:R {w:2.5}]->(b:N {v:3}), (:M {s:'hello'})")
        body, version_bytes = self._dump_body("src")

        # body: <RDB_TYPE_MODULE_2><0x81><8-byte big-endian module type id>...
        self.env.assertEqual(body[0], 7)
        self.env.assertEqual(body[1], 0x81)
        type_id = int.from_bytes(body[2:10], 'big')
        encver  = type_id & 1023
        self.env.assertEqual(type_id, self._module_type_id('graphdata', encver))

        meta_id = self._module_type_id('graphmeta', encver)
        meta    = body[:2] + meta_id.to_bytes(8, 'big') + body[10:]

        # control: the retagged, untruncated payload loads as a graphmeta key
        self.conn.flushall()
        self.conn.restore('meta', 0, self._reframe(meta, version_bytes))
        self.env.assertEqual(self.conn.type('meta'), 'graphmeta')

        n = len(meta)
        for off in [15, 25, 40, n // 3, n // 2, n - 40, n - 12]:
            self._assert_restore_rejected(self._reframe(meta[:off],
                                                        version_bytes))
