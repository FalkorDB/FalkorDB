from common import *
from rdb_utils import *

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

    # rebuild a valid DUMP payload from a (possibly truncated) module body:
    # <body><2-byte RDB version><8-byte CRC64>. without a correct footer Redis
    # rejects the payload before the module decoder ever runs
    @staticmethod
    def _reframe(body, version_bytes):
        payload = body + version_bytes
        return payload + crc64(payload).to_bytes(8, 'little')

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
        self.env.assertEqual(crc64(full[:-8]),
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
