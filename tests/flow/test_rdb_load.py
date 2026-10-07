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

    # regression test for the RESTORE ... REPLACE crash (#259 / #2506):
    # a full-text indexed graph dumped and restored with REPLACE onto its own
    # (still live) key used to decode the payload into the existing graph
    # context, duplicating its schema. that drove Schema_AddIndex to return a
    # NULL index which _RdbLoadIndex handed to Index_SetLanguage -> SIGSEGV
    # (and, with lazy-free, a double free of the live context on the BIO thread)
    def test_restore_replace_indexed_graph(self):
        self.conn.flushall()

        g = self.db.select_graph('replace_me')
        g.query("CREATE (:Person {name:'alice', bio:'hello world'}),"
                "       (:Person {name:'bob',   bio:'bonjour monde'})")

        # full-text index carrying a language (exercises Index_SetLanguage)
        g.query("CALL db.idx.fulltext.createNodeIndex("
                "{label:'Person', language:'english'}, 'name', 'bio')")

        payload = self.conn.dump('replace_me')

        # restore with REPLACE onto the live key - twice, to make sure the old
        # context is freed cleanly and the operation is repeatable
        for _ in range(2):
            self.conn.restore('replace_me', 0, payload, replace=True)

            # server must still be alive and the graph intact
            self.env.assertTrue(self.conn.ping())

            result = g.query("MATCH (n:Person) RETURN n.name ORDER BY n.name")
            self.env.assertEqual(result.result_set, [['alice'], ['bob']])

            # the full-text index survived and is usable
            idx = g.query("CALL db.idx.fulltext.queryNodes('Person', 'alice') "
                          "YIELD node RETURN node.name")
            self.env.assertEqual(idx.result_set, [['alice']])
