from common import *


class testRdbLoad():
    def __init__(self):
        self.env, self.db = Env(moduleArgs='VKEY_MAX_ENTITY_COUNT 10')
        self.conn = self.env.getConnection()

    # assert that |keyspace| == `n`
    def validate_key_count(self, n):
        keys = self.conn.keys('*')
        self.env.assertEqual(len(keys), n)

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
