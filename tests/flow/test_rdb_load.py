import time

from common import *
from index_utils import (
    create_edge_range_index,
    create_node_range_index,
    wait_for_indices_to_sync,
)


class testRdbLoad():
    def __init__(self):
        self.env, self.db = Env(moduleArgs='VKEY_MAX_ENTITY_COUNT 10')
        self.conn = self.env.getConnection()

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

    # RESTORE ... REPLACE of a graph's own DUMP, over a live key which already
    # holds an index, decodes the index into a schema that already contains it.
    # The C engine dereferenced NULL in that case and took the whole server
    # down, see https://github.com/FalkorDB/FalkorDB/issues/2506
    def test_restore_replace_indexed_graph(self):
        self.conn.flushall()

        graph = self.db.select_graph("indexed")
        graph.query("CREATE (:N {v: 1})-[:R {w: 2}]->(:N {v: 2})")

        create_node_range_index(graph, "N", "v", sync=True)
        create_edge_range_index(graph, "R", "w", sync=True)

        # warm the key, both indexes are resolved before the graph is dumped
        self.env.assertEqual(
            graph.ro_query("MATCH (n:N) WHERE n.v = 1 RETURN n.v").result_set,
            [[1]])
        self.env.assertEqual(
            graph.ro_query("MATCH ()-[e:R]->() WHERE e.w = 2 RETURN e.w").result_set,
            [[2]])

        blob = self.conn.dump("indexed")
        self.env.assertIsNotNone(blob)

        # replace the live key with its own dump, repeatedly:
        # each restore decodes into the graph left behind by the previous one
        for _ in range(3):
            self.conn.restore("indexed", 0, blob, replace=True)
            # the server must still be alive
            self.env.assertTrue(self.conn.ping())

        # both indexes survived the replace and are usable
        wait_for_indices_to_sync(graph)
        indexes = graph.ro_query("""CALL db.indexes()
                                 YIELD label, status
                                 RETURN label, status ORDER BY label""").result_set
        self.env.assertEqual(indexes, [['N', 'OPERATIONAL'], ['R', 'OPERATIONAL']])

        self.env.assertContains('Node By Index Scan',
                                str(graph.explain("MATCH (n:N) WHERE n.v = 1 RETURN n")))
        self.env.assertContains('Edge By Index Scan',
                                str(graph.explain("MATCH ()-[e:R]->() WHERE e.w = 2 RETURN e")))

        # data survived the replace
        self.env.assertEqual(
            graph.ro_query("MATCH (n:N) RETURN n.v ORDER BY n.v").result_set,
            [[1], [2]])
        self.env.assertEqual(
            graph.ro_query("MATCH ()-[e:R]->() WHERE e.w = 2 RETURN e.w").result_set,
            [[2]])

        self.conn.flushall()

    # same as above for a fulltext index: the crash was a NULL dereference
    # while restoring the index language, which only fulltext indexes set
    def test_restore_replace_fulltext_indexed_graph(self):
        self.conn.flushall()

        graph = self.db.select_graph("fulltext")
        graph.query("CREATE (:P {name: 'alice in wonderland'})")
        graph.query("""CREATE FULLTEXT INDEX FOR (n:P) ON (n.name)
                    OPTIONS {language: 'english', stopwords: ['in']}""")
        wait_for_indices_to_sync(graph)

        # warm the key
        q = "CALL db.idx.fulltext.queryNodes('P', 'alice') YIELD node RETURN node.name"
        self.env.assertEqual(graph.ro_query(q).result_set, [['alice in wonderland']])

        blob = self.conn.dump("fulltext")
        self.conn.restore("fulltext", 0, blob, replace=True)
        self.env.assertTrue(self.conn.ping())

        # index survived the replace, language and stopwords included
        wait_for_indices_to_sync(graph)
        index = graph.ro_query("""CALL db.indexes()
                               YIELD label, language, stopwords, status
                               RETURN label, language, stopwords, status""").result_set
        self.env.assertEqual(index, [['P', 'english', ['in'], 'OPERATIONAL']])
        self.env.assertEqual(graph.ro_query(q).result_set, [['alice in wonderland']])

        self.conn.flushall()
