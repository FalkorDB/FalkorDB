import redis
from common import Env
from test_rdb_load import capture_restore_payload

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
        for opt in ("efRuntime:1048577", "M:1048577", "efConstruction:1048577"):
            with self.env.assertResponseError():
                g.query("CREATE VECTOR INDEX FOR (n:V) ON (n.v) "
                        "OPTIONS {dimension:2, similarityFunction:'cosine', %s}" % opt)
        with self.env.assertResponseError():
            g.query("CREATE VECTOR INDEX FOR (n:V) ON (n.v) "
                    "OPTIONS {dimension:16777217, similarityFunction:'cosine'}")
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
