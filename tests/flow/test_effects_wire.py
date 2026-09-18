"""Effects v3 -- the bytes on the wire: what a reader refuses, what compression does to a payload, and that the same write twice produces the same bytes.

See `effects_common.py` for the shared fixture and why these are split.
"""

import time


from common import *
from constraint_utils import (create_mandatory_node_constraint, create_unique_node_constraint)
from graph_utils import graph_eq
from index_utils import (create_node_range_index, wait_for_indices_to_sync)

from effects_common import _EffectsBase, zstd_raw_frame


class testEffects_00_UnreadableBuffer(_EffectsBase):
    """A `GRAPH.EFFECT` payload this build cannot read must be refused, leave
    the graph exactly as it was, and take nothing down.

    With v2 deleted, "version 2" is no longer an old format — it is a version
    this build has never heard of, and so is anything else that is not 3. The
    same goes for a flags bit outside `KNOWN_FLAGS`: `open_payload` refuses
    both rather than decoding the records it happens to recognise, because a
    reader that guesses corrupts itself with the rest of the buffer.

    Every case here is *client-sent*, which is the half that has to be safe
    rather than loud. `divergence_guard::is_replayed` deliberately differs from
    C, which guards `GRAPH.EFFECT` unconditionally: `GRAPH.EFFECT` is a payload
    any client can send, so treating a bad one as divergence would hand anyone
    a way to force a replica to resync — or, under `LOADING`, to `exit(1)`. So
    the assertions are: an error reply, no resync, no data change, both servers
    still serving.

    The replayed half needs no separate case per malformed shape:
    `graph_effect` funnels every `ApplyError` through one `is_replayed` check,
    and `testEffects_06c_DivergenceForcesResync` already pins that path end
    to end.
    """

    GRAPH_ID = "effects_unreadable"

    def __init__(self):
        self._setup()

    #-------------------------------------------------------------------------
    # helpers
    #-------------------------------------------------------------------------

    @staticmethod
    def _framed(version, flags, body=b""):
        return bytes([version, flags]) + body

    @staticmethod
    def _compressed(plain_len, checksum, frame, comp_len=None):
        """A payload that *claims* to be a compressed v3 frame:
        `u8 3 · u8 FLAG_COMPRESSED · u32 plain_len · u32 comp_len · u32 crc32 · frame`.

        `comp_len` defaults to the frame's real length, so a caller that wants
        to corrupt one field corrupts only that one. Lying about it instead is
        how the framing itself is tested — the reader takes exactly this many
        bytes and rejects anything after them."""
        if comp_len is None:
            comp_len = len(frame)
        return (b"\x03\x01"
                + plain_len.to_bytes(4, "little")
                + comp_len.to_bytes(4, "little")
                + checksum.to_bytes(4, "little")
                + frame)

    def _refused(self, buf, what, key=None):
        """Send `buf` and require an error reply. Returns the message."""
        try:
            self.master.execute_command(
                "GRAPH.EFFECT", key or self.GRAPH_ID, buf)
        except ResponseError as e:
            return str(e)
        raise AssertionError(f"{what} was accepted; it must be refused")

    def _still_healthy(self, full_before):
        # No resync was scheduled, and neither server went away. Both matter:
        # `on_failure`'s two arms are a forced `REPLICAOF` and `exit(1)`.
        time.sleep(0.5)
        self.env.assertEqual(self.master.info()["sync_full"], full_before)
        self.env.assertEqual(self.master.ping(), True)
        self.env.assertEqual(self.replica.ping(), True)

    #-------------------------------------------------------------------------
    # tests
    #-------------------------------------------------------------------------

    def test01_a_version_this_build_cannot_read_is_refused(self):
        self.set_effects_config()
        self.query_and_sync("CREATE (:U {v: 1}), (:U {v: 2})")
        full_before = self.master.info()["sync_full"]

        # 2 is the interesting one: it is what every shipped C engine writes,
        # so this is the exact payload a mixed-version pair produces.
        for version in (2, 4, 0, 255, 1):
            msg = self._refused(
                # a body that would be a well-formed record at v3, so the
                # refusal is the version byte and nothing else
                self._framed(version, 0, b"\x03\x00\x00\x00\x01\x00\x00\x00"),
                f"a version-{version} buffer")
            self.env.assertContains("version", msg.lower())

        self._still_healthy(full_before)
        # ... and nothing was applied on the way to the error
        self.assert_agree("MATCH (n:U) RETURN count(n), sum(n.v)", [[2, 3]])
        self.assert_graph_eq()

    @staticmethod
    def _add_schema_record(label_id, name):
        """`9 ADD_SCHEMA` — `u32 opcode · u32 entity_tag · i32 id · cstring`.

        Hand-assembled rather than captured off the wire because the point is
        for the *body* to be valid v3 while the header says otherwise, and
        MONITOR renders a payload as escaped text, not bytes.
        """
        return (b"\x09\x00\x00\x00"                   # opcode 9
                + b"\x00\x00\x00\x00"                 # entity tag: node
                + int(label_id).to_bytes(4, "little", signed=True)
                + (len(name) + 1).to_bytes(8, "little")  # length includes the NUL
                + name.encode() + b"\x00")

    def test01b_a_future_version_whose_body_is_readable_is_still_refused(self):
        # The case the version check exists for, and the one a message-only
        # assertion would miss: a payload whose *records* this build reads
        # perfectly well, announced as a version it does not know. Decoding the
        # records it recognises and stopping at the first it does not is exactly
        # how a reader half-applies a future buffer and corrupts itself with the
        # rest; refusing on the header byte is how it does not.
        self.set_effects_config()
        key = "effects_unreadable_future"
        g = Graph(self.master, key)
        # One label, so the injected record's id 1 is the id this graph would
        # assign next and the apply path has no other reason to refuse it.
        g.query("CREATE (:Anchor)")
        self.wait_for_replica_offset()

        record = self._add_schema_record(1, "Injected")
        # Sanity: the same record under the right version *is* accepted, so the
        # refusal below is the version byte and not a malformed body.
        self.env.assertEqual(
            self.master.execute_command("GRAPH.EFFECT", key, b"\x03\x00" + record),
            "OK")
        self.env.assertEqual(
            g.ro_query("CALL db.labels() YIELD label RETURN label ORDER BY label"
                       ).result_set, [['Anchor'], ['Injected']])

        # And now the same shape at a version this build cannot read. Id 2 this
        # time, since 'Injected' took 1.
        full_before = self.master.info()["sync_full"]
        for version in (2, 4):
            self._refused(
                bytes([version, 0]) + self._add_schema_record(2, f"Future{version}"),
                f"a valid record announced as version {version}", key=key)

        # Nothing was applied — the assertion a wrong error message cannot make.
        self.env.assertEqual(
            g.ro_query("CALL db.labels() YIELD label RETURN label ORDER BY label"
                       ).result_set, [['Anchor'], ['Injected']])
        self._still_healthy(full_before)

    def test02_a_flags_bit_this_build_does_not_know_is_refused(self):
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]

        # bit 0 is FLAG_COMPRESSED; everything above it is reserved, and a
        # reader that ignored the reserved bits would go on to parse the body
        # of a payload whose framing it does not understand.
        for flags in (0x02, 0x04, 0x80, 0xff):
            msg = self._refused(
                self._framed(3, flags, b"\x03\x00\x00\x00\x01\x00\x00\x00"),
                f"a buffer with flags 0x{flags:02x}")
            self.env.assertContains("flag", msg.lower())

        self._still_healthy(full_before)
        self.assert_agree("MATCH (n:U) RETURN count(n), sum(n.v)", [[2, 3]])

    def test03_a_truncated_header_is_refused(self):
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]

        # One byte: the version reads, the flags byte is not there. Two bytes
        # is a *valid empty* payload, which the handler answers OK — that is
        # the documented no-op, and asserting it here is what stops a future
        # "reject short buffers" from breaking every write that changed
        # nothing.
        self._refused(self._framed(3, 0)[:1], "a one-byte buffer")
        self.env.assertEqual(
            self.master.execute_command("GRAPH.EFFECT", self.GRAPH_ID, b"\x03\x00"),
            "OK")
        # and a genuinely empty argument, which short-circuits before the
        # decoder is reached at all
        self.env.assertEqual(
            self.master.execute_command("GRAPH.EFFECT", self.GRAPH_ID, b""), "OK")

        self._still_healthy(full_before)
        self.assert_agree("MATCH (n:U) RETURN count(n), sum(n.v)", [[2, 3]])

    def test04_a_corrupt_compressed_frame_is_refused(self):
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]

        # Framed as compressed, with a body zstd cannot inflate.
        msg = self._refused(self._compressed(64, 0, b"not a zstd frame at all"),
                            "a compressed buffer with a corrupt frame")
        self.env.assertContains("compress", msg.lower())

        # And an oversized declared plaintext length. `open_payload` passes it
        # to `zstd::bulk::decompress` as the allocation *ceiling*, which is
        # what stops a 100-byte frame of zeros from inflating to gigabytes —
        # the reason it is `bulk::decompress` and not `stream::decode_all`.
        # A refusal is the assertion; surviving it is the point.
        self._refused(self._compressed(0xffff_ffff, 0, b"not a zstd frame at all"),
                      "a compressed buffer declaring a 4 GiB plaintext")

        self._still_healthy(full_before)
        self.assert_agree("MATCH (n:U) RETURN count(n), sum(n.v)", [[2, 3]])

    def test05_a_valid_frame_with_the_wrong_checksum_is_refused(self):
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]

        # A well-formed record, compressed for real, with the CRC deliberately
        # wrong. The checksum is over the *plaintext* rather than the frame so
        # that C can match it without vendoring zstd's own framing, which
        # means it is the last line of defence before the records are parsed.
        plain = b"\x03\x00\x00\x00\x01\x00\x00\x00"
        frame = zstd_raw_frame(plain)
        msg = self._refused(self._compressed(len(plain), 0xdead_beef, frame),
                            "a compressed buffer with a wrong checksum")
        self.env.assertContains("checksum", msg.lower())

        # A frame that inflates to fewer bytes than the header declares: the
        # ceiling is an upper bound, so this one gets past the allocation and
        # has to be caught by the equality check behind it.
        msg = self._refused(self._compressed(len(plain) + 16, 0, frame),
                            "a compressed buffer whose declared length is too big")
        self.env.assertContains("declares", msg.lower())

        self._still_healthy(full_before)
        self.assert_agree("MATCH (n:U) RETURN count(n), sum(n.v)", [[2, 3]])

    def test06_a_client_cannot_send_an_effect_to_a_replica_at_all(self):
        # The whole reason `is_replayed` can afford to be lenient: reaching a
        # replica's apply path from a client needs the replica to be writable,
        # and it is not. If this ever starts succeeding, the leniency in
        # `divergence_guard::is_replayed` stops being a safe trade.
        self.set_effects_config()
        try:
            self.replica.execute_command(
                "GRAPH.EFFECT", self.GRAPH_ID, b"\x03\x00\x00\x00\x00\x00\x01\x00\x00\x00")
            raise AssertionError("a replica accepted a client's GRAPH.EFFECT")
        except ResponseError as e:
            self.env.assertContains("read only replica", str(e))


#-----------------------------------------------------------------------------
# 6. compression must be transparent
#-----------------------------------------------------------------------------


class testEffects_06_Compression(_EffectsBase):
    """EFFECTS_COMPRESSION only changes the framing, never the outcome.

    The same workload is applied twice, into two graph keys, once with
    compression off and once on, and the two replicas' views are compared
    against each other as well as against their primaries.
    """

    GRAPH_ID = "effects_compression"

    # Deliberately mixed: creates, an update, a removal, a delete and an edge,
    # so the compressed frame contains several record types.
    WORKLOAD = [
        """UNWIND range(1, 5000) AS x
           CREATE (:C {v: x, s: 'padding-padding-padding-' + x, l: [x, x + 1, x + 2]})""",
        "MATCH (n:C) WHERE n.v % 3 = 0 SET n.t = 'touched', n.s = NULL",
        "MATCH (n:C) WHERE n.v % 7 = 0 DELETE n",
        "UNWIND range(1, 500) AS x CREATE (:C {v: -x})",
        "MATCH (a:C {v: 1}), (b:C {v: 2}) CREATE (a)-[:LINK {w: 1.5}]->(b)",
    ]

    PROBES = [
        "MATCH (n:C) RETURN count(n)",
        "MATCH (n:C) RETURN count(n.s), count(n.t), sum(n.v)",
        "MATCH (n:C) RETURN sum(size(n.l))",
        "MATCH (n:C) WHERE n.v = 3 RETURN n.t, n.s, n.l",
        "MATCH ()-[e:LINK]->() RETURN count(e), sum(e.w)",
        "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey",
    ]

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT')

    def _run_workload(self, key, compression):
        self.set_effects_config(compression)
        self.master_graph  = Graph(self.master,  key)
        self.replica_graph = Graph(self.replica, key)
        self.monitor_mark()
        for q in self.WORKLOAD:
            self.query_and_sync(q)
        payloads = self.effect_payloads(self.monitor_mark(), key)
        answers = []
        for p in self.PROBES:
            m, r = self.probe(p)
            self.env.assertEqual(r, m)
            answers.append(m)
        return answers, payloads

    def test03_a_constraint_announcement_is_sealed_like_any_other_payload(self):
        """A constraint announcement goes out through the format, not around it.

        Both constraint sites used to call `ctx.replicate("GRAPH.EFFECT", ..)`
        themselves, on a buffer the format had never finished — so
        EFFECTS_COMPRESSION silently did not apply to them while it applied to
        every other payload. They go through `EffectsPayload::replicate` now,
        which is the only place that seals.

        The constraint is deliberately wide. `maybe_compress` refuses a frame
        that does not pay for its own 8 bytes of length and checksum, and a
        two-property constraint does not — so a narrow one would leave this
        asserting nothing, whichever way the code went.
        """
        key = "effects_comp_constraint"
        self.master_graph  = Graph(self.master,  key)
        self.replica_graph = Graph(self.replica, key)
        self.set_effects_config(64)

        props = [f"property_with_a_deliberately_long_name_{i:02d}" for i in range(12)]
        # Carrying every property the constraint will require, so it settles
        # OPERATIONAL. A node missing them is a legitimate FAILED, which would
        # test the failure path rather than this one.
        self.query_and_sync(
            "CREATE (:Wide {%s})" % ", ".join(f"{p}: {i}" for i, p in enumerate(props)))

        self.monitor_mark()
        create_mandatory_node_constraint(self.master_graph, 'Wide', *props)
        self.wait_for_constraint_settled(self.master_graph, 'Wide')
        self.wait_for_replica_offset()
        payloads = self.effect_payloads(self.monitor_mark(), key)

        self.env.assertGreater(len(payloads), 0)
        # A plain `raise`, not `env.assertTrue(cond, msg)` — RLTest's second
        # positional argument is `depth`, an int, so a message there is a
        # TypeError rather than a failure report.
        if not any(p.startswith(self.HEADER_COMPRESSED) for p in payloads):
            raise AssertionError(
                f"the constraint announcement was not sealed: {len(payloads)} "
                f"payload(s), none framed compressed")

        # and it is still a constraint on the far side
        rows = self.wait_for_constraint_settled(self.replica_graph, 'Wide')
        self.env.assertEqual(rows, self.constraint_rows(self.master_graph, 'Wide'))
        self.env.assertEqual(rows[0][0], 'MANDATORY')
        self.env.assertEqual(rows[0][4], 'OPERATIONAL')

    def test01_compressed_and_uncompressed_reach_the_same_state(self):
        plain, plain_payloads = self._run_workload("effects_comp_off", 0)
        zstd,  zstd_payloads  = self._run_workload("effects_comp_on", 1024)

        # Asserting on the flags byte is what stops this from quietly becoming
        # a second copy of the uncompressed run the day the threshold stops
        # being reached.
        self.env.assertGreater(len(plain_payloads), 0)
        self.env.assertGreater(len(zstd_payloads), 0)
        # with compression off, nothing is framed as compressed
        for p in plain_payloads:
            self.env.assertTrue(p.startswith(self.HEADER_PLAIN))
        # with it on, at least one buffer cleared the 1024-byte floor. Not all
        # of them: the floor is a minimum, and the small buffers in this
        # workload stay plain by design.
        self.env.assertGreater(
            sum(1 for p in zstd_payloads if p.startswith(self.HEADER_COMPRESSED)), 0)

        # and the two runs agree probe for probe
        for p, a, b in zip(self.PROBES, plain, zstd):
            self.env.assertEqual(b, a)

    def test02_a_compressed_graph_is_whole(self):
        # graph_eq over the compressed run, including its indexes and
        # constraints, rather than only the aggregate probes above.
        self.set_effects_config(1024)
        self.master_graph  = Graph(self.master,  "effects_comp_on")
        self.replica_graph = Graph(self.replica, "effects_comp_on")
        create_node_range_index(self.master_graph, 'C', 'v', sync=True)
        create_unique_node_constraint(self.master_graph, 'C', 'v')
        self.wait_for_constraint_settled(self.master_graph, 'C')
        self.wait_for_constraint_settled(self.replica_graph, 'C')
        self.wait_for_replica_offset()
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)
        self.assert_graph_eq()

        # leave the shared server with compression off
        self.set_effects_config(0)


#-----------------------------------------------------------------------------
# 8. the same write produces the same bytes
#-----------------------------------------------------------------------------


class testEffects_08_ByteDeterminism(_EffectsBase):
    """The bytes, not just the state.

    Every other class here asserts that the replica *agrees* — which an encoder
    could satisfy while emitting a different payload every time. That is not
    enough for #2698: a second implementation is validated by producing
    **identical** bytes for the same write, so an encoder nobody has pinned is
    a spec nobody can implement against.

    Four things in the format exist only for this, and none of them was covered
    end to end: record groups are sorted, `optimize()` is called on every
    bitmap, a bitmap is built by range rather than id by id, and the collapse
    rule is a function of the ids rather than of when the encoder looked.
    """

    GRAPH_ID = "effects_determinism"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT')

    def _payload_for(self, key, query):
        """The single effect payload `query` produces against a fresh `key`."""
        self.monitor_mark()
        self.db.select_graph(key).query(query)
        self.wait_for_replica_offset()
        window = self.monitor_mark()
        payloads = self.effect_payloads(window, key)
        if len(payloads) != 1:
            raise AssertionError(
                f"expected exactly one effect for {key}, got {len(payloads)}")
        return payloads[0]

    def test01_the_same_write_twice_produces_identical_bytes(self):
        # Two graphs, empty and therefore identical, given the same query. Ids
        # are allocated densely from zero, so the two writes describe the same
        # entities and must serialize the same way.
        #
        # A shape deliberately wide enough to exercise the parts that could
        # differ run to run: several labels so the label sets have to be
        # grouped, differing property shapes so there is more than one record
        # partition, and enough rows that grouping is not trivially ordered.
        self.set_effects_config()
        q = """
            UNWIND range(0, 199) AS i
            CREATE (:A:B {v: i, s: 'x' + i}),
                   (:B {v: i}),
                   (:A {w: i * 2, t: true})
        """
        first = self._payload_for(f"{self.GRAPH_ID}_det_1", q)
        second = self._payload_for(f"{self.GRAPH_ID}_det_2", q)
        if first != second:
            raise AssertionError(
                "the same write serialized differently — record order, bitmap "
                "construction or the collapse rule is not a function of the ids")
        self.env.assertTrue(len(first) > 0)

    def test02_a_supernode_fanout_replicates(self):
        # `Repeat`'s end-to-end case. Every edge out of one node carries the
        # same source id, which is one segment rather than one per edge — the
        # shape that has no compact form without it.
        self.set_effects_config()
        g = self.master_graph
        g.query("CREATE (:Hub {name: 'hub'})")
        self.query_and_sync("""
            MATCH (h:Hub {name: 'hub'})
            UNWIND range(0, 999) AS i
            CREATE (h)-[:OUT {i: i}]->(:Leaf {i: i})
        """)
        self.assert_agree("MATCH (:Hub)-[r:OUT]->(:Leaf) RETURN count(r)", [[1000]])
        # Every edge really does leave the one hub, so the source list is one
        # repeated id rather than a thousand distinct ones.
        self.assert_agree(
            "MATCH (h:Hub)-[:OUT]->() RETURN count(DISTINCT h)", [[1]])
        self.assert_graph_eq()

    def test03_a_permuted_id_order_is_not_silently_sorted(self):
        # Row *k* belongs to the k-th id **as written**. Nothing may reorder a
        # list to make an encoding eligible, so a scrambled update order has to
        # land each value on its own entity.
        self.set_effects_config()
        self.query_and_sync(
            "UNWIND range(0, 49) AS i CREATE (:P {v: i})")
        # A deliberately non-ascending traversal: the multiplier scatters the
        # match order, so the ids reach the emitter out of order.
        self.query_and_sync("""
            UNWIND range(0, 49) AS i
            WITH (i * 37) % 50 AS k
            MATCH (n:P {v: k})
            SET n.tag = 'tag-' + k
        """)
        self.assert_agree(
            "MATCH (n:P) RETURN n.v, n.tag ORDER BY n.v",
            [[i, f"tag-{i}"] for i in range(50)])
        self.assert_graph_eq()
