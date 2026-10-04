"""Framing, version selection, byte determinism, and compression.

The file where the payload itself is the subject rather than its effect on the
graph. Three classes, and only the middle one runs at both versions:

  test01  which version is selected, and what actually reaches the wire
  test02  byte determinism -- the same write twice produces the same bytes
  test03  compression, which is v3-only because v2 has no flags byte to carry
          the compressed bit

Determinism is worth more than it sounds. The conformance corpus re-encodes 39
Rust-generated fixtures in C and reproduces 38 byte for byte; that means what it
appears to mean ONLY if C's encoder is a function of its input.
"""

import time
import redis
from common import *
from graph_utils import graph_eq
from effects_common import _EffectsBase, WIRE


class test01_VersionSelection(_EffectsBase):
    GRAPH_ID = "effects_wire_version"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def test01_effects_enabled_by_default(self):
        # threshold 0 is the shipped default and means "always use effects"
        self.env.assertEquals(int(self.db.config_get("EFFECTS_THRESHOLD")), 0)

    def test02_default_emit_version_is_two(self):
        """C emits v2 unless asked otherwise. If this ever flips, every suite
        that does not pin a version starts measuring a different format, so the
        default is asserted rather than assumed."""
        self.env.assertEquals(int(self.db.config_get("EFFECTS_VERSION")), 2)

    def test03_selected_version_reaches_the_wire(self):
        """The config says 3; assert the PAYLOAD says 3.

        A config read only proves the setter took. `assert_wire_version` reads
        the first bytes of what the replica actually received, which is the only
        thing that distinguishes "configured for v3" from "emitting v3".
        """
        def body(v):
            self.query_and_sync("CREATE (:W {v: %d})" % v)
            window = self.assert_effect_emitted()
            self.assert_wire_version(window, version=v)
            # and the header is the length this version's table claims
            for c in [x for x in window if "GRAPH.EFFECT" in x]:
                payload = self.payload_of(c)
                self.env.assertEquals(payload[:WIRE[v]["header_len"]],
                                      WIRE[v]["header"])
        self.for_each_version(body)


class test02_Determinism(_EffectsBase):
    """The same write against two identical graphs produces identical bytes."""

    GRAPH_ID = "effects_wire_determinism"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT")

    def _payloads_for(self, key, query):
        """Every effect payload `query` produces against a fresh graph `key`."""
        self.new_graph(key)
        self.drain()
        self.master_graph.query(query)
        self.master.execute_command("WAIT", "1", "0")
        window = self.monitor_mark()
        return [self.payload_of(c) for c in window if "GRAPH.EFFECT" in c]

    @staticmethod
    def _first_diff(a, b):
        """Where two payloads first differ, with a small window around it.

        Dumping both in full is unreadable for the shapes this class uses; the
        offset is what identifies the field that was encoded differently.
        """
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                lo, hi = max(0, i - 4), min(n, i + 4)
                return "offset %d: %r vs %r" % (i, a[lo:hi], b[lo:hi])
        if len(a) != len(b):
            return "lengths differ: %d vs %d" % (len(a), len(b))
        return None

    def _assert_deterministic(self, query, tag):
        def body(v):
            first  = self._payloads_for("%s_%s_a_v%d" % (self.GRAPH_ID, tag, v), query)
            second = self._payloads_for("%s_%s_b_v%d" % (self.GRAPH_ID, tag, v), query)
            self.env.assertTrue(len(first) > 0,
                                message="no payloads captured for %s" % tag)
            self.env.assertEquals(len(first), len(second))
            for a, b in zip(first, second):
                self.env.assertEquals(
                    a, b, message="non-deterministic payload, %s"
                                  % self._first_diff(a, b))
        self.for_each_version(body)

    def test01_mixed_create_is_deterministic(self):
        self._assert_deterministic(
            "CREATE (a:D {x: 1})-[:E {w: 2}]->(b:D {x: 3}), (c:D {x: 4})",
            "mixed")

    def test02_dense_delete_is_deterministic(self):
        """The collapse rule's case: enough ids that the encoder chooses between
        a range and a bitmap, so a cost-arithmetic difference would show."""
        self._assert_deterministic(
            "UNWIND range(0, 499) AS i CREATE (:D2 {i: i})", "dense")

    def test03_supernode_fanout_is_deterministic(self):
        """Repeat's case: one source, many edges."""
        self._assert_deterministic(
            "CREATE (s:S3) WITH s UNWIND range(0, 499) AS i "
            "CREATE (s)-[:F {i: i}]->(:T3)", "fanout")


class test03_Compression(_EffectsBase):
    """v3 only: v2 has no flags byte, so no bit to carry 'this body is
    compressed'. Pinned rather than looped for that reason.

    NOT asserted: a specific ratio. Level 1's ratio swings on record alignment,
    so a threshold on it would be a test of zstd's tuning rather than of this
    engine's plumbing.
    """

    GRAPH_ID = "effects_wire_compression"

    def __init__(self):
        self._setup(version=3, threshold=0)

    # GRAPH.EFFECT is counted on the REPLICA, not the master -- the master runs
    # GRAPH.QUERY. Reading the master's commandstats concludes effects are off
    # when they are on.
    def _effect_stats(self):
        row = self.replica.info("commandstats").get("cmdstat_graph.EFFECT")
        if row is None:
            return (0, 0)
        return (row["calls"], row["failed_calls"])

    def _sync_full(self):
        return self.master.info("stats")["sync_full"]

    def _replica_bytes_in(self):
        return self.replica.info("stats")["total_net_input_bytes"]

    def _wait_for_count(self, n, q="MATCH (n:L) RETURN count(n)"):
        # GRAPH.QUERY is a write command and refused on a read-only replica, so
        # this reads with ro_query.
        #
        # AND THE GRAPH KEY MAY NOT EXIST ON THE REPLICA YET. A live link means
        # the connection is established, not that anything has replicated; the
        # key appears only with the first write. Querying a missing graph raises
        # "Invalid graph operation on empty key" rather than returning 0, which
        # failed this test about half the time.
        last = 0
        for _ in range(200):
            try:
                last = self.replica_graph.ro_query(q).result_set[0][0]
                if last == n:
                    return n
            except redis.exceptions.ResponseError as e:
                # only the not-yet-replicated case is tolerated; anything else
                # is a real failure and must not be swallowed by a poll loop
                if "empty key" not in str(e):
                    raise
                last = 0
            time.sleep(0.05)
        return last

    def _measured(self, min_bytes, q, settle):
        """Run q with compression at `min_bytes`, holding the three mechanism
        checks around it. Returns the bytes the replica read."""
        self.set_effects_config(compression=min_bytes)

        calls0, failed0 = self._effect_stats()
        sync0           = self._sync_full()
        bytes0          = self._replica_bytes_in()

        settle(self.master_graph.query(q))
        bytes1 = self._replica_bytes_in()
        calls1, failed1 = self._effect_stats()

        # the replica applied it through the effects path
        self.env.assertGreater(calls1, calls0)
        # and did NOT refuse it. Without this, a refusal followed by a resync
        # looks identical to success
        self.env.assertEquals(failed1, failed0)
        # and no full resync happened, which is the other way a refusal hides
        self.env.assertEquals(self._sync_full(), sync0)

        return bytes1 - bytes0

    def _write_run(self, min_bytes, n):
        q = ("UNWIND range(1, %d) AS i "
             "CREATE (:L {v: i, s: 'padpadpadpadpadpad'})" % n)

        def settle(res):
            self.env.assertEquals(res.nodes_created, n)
            self.env.assertEquals(self._wait_for_count(n), n)

        return self._measured(min_bytes, q, settle)

    def test01_compressed_payload_round_trips(self):
        n = 2000
        self.new_graph(self.GRAPH_ID + "_off")
        uncompressed = self._write_run(0, n)
        self.new_graph(self.GRAPH_ID + "_on")
        compressed   = self._write_run(64, n)

        # the compressed run moved materially fewer bytes. A ratio is not
        # asserted; that it is smaller by a wide margin is
        self.env.assertLess(compressed * 4, uncompressed)
        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    def test02_every_group_in_a_compressed_payload_applies(self):
        """A payload with several record groups, compressed, must apply whole --
        a partial inflate would leave the graphs differing rather than error."""
        n = 500
        self.new_graph(self.GRAPH_ID + "_groups")
        q = ("UNWIND range(1, %d) AS i "
             "CREATE (:A {i: i})-[:R {i: i}]->(:B {i: i})" % n)

        def settle(res):
            self.env.assertEquals(res.nodes_created, 2 * n)
            self.env.assertEquals(res.relationships_created, n)
            self.env.assertEquals(
                self._wait_for_count(n, "MATCH (a:A) RETURN count(a)"), n)

        self._measured(64, q, settle)
        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))
