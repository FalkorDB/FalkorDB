"""Effects replication over the v3 wire format.

`test_effects.py` covers the *shapes* a graph mutation can take and asserts the
replica ends up equal. This file is about the wire itself: what v3 puts on it,
and the replica-side apply path that reads it. Everything here needs a primary
and a replica, and asserts on both.

Ordering matters, so the classes carry numeric prefixes: RLTest discovers
classes via `dir(module)`, which is sorted, and the three classes at the end
take the topology apart — `06d` restarts the primary, `06e` stops it with a
deliberately unloadable AOF, and `07` promotes the replica. Each puts the link
back, but running them last means a failure part-way through one cannot strand
the topology for a class that has not run yet. File order matches run order, so
read them in the order they are written.

Every class sets its own compression, rather than inheriting it: it is a
process-global that resets on restart, and RLTest hands consecutive classes with
identical `Env(...)` parameters the *same* server — so whatever the previous
class left behind is what this one would otherwise start with. `06e` is the one
class with a server of its own, because it passes `enableDebugCommand=True` and
turns `appendonly` on.
"""

import itertools
import threading
import time

from common import *
from constraint_utils import (create_mandatory_node_constraint,
                              create_unique_node_constraint,
                              drop_unique_node_constraint,
                              get_constraint, list_constraints)
from graph_utils import graph_eq
from index_utils import (create_node_range_index, drop_node_range_index,
                         list_indicies, wait_for_indices_to_sync)

# A plain Redis key the tests SET on the primary purely so its replicated form
# shows up in the replica's MONITOR feed as a fence post. See `monitor_mark`.
MONITOR_MARK_KEY = "__effects_v3_mark__"


class _EffectsV3Base():
    """A primary/replica pair with v3 selected, plus the waiting helpers.

    Not discovered as a test: RLTest only collects module-level names starting
    with "test".
    """

    # Overridden per class so no two classes share a graph key. RLTest may hand
    # them the same server.
    GRAPH_ID = "effects_v3"

    #-------------------------------------------------------------------------
    # setup
    #-------------------------------------------------------------------------

    def _setup(self, compression=0, enable_debug=False):
        # replication under sanitizer is unreliable, as test_replication.py notes
        if SANITIZER:
            Environment.skip(None)

        # `enable_debug` is only for `testEffectsV3_06e_AofReplay`, which needs
        # DEBUG LOADAOF. It also changes the `Env(...)` parameters, so RLTest
        # hands that class a server of its own rather than the shared one —
        # which is wanted anyway, since it turns `appendonly` on.
        self.env, self.db = Env(env='oss', useSlaves=True,
                                enableDebugCommand=enable_debug)
        self.master  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()

        self.monitor = []
        self.monitor_filter = ()
        # Per-instance, not the module-level flag test_effects.py and
        # test_constraint.py use: several classes here each attach their own
        # MONITOR, and a global latched True by the first would let the rest
        # run before their listener was on the socket.
        self.monitor_attached = False
        self.monitor_thread = None
        self._marks = itertools.count(1)

        # A previous class in a reused env may have left keys behind.
        self.master.flushall()

        # The link takes seconds to come up and RLTest does not gate on it.
        self.wait_for_replica_link()
        self.set_effects_config(compression)

        self.master_graph  = Graph(self.master,  self.GRAPH_ID)
        self.replica_graph = Graph(self.replica, self.GRAPH_ID)

    def set_effects_config(self, compression=0):
        # Effects are the only replication mechanism and v3 the only format, so
        # compression is the one thing left to choose.
        self.db.config_set("EFFECTS_COMPRESSION", compression)

    #-------------------------------------------------------------------------
    # waiting
    #-------------------------------------------------------------------------

    def wait_for_replica_link(self, timeout=60):
        """Block until the replica reports a live, finished link.

        Polled rather than slept: the handshake takes a few seconds and its
        duration is not something a test can predict.
        """
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            last = self.replica.info("replication")
            if (last.get("master_link_status") == "up"
                    and int(last.get("master_sync_in_progress", 1)) == 0):
                return
            time.sleep(0.1)
        raise AssertionError(
            f"replica link did not come up within {timeout}s; last INFO "
            f"replication: {last}")

    def wait_for_replica_offset(self, timeout=180):
        """Block until the replica has applied everything the primary has
        propagated so far.

        WAIT is not usable here. It blocks on the calling client's own `woff`,
        which Redis only advances when *that* client's command propagated
        something. Two of the writes these tests care about are issued by
        threads inside the module — the constraint validation thread's
        re-announcement, and the post-promotion hook — over contexts this
        connection knows nothing about, so `self.master`'s `woff` never covers
        them and WAIT takes its "already acked" fast path. Compare offsets.
        """
        target = self.master.info("replication")["master_repl_offset"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.replica.info("replication").get("slave_repl_offset", -1) >= target:
                return
            time.sleep(0.01)
        raise AssertionError(
            f"replica did not reach primary offset {target} within {timeout}s")

    def query_and_sync(self, q, params=None):
        """Write on the primary, then block until the replica has applied it."""
        res = self.master_graph.query(q, params)
        self.wait_for_replica_offset()
        return res

    #-------------------------------------------------------------------------
    # reading both sides
    #-------------------------------------------------------------------------

    def probe(self, q, params=None):
        """Run one read query on both sides, returning (primary, replica).

        Both go through GRAPH.RO_QUERY: GRAPH.QUERY against a replica is
        rejected with READONLY regardless of what the query actually does.
        """
        m = self.master_graph.ro_query(q, params).result_set
        r = self.replica_graph.ro_query(q, params).result_set
        return m, r

    def assert_agree(self, q, expected, params=None):
        """The primary produces `expected`, and the replica produces the same.

        Both halves matter. Equality alone would also pass on the day the
        primary stops doing the thing under test.
        """
        m, r = self.probe(q, params)
        self.env.assertEqual(m, expected)
        self.env.assertEqual(r, m)

    def assert_graph_eq(self):
        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    def constraint_rows(self, graph, label):
        q = """CALL db.constraints()
               YIELD type, label, properties, entitytype, status
               WHERE label = $lbl
               RETURN type, label, properties, entitytype, status
               ORDER BY type, properties"""
        return graph.ro_query(q, {'lbl': label}).result_set

    def wait_for_constraint_settled(self, graph, label, timeout=120):
        """Poll until `label` has at least one constraint and none of them read
        UNDER CONSTRUCTION. Returns the rows.

        Bounded, unlike `constraint_utils.wait_on_constraint`, which loops
        forever — a stuck-pending replica is exactly the regression this file
        is here to catch, and it should fail rather than hang.
        """
        deadline = time.time() + timeout
        rows = None
        while time.time() < deadline:
            rows = self.constraint_rows(graph, label)
            if rows and all(row[4] != 'UNDER CONSTRUCTION' for row in rows):
                return rows
            time.sleep(0.05)
        raise AssertionError(
            f"constraint on {label} did not settle within {timeout}s; last "
            f"read: {rows}")

    #-------------------------------------------------------------------------
    # MONITOR on the replica
    #-------------------------------------------------------------------------

    def start_monitor(self, *interesting):
        """Attach MONITOR to the replica, recording commands whose text
        contains one of `interesting` (the fence-post key is always kept)."""
        self.monitor_filter = tuple(interesting) + (MONITOR_MARK_KEY,)
        # daemon=True so a stuck listen() cannot outlive the test process
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        deadline = time.time() + 30
        while not self.monitor_attached:
            if time.time() > deadline:
                raise AssertionError("MONITOR did not attach to the replica within 30s")
            time.sleep(0.05)

    def _monitor_loop(self):
        try:
            with self.replica.monitor() as m:
                self.monitor_attached = True
                for cmd in m.listen():
                    if any(f in cmd['command'] for f in self.monitor_filter):
                        self.monitor.append(cmd['command'])
        except Exception:
            pass

    def monitor_mark(self, timeout=60):
        """Fence the replica's MONITOR feed, and return everything recorded
        before the fence (dropping it from the buffer).

        MONITOR is asynchronous with respect to replication: the replica
        applies a replicated command and only *then* writes the feed line to
        the monitoring client, so an offset-based wait can return before the
        line has reached this process. Clearing the buffer at that point drops
        nothing and the line lands in the *next* window instead — which is how
        a constraint announced twice reads as three effects. Bracketing a
        window with two marks removes the race entirely: when the mark's own
        line is visible, every line the replica produced before it is too.

        The fence is a plain SET on the primary. It replicates verbatim, so it
        appears in the replica's feed carrying a token this process chose.
        """
        token = f"mark-{next(self._marks)}"
        self.master.set(MONITOR_MARK_KEY, token)
        deadline = time.time() + timeout
        while time.time() < deadline:
            for i, cmd in enumerate(self.monitor):
                if token in cmd:
                    window = list(self.monitor[:i])
                    del self.monitor[:i + 1]
                    return window
            time.sleep(0.02)
        raise AssertionError(
            f"MONITOR fence {token} never appeared in the replica's feed")

    @staticmethod
    def count_in(window, cmd):
        return sum(1 for c in window if cmd in c)

    # A v3 buffer opens with `u8 version · u8 flags`, and bit 0 of the flags
    # byte is FLAG_COMPRESSED. MONITOR renders the two as escape sequences.
    HEADER_PLAIN      = r'\x03\x00'
    HEADER_COMPRESSED = r'\x03\x01'

    @staticmethod
    def effect_payloads(window, key):
        """The payload text of every GRAPH.EFFECT in `window` aimed at `key`.

        Sliced by locating the command prefix rather than by splitting on
        spaces: an effects buffer is binary, 0x20 occurs in it freely, and
        MONITOR renders it as a literal space — so the payload is "everything
        after the graph key", not "the third token". For the same reason a
        header test has to be anchored with startswith: the two header bytes
        also occur, meaninglessly, all over the body.
        """
        prefix = f"GRAPH.EFFECT {key} "
        return [c[c.index(prefix) + len(prefix):]
                for c in window if c.startswith(prefix)]


#-----------------------------------------------------------------------------
# 0. a payload this build cannot read
#-----------------------------------------------------------------------------

# ── zstd, for the compressed-framing cases below ──────────────────────────
#
# Only the two tests that need a *valid* frame use it; the corrupt-frame and
# oversized-length cases are constructed from garbage on purpose. Guarded
# because `compression.zstd` is 3.14+ and `zstandard` is not in
# tests/requirements.txt — a missing zstd must skip two assertions, not the
# file.
try:
    from compression import zstd as _zstd

    def zstd_compress(data):
        return _zstd.compress(data)
except ImportError:  # pragma: no cover - depends on the interpreter
    try:
        import zstandard as _zstandard

        def zstd_compress(data):
            return _zstandard.ZstdCompressor().compress(data)
    except ImportError:
        zstd_compress = None


class testEffectsV3_00_UnreadableBuffer(_EffectsV3Base):
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
    and `testEffectsV3_06c_DivergenceForcesResync` already pins that path end
    to end.
    """

    GRAPH_ID = "effects_v3_unreadable"

    def __init__(self):
        self._setup()

    #-------------------------------------------------------------------------
    # helpers
    #-------------------------------------------------------------------------

    @staticmethod
    def _framed(version, flags, body=b""):
        return bytes([version, flags]) + body

    @staticmethod
    def _compressed(plain_len, checksum, frame):
        """A payload that *claims* to be a compressed v3 frame:
        `u8 3 · u8 FLAG_COMPRESSED · u32 plain_len · u32 crc32 · frame`."""
        return (b"\x03\x01"
                + plain_len.to_bytes(4, "little")
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
        key = "effects_v3_unreadable_future"
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
        if zstd_compress is None:
            Environment.skip(None)
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]

        # A well-formed record, compressed for real, with the CRC deliberately
        # wrong. The checksum is over the *plaintext* rather than the frame so
        # that C can match it without vendoring zstd's own framing, which
        # means it is the last line of defence before the records are parsed.
        plain = b"\x03\x00\x00\x00\x01\x00\x00\x00"
        frame = zstd_compress(plain)
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
# 1. a null property value means REMOVE
#-----------------------------------------------------------------------------

class testEffectsV3_01_NullIsRemove(_EffectsV3Base):
    """`SET x = NULL` removes the property, and the replica must remove it too.

    Regression for a real v3 divergence: the apply side filtered nulls out of a
    record's value rows before merging them, so `MATCH (n) SET n.x = NULL`
    removed the property on the primary and did nothing at all on the replica.
    Nothing surfaced it — the link stayed healthy, no error anywhere, and the
    two only re-converged on the next full resync.

    FalkorDB never stores a null property value, which is what makes a null on
    the wire unambiguous: it can only mean "remove this attribute".
    """

    GRAPH_ID = "effects_v3_null"

    def __init__(self):
        self._setup()

    def test01_node_property_set_to_null(self):
        self.query_and_sync("CREATE (:N {id: 1, x: 'gone', y: 'kept'})")
        # baseline — both sides can see the property before it is removed
        self.assert_agree("MATCH (n:N) RETURN count(n.x)", [[1]])

        res = self.query_and_sync("MATCH (n:N) SET n.x = NULL")
        self.env.assertEqual(res.properties_removed, 1)

        # count(expr) skips nulls, so a value surviving on either side is a 1
        self.assert_agree("MATCH (n:N) RETURN count(n.x)", [[0]])
        # ... and the property is *absent*, not merely reading as null
        self.assert_agree("MATCH (n:N) RETURN 'x' IN keys(n)", [[False]])
        # the untouched sibling is still there on both sides
        self.assert_agree("MATCH (n:N) RETURN n.y", [['kept']])
        # keys() enumerates registered attributes in a deterministic order, so
        # the two sides' whole property shape is comparable
        m, r = self.probe("MATCH (n:N) RETURN keys(n)")
        self.env.assertEqual(m, [[['id', 'y']]])
        self.env.assertEqual(r, m)

        self.assert_graph_eq()

    def test02_edge_property_set_to_null(self):
        self.query_and_sync("CREATE ()-[:R {id: 1, x: 'gone', y: 'kept'}]->()")
        self.assert_agree("MATCH ()-[e:R]->() RETURN count(e.x)", [[1]])

        res = self.query_and_sync("MATCH ()-[e:R]->() SET e.x = NULL")
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_agree("MATCH ()-[e:R]->() RETURN count(e.x)", [[0]])
        self.assert_agree("MATCH ()-[e:R]->() RETURN 'x' IN keys(e)", [[False]])
        self.assert_agree("MATCH ()-[e:R]->() RETURN e.y", [['kept']])

        self.assert_graph_eq()

    def test03_one_of_several_properties_set_to_null(self):
        # A null in the middle of a multi-assignment SET: the record carries
        # three attribute columns, one of which is a removal.
        self.query_and_sync("CREATE (:M {a: 1, b: 2, c: 3, d: 4})")

        res = self.query_and_sync("MATCH (n:M) SET n.a = 10, n.b = NULL, n.c = 30")
        # `a` and `c` are overwrites, which FalkorDB accounts for as a removal
        # plus a set; `b` is the only pure removal. So two sets, three removals.
        self.env.assertEqual(res.properties_set, 2)
        self.env.assertEqual(res.properties_removed, 3)

        self.assert_agree("MATCH (n:M) RETURN n.a, n.c, n.d", [[10, 30, 4]])
        self.assert_agree("MATCH (n:M) RETURN 'b' IN keys(n)", [[False]])
        self.assert_agree("MATCH (n:M) RETURN count(n.b)", [[0]])

        # ... and the same via map addition, which reaches the same record
        res = self.query_and_sync("MATCH (n:M) SET n += {a: NULL, e: 5}")
        self.env.assertEqual(res.properties_set, 1)
        self.env.assertEqual(res.properties_removed, 1)
        self.assert_agree("MATCH (n:M) RETURN n.c, n.d, n.e", [[30, 4, 5]])
        self.assert_agree("MATCH (n:M) RETURN 'a' IN keys(n)", [[False]])

        self.assert_graph_eq()

    def test04_whole_shape_null(self):
        # Every column of the record is a removal.
        self.query_and_sync("CREATE (:W {a: 1, b: 2, c: 3})")
        self.query_and_sync("CREATE ()-[:WR {a: 1, b: 2}]->()")

        res = self.query_and_sync("MATCH (n:W) SET n = {a: NULL, b: NULL, c: NULL}")
        self.env.assertEqual(res.properties_removed, 3)
        self.assert_agree("MATCH (n:W) RETURN keys(n)", [[[]]])

        res = self.query_and_sync("MATCH ()-[e:WR]->() SET e = {a: NULL, b: NULL}")
        self.env.assertEqual(res.properties_removed, 2)
        self.assert_agree("MATCH ()-[e:WR]->() RETURN keys(e)", [[[]]])

        # the empty-map form of the same thing
        self.query_and_sync("MATCH (n:W) SET n.z = 1")
        res = self.query_and_sync("MATCH (n:W) SET n = {}")
        self.env.assertEqual(res.properties_removed, 1)
        self.assert_agree("MATCH (n:W) RETURN keys(n)", [[[]]])

        self.assert_graph_eq()

    def test05_null_arriving_as_a_query_parameter(self):
        # The literal NULL and a null-valued parameter take different routes
        # through the planner; both must reach the wire as a removal.
        self.query_and_sync("CREATE (:Param {id: 7, v: 'here'})")
        res = self.master_graph.query("MATCH (n:Param) SET n.v = $new",
                                      {'new': None})
        self.wait_for_replica_offset()
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_agree("MATCH (n:Param) RETURN count(n.v)", [[0]])
        self.assert_agree("MATCH (n:Param) RETURN 'v' IN keys(n)", [[False]])
        self.assert_agree("MATCH (n:Param) RETURN n.id", [[7]])

        self.assert_graph_eq()

    def test06_null_in_only_some_rows_of_one_record(self):
        # The shape most likely to break: one record, many entities, and the
        # column is a real value for some rows and a removal for others. A
        # filtering apply path either drops the whole column or misaligns the
        # values against the ids.
        self.query_and_sync(
            "UNWIND range(1, 1000) AS i CREATE (:P {id: i, x: 'v'})")
        self.assert_agree("MATCH (p:P) RETURN count(p.x)", [[1000]])

        res = self.query_and_sync(
            "MATCH (p:P) SET p.x = CASE WHEN p.id % 2 = 0 THEN NULL ELSE 'v2' END")
        # 500 nulls are pure removals; the 500 overwrites are each accounted
        # for as a removal plus a set.
        self.env.assertEqual(res.properties_removed, 1000)
        self.env.assertEqual(res.properties_set, 500)

        self.assert_agree("MATCH (p:P) RETURN count(p)", [[1000]])
        self.assert_agree("MATCH (p:P) RETURN count(p.x)", [[500]])
        # the survivors are exactly the odd ids, on both sides — this is what
        # catches a column that landed against the wrong entities
        self.assert_agree(
            "MATCH (p:P) WHERE p.x IS NOT NULL RETURN count(p), sum(p.id % 2)",
            [[500, 500]])
        self.assert_agree(
            "MATCH (p:P) WHERE p.x IS NULL RETURN count(p), sum(p.id % 2)",
            [[500, 0]])
        self.assert_agree("MATCH (p:P) WHERE p.x IS NOT NULL RETURN DISTINCT p.x",
                          [['v2']])

        self.assert_graph_eq()

    def test07_null_removes_every_property_of_many_entities(self):
        self.query_and_sync(
            "UNWIND range(1, 500) AS i CREATE (:Q {id: i, x: i, y: i})")
        res = self.query_and_sync("MATCH (q:Q) SET q.x = NULL, q.y = NULL")
        self.env.assertEqual(res.properties_removed, 1000)

        self.assert_agree("MATCH (q:Q) RETURN count(q), count(q.x), count(q.y)",
                          [[500, 0, 0]])
        self.assert_agree("MATCH (q:Q) RETURN DISTINCT keys(q)", [[['id']]])

        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# 2. constraints replicate as GRAPH.EFFECT, not verbatim GRAPH.CONSTRAINT
#-----------------------------------------------------------------------------

class testEffectsV3_02_ConstraintAsEffect(_EffectsV3Base):
    """A constraint is announced as an effect, carrying its status.

    The announcement is of the *outcome*: the replica installs the status this
    node reached instead of validating on its own, because an independent scan
    runs at a different time against different interleavings and could
    legitimately disagree. v2 had no way to carry a status, so it replicated the
    command twice and let the replica read the repeat as an activation signal.

    MANDATORY constraints are used throughout: they need no supporting index,
    so the only thing the window contains is the constraint announcement.
    """

    GRAPH_ID = "effects_v3_constraint"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT', 'GRAPH.CONSTRAINT')

    def test01_v3_announces_as_one_effect(self):
        self.set_effects_config()
        # under the 10_000-entity async threshold, so validation runs inline
        # and the status is already settled when the command returns — one
        # announcement, nothing to re-announce
        self.query_and_sync("CREATE (:Small {v: 1})")

        self.monitor_mark()
        create_mandatory_node_constraint(self.master_graph, 'Small', 'v')
        self.wait_for_constraint_settled(self.master_graph, 'Small')
        self.wait_for_replica_offset()
        window = self.monitor_mark()

        self.env.assertEqual(self.count_in(window, 'GRAPH.EFFECT'), 1)
        self.env.assertEqual(self.count_in(window, 'GRAPH.CONSTRAINT'), 0)

        # the status the primary reached is the status the replica holds
        master_rows  = self.constraint_rows(self.master_graph,  'Small')
        replica_rows = self.wait_for_constraint_settled(self.replica_graph, 'Small')
        self.env.assertEqual(master_rows,
                             [['MANDATORY', 'Small', ['v'], 'NODE', 'OPERATIONAL']])
        self.env.assertEqual(replica_rows, master_rows)

    def test02_v3_announces_a_drop_as_an_effect(self):
        self.set_effects_config()
        self.monitor_mark()
        self.master_graph.execute_command(
            "GRAPH.CONSTRAINT", "DROP", self.GRAPH_ID, "MANDATORY", "NODE",
            "Small", "PROPERTIES", 1, "v")
        self.wait_for_replica_offset()
        window = self.monitor_mark()

        self.env.assertEqual(self.count_in(window, 'GRAPH.EFFECT'), 1)
        self.env.assertEqual(self.count_in(window, 'GRAPH.CONSTRAINT'), 0)

        self.env.assertEqual(self.constraint_rows(self.master_graph, 'Small'), [])
        self.env.assertEqual(self.constraint_rows(self.replica_graph, 'Small'), [])

    def test04_a_failed_constraint_replicates_its_failure(self):
        # The status travels, so FAILED has to arrive as FAILED. A replica that
        # re-derived the status would reach the same answer here, which is
        # precisely why the interesting assertion is that it did not have to.
        self.set_effects_config()
        self.query_and_sync(
            "CREATE (:Dup {v: 1}), (:Dup {v: 1})")

        self.monitor_mark()
        create_unique_node_constraint(self.master_graph, 'Dup', 'v')
        self.wait_for_constraint_settled(self.master_graph, 'Dup')
        self.wait_for_replica_offset()
        window = self.monitor_mark()
        self.env.assertEqual(self.count_in(window, 'GRAPH.CONSTRAINT'), 0)
        self.env.assertGreater(self.count_in(window, 'GRAPH.EFFECT'), 0)

        master_rows  = self.constraint_rows(self.master_graph,  'Dup')
        replica_rows = self.wait_for_constraint_settled(self.replica_graph, 'Dup')
        self.env.assertEqual(master_rows,
                             [['UNIQUE', 'Dup', ['v'], 'NODE', 'FAILED']])
        self.env.assertEqual(replica_rows, master_rows)


#-----------------------------------------------------------------------------
# 3. an asynchronously validated constraint is announced twice and converges
#-----------------------------------------------------------------------------

class testEffectsV3_03_ConstraintConvergence(_EffectsV3Base):
    """Above the async threshold a constraint is announced twice — once UNDER
    CONSTRUCTION, once with the settled status — and the replica must end with
    exactly ONE constraint at the settled status.

    The second announcement comes from the module's own validation thread, not
    from the client's command, and the apply side upserts rather than inserts.
    Get either wrong and the replica ends with two constraints, or with one
    stuck at UNDER CONSTRUCTION forever.
    """

    GRAPH_ID = "effects_v3_async_constraint"

    # > 10_000 entities of the label is what pushes validation off the main
    # thread in Graph::create_constraint. At or below it the status is settled
    # before the command returns and there is nothing to re-announce.
    BIG = 10_500

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT', 'GRAPH.CONSTRAINT')

    def test01_two_announcements_leave_exactly_one_constraint(self):
        self.set_effects_config()
        self.query_and_sync(
            f"UNWIND range(1, {self.BIG}) AS x CREATE (:Big {{v: x}})")
        self.assert_agree("MATCH (n:Big) RETURN count(n)", [[self.BIG]])

        self.monitor_mark()
        res = create_mandatory_node_constraint(self.master_graph, 'Big', 'v')
        # the command returns before validation finishes
        self.env.assertEqual(res, 'PENDING')

        master_rows  = self.wait_for_constraint_settled(self.master_graph,  'Big')
        replica_rows = self.wait_for_constraint_settled(self.replica_graph, 'Big')
        self.wait_for_replica_offset()
        window = self.monitor_mark()

        # exactly two: UNDER CONSTRUCTION, then the settled status
        self.env.assertEqual(self.count_in(window, 'GRAPH.EFFECT'), 2)
        self.env.assertEqual(self.count_in(window, 'GRAPH.CONSTRAINT'), 0)

        # the convergence property — one constraint, not two, at the settled
        # status and not stuck pending
        self.env.assertEqual(master_rows,
                             [['MANDATORY', 'Big', ['v'], 'NODE', 'OPERATIONAL']])
        self.env.assertEqual(replica_rows, master_rows)
        self.assert_agree(
            "CALL db.constraints() YIELD label WHERE label = 'Big' RETURN count(1)",
            [[1]])
        self.assert_agree(
            "CALL db.constraints() YIELD status RETURN count(1)", [[1]])

    def test02_re_announcement_does_not_duplicate_a_unique_constraint(self):
        # Same convergence, via UNIQUE — which also drags a supporting index
        # onto the wire ahead of the constraint, so the announcements are not
        # the only records in the window.
        self.set_effects_config()
        self.query_and_sync(
            f"UNWIND range(1, {self.BIG}) AS x CREATE (:BigU {{u: x}})")

        create_unique_node_constraint(self.master_graph, 'BigU', 'u')
        master_rows  = self.wait_for_constraint_settled(self.master_graph,  'BigU')
        replica_rows = self.wait_for_constraint_settled(self.replica_graph, 'BigU')

        self.env.assertEqual(master_rows,
                             [['UNIQUE', 'BigU', ['u'], 'NODE', 'OPERATIONAL']])
        self.env.assertEqual(replica_rows, master_rows)
        self.assert_agree(
            "CALL db.constraints() YIELD label WHERE label = 'BigU' RETURN count(1)",
            [[1]])

        # the constraint is enforced on the primary...
        try:
            self.master_graph.query("CREATE (:BigU {u: 1})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation", str(e))

        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)
        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# 4. bulk shapes round-trip
#-----------------------------------------------------------------------------

class testEffectsV3_04_Shapes(_EffectsV3Base):
    """The record shapes the v3 writer has to partition and the reader has to
    reassemble: multiple labels, differing property shapes in one query,
    points, lists, large batches, deletes interleaved with creates, and ids
    recycled by a delete-then-recreate.
    """

    GRAPH_ID = "effects_v3_shapes"

    def __init__(self):
        self._setup()

    def test01_multi_label_nodes(self):
        # Three label sets in one CREATE is three record partitions: the writer
        # groups by label set, so this is the smallest query that exercises
        # partitioning at all.
        res = self.query_and_sync("CREATE (:A), (:B), (:A:B)")
        self.env.assertEqual(res.nodes_created, 3)
        self.env.assertEqual(res.labels_added, 2)

        self.assert_agree("MATCH (n) RETURN count(n)", [[3]])
        self.assert_agree("MATCH (n:A) RETURN count(n)", [[2]])
        self.assert_agree("MATCH (n:B) RETURN count(n)", [[2]])
        self.assert_agree("MATCH (n:A:B) RETURN count(n)", [[1]])
        self.assert_agree(
            "CALL db.labels() YIELD label RETURN label ORDER BY label",
            [['A'], ['B']])

        # a label added after the fact, and one removed
        self.query_and_sync("MATCH (n:A) WHERE NOT n:B SET n:C:D")
        self.assert_agree("MATCH (n:C:D) RETURN count(n)", [[1]])
        self.query_and_sync("MATCH (n:C) REMOVE n:C")
        self.assert_agree("MATCH (n:C) RETURN count(n)", [[0]])
        self.assert_agree("MATCH (n:D) RETURN count(n)", [[1]])

        self.assert_graph_eq()

    def test02_differing_property_shapes_in_one_query(self):
        # Same label, four different property shapes: another four partitions,
        # this time keyed on the attribute set rather than the label set.
        res = self.query_and_sync(
            "CREATE (:S {a: 1}), (:S {a: 1, b: 2}), (:S {b: 2, c: 3}), (:S)")
        self.env.assertEqual(res.nodes_created, 4)

        m, r = self.probe("MATCH (n:S) RETURN keys(n) ORDER BY keys(n)")
        self.env.assertEqual(m, [[[]], [['a']], [['a', 'b']], [['b', 'c']]])
        self.env.assertEqual(r, m)
        self.assert_agree("MATCH (n:S) RETURN sum(n.a), sum(n.b), sum(n.c)",
                          [[2, 4, 3]])

        # and the same for edges
        res = self.query_and_sync(
            "CREATE ()-[:E {x: 1}]->(), ()-[:E]->(), ()-[:F {y: 's', z: [1, 2]}]->()")
        self.env.assertEqual(res.relationships_created, 3)
        m, r = self.probe(
            "MATCH ()-[e]->() RETURN type(e), keys(e) ORDER BY type(e), keys(e)")
        self.env.assertEqual(m, [['E', []], ['E', ['x']], ['F', ['y', 'z']]])
        self.env.assertEqual(r, m)

        self.assert_graph_eq()

    def test03_points_and_lists(self):
        self.query_and_sync("""CREATE (:Geo {
                                    p: point({latitude: 32.07, longitude: 34.79}),
                                    l: [1, 'two', [3.5], true],
                                    empty: []
                               })""")
        self.query_and_sync("""CREATE ()-[:GEO {
                                    p: point({latitude: -12.5, longitude: 0.0}),
                                    l: ['a', [1, 2], 3]
                               }]->()""")

        # Nested lists and points are structured values; comparing the rendered
        # value on both sides is what pins that the wire carried the structure
        # rather than something that merely counts the same.
        m, r = self.probe("MATCH (n:Geo) RETURN n.p, n.l, n.empty")
        self.env.assertEqual(len(m), 1)
        self.env.assertEqual(r, m)
        m, r = self.probe("MATCH ()-[e:GEO]->() RETURN e.p, e.l")
        self.env.assertEqual(len(m), 1)
        self.env.assertEqual(r, m)

        self.assert_agree("MATCH (n:Geo) RETURN size(n.l), size(n.empty)", [[4, 0]])
        self.assert_agree(
            "MATCH (n:Geo) RETURN n.p.latitude > 32.0 AND n.p.longitude > 34.0",
            [[True]])

        self.assert_graph_eq()

    def test04_large_batch(self):
        # ~100k nodes in one query: one record with 100k ids, which is where
        # the id-list encodings and the block sizing actually get exercised.
        # Aggregate probes rather than graph_eq — comparing 100k whole nodes
        # twice measures the client, not the wire.
        N = 100_000
        res = self.query_and_sync(
            f"UNWIND range(1, {N}) AS x CREATE (:Bulk {{v: x, s: 's' + x}})")
        self.env.assertEqual(res.nodes_created, N)

        self.assert_agree("MATCH (n:Bulk) RETURN count(n)", [[N]])
        self.assert_agree(
            "MATCH (n:Bulk) RETURN sum(n.v), min(n.v), max(n.v)",
            [[N * (N + 1) // 2, 1, N]])
        # a spot check that the values landed against the right entities, not
        # merely that the right number of them arrived
        self.assert_agree(
            "MATCH (n:Bulk) WHERE n.s = 's50000' RETURN n.v", [[50000]])
        self.assert_agree(
            "MATCH (n:Bulk) WHERE n.s <> 's' + n.v RETURN count(n)", [[0]])

        # a large batch of edges over those nodes
        E = 20_000
        res = self.query_and_sync(
            f"""MATCH (a:Bulk), (b:Bulk)
                WHERE a.v <= {E} AND b.v = a.v + 1
                CREATE (a)-[:NEXT {{w: a.v}}]->(b)""")
        self.env.assertEqual(res.relationships_created, E)
        self.assert_agree("MATCH ()-[e:NEXT]->() RETURN count(e), sum(e.w)",
                          [[E, E * (E + 1) // 2]])

    def test05_deletes_interleaved_with_creates(self):
        # One query that deletes and creates, so a single effects buffer holds
        # both a delete record and a create record.
        res = self.query_and_sync(
            """MATCH (n:Bulk) WHERE n.v % 2 = 0 DELETE n
               WITH count(1) AS x
               UNWIND range(1, 100) AS i CREATE (:Fresh {i: i})""")
        self.env.assertEqual(res.nodes_deleted, 50_000)
        self.env.assertEqual(res.nodes_created, 100)

        self.assert_agree("MATCH (n:Bulk) RETURN count(n)", [[50_000]])
        self.assert_agree("MATCH (n:Bulk) WHERE n.v % 2 = 0 RETURN count(n)", [[0]])
        self.assert_agree("MATCH (n:Fresh) RETURN count(n), sum(n.i)", [[100, 5050]])
        # the edges hanging off the deleted nodes went with them, on both sides
        self.assert_agree("MATCH ()-[e:NEXT]->() RETURN count(e)", [[0]])

        # and an edge delete in the same shape
        self.query_and_sync(
            """MATCH (a:Fresh {i: 1}), (b:Fresh {i: 2})
               CREATE (a)-[:TMP]->(b), (b)-[:TMP]->(a)""")
        res = self.query_and_sync(
            """MATCH ()-[e:TMP]->() DELETE e
               WITH count(1) AS x
               CREATE (:AfterEdgeDelete)""")
        self.env.assertEqual(res.relationships_deleted, 2)
        self.assert_agree("MATCH ()-[e:TMP]->() RETURN count(e)", [[0]])
        self.assert_agree("MATCH (n:AfterEdgeDelete) RETURN count(n)", [[1]])

    def test06_delete_then_recreate_recycles_ids(self):
        # A create record after a delete hands out ids the delete freed. v3
        # refuses a node id the replica cannot legitimately hold, so a recycled
        # id is exactly the case where that check has to agree with the
        # primary's allocator rather than fight it.
        before = self.master_graph.ro_query(
            "MATCH (n:Fresh) RETURN min(ID(n)), max(ID(n))").result_set[0]

        self.query_and_sync("MATCH (n:Fresh) DELETE n")
        self.assert_agree("MATCH (n:Fresh) RETURN count(n)", [[0]])

        res = self.query_and_sync(
            "UNWIND range(1, 100) AS i CREATE (:Recycled {i: i})")
        self.env.assertEqual(res.nodes_created, 100)

        self.assert_agree("MATCH (n:Recycled) RETURN count(n), sum(n.i)",
                          [[100, 5050]])
        # the ids really were reused — the new nodes sit inside the range the
        # deleted ones occupied, so this is not just "100 nodes arrived"
        m, r = self.probe("MATCH (n:Recycled) RETURN min(ID(n)), max(ID(n))")
        self.env.assertEqual(r, m)
        self.env.assertLessEqual(m[0][0], before[1])

        # round-trip it again, this time deleting and recreating in one query
        res = self.query_and_sync(
            """MATCH (n:Recycled) DELETE n
               WITH count(1) AS x
               UNWIND range(1, 100) AS i CREATE (:Recycled2 {i: i})""")
        self.env.assertEqual(res.nodes_deleted, 100)
        self.env.assertEqual(res.nodes_created, 100)
        self.assert_agree("MATCH (n:Recycled) RETURN count(n)", [[0]])
        self.assert_agree("MATCH (n:Recycled2) RETURN count(n), sum(n.i)",
                          [[100, 5050]])

    def test06b_create_delete_and_recreate_one_id_in_one_buffer(self):
        # The case test06 does *not* reach. A query that commits three times
        # puts all three commits in ONE effects buffer, and the allocator
        # recycles a freed id across commits — so the buffer creates an id,
        # deletes it, and creates it again. The replica has to accept the third
        # record: the delete released the id.
        #
        # It needs a graph whose recycle bin is empty, because `reserve_node`
        # hands out the *smallest* freed id. On a graph with older free ids the
        # third commit gets one of those instead and the collision never
        # happens — which is exactly why the shared-graph tests miss this.
        fresh = "effects_v3_recycle_one_buffer"
        m = Graph(self.master, fresh)
        r = Graph(self.replica, fresh)

        res = m.query("CREATE (n:Doomed) WITH n DELETE n "
                      "WITH 1 AS z CREATE (:Reborn)")
        self.env.assertEqual(res.nodes_created, 2)
        self.env.assertEqual(res.nodes_deleted, 1)
        self.wait_for_replica_offset()

        # The primary kept exactly one node, on the recycled id 0.
        self.env.assertEqual(
            m.ro_query("MATCH (n) RETURN count(n), labels(n)[0], ID(n)").result_set,
            [[1, "Reborn", 0]])
        # And the replica must hold the same. Refusing the recreate discards the
        # whole payload, so the failure here is an empty graph, not a wrong one.
        self.env.assertEqual(
            r.ro_query("MATCH (n) RETURN count(n), labels(n)[0], ID(n)").result_set,
            m.ro_query("MATCH (n) RETURN count(n), labels(n)[0], ID(n)").result_set)
        self.env.assertTrue(graph_eq(m, r))

    def test07_everything_still_agrees(self):
        # A whole-graph comparison once the shapes above have all been applied.
        # Cheap now: test05 removed the bulk of the nodes.
        self.query_and_sync("MATCH (n:Bulk) DELETE n")
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)
        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# 4b. every value tag, over a live wire
#-----------------------------------------------------------------------------

class testEffectsV3_04b_ValueTypes(_EffectsV3Base):
    """Every `SIValue` tag the v3 codec writes, replicated for real.

    `graph/src/effects/v3/value.rs` pins these byte for byte, and that is a
    different claim: the pins say the encoder and the decoder agree about the
    bytes, not that a value a query produced survives being written, framed,
    replicated, applied, folded into the replica's attribute set and read back.
    The two have come apart before — the null-is-remove divergence
    `testEffectsV3_01_NullIsRemove` exists for was a value the codec carried
    correctly and the apply path dropped.

    Points and lists are already covered by
    `testEffectsV3_04_Shapes.test03_points_and_lists`; this is the rest of the
    tag list, plus the numeric and string edges that only a live round trip
    reaches. Maps are deliberately absent: `T_MAP` exists in the codec for
    index `OPTIONS`, and a map is not a legal property value ("Property values
    can only be of primitive types or arrays of primitive types"), so there is
    no query that puts one on this path.
    """

    GRAPH_ID = "effects_v3_value_types"

    def __init__(self):
        self._setup()

    def _round_trips(self, label, props, probe, expected):
        """Store `props` on a node and on an edge, and require both sides to
        read back `expected` through `probe`."""
        assignments = ", ".join(f"{k}: {v}" for k, v in props.items())
        self.query_and_sync(f"CREATE (:{label} {{{assignments}}})")
        self.query_and_sync(f"CREATE ()-[:{label}R {{{assignments}}}]->()")
        self.assert_agree(f"MATCH (n:{label}) RETURN {probe('n')}", expected)
        self.assert_agree(f"MATCH ()-[e:{label}R]->() RETURN {probe('e')}", expected)

    def test01_temporal_values(self):
        # T_DATE, T_TIME, T_DATETIME and T_DURATION are four separate tags over
        # the same i64 payload, so a swapped pair round-trips as a number and
        # reads back as the wrong type. Comparing the *rendered* value is what
        # distinguishes them.
        self.set_effects_config()
        self._round_trips(
            'Temporal',
            {'d':  "date('2024-01-15')",
             't':  "localtime('10:30:00')",
             'dt': "localdatetime('2024-01-15T10:30:00')",
             'du': "duration({days: 3, hours: 4})"},
            lambda v: (f"toString({v}.d), toString({v}.t), "
                       f"toString({v}.dt), toString({v}.du)"),
            [['2024-01-15', '10:30:00', '2024-01-15T10:30:00', 'P3DT4H']])
        self.assert_graph_eq()

    def test02_vecf32(self):
        # T_VECTOR_F32 is `u32 dim · f32 x dim`, the one value whose payload is
        # length-prefixed rather than fixed or NUL-terminated.
        self.set_effects_config()
        # A vector is not a stringifiable type, so the content is pinned by
        # distance to a literal built the same way: zero to itself, non-zero to
        # a vector one component away.
        self._round_trips(
            'Vec',
            {'v': "vecf32([1.5, -2.25, 0.0, 3.0e10])", 'one': "vecf32([0.5])"},
            lambda x: (f"vec.euclideanDistance({x}.v, vecf32([1.5, -2.25, 0.0, 3.0e10])), "
                       f"vec.euclideanDistance({x}.one, vecf32([0.5])), "
                       f"vec.euclideanDistance({x}.one, vecf32([1.5]))"),
            [[0.0, 0.0, 1.0]])
        self.assert_graph_eq()

    def test03_numeric_edges(self):
        # i64 at both ends, and the float values whose bit patterns a
        # round-trip through anything but `to_le_bytes` mangles: signed zero
        # (which compares equal to 0.0, so only the *sign* separates them) and
        # the infinities.
        self.set_effects_config()
        self._round_trips(
            'Num',
            {'imax': "9223372036854775807",
             'imin': "-9223372036854775808",
             'zero': "0.0",
             'negzero': "-0.0",
             'inf': "1.0/0.0",
             'neginf': "-1.0/0.0",
             # not f64::MAX: the parser folds 1.797...e308 to infinity, and
             # refuses a subnormal like 5.0e-324 outright ("unhandled type in
             # inlined properties"). Both sides agree on those, so they are a
             # parser limit rather than anything this file is about.
             'huge': "1.0e300",
             'small': "-1.0e300"},
            lambda v: (f"{v}.imax, {v}.imin, {v}.inf, {v}.neginf, "
                       f"{v}.huge, {v}.small, "
                       # `1/x` is the only expression that tells -0.0 from 0.0
                       f"1.0/{v}.negzero, 1.0/{v}.zero"),
            [[9223372036854775807, -9223372036854775808,
              float('inf'), float('-inf'),
              1.0e300, -1.0e300,
              float('-inf'), float('inf')]])
        self.assert_graph_eq()

    def test04_booleans_are_not_integers(self):
        # T_BOOL is a tag plus one byte, T_INT64 a tag plus eight. Losing the
        # distinction reads back as 0/1, which most probes cannot see.
        self.set_effects_config()
        self._round_trips(
            'Bool', {'t': "true", 'f': "false", 'i': "1", 'z': "0"},
            lambda v: (f"{v}.t, {v}.f, toString({v}.t), toString({v}.i), "
                       f"{v}.t = {v}.i, {v}.f = {v}.z"),
            [[True, False, 'true', '1', False, False]])
        self.assert_graph_eq()

    def test05_string_edges(self):
        # `write_string` is a *C* string: a length that includes the
        # terminator, then the bytes, then a NUL. So the edges are the empty
        # string (length 1, no bytes), multi-byte UTF-8 (a length in bytes, not
        # characters), and bytes that the RESP framing around the payload would
        # be sensitive to if the payload were not length-prefixed.
        self.set_effects_config()
        params = {
            'empty':   '',
            'space':   ' ',
            'crlf':    'a\r\nb',
            'quote':   'he said "hi" and \\ then',
            'unicode': 'שלום · 世界 · 🐦‍🔥',
            'long':    'x' * 1_000_000,
        }
        self.master_graph.query(
            """CREATE (:Str {empty: $empty, space: $space, crlf: $crlf,
                             quote: $quote, unicode: $unicode, long: $long})""",
            params)
        self.master_graph.query(
            """CREATE ()-[:StrR {empty: $empty, space: $space, crlf: $crlf,
                                 quote: $quote, unicode: $unicode, long: $long}]->()""",
            params)
        self.wait_for_replica_offset()

        for entity, pattern in (('n', "MATCH (n:Str)"),
                                ('e', "MATCH ()-[e:StrR]->()")):
            q = (f"{pattern} RETURN {entity}.empty, {entity}.space, "
                 f"{entity}.crlf, {entity}.quote, {entity}.unicode, "
                 f"size({entity}.long), {entity}.long = $long")
            m, r = self.probe(q, {'long': params['long']})
            self.env.assertEqual(
                m, [['', ' ', 'a\r\nb', params['quote'], params['unicode'],
                     1_000_000, True]])
            self.env.assertEqual(r, m)

        # An interior NUL is the one string shape the writer calls out — it
        # `debug_assert`s against one, because C reads these as C strings and
        # would truncate at it. There is no case for it here: the parser
        # refuses a NUL in a parameter ("Failed to parse the value of parameter
        # 's'") and truncates a query string at one, so no query can put such a
        # value on this path in the first place.

        self.assert_graph_eq()

    def test06_list_edges(self):
        # T_ARRAY is a count and then values, so nesting is the interesting
        # axis rather than length: depth costs a tag per level and a decoder
        # that recursed without a bound is what the codec's depth cap is for.
        self.set_effects_config()
        self.query_and_sync("""CREATE (:List {
            empty:  [],
            nested: [[1, [2, [3, [4, [5]]]]]],
            mixed:  [1, 'two', 3.5, true, [6]],
            long:   range(1, 10000)
        })""")
        self.assert_agree(
            """MATCH (n:List) RETURN size(n.empty), size(n.nested),
                      size(n.mixed), size(n.long),
                      n.nested[0][1][1][1][1][0],
                      n.mixed[1], n.mixed[4][0],
                      reduce(s = 0, x IN n.long | s + x)""",
            [[0, 1, 5, 10000, 5, 'two', 6, 50005000]])
        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# 4c. shapes the earlier classes do not reach
#-----------------------------------------------------------------------------

class testEffectsV3_04c_HarderShapes(_EffectsV3Base):
    """Shapes with structure the writer's partitioning has to survive: an edge
    whose endpoints are one node, a very wide attribute set, and the operators
    that commit more than once so their records share a buffer.

    `testEffectsV3_04_Shapes` covers volume and interleaving. These are the
    shapes where the *blocks* are unusual — `IdSet` and `IdList` are distinct
    types precisely because edge endpoints repeat, and a self-loop is the
    smallest case where the same id appears in both endpoint lists.
    """

    GRAPH_ID = "effects_v3_harder_shapes"

    def __init__(self):
        self._setup()

    def test01_an_edge_whose_endpoints_are_the_same_node(self):
        # `IdList` exists because endpoints deduplicate neither by value nor by
        # position; a self-loop is where a set-shaped encoding would collapse
        # src and dst into one entry and misalign everything after it.
        self.set_effects_config()
        res = self.query_and_sync(
            "CREATE (a:Loop {v: 1}) WITH a CREATE (a)-[:SELF {w: 1}]->(a)")
        self.env.assertEqual(res.relationships_created, 1)
        self.assert_agree(
            "MATCH (a:Loop)-[e:SELF]->(b) RETURN ID(a) = ID(b), count(e)",
            [[True, 1]])

        # A batch of them, so the record carries many rows whose two endpoint
        # lists are identical.
        self.query_and_sync("UNWIND range(1, 500) AS i CREATE (:Ring {i: i})")
        res = self.query_and_sync(
            "MATCH (n:Ring) CREATE (n)-[:SELF2 {w: n.i}]->(n)")
        self.env.assertEqual(res.relationships_created, 500)
        self.assert_agree(
            """MATCH (a:Ring)-[e:SELF2]->(b) WHERE ID(a) = ID(b)
               RETURN count(e), sum(e.w)""",
            [[500, 125250]])

        # ... and they come off cleanly, endpoints and all
        res = self.query_and_sync("MATCH (n:Ring) WHERE n.i % 2 = 0 DETACH DELETE n")
        self.env.assertEqual(res.nodes_deleted, 250)
        self.env.assertEqual(res.relationships_deleted, 250)
        self.assert_agree("MATCH ()-[e:SELF2]->() RETURN count(e)", [[250]])
        self.assert_graph_eq()

    def test02_a_very_wide_attribute_set(self):
        # `AttrSet` is `u16 n` ids and then `count x n` values, so a wide shape
        # is the case where the reader's row/column arithmetic can go wrong
        # without the totals changing.
        self.set_effects_config()
        W = 300
        props = ", ".join(f"p{i}: {i}" for i in range(W))
        self.query_and_sync(f"CREATE (:Wide {{{props}}})")
        self.assert_agree("MATCH (n:Wide) RETURN size(keys(n))", [[W]])
        # every column against its own id, not just the count
        self.assert_agree(
            f"""MATCH (n:Wide)
                RETURN size([k IN keys(n) WHERE n[k] <> toInteger(substring(k, 1))])""",
            [[0]])

        # the same width across many entities in one record
        self.query_and_sync(
            f"UNWIND range(1, 50) AS i CREATE (:Wide2 {{{props}, i: i}})")
        self.assert_agree(
            "MATCH (n:Wide2) RETURN count(n), sum(n.p299), sum(n.i)",
            [[50, 50 * 299, 1275]])
        self.assert_graph_eq()

    def test03_operators_that_commit_more_than_once(self):
        # MERGE, FOREACH and a chain of CREATEs each drive `CommitOp` several
        # times in one query, and every commit appends to the *same* buffer.
        # That is the shape where a payload compressed or finished per commit
        # produces something unreadable, and where a record written against a
        # stale schema baseline names an id the earlier commit had not yet
        # announced.
        self.set_effects_config()
        res = self.query_and_sync(
            "UNWIND range(1, 100) AS i MERGE (:Merged {i: i})")
        self.env.assertEqual(res.nodes_created, 100)
        # a second run of the same MERGE creates nothing, so the buffer is
        # empty and nothing must be replicated for it either
        res = self.query_and_sync(
            "UNWIND range(1, 100) AS i MERGE (:Merged {i: i})")
        self.env.assertEqual(res.nodes_created, 0)
        self.assert_agree("MATCH (n:Merged) RETURN count(n), sum(n.i)",
                          [[100, 5050]])

        # MERGE that matches some and creates others, in one query
        res = self.query_and_sync(
            "UNWIND range(50, 150) AS i MERGE (:Merged {i: i})")
        self.env.assertEqual(res.nodes_created, 50)
        self.assert_agree("MATCH (n:Merged) RETURN count(n), sum(n.i)",
                          [[150, 11325]])

        # MERGE on a relationship, which commits both endpoints and the edge
        self.query_and_sync(
            """UNWIND range(1, 50) AS i
               MATCH (a:Merged {i: i}), (b:Merged {i: i + 1})
               MERGE (a)-[:MERGED {w: i}]->(b)""")
        self.assert_agree("MATCH ()-[e:MERGED]->() RETURN count(e), sum(e.w)",
                          [[50, 1275]])

        # FOREACH, which commits once per iteration
        res = self.query_and_sync(
            "FOREACH (i IN range(1, 100) | CREATE (:Foreach {i: i}))")
        self.env.assertEqual(res.nodes_created, 100)
        self.assert_agree("MATCH (n:Foreach) RETURN count(n), sum(n.i)",
                          [[100, 5050]])

        # a nested FOREACH, and one that mutates rather than creates
        self.query_and_sync(
            """FOREACH (i IN range(1, 10) |
                 FOREACH (j IN range(1, 10) |
                   CREATE (:Nested {v: i * 10 + j})))""")
        self.assert_agree("MATCH (n:Nested) RETURN count(n), sum(n.v)",
                          [[100, 6050]])
        self.query_and_sync(
            "MATCH (n:Foreach) WITH collect(n) AS ns "
            "FOREACH (n IN ns | SET n.touched = true, n.i = n.i + 1000)")
        self.assert_agree(
            "MATCH (n:Foreach) RETURN count(n.touched), sum(n.i)",
            [[100, 105050]])

        # several CREATE clauses in one query, each its own commit
        res = self.query_and_sync(
            """CREATE (:Chain {s: 1})
               WITH 1 AS x CREATE (:Chain {s: 2})
               WITH 1 AS x CREATE (:Chain {s: 3})
               WITH 1 AS x MATCH (c:Chain) SET c.seen = true""")
        self.env.assertEqual(res.nodes_created, 3)
        self.assert_agree("MATCH (n:Chain) RETURN count(n), count(n.seen)",
                          [[3, 3]])

        self.assert_graph_eq()

    def test04_a_delete_that_takes_its_own_edges_with_it(self):
        # A node and the edges that hang off it in one buffer: the delete
        # record for the edges has to be readable *after* the node's, since
        # applying them the other way round leaves an edge whose endpoint is
        # gone. Self-loops are in the mix on purpose — they appear in both
        # endpoint lists of the record being deleted.
        self.set_effects_config()
        self.query_and_sync(
            """UNWIND range(1, 200) AS i CREATE (:Hub {i: i})""")
        self.query_and_sync(
            """MATCH (a:Hub {i: 1}), (b:Hub) WHERE b.i > 1
               CREATE (a)-[:OUT {w: b.i}]->(b), (b)-[:IN {w: b.i}]->(a)""")
        self.query_and_sync("MATCH (a:Hub {i: 1}) CREATE (a)-[:OWN]->(a)")
        self.assert_agree(
            "MATCH (a:Hub {i: 1}) RETURN size((a)--())", [[399]])

        res = self.query_and_sync("MATCH (a:Hub {i: 1}) DETACH DELETE a")
        self.env.assertEqual(res.nodes_deleted, 1)
        self.env.assertEqual(res.relationships_deleted, 399)
        # scoped by type: the class shares one graph, and the earlier tests
        # left edges of their own in it
        self.assert_agree(
            "MATCH ()-[e:OUT|IN|OWN]->() RETURN count(e)", [[0]])
        self.assert_agree("MATCH (n:Hub) RETURN count(n)", [[199]])

        # and deleting a node and one of its own edges explicitly, in one query
        self.query_and_sync(
            """MATCH (a:Hub {i: 2}), (b:Hub {i: 3})
               CREATE (a)-[:E1]->(b), (a)-[:E2]->(b)""")
        res = self.query_and_sync(
            "MATCH (a:Hub {i: 2})-[e:E1]->() DELETE e, a")
        self.env.assertEqual(res.nodes_deleted, 1)
        self.env.assertEqual(res.relationships_deleted, 2)
        self.assert_agree("MATCH ()-[e:E1|E2]->() RETURN count(e)", [[0]])
        self.assert_graph_eq()


    def test05_an_edge_updated_and_deleted_in_one_query(self):
        # UPDATE_EDGE carries its relationship type, which the emitter reads off
        # the graph *after* commit has applied. `SET e.x = 1 DELETE e` leaves
        # the edge in `existing_relationships_attrs` while its row in the type
        # matrix is already gone, so a panicking type lookup takes the primary
        # down here — on a legitimate query, before anything reaches a replica.
        self.set_effects_config()
        self.query_and_sync(
            """CREATE (a:Doomed {i: 1})-[:GONE {w: 0}]->(b:Doomed {i: 2})""")

        res = self.query_and_sync(
            "MATCH (a:Doomed)-[e:GONE]->(b) SET e.w = 99 DELETE e")
        self.env.assertEqual(res.relationships_deleted, 1)
        self.assert_agree("MATCH ()-[e:GONE]->() RETURN count(e)", [[0]])
        # both endpoints survive: only the edge was named
        self.assert_agree("MATCH (n:Doomed) RETURN count(n)", [[2]])

        # and in bulk, where the update and the delete land in one buffer with
        # many rows rather than one
        self.query_and_sync(
            """UNWIND range(1, 200) AS i
               CREATE (:Src {i: i})-[:ALSOGONE {w: i}]->(:Dst {i: i})""")
        res = self.query_and_sync(
            "MATCH ()-[e:ALSOGONE]->() SET e.w = e.w * 2 DELETE e")
        self.env.assertEqual(res.relationships_deleted, 200)
        self.assert_agree("MATCH ()-[e:ALSOGONE]->() RETURN count(e)", [[0]])
        self.assert_graph_eq()

    def test06_edges_of_two_types_updated_in_one_query(self):
        # The type is stated once per record, so it is part of the group key:
        # two types sharing an attribute shape must split into two records
        # rather than land in one under whichever type was seen first.
        self.set_effects_config()
        self.query_and_sync(
            """UNWIND range(1, 50) AS i
               CREATE (a:Two {i: i}), (b:Two {i: -i}),
                      (a)-[:TA {w: i}]->(b), (a)-[:TB {w: i}]->(b)""")

        res = self.query_and_sync(
            "MATCH ()-[e:TA|TB]->() SET e.w = e.w + 1000")
        self.env.assertEqual(res.properties_set, 100)
        self.assert_agree(
            """MATCH ()-[e:TA]->() RETURN count(e), min(e.w), max(e.w)""",
            [[50, 1001, 1050]])
        self.assert_agree(
            """MATCH ()-[e:TB]->() RETURN count(e), min(e.w), max(e.w)""",
            [[50, 1001, 1050]])
        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# 4d. GRAPH.RECORD is a real write
#-----------------------------------------------------------------------------

class testEffectsV3_04d_RecordCommand(_EffectsV3Base):
    """`GRAPH.RECORD` replicates its write as an effect, like `GRAPH.QUERY`.

    RECORD adds an operator trace to a normal write; it is not a dry run.
    `record_mut` reaches `finish_write` — the same tail `GRAPH.QUERY` uses — so
    the write commits and ships. Nothing exercised that: RECORD's only other
    references are a TUI visualiser and an e2e *read*, so the write half had no
    automated coverage at all, and a RECORD that quietly stopped replicating
    would look exactly like one that worked.

    It is also registered `write`, which means Redis lets it through on a
    master only — so a RECORD reaching a replica through the stream would be
    the command itself, not an effect. Hence the feed assertions.
    """

    GRAPH_ID = "effects_v3_record"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT', 'GRAPH.RECORD', 'graph.RECORD')

    def record(self, query):
        return self.master.execute_command("GRAPH.RECORD", self.GRAPH_ID, query)

    def test01_a_recorded_write_ships_as_an_effect(self):
        self.set_effects_config()
        self.monitor_mark()
        self.record("CREATE (:R {v: 1}), (:R {v: 2})")
        self.wait_for_replica_offset()
        window = self.monitor_mark()

        # One effect, and the command itself did not go verbatim.
        self.env.assertEqual(self.count_in(window, 'GRAPH.EFFECT'), 1)
        self.env.assertEqual(
            sum(1 for c in window if 'GRAPH.RECORD' in c.upper()), 0)
        self.assert_agree("MATCH (n:R) RETURN count(n), sum(n.v)", [[2, 3]])
        self.assert_graph_eq()

    def test02_recorded_updates_and_deletes(self):
        self.set_effects_config()
        self.record("MATCH (n:R) SET n.v = n.v * 10, n.tag = 'set'")
        self.wait_for_replica_offset()
        self.assert_agree("MATCH (n:R) RETURN sum(n.v), count(n.tag)", [[30, 2]])

        # a removal through RECORD, which is the path the null-is-remove
        # divergence would have hidden in
        self.record("MATCH (n:R) SET n.tag = NULL")
        self.wait_for_replica_offset()
        self.assert_agree("MATCH (n:R) RETURN count(n.tag)", [[0]])

        self.record("MATCH (n:R) WHERE n.v = 10 DELETE n")
        self.wait_for_replica_offset()
        self.assert_agree("MATCH (n:R) RETURN count(n), sum(n.v)", [[1, 20]])
        self.assert_graph_eq()

    def test03_recorded_index_ddl(self):
        # Index DDL replicates from the plan rather than from the mutation
        # counters, so it is the one write that ships an effect while every
        # statistic stays zero — and RECORD takes the same path.
        self.set_effects_config()
        self.monitor_mark()
        self.record("CREATE INDEX FOR (n:R) ON (n.v)")
        self.wait_for_replica_offset()
        window = self.monitor_mark()
        self.env.assertEqual(self.count_in(window, 'GRAPH.EFFECT'), 1)
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)
        self.env.assertEqual(list_indicies(self.replica_graph).result_set,
                             list_indicies(self.master_graph).result_set)

        self.record("DROP INDEX FOR (n:R) ON (n.v)")
        self.wait_for_replica_offset()
        self.env.assertEqual(list_indicies(self.replica_graph).result_set,
                             list_indicies(self.master_graph).result_set)

    def test04_a_recorded_write_that_fails_replicates_nothing(self):
        # `record_mut` routes a failed write to `abandon_write`, which drops the
        # private version without publishing. Nothing must reach the wire — a
        # buffer sent for a write the master rolled back is the one failure mode
        # that leaves the replica *ahead* of the master.
        #
        # RECORD does not answer a failed write with an error: it answers with
        # the trace, and the failing operator's row carries the message. That is
        # the point of the command, and it is also why this needs testing — the
        # caller sees a successful reply either way, so a RECORD that rolled
        # back on the master and still shipped its buffer would look identical
        # from the client.
        self.set_effects_config()
        self.query_and_sync("CREATE (:Guard {u: 1})")
        create_node_range_index(self.master_graph, 'Guard', 'u', sync=True)
        create_unique_node_constraint(self.master_graph, 'Guard', 'u')
        rows = self.wait_for_constraint_settled(self.master_graph, 'Guard')
        self.env.assertEqual(rows[0][4], 'OPERATIONAL',
                             message="the constraint has to be enforcing for this to prove anything")
        self.wait_for_constraint_settled(self.replica_graph, 'Guard')
        self.wait_for_replica_offset()

        self.monitor_mark()
        trace, _plan = self.record("CREATE (:Guard {u: 1})")
        # the violation is reported, inside the trace
        failures = [row for row in trace if row[1] == 0]
        self.env.assertEqual(len(failures), 1)
        self.env.assertContains("unique constraint violation", failures[0][2])

        self.wait_for_replica_offset()
        window = self.monitor_mark()
        self.env.assertEqual(self.count_in(window, 'GRAPH.EFFECT'), 0)
        self.assert_agree("MATCH (n:Guard) RETURN count(n)", [[1]])
        self.assert_graph_eq()

        # And the rolled-back write released the MVCC write slot: `abandon_write`
        # owns that, and a leak here would wedge every later write on this graph
        # rather than fail visibly.
        self.query_and_sync("CREATE (:AfterFailure {v: 1})")
        self.assert_agree("MATCH (n:AfterFailure) RETURN count(n)", [[1]])


#-----------------------------------------------------------------------------
# 4e. the commands that still replicate verbatim
#-----------------------------------------------------------------------------

class testEffectsV3_04e_VerbatimCommands(_EffectsV3Base):
    """The commands that were deliberately left replicating verbatim, and the
    effects that follow them.

    Removing query replay removed it for *queries*. `GRAPH.COPY`,
    `GRAPH.RESTORE`, `GRAPH.DELETE` and `GRAPH.UDF` are commands rather than
    Cypher, so `replicate_verbatim` (or, for COPY, an explicit
    `ctx.replicate("GRAPH.RESTORE", ...)`) is still how they travel — there is
    no effect record for "here is a whole serialized graph".

    The interesting half is not that they arrive; it is what happens to the
    *next* effect. Every data record identifies labels, types and attributes by
    a bare id, and after one of these commands the replica's dictionaries were
    built by a different mechanism than the master's — a decoder blob, or a
    deletion. If the two end up numbering anything differently, the next
    effect is refused with `IdMismatch` and the replica resyncs in a loop. So
    each case here is followed by writes that introduce *new* schema and new
    attributes, and by an assertion that nothing was refused.
    """

    GRAPH_ID = "effects_v3_verbatim"

    def __init__(self):
        self._setup()

    def _nothing_was_refused(self, full_before, diverged_before):
        self.env.assertEqual(
            self.master.info()["sync_full"], full_before,
            message="an effect was refused after a verbatim command, forcing a resync")
        self.env.assertEqual(self.replica_log_diverged(), diverged_before)

    def replica_log_diverged(self):
        """How many divergence refusals the replica has logged so far.

        `sync_full` alone would miss a refusal on a graph the guard could not
        find a master address for — that arm exits instead of resyncing, so it
        would show up as a dead replica rather than as a resync, and counting
        the log lines catches both.
        """
        path = self.env.log_path
        if not path:
            return 0
        slave = path.replace("master-1", "slave-2")
        total = 0
        for p in (path, slave):
            if os.path.exists(p):
                with open(p, "rb") as f:
                    total += f.read().count(b"diverged")
        return total

    @staticmethod
    def udf_names(con):
        return sorted(lib[1] for lib in con.execute_command("GRAPH.UDF", "LIST"))

    def replica_udf_names(self):
        self.replica.config_set("slave-read-only", "no")
        try:
            return self.udf_names(self.replica)
        finally:
            self.replica.config_set("slave-read-only", "yes")

    def test01_copy_replicates_as_restore_and_effects_follow(self):
        self.set_effects_config()
        src, dst = self.GRAPH_ID, self.GRAPH_ID + "_copy"
        self.query_and_sync(
            "CREATE (:P {a: 1, b: 'x'})-[:E {w: 2}]->(:Q {c: 3})")

        full_before = self.master.info()["sync_full"]
        diverged_before = self.replica_log_diverged()

        # COPY forks, and a fork can be refused under memory pressure; the
        # retry is what test_graph_copy.py does for the same reason.
        deadline = time.time() + 60
        while True:
            try:
                self.master.execute_command("GRAPH.COPY", src, dst)
                break
            except ResponseError as e:
                if "fork" not in str(e).lower() or time.time() > deadline:
                    raise
                time.sleep(1)
        self.wait_for_replica_offset()

        m_dst, r_dst = Graph(self.master, dst), Graph(self.replica, dst)
        for q, expected in (
                ("MATCH (n) RETURN count(n)", [[2]]),
                ("MATCH ()-[e]->() RETURN count(e), sum(e.w)", [[1, 2]]),
                ("CALL db.labels() YIELD label RETURN label", [['P'], ['Q']]),
                ("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey",
                 [['a'], ['b'], ['c'], ['w']])):
            mv = m_dst.ro_query(q).result_set
            self.env.assertEqual(mv, expected)
            self.env.assertEqual(r_dst.ro_query(q).result_set, mv)

        # Now effects into the copy, introducing a new attribute and a new
        # label — the ids the replica has to agree about.
        m_dst.query("MATCH (n:P) SET n.zzz = 9, n.b = 'y'")
        m_dst.query("CREATE (:NewLabel {brandnew: 1})")
        self.wait_for_replica_offset()
        for q in ("MATCH (n) RETURN count(n)",
                  "MATCH (n:P) RETURN n.zzz, n.b",
                  "MATCH (n:NewLabel) RETURN n.brandnew",
                  "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey"):
            mv = m_dst.ro_query(q).result_set
            self.env.assertEqual(r_dst.ro_query(q).result_set, mv,
                                 message=f"disagreed after RESTORE: {q}")
        self.env.assertTrue(graph_eq(m_dst, r_dst))
        self._nothing_was_refused(full_before, diverged_before)

    def test02_delete_reaches_the_replica_and_the_rebuild_replays(self):
        # A graph key deleted on the master must go on the replica too, and the
        # writes that recreate it must land in the *new* graph rather than
        # against the dictionaries of the old one.
        self.set_effects_config()
        key = self.GRAPH_ID + "_copy"
        full_before = self.master.info()["sync_full"]
        diverged_before = self.replica_log_diverged()

        self.master.execute_command("GRAPH.DELETE", key)
        self.wait_for_replica_offset()
        self.env.assertEqual(self.master.exists(key), 0)
        self.env.assertEqual(self.replica.exists(key), 0)

        m, r = Graph(self.master, key), Graph(self.replica, key)
        m.query("UNWIND range(1, 50) AS i CREATE (:Fresh {q: i})")
        m.query("MATCH (n:Fresh) WHERE n.q % 5 = 0 SET n.extra = 'e'")
        self.wait_for_replica_offset()
        for q in ("MATCH (n) RETURN count(n), sum(n.q), count(n.extra)",
                  "CALL db.labels() YIELD label RETURN label",
                  "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey"):
            mv = m.ro_query(q).result_set
            self.env.assertEqual(r.ro_query(q).result_set, mv,
                                 message=f"disagreed after GRAPH.DELETE: {q}")
        # the old graph's labels are gone rather than inherited
        self.env.assertEqual(
            m.ro_query("CALL db.labels() YIELD label RETURN label").result_set,
            [['Fresh']])
        self.env.assertTrue(graph_eq(m, r))
        self._nothing_was_refused(full_before, diverged_before)

    def test03_udf_libraries_reach_the_replica(self):
        # UDFs are process state rather than graph state, so they have no
        # effect record and never will. They must still replicate, because a
        # promoted replica has to be able to answer a query that calls one.
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]
        diverged_before = self.replica_log_diverged()

        script = """
        function Doubled (x) { return x * 2; }
        falkor.register ('Doubled', Doubled);
        """
        self.db.udf_load("EffectsV3Udf", script, True)
        self.wait_for_replica_offset()

        self.env.assertContains("EffectsV3Udf", self.udf_names(self.master))
        # `GRAPH.UDF` carries Redis's `write` flag for every subcommand, LIST
        # included, so reading the replica's libraries needs the read-only flag
        # lifted — the same dance `testEffectsV3_05b_IndexDDLMechanism` does for
        # GRAPH.EXPLAIN, and lifted only for the read.
        self.env.assertContains("EffectsV3Udf", self.replica_udf_names())
        # and it is callable there, not merely listed
        self.assert_agree("RETURN EffectsV3Udf.Doubled(21)", [[42]])

        self.db.udf_delete("EffectsV3Udf")
        self.wait_for_replica_offset()
        self.env.assertEqual(self.replica_udf_names(),
                             self.udf_names(self.master))
        self.env.assertTrue("EffectsV3Udf" not in self.replica_udf_names())
        self._nothing_was_refused(full_before, diverged_before)


#-----------------------------------------------------------------------------
# 5. indexes and constraints together
#-----------------------------------------------------------------------------

class testEffectsV3_05_IndexAndConstraint(_EffectsV3Base):
    """An index, a unique constraint that depends on it, and the drops — with
    db.indexes() and db.constraints() compared on both sides at every step.

    Only range indexes here: db.idx.fulltext.createNodeIndex is a no-op stub on
    this branch and would pin nothing.
    """

    GRAPH_ID = "effects_v3_index_constraint"

    def _both_agree_on_schema(self):
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)
        m_idx = list_indicies(self.master_graph).result_set
        r_idx = list_indicies(self.replica_graph).result_set
        self.env.assertEqual(r_idx, m_idx)
        m_con = list_constraints(self.master_graph)
        r_con = list_constraints(self.replica_graph)
        self.env.assertEqual(r_con, m_con)
        return m_idx, m_con

    def __init__(self):
        self._setup()

    def test01_index_then_constraint_then_drops(self):
        self.query_and_sync(
            "CREATE (:Person {email: 'a@b.c', name: 'a'}), (:Person {email: 'd@e.f', name: 'd'})")

        # 1. the index alone
        create_node_range_index(self.master_graph, 'Person', 'email', sync=True)
        self.wait_for_replica_offset()
        idx, con = self._both_agree_on_schema()
        self.env.assertEqual(len(idx), 1)
        self.env.assertEqual(len(con), 0)

        # 2. a unique constraint that depends on that index
        create_unique_node_constraint(self.master_graph, 'Person', 'email')
        self.wait_for_constraint_settled(self.master_graph, 'Person')
        self.wait_for_constraint_settled(self.replica_graph, 'Person')
        self.wait_for_replica_offset()
        idx, con = self._both_agree_on_schema()
        self.env.assertEqual(len(idx), 1)
        self.env.assertEqual(len(con), 1)
        self.env.assertEqual(con[0].status, 'OPERATIONAL')

        # the index cannot be dropped while it supports the constraint, and
        # that refusal must leave both sides exactly as they were
        try:
            drop_node_range_index(self.master_graph, 'Person', 'email')
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Index supports constraint", str(e))
        self.wait_for_replica_offset()
        idx, con = self._both_agree_on_schema()
        self.env.assertEqual(len(idx), 1)
        self.env.assertEqual(len(con), 1)

        # 3. drop the constraint, then the index
        drop_unique_node_constraint(self.master_graph, 'Person', 'email')
        self.wait_for_replica_offset()
        idx, con = self._both_agree_on_schema()
        self.env.assertEqual(len(idx), 1)
        self.env.assertEqual(len(con), 0)

        drop_node_range_index(self.master_graph, 'Person', 'email')
        self.wait_for_replica_offset()
        idx, con = self._both_agree_on_schema()
        self.env.assertEqual(len(idx), 0)
        self.env.assertEqual(len(con), 0)

        self.assert_graph_eq()

    def test02_index_arrives_usable_not_merely_listed(self):
        # An index the replica lists but never populated would satisfy
        # db.indexes() and still answer queries wrongly. Compare the plan and
        # the answer, the way test_replication does.
        self.query_and_sync(
            "UNWIND range(1, 2000) AS i CREATE (:Doc {n: i, tag: 't' + (i % 7)})")
        create_node_range_index(self.master_graph, 'Doc', 'n', sync=True)
        self.wait_for_replica_offset()
        self._both_agree_on_schema()

        q = "MATCH (d:Doc {n: 1234}) RETURN d.tag"
        master_plan = str(self.master_graph.explain(q))
        self.env.assertContains("Index Scan", master_plan)
        # Every FalkorDB command carries Redis's `write` flag, GRAPH.EXPLAIN
        # included, so a read-only replica refuses it with READONLY. Lift the
        # flag only for the duration of the comparison — leaving the replica
        # writable would let a stray write hide a divergence.
        self.replica.config_set("slave-read-only", "no")
        try:
            replica_plan = str(self.replica_graph.explain(q))
        finally:
            self.replica.config_set("slave-read-only", "yes")
        self.env.assertEqual(replica_plan, master_plan)
        self.assert_agree(q, [['t' + str(1234 % 7)]])

        # a range scan over the whole index, not just one key
        self.assert_agree(
            "MATCH (d:Doc) WHERE d.n > 1990 RETURN count(d), sum(d.n)",
            [[10, sum(range(1991, 2001))]])

        # and it keeps agreeing after the indexed values are mutated, including
        # a removal
        self.query_and_sync("MATCH (d:Doc) WHERE d.n <= 10 SET d.n = d.n + 100000")
        self.assert_agree("MATCH (d:Doc) WHERE d.n > 100000 RETURN count(d)", [[10]])
        self.query_and_sync("MATCH (d:Doc) WHERE d.n > 100000 SET d.n = NULL")
        self.assert_agree("MATCH (d:Doc) WHERE d.n > 100000 RETURN count(d)", [[0]])
        self.assert_agree("MATCH (d:Doc) RETURN count(d), count(d.n)", [[2000, 1990]])

        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# 6. compression must be transparent
#-----------------------------------------------------------------------------

class testEffectsV3_05b_IndexDDLMechanism(_EffectsV3Base):
    """*How* index DDL reaches the replica, not just whether the state matches.

    The resulting index looks the same however it arrived, so a state-only
    assertion cannot tell an effect from a replayed query. These read the feed.
    v2 could not encode `OPTIONS {...}` at all and had to replicate the
    statement verbatim; v3 puts the evaluated map on the wire.
    """

    GRAPH_ID = "effects_v3_index_ddl"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT', 'GRAPH.QUERY', 'graph.query')

    @staticmethod
    def _verbatim(window):
        return sum(1 for c in window if 'GRAPH.QUERY' in c.upper())

    def test01_create_and_drop_replicate_as_effects(self):
        self.set_effects_config()
        self.monitor_mark()
        self.master_graph.query("CREATE INDEX FOR (n:P) ON (n.name)")
        self.wait_for_replica_offset()
        window = self.monitor_mark()
        self.env.assertEqual(self.count_in(window, 'GRAPH.EFFECT'), 1)
        self.env.assertEqual(self._verbatim(window), 0)

        self.monitor_mark()
        self.master_graph.query("DROP INDEX FOR (n:P) ON (n.name)")
        self.wait_for_replica_offset()
        window = self.monitor_mark()
        self.env.assertEqual(self.count_in(window, 'GRAPH.EFFECT'), 1)
        self.env.assertEqual(self._verbatim(window), 0)
        self.env.assertEqual(list_indicies(self.replica_graph).result_set,
                             list_indicies(self.master_graph).result_set)

    def test02_options_ride_the_effect_and_survive(self):
        self.set_effects_config()
        self.monitor_mark()
        self.master_graph.query(
            "CREATE FULLTEXT INDEX FOR (n:D) ON (n.body) "
            "OPTIONS {language: 'german', stopwords: ['der', 'die']}")
        self.wait_for_replica_offset()
        window = self.monitor_mark()
        # the case v2 had to send verbatim
        self.env.assertEqual(self.count_in(window, 'GRAPH.EFFECT'), 1)
        self.env.assertEqual(self._verbatim(window), 0)

        # and the map itself round-tripped, rather than the replica falling back
        # to a default-configured fulltext index
        q = ("CALL db.indexes() YIELD label, language, stopwords "
             "WHERE label = 'D' RETURN language, stopwords")
        self.assert_agree(q, [['german', ['der', 'die']]])

class testEffectsV3_06_Compression(_EffectsV3Base):
    """EFFECTS_COMPRESSION only changes the framing, never the outcome.

    The same workload is applied twice, into two graph keys, once with
    compression off and once on, and the two replicas' views are compared
    against each other as well as against their primaries.
    """

    GRAPH_ID = "effects_v3_compression"

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

    def test01_compressed_and_uncompressed_reach_the_same_state(self):
        plain, plain_payloads = self._run_workload("effects_v3_comp_off", 0)
        zstd,  zstd_payloads  = self._run_workload("effects_v3_comp_on", 1024)

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
        self.master_graph  = Graph(self.master,  "effects_v3_comp_on")
        self.replica_graph = Graph(self.replica, "effects_v3_comp_on")
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
# 7. a promoted replica finishes what it inherited
#-----------------------------------------------------------------------------

class testEffectsV3_06b_ConstraintSettlingRaces(_EffectsV3Base):
    """What happens to a constraint left UNDER CONSTRUCTION when the settle is
    interrupted.

    Both tests need the gap between the two announcements to be wide enough to
    act inside, and use the same lever `testEffectsV3_07_PromotedReplica`
    documents: a UNIQUE constraint over three indexed properties of a million
    nodes sits there for roughly 400ms on a release build. MANDATORY will not
    do — its validation is a bare scan that settles in under 4ms even at that
    size.

    A constraint stranded UNDER CONSTRUCTION is unrecoverable, which is what
    makes these worth the setup cost: it never enforces, so the master accepts
    writes it should reject, and nothing can re-drive it because
    `GRAPH.CONSTRAINT CREATE` answers `Constraint already exists` from then on.
    """

    GRAPH_ID = "effects_v3_settling_races"

    N = 1_000_000
    PROPS = ('a', 'b', 'c')

    def __init__(self):
        self._setup()

    def _build(self):
        self.set_effects_config()
        props = ", ".join(f"{p}: x" for p in self.PROPS)
        self.query_and_sync(
            f"UNWIND range(1, {self.N}) AS x CREATE (:Huge {{{props}}})")
        for p in self.PROPS:
            create_node_range_index(self.master_graph, 'Huge', p)
        wait_for_indices_to_sync(self.master_graph)
        self.wait_for_replica_offset()

    def _create_pending(self):
        res = self.master_graph.execute_command(
            "GRAPH.CONSTRAINT", "CREATE", self.GRAPH_ID, "UNIQUE", "NODE",
            "Huge", "PROPERTIES", len(self.PROPS), *self.PROPS)
        self.env.assertEqual(res, 'PENDING')

    def test01_a_paused_escalation_is_retried_not_abandoned(self):
        # The settling thread has to become a writer to publish the outcome, and
        # `CLIENT PAUSE ... WRITE` refuses that escalation. It used to exit on the
        # refusal, stranding the constraint for the life of the process.
        self._build()
        self._create_pending()

        # Inside the validation window, so the escalation at the end of it is the
        # call that gets refused.
        self.master.execute_command("CLIENT", "PAUSE", 3000, "WRITE")
        self.env.assertEqual(
            self.constraint_rows(self.master_graph, 'Huge'),
            [['UNIQUE', 'Huge', list(self.PROPS), 'NODE', 'UNDER CONSTRUCTION']],
            message="the pause must land while the constraint is still building")

        # It waits the window out and settles. The timeout is well past the 3s
        # pause, so a failure here means it gave up rather than that it was slow.
        settled = self.wait_for_constraint_settled(self.master_graph, 'Huge')
        self.env.assertEqual(
            settled, [['UNIQUE', 'Huge', list(self.PROPS), 'NODE', 'OPERATIONAL']])

        # And the replica hears the settled status, so the retry re-announces
        # rather than only fixing the master.
        self.wait_for_replica_offset()
        self.env.assertEqual(
            self.wait_for_constraint_settled(self.replica_graph, 'Huge'), settled)

        # It really is enforcing now, not merely labelled OPERATIONAL.
        rejected = None
        try:
            self.master_graph.query("CREATE (:Huge {a: 1, b: 1, c: 1})")
        except Exception as e:
            rejected = str(e).lower()
        self.env.assertTrue(
            rejected is not None and "unique constraint violation" in rejected,
            message=f"a duplicate must be rejected, got {rejected!r}")

    def test02_a_drop_during_validation_is_not_resurrected(self):
        # `drop_constraint` has no UNDER CONSTRUCTION guard, and
        # `apply_constraint_validation_results` matches on the constraint id, so a
        # drop mid-validation makes the apply a silent no-op. The settling thread
        # then read "no constraint found" as OPERATIONAL and announced a CREATE,
        # leaving the replica enforcing a constraint the master did not have.
        self.master_graph.execute_command("GRAPH.DELETE", self.GRAPH_ID)
        self.wait_for_replica_offset()
        self._build()
        self._create_pending()

        # Inside the window again.
        self.master_graph.execute_command(
            "GRAPH.CONSTRAINT", "DROP", self.GRAPH_ID, "UNIQUE", "NODE",
            "Huge", "PROPERTIES", len(self.PROPS), *self.PROPS)

        # Give the settling thread time to finish and announce if it is going to.
        # There is no state transition to poll for here — the assertion is that
        # nothing arrives — so this waits on the replication stream instead.
        for _ in range(40):
            self.wait_for_replica_offset()
            time.sleep(0.05)

        # Gone on both sides. Before the guard the replica held a UNIQUE
        # constraint at OPERATIONAL while the master held none.
        self.assert_agree(
            "CALL db.constraints() YIELD label RETURN count(1)", [[0]])

        # And the replica is not enforcing a constraint that no longer exists.
        # Checked through the master, because a replica refuses writes.
        self.master_graph.query("CREATE (:Huge {a: 1, b: 1, c: 1})")
        self.wait_for_replica_offset()
        self.assert_agree("MATCH (n:Huge {a: 1, b: 1}) RETURN count(n)", [[2]])


    def test03_deleting_the_graph_mid_validation_stops_the_settler(self):
        # `GraphUnregistered` is one of the two conditions that end the retry
        # loop rather than being waited out — waiting cannot bring a deleted
        # graph back, and the loop is otherwise unbounded, so getting this wrong
        # leaves a thread retrying against a graph nobody holds until the process
        # exits. The settler must also not resurrect the key by committing into
        # it after the delete.
        self.master_graph.execute_command("GRAPH.DELETE", self.GRAPH_ID)
        self.wait_for_replica_offset()
        self._build()
        self._create_pending()

        # Inside the validation window.
        self.master_graph.execute_command("GRAPH.DELETE", self.GRAPH_ID)

        # Long enough for the settler to have finished and, if it were going to,
        # committed or announced.
        for _ in range(40):
            self.wait_for_replica_offset()
            time.sleep(0.05)

        # Gone on both sides and not recreated behind our backs.
        self.env.assertEqual(self.master.exists(self.GRAPH_ID), 0)
        self.env.assertEqual(self.replica.exists(self.GRAPH_ID), 0)

        # Both servers are still healthy — a panic in that thread would abort the
        # process, and the panic hook in module_init exits.
        self.env.assertEqual(self.master.ping(), True)
        self.env.assertEqual(self.replica.ping(), True)

        # And the pair still replicates, so nothing was left wedged.
        self.master_graph.query("CREATE (:After {v: 1})")
        self.wait_for_replica_offset()
        self.assert_agree("MATCH (n:After) RETURN count(n)", [[1]])


class testEffectsV3_06c_DivergenceForcesResync(_EffectsV3Base):
    """A replica that cannot apply an effect must repair itself, not carry on.

    v3 detects far more divergence than v2 did — thirteen distinct checks
    against almost none — and detection without repair is worse than useless:
    Redis does not break a replication link over a module command's error
    reply, so the offset advances and the replica keeps serving data it has
    already proved wrong.

    The fix is C's: discard the cached replication state with `REPLICAOF NO
    ONE` so the reconnect cannot be satisfied by a partial resync, then
    reattach. `sync_full` on the master is what distinguishes that from a mere
    reconnect.
    """

    GRAPH_ID = "effects_v3_divergence"

    def __init__(self):
        self._setup()

    def test01_a_refused_effect_forces_a_full_resync(self):
        self.set_effects_config()
        self.query_and_sync("CREATE (:P {v: 1}), (:P {v: 2}), (:P {v: 3})")

        target = self.master_graph.ro_query(
            "MATCH (n:P) RETURN id(n) ORDER BY id(n) LIMIT 1").result_set[0][0]

        # Diverge the replica behind the master's back — a stand-in for the
        # bug or the earlier undetected inconsistency this guard exists for.
        self.replica.config_set("slave-read-only", "no")
        Graph(self.replica, self.GRAPH_ID).query(
            f"MATCH (n) WHERE id(n) = {target} DELETE n")
        self.replica.config_set("slave-read-only", "yes")
        self.env.assertEqual(
            self.replica_graph.ro_query("MATCH (n:P) RETURN count(n)").result_set,
            [[2]],
            message="the replica must actually be diverged for this to prove anything")

        full_before = self.master.info()["sync_full"]

        # Deleting the same node on the master ships a DELETE_NODE naming an id
        # the replica no longer holds. v3 refuses it — "already in the recycle
        # bin" — which is exactly the detection that used to end in a shrug.
        self.master_graph.query(f"MATCH (n) WHERE id(n) = {target} DELETE n")

        deadline = time.time() + 60
        while time.time() < deadline:
            if self.master.info()["sync_full"] > full_before:
                break
            time.sleep(0.1)
        else:
            raise AssertionError(
                f"no full resync within 60s; sync_full still {full_before}")

        # And the resync actually repaired it, rather than merely happening.
        self.wait_for_replica_link()
        deadline = time.time() + 60
        while time.time() < deadline:
            m = self.master_graph.ro_query("MATCH (n:P) RETURN count(n)").result_set
            r = self.replica_graph.ro_query("MATCH (n:P) RETURN count(n)").result_set
            if m == r:
                break
            time.sleep(0.1)
        self.assert_agree("MATCH (n:P) RETURN count(n)", [[2]])
        self.assert_graph_eq()

    def test02_a_client_sent_effect_does_not_force_a_resync(self):
        # The guard keys off the command being replayed, not off the failure.
        # `GRAPH.EFFECT` is a payload any client can send, so treating a
        # malformed one as divergence would hand anyone a way to resync — or
        # shut down — a replica at will. C guards this path unconditionally;
        # this is a deliberate difference.
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]

        # A v3 buffer whose opcode is nonsense: refused by the decoder.
        try:
            self.master.execute_command(
                "GRAPH.EFFECT", self.GRAPH_ID, b"\x03\x00\xff\xff\xff\xff")
            self.env.assertTrue(False, message="a malformed effect must be refused")
        except ResponseError:
            pass

        # Nothing was scheduled: no resync, and the server is still here.
        time.sleep(1)
        self.env.assertEqual(self.master.info()["sync_full"], full_before)
        self.env.assertEqual(self.master.ping(), True)
        self.env.assertEqual(self.replica.ping(), True)


class testEffectsV3_06d_FirstReplicaAttach(_EffectsV3Base):
    """A write that lands while a replica is doing its first full sync must
    still reach that replica.

    `REPLICATION_CONSUMERS` (`src/graph_core.rs`) is a process-global sticky
    flag, and `execute_query_write` reads it to decide whether to build an
    effects buffer at all. Nothing latches it until Redis fires
    `ReplicaChange`, and Redis fires that from `replicaPutOnline` — *after* the
    snapshot has been delivered. So every write between the sync's fork and the
    replica coming online runs with `build_effects` false and produces no
    buffer.

    That used to be harmless: `replicate_effects` fell back to
    `ctx.replicate("GRAPH.QUERY", query)` when there was no buffer, so the
    write still propagated, as a replay. The fallback went with v2. Now nothing
    is propagated, and nothing notices: no effect ever arrives to be refused,
    so the divergence guard never runs, the link reports healthy, and the
    replica is permanently short one write.

    The window is the fork's snapshot, so `rdb-key-save-delay` widens it and
    `repl-diskless-sync-delay 0` stops the master idling *before* the fork —
    a write in that earlier window would be captured by the snapshot and prove
    nothing.

    Needs a master process that has never had a replica, which is why it
    restarts one: the flag is never cleared. That is also why it runs late and
    puts the topology back in a `finally`.
    """

    GRAPH_ID = "effects_v3_first_attach"

    # Per key, in microseconds, applied inside the forked child. One graph key,
    # so this is the whole width of the window.
    SAVE_DELAY_US = 3_000_000

    def __init__(self):
        self._setup()

    def test01_a_write_during_the_first_full_sync_is_not_lost(self):
        info = self.replica.info("replication")
        host, port = info["master_host"], info["master_port"]
        try:
            self._write_during_first_sync(host, port)
        finally:
            # Hand the topology back for the classes that follow, whatever
            # happened above.
            try:
                self.master.config_set("rdb-key-save-delay", 0)
            except Exception:
                pass
            self.replica.execute_command("REPLICAOF", host, port)
            self.wait_for_replica_link()

    def _fresh_master(self, host, port):
        """Restart the master so `REPLICATION_CONSUMERS` starts false again.

        The replica is detached first: left attached it would reconnect the
        instant the master came back and latch the flag before the test could
        do anything.
        """
        self.replica.execute_command("REPLICAOF", "NO", "ONE")
        self.env.envRunner.stopEnv(masters=True, slaves=False)
        self.env.envRunner.startEnv(masters=True, slaves=False)
        # The pooled connections died with the old process, and these clients
        # are built with no retry policy on purpose (see common.NO_RETRY), so a
        # stale one raises rather than reconnecting. Take fresh ones.
        self.master = self.env.getConnection()
        self.master_graph  = Graph(self.master,  self.GRAPH_ID)
        self.replica_graph = Graph(self.replica, self.GRAPH_ID)
        self.master.flushall()
        # A restarted process is back at the compression default, so there is
        # nothing to set — and `self.db` still wraps the dead pool.
        self.env.assertEqual(
            self.master.info("replication")["role"], "master",
            message="the restarted master must not have come back as a replica")
        self.env.assertEqual(
            self.master.info()["sync_full"], 0,
            message="a master that has already served a full sync has the flag latched")

    def _write_during_first_sync(self, host, port):
        self._fresh_master(host, port)

        # The seed registers the label and the attribute before the fork, on
        # purpose: the write inside the window then introduces no new schema and
        # no new attribute id, so no *later* effect can trip a divergence check
        # on its behalf. That is what makes the loss permanent rather than
        # something the guard eventually repairs with a resync.
        #
        # It is also the control. One key, so the child's per-key delay is the
        # whole window; if the seed is missing too then the sync itself failed
        # and nothing here says anything about propagation.
        self.master_graph.query("CREATE (:Seed {v: 1})")
        self.master.config_set("repl-diskless-sync-delay", 0)
        self.master.config_set("rdb-key-save-delay", self.SAVE_DELAY_US)

        self.replica.execute_command("REPLICAOF", host, port)

        # Wait for the child to exist. `rdb_bgsave_in_progress` rather than the
        # replica's `master_sync_in_progress`, which is already 1 while the
        # master is still deciding to fork — a write in *that* window would be
        # captured by the snapshot and prove nothing.
        deadline = time.time() + 60
        while time.time() < deadline:
            if int(self.master.info("persistence")["rdb_bgsave_in_progress"]) == 1:
                break
            time.sleep(0.005)
        else:
            raise AssertionError("the master never forked for the replica's full sync")

        self.master_graph.query("CREATE (:Seed {v: 2})")

        # Still forking, so the write really did land after the snapshot. An
        # assertion rather than a branch: had the child already finished, the
        # snapshot would carry the write and the rest would pass for the wrong
        # reason.
        self.env.assertEqual(
            int(self.master.info("persistence")["rdb_bgsave_in_progress"]), 1,
            message="the write must land while the snapshot is still being taken")
        self.master.config_set("rdb-key-save-delay", 0)

        self.wait_for_replica_link()
        self.wait_for_replica_offset()

        self.env.assertEqual(
            self.master.info()["sync_full"], 1,
            message="this must be the first full sync this master process served")

        self.assert_agree("MATCH (n:Seed) RETURN count(n), sum(n.v)", [[2, 3]])
        self.assert_graph_eq()

        # And it stays wrong. Three more writes of the same shape, each of which
        # the replica applies without complaint, because none of them names
        # anything it does not already hold — so the offsets keep matching, the
        # link keeps reporting healthy, and nothing ever forces the resync that
        # would repair it.
        for v in (3, 4, 5):
            self.query_and_sync(f"CREATE (:Seed {{v: {v}}})")
        self.assert_agree("MATCH (n:Seed) RETURN count(n), sum(n.v)", [[5, 15]])
        self.env.assertEqual(
            self.master.info()["sync_full"], 1,
            message="nothing detected the drift, so no second full sync happened")


class testEffectsV3_06e_AofReplay(_EffectsV3Base):
    """A graph rebuilds from an AOF whose only graph records are effects.

    Effects go to the AOF as well as to the replication stream —
    `execute_query_write` builds a buffer when `ContextFlags::AOF` is set, quite
    apart from whether a replica is attached — and `GRAPH.EFFECT` re-propagates
    verbatim, so it lands in the AOF of every node that applies one.

    That path is newly load-bearing. Query replay used to be an alternative
    recording, chosen per query by `EFFECTS_THRESHOLD`; now a graph's entire
    history on disk is the RDB base plus a stream of `GRAPH.EFFECT`, and
    replaying it is the only way an AOF-configured instance comes back. It is
    also the one place a failure to apply is fatal rather than recoverable:
    `divergence_guard::on_failure` calls `exit(1)` under `LOADING`, because a
    resync cannot repair state that is already baked into local disk.

    AOF is enabled here on an empty dataset on purpose, so the base file holds
    nothing and every byte of the graph has to arrive as an effect.

    Left with `appendonly no`, by `test09`: it is server state, and the classes
    that follow share this server.
    """

    GRAPH_ID = "effects_v3_aof"

    # Mixed on purpose: creates, a multi-commit query, an update, a removal, a
    # delete that frees ids, an edge, a self-loop, and schema DDL — so the AOF
    # holds every record type the emitter writes rather than only CREATE_NODE.
    WORKLOAD = [
        """UNWIND range(1, 3000) AS i
           CREATE (:A {id: i, s: 'v' + i, l: [i, i + 1], f: i * 1.5, b: i % 2 = 0})""",
        "MATCH (a:A), (b:A) WHERE b.id = a.id + 1 AND a.id <= 500 CREATE (a)-[:R {w: a.id}]->(b)",
        "MATCH (a:A {id: 1}) CREATE (a)-[:SELF]->(a)",
        "MATCH (n:A) WHERE n.id % 3 = 0 SET n.s = NULL, n:Extra, n.f = n.f + 0.5",
        "MATCH (n:A) WHERE n.id % 7 = 0 DETACH DELETE n",
        # removals in both of the shapes that reach the wire as a null value
        # row: a whole-shape node SET, and an edge property
        "MATCH (n:A) WHERE n.id % 13 = 0 SET n = {id: n.id, only: true}",
        "MATCH ()-[e:R]->() SET e.tag = 'x'",
        "MATCH ()-[e:R]->() WHERE e.w % 4 = 0 SET e.tag = NULL",
        "UNWIND range(1, 200) AS i MERGE (:Merged {i: i})",
        "CREATE (:Geo {p: point({latitude: 1.5, longitude: 2.5}), d: date('2024-01-15')})",
        "CREATE INDEX FOR (n:A) ON (n.id)",
    ]

    PROBES = [
        "MATCH (n) RETURN count(n)",
        "MATCH (n:A) RETURN count(n), count(n.s), sum(n.id), sum(n.f), count(n.b)",
        "MATCH (n:A) RETURN sum(size(n.l))",
        "MATCH (n:Extra) RETURN count(n)",
        "MATCH (n:Merged) RETURN count(n), sum(n.i)",
        "MATCH ()-[e:R]->() RETURN count(e), sum(e.w), count(e.tag)",
        "MATCH (n:A) RETURN count(n.only), count(n.l), count(n.f)",
        "MATCH (a)-[e:SELF]->(b) RETURN count(e), ID(a) = ID(b)",
        "MATCH (n:Geo) RETURN n.p, toString(n.d)",
        "MATCH (n:A) WHERE n.id > 2990 RETURN count(n), collect(n.id)",
        "CALL db.labels() YIELD label RETURN label ORDER BY label",
        "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey",
        "CALL db.indexes() YIELD label, properties RETURN label, properties ORDER BY label",
        "CALL db.constraints() YIELD type, label, properties, status RETURN type, label, properties, status",
    ]

    def __init__(self):
        self._setup(enable_debug=True)

    #-------------------------------------------------------------------------
    # helpers
    #-------------------------------------------------------------------------

    def enable_aof(self):
        """Turn on `appendonly` and wait for the initial rewrite to land.

        Enabling it forks to write the base file; a `DEBUG LOADAOF` issued
        before that finished would load a manifest still being assembled.
        """
        self.master.config_set("appendonly", "yes")
        self.wait_for_rewrite()
        info = self.master.info("persistence")
        self.env.assertEqual(int(info["aof_enabled"]), 1)

    def wait_for_rewrite(self, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            info = self.master.info("persistence")
            if (int(info["aof_rewrite_in_progress"]) == 0
                    and int(info.get("aof_rewrite_scheduled", 0)) == 0):
                self.env.assertEqual(
                    info["aof_last_bgrewrite_status"], "ok",
                    message="the AOF rewrite failed; nothing below can be trusted")
                return
            time.sleep(0.05)
        raise AssertionError("the AOF rewrite did not finish")

    def snapshot(self):
        return [self.master_graph.ro_query(p).result_set for p in self.PROBES]

    def reload_from_aof(self):
        """Throw the dataset away and rebuild it from the AOF.

        `DEBUG LOADAOF` flushes the AOF buffer, empties the keyspace and
        replays the manifest — the same code path a restart takes, without the
        restart. It does not propagate, so the replica keeps the copy it
        already had and stays a second, independent witness.
        """
        self.master.execute_command("DEBUG", "LOADAOF")

    #-------------------------------------------------------------------------
    # tests
    #-------------------------------------------------------------------------

    def test01_a_graph_rebuilds_from_an_effects_only_aof(self):
        self.set_effects_config()
        self.enable_aof()

        for q in self.WORKLOAD:
            self.query_and_sync(q)
        create_node_range_index(self.master_graph, 'Merged', 'i', sync=True)
        create_unique_node_constraint(self.master_graph, 'Merged', 'i')
        self.wait_for_constraint_settled(self.master_graph, 'Merged')
        self.wait_for_replica_offset()
        wait_for_indices_to_sync(self.master_graph)

        before = self.snapshot()
        # not vacuously true: the workload has to have produced something
        self.env.assertEqual(before[0], [[3000 - 428 + 200 + 1]])

        self.reload_from_aof()

        # Still serving, which is the first thing to check: a failure to apply
        # while `LOADING` is `exit(1)` by design, so a broken replay shows up
        # as a dead server rather than as a wrong answer.
        self.env.assertEqual(self.master.ping(), True)
        wait_for_indices_to_sync(self.master_graph)
        after = self.snapshot()
        for probe, b, a in zip(self.PROBES, before, after):
            self.env.assertEqual(a, b, message=f"changed across the reload: {probe}")

        # And the replica — which never reloaded — agrees with what came back
        # off disk, so the AOF and the replication stream carried the same
        # thing.
        self.assert_graph_eq()

    def test02_a_rewrite_compacts_and_the_effects_after_it_still_replay(self):
        # After `BGREWRITEAOF` the base file is an RDB of the current state and
        # the incremental file starts empty, so this is the *other* AOF shape:
        # a base that carries the graph plus effects layered on top of it. Both
        # have to replay, in that order.
        self.set_effects_config()
        self.master.execute_command("BGREWRITEAOF")
        self.wait_for_rewrite()

        self.query_and_sync(
            "UNWIND range(1, 500) AS i CREATE (:Post {i: i, s: 'p' + i})")
        self.query_and_sync("MATCH (n:Post) WHERE n.i % 5 = 0 SET n.s = NULL")
        self.query_and_sync("MATCH (n:Post) WHERE n.i % 11 = 0 DELETE n")
        self.query_and_sync("MATCH (a:Post {i: 1}), (b:Post {i: 2}) CREATE (a)-[:P]->(b)")

        extra = ["MATCH (n:Post) RETURN count(n), count(n.s), sum(n.i)",
                 "MATCH ()-[e:P]->() RETURN count(e)"]
        before = self.snapshot() + [
            self.master_graph.ro_query(p).result_set for p in extra]

        self.reload_from_aof()
        self.env.assertEqual(self.master.ping(), True)
        wait_for_indices_to_sync(self.master_graph)
        after = self.snapshot() + [
            self.master_graph.ro_query(p).result_set for p in extra]
        for probe, b, a in zip(self.PROBES + extra, before, after):
            self.env.assertEqual(a, b, message=f"changed across the reload: {probe}")
        self.assert_graph_eq()

    def test03_a_graph_deleted_and_rebuilt_replays_in_order(self):
        # `GRAPH.DELETE` replicates verbatim while the writes around it are
        # effects, so the AOF holds a mix — and a replay that applied the
        # effects against the pre-delete graph would rebuild the wrong thing.
        self.set_effects_config()
        key = "effects_v3_aof_recreate"
        g = Graph(self.master, key)
        g.query("CREATE (:Old {v: 1}), (:Old {v: 2})")
        self.master.execute_command("GRAPH.DELETE", key)
        g.query("UNWIND range(1, 50) AS i CREATE (:New {i: i})")
        self.wait_for_replica_offset()

        before = g.ro_query(
            "MATCH (n) RETURN count(n), labels(n)[0], sum(n.i)").result_set
        self.env.assertEqual(before, [[50, 'New', 1275]])

        self.reload_from_aof()
        self.env.assertEqual(self.master.ping(), True)
        self.env.assertEqual(
            Graph(self.master, key).ro_query(
                "MATCH (n) RETURN count(n), labels(n)[0], sum(n.i)").result_set,
            before)
        # the deleted label is gone rather than resurrected by the replay
        self.env.assertEqual(
            Graph(self.master, key).ro_query(
                "CALL db.labels() YIELD label RETURN label").result_set,
            [['New']])

    def test08_a_poisoned_aof_takes_the_server_down_rather_than_loading_it(self):
        """An effect that will not apply while `LOADING` is `exit(1)`.

        The other arm of `divergence_guard::on_failure` is a forced resync, and
        it cannot help here: the divergence is already baked into this
        instance's own disk, and continuing would replay the rest of the file
        against a dataset that is known wrong. So the guard refuses to come up.
        `testEffectsV3_06c_DivergenceForcesResync` covers the resync arm; this
        is the only way to reach the other one, because `LOADING` is not a flag
        a client can set.

        The effect is injected into the AOF by hand — appended as RESP after a
        clean shutdown, so nothing races the server's own writer. It names a
        label id the replaying graph would not assign, which is `IdMismatch`:
        well-formed bytes describing a graph this instance does not have, which
        is the failure v3 exists to make loud.
        """
        # A dead master must not drag a resyncing replica with it.
        info = self.replica.info("replication")
        host, port = info["master_host"], info["master_port"]
        self.replica.execute_command("REPLICAOF", "NO", "ONE")

        runner = self.env.envRunner
        try:
            self._poisoned_start_must_exit(runner)
        finally:
            # Bring the master back however the above ended, then the link.
            if runner.masterProcess is None:
                runner.startEnv(masters=True, slaves=False)
            self.master = self.env.getConnection()
            self.master.flushall()
            self.master_graph  = Graph(self.master,  self.GRAPH_ID)
            self.replica_graph = Graph(self.replica, self.GRAPH_ID)
            self.replica.execute_command("REPLICAOF", host, port)
            self.wait_for_replica_link()

        # The pair still works, so the recovery was real and not a husk.
        self.query_and_sync("CREATE (:AfterPoison {v: 1})")
        self.assert_agree("MATCH (n:AfterPoison) RETURN count(n)", [[1]])

    def _poisoned_start_must_exit(self, runner):
        runner.stopEnv(masters=True, slaves=False)

        incr = self._aof_incr_path(runner)
        clean_len = os.path.getsize(incr)
        with open(incr, "ab") as f:
            f.write(self._resp(b"GRAPH.EFFECT", self.GRAPH_ID.encode(),
                               self._bad_schema_effect()))

        log_before = self._log_size(runner)
        # Spawned here rather than through `startEnv`, which is built to wait
        # for a server that comes up: this one must not, and the exit code is
        # the assertion.
        args = list(runner.masterCmdArgs) + ["--appendonly", "yes"]
        proc = subprocess.Popen(args, cwd=runner.dbDirPath,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            rc = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise AssertionError(
                "the server came up on a poisoned AOF instead of refusing to")
        finally:
            # Whatever happened, the file goes back to what the server wrote,
            # so the recovery start in the caller's `finally` has a loadable
            # AOF.
            with open(incr, "r+b") as f:
                f.truncate(clean_len)

        self.env.assertEqual(rc, 1, message=f"expected exit(1), got {rc}")

        # And it exited for *this* reason. An exit code alone would also be
        # satisfied by a server that failed to bind its port.
        tail = self._log_tail(runner, log_before)
        self.env.assertContains("Diverged applying GRAPH.EFFECT", tail)
        self.env.assertContains("while loading from disk", tail)
        self.env.assertContains("shutting down", tail)

    #-------------------------------------------------------------------------
    # poisoned-AOF plumbing
    #-------------------------------------------------------------------------

    @staticmethod
    def _resp(*args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        return out

    @staticmethod
    def _bad_schema_effect():
        """A v3 buffer holding one `ADD_SCHEMA` record for node label id 9999.

        `u8 3 · u8 0 · u32 9 · u32 0 · i32 id · u64 len+1 · name · NUL`. The id
        is far past anything the replay could have reached, so the apply path
        refuses it with `IdMismatch` — the bytes are well formed, which is the
        point: a decode error would prove nothing about divergence.
        """
        name = b"PoisonedLabel"
        return (b"\x03\x00"
                + b"\x09\x00\x00\x00"
                + b"\x00\x00\x00\x00"
                + (9999).to_bytes(4, "little", signed=True)
                + (len(name) + 1).to_bytes(8, "little") + name + b"\x00")

    @staticmethod
    def _aof_incr_path(runner):
        d = os.path.join(runner.dbDirPath, "appendonlydir")
        incr = sorted(f for f in os.listdir(d) if f.endswith(".incr.aof"))
        if not incr:
            raise AssertionError(f"no incremental AOF under {d}: {os.listdir(d)}")
        return os.path.join(d, incr[-1])

    @staticmethod
    def _master_log(runner):
        name = runner._getFileName('master', '.log')
        for base in (runner.dbDirPath, getattr(runner, 'outputFilesFormat', None)):
            if base and os.path.exists(os.path.join(str(base), name)):
                return os.path.join(str(base), name)
        return os.path.join(runner.dbDirPath, name)

    def _log_size(self, runner):
        path = self._master_log(runner)
        return os.path.getsize(path) if os.path.exists(path) else 0

    def _log_tail(self, runner, offset):
        path = self._master_log(runner)
        if not os.path.exists(path):
            raise AssertionError(f"no master log at {path}")
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode(errors="replace")

    def test09_leave_appendonly_off(self):
        # Server state, and the classes after this one share the server.
        self.master.config_set("appendonly", "no")
        self.env.assertEqual(
            int(self.master.info("persistence")["aof_enabled"]), 0)


#-----------------------------------------------------------------------------
# 8. the same write produces the same bytes
#-----------------------------------------------------------------------------

class testEffectsV3_08_ByteDeterminism(_EffectsV3Base):
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

    GRAPH_ID = "effects_v3_determinism"

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


class testEffectsV3_07_PromotedReplica(_EffectsV3Base):
    """A replica promoted while holding a constraint UNDER CONSTRUCTION must
    settle it itself.

    A replica never validates — it installs the status the primary announced.
    That is right while it is a replica and wrong the moment it is promoted: a
    constraint the old primary was still building becomes this node's to
    finish, and until it does the constraint is neither enforcing nor failed,
    so writes that should be rejected are not.

    The window this test needs is the gap between the two announcements. A
    UNIQUE constraint over three indexed properties of a million nodes spends
    about 400ms there (measured on a release build), against roughly 1.5ms to
    confirm the replica has the first announcement and cut the link — so the
    promotion lands inside it with two orders of magnitude to spare. A slower
    build widens the gap without widening the two round trips, so the margin
    only grows. MANDATORY will not do: its validation is a bare scan and
    settles in under 4ms even at a million nodes.

    This class runs last because it takes the topology apart. It puts the link
    back before returning either way.
    """

    GRAPH_ID = "effects_v3_promotion"

    N = 1_000_000
    PROPS = ('a', 'b', 'c')

    def __init__(self):
        self._setup()
        info = self.replica.info("replication")
        self.replicaof_host = info["master_host"]
        self.replicaof_port = info["master_port"]

    def test01_promotion_settles_an_inherited_constraint(self):
        try:
            self._promote_mid_validation()
        finally:
            # Hand the topology back. A later class — or a rerun in a reused
            # env — gets a replica again, and the resync wipes whatever this
            # test left on the promoted node.
            self.replica.execute_command(
                "REPLICAOF", self.replicaof_host, self.replicaof_port)
            self.wait_for_replica_link()

    def _promote_mid_validation(self):
        self.set_effects_config()
        props = ", ".join(f"{p}: x" for p in self.PROPS)
        self.query_and_sync(
            f"UNWIND range(1, {self.N}) AS x CREATE (:Huge {{{props}}})")
        for p in self.PROPS:
            create_node_range_index(self.master_graph, 'Huge', p)
        wait_for_indices_to_sync(self.master_graph)
        self.wait_for_replica_offset()
        wait_for_indices_to_sync(self.replica_graph)
        self.assert_agree("MATCH (n:Huge) RETURN count(n)", [[self.N]])

        # Create the constraint and cut the link as soon as the replica has the
        # first announcement — before the settled one can reach it.
        #
        # One constraint, because one is all this test needs — not because two
        # is impossible. Two `GRAPH.CONSTRAINT CREATE`s issued concurrently on
        # different labels of the same graph *do* both answer PENDING and sit
        # UNDER CONSTRUCTION together (measured: two 400k-node labels, both
        # observed pending at once), so `settle_async_constraint`'s
        # `for c in announce` really can iterate more than once — reached
        # through `enforce_pending_constraints_after_promotion`, which collects
        # every pending constraint of a graph into one slice. That case is
        # still uncovered; it needs a replica holding two pending constraints
        # at the instant it is promoted.
        res = self.master_graph.execute_command(
            "GRAPH.CONSTRAINT", "CREATE", self.GRAPH_ID, "UNIQUE", "NODE",
            "Huge", "PROPERTIES", len(self.PROPS), *self.PROPS)
        self.env.assertEqual(res, 'PENDING')
        self.wait_for_replica_offset()
        inherited = self.constraint_rows(self.replica_graph, 'Huge')
        self.replica.execute_command("REPLICAOF", "NO", "ONE")

        # What it was holding at the instant it stopped being a replica. If
        # this is not UNDER CONSTRUCTION the promotion had nothing to finish
        # and the rest of the test would prove nothing, so it is an assertion
        # rather than a branch.
        self.env.assertEqual(
            inherited, [['UNIQUE', 'Huge', list(self.PROPS), 'NODE', 'UNDER CONSTRUCTION']])
        self.env.assertEqual(self.replica.info("replication")["role"], "master")

        # It settles on its own, as a node that is no longer a replica: there
        # is nothing left to learn the status from.
        settled = self.wait_for_constraint_settled(self.replica_graph, 'Huge')
        self.env.assertEqual(
            settled, [['UNIQUE', 'Huge', list(self.PROPS), 'NODE', 'OPERATIONAL']])
        self.env.assertEqual(self.replica.info("replication")["role"], "master")
        # exactly one constraint — the promotion hook updates in place
        self.env.assertEqual(
            self.replica_graph.ro_query(
                "CALL db.constraints() YIELD status RETURN count(1)"
            ).result_set, [[1]])

        # the old primary reached the same answer independently
        self.env.assertEqual(
            self.wait_for_constraint_settled(self.master_graph, 'Huge'),
            [['UNIQUE', 'Huge', list(self.PROPS), 'NODE', 'OPERATIONAL']])

        # and the promoted node enforces it, which is the point of finishing it
        promoted = Graph(self.replica, self.GRAPH_ID)
        try:
            promoted.query("CREATE (:Huge {a: 1, b: 1, c: 1})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation", str(e))
