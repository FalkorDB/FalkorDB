import time
import threading
from common import *

GRAPH_ID = "effects_v3_determinism"

# Is C's v3 encoder DETERMINISTIC? Same write, same bytes.
#
# Every other v3 test asserts the replica AGREES, which an encoder could satisfy
# while emitting a different payload every time. Four things in the format exist
# only to make the bytes a function of the ids -- record groups are sorted,
# optimize() is called on every bitmap, a bitmap is built by range rather than id
# by id, and the collapse rule is evaluated against the run rather than against
# whenever the encoder happened to look. None of them has ever been checked end
# to end on this engine.
#
# WHY THIS IS A PRECONDITION AND NOT A NICETY. The conformance corpus decodes 39
# Rust-generated fixtures and re-encodes them in C, and 38 reproduce the bytes
# exactly. That result means what it appears to mean only if C's encoder is a
# function of its input. If it is not, those 38 are 38 coincidences and every
# fixture claim rests on nothing. So this test is upstream of the corpus, not
# beside it.
#
# Ported from the intention of Rust's testEffectsV3_08_ByteDeterminism, not from
# its code. Two of its assumptions are deliberately NOT carried over:
#
#   * it requires exactly one payload per query. C emits one effect per FIELD for
#     some records where v3 is one record per statement, so a count of one is a
#     property of Rust's emitter rather than of determinism. This compares the
#     whole SEQUENCE of payloads instead, which is the actual claim: the same
#     write produces the same bytes, however many buffers it takes.
#   * it drives its own base class. This reuses the MONITOR idiom already in
#     tests/flow/test_effects.py, which has captured the whole command dict since
#     it was written and only ever asked whether an effect appeared.


class testEffectsV3Determinism():
    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        # EFFECTS_THRESHOLD 0 forces every write onto the effects path rather
        # than letting a small one replicate verbatim, which would leave nothing
        # to compare. EFFECTS_VERSION 3 because C still emits v2 by default.
        self.env, self.db = Env(env='oss', useSlaves=True,
                moduleArgs='EFFECTS_THRESHOLD 0 EFFECTS_VERSION 3')

        self.master  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()

        self.monitor  = []
        self._attached = threading.Event()

        self.master.wait(1, 0)

        self._monitor = threading.Thread(target=self._monitor_thread, daemon=True)
        self._monitor.start()
        if not self._attached.wait(timeout=30):
            raise Exception("monitor never attached to the replica")

    def __del__(self):
        try:
            self.replica.shutdown()
        except Exception:
            pass

    def _monitor_thread(self):
        # the replica's MONITOR, not the master's: the master runs GRAPH.QUERY
        # and it is the replica that receives GRAPH.EFFECT
        try:
            with self.replica.monitor() as m:
                self._attached.set()
                for cmd in m.listen():
                    if 'GRAPH.EFFECT' in cmd['command']:
                        self.monitor.append(cmd)
        except Exception:
            pass

    @staticmethod
    def _first_diff(a, b):
        """Where two payloads first differ, and a small window around it.

        assertEqual on two payloads dumps both in full -- 223KB for the shapes
        below, which is unreadable and buries the one fact that matters. The
        round-trip harness learned this already: report the OFFSET, because the
        offset is what identifies the field that was encoded differently.
        """
        n = min(len(a), len(b))
        at = next((i for i in range(n) if a[i] != b[i]), n)
        lo, hi = max(0, at - 12), at + 12
        return (f"first differ at offset {at} of {len(a)}/{len(b)}\n"
                f"      a[{lo}:{hi}] = {a[lo:hi]!r}\n"
                f"      b[{lo}:{hi}] = {b[lo:hi]!r}")

    def _payloads_for(self, key, query):
        """Every effect payload `query` produces against a fresh `key`.

        SLICED BY PREFIX, NEVER SPLIT ON WHITESPACE. An effects buffer is
        binary: 0x20 occurs in it freely and MONITOR renders it as a literal
        space, so the payload is "everything after the graph key" and not "the
        third token". A helper that tokenised would truncate every payload at
        its first 0x20 and compare prefixes -- which passes happily on bytes
        that are not deterministic at all, and is why this is spelled out here
        rather than left to the reader. C's existing monitor helpers have never
        had to get this right because they only ever test that an effect
        appeared.
        """
        mark   = len(self.monitor)
        prefix = f"GRAPH.EFFECT {key} "

        def matched():
            out = []
            for cmd in self.monitor[mark:]:
                c = cmd['command']
                i = c.find(prefix)
                if i >= 0:
                    out.append(c[i + len(prefix):])
            return out

        Graph(self.master, key).query(query)
        self.master.wait(1, 0)

        # WAIT confirms the REPLICA acked the write. It says nothing about when
        # MONITOR delivers it to this client, which is a separate asynchronous
        # stream -- so the two are not ordered with respect to each other.
        #
        # An earlier version waited for the monitor count to stop changing and
        # treated that as "everything has arrived". It is also what an empty
        # stream looks like: two consecutive samples of a count that has not
        # moved yet are indistinguishable from a count that will not move. That
        # made this helper intermittently return an empty window, which failed
        # here as "no payload reached the replica" and would elsewhere have
        # compared a PARTIAL window and passed. Same family as everything else
        # in this project: green because the mechanism never engaged.
        #
        # So: wait for at least one matching payload before deciding anything,
        # and only then settle briefly for stragglers.
        deadline = time.time() + 30
        while time.time() < deadline and not matched():
            time.sleep(0.1)

        # Then settle for stragglers -- a query can produce several payloads and
        # only the first may have landed.
        #
        # Two deliberate choices here. It watches the MATCHED count rather than
        # len(self.monitor), because global traffic from another key would keep
        # the loop alive without telling us anything about this window; and it
        # requires several consecutive quiet samples rather than one, because a
        # single quiet sample is the same race as before on a smaller scale.
        #
        # RESIDUAL, considered rather than missed: this can still close early if
        # a payload is delayed past the whole quiet period. What it cannot do any
        # more is close early SILENTLY -- an under-captured window shows up as
        # len(first) != len(second), which every test below asserts on. That is
        # the difference between a flaky failure and a partial comparison that
        # passes, and only the second kind is dangerous.
        QUIET_SAMPLES = 4
        quiet = 0
        n = len(matched())
        while time.time() < deadline and quiet < QUIET_SAMPLES:
            time.sleep(0.25)
            m = len(matched())
            quiet = quiet + 1 if m == n else 0
            n = m

        return matched()

    def test01_the_same_write_twice_produces_identical_bytes(self):
        # Two graphs, both empty and therefore identical, given the same query.
        # Ids allocate densely from zero, so the two writes describe the same
        # entities and must serialize the same way.
        #
        # The shape is deliberately wide enough to engage the parts that could
        # differ run to run: several labels so label sets have to be grouped,
        # differing property shapes so there is more than one record partition,
        # and enough rows that any grouping order is not trivially the insertion
        # order.
        q = """
            UNWIND range(0, 199) AS i
            CREATE (:A:B {v: i, s: 'x' + i}),
                   (:B {v: i}),
                   (:A {w: i * 2, t: true})
        """

        first  = self._payloads_for(f"{GRAPH_ID}_1", q)
        second = self._payloads_for(f"{GRAPH_ID}_2", q)

        # if this fires, the write produced nothing to compare and the test
        # proved nothing -- a clean result here would be indistinguishable from
        # a harness that never engaged
        self.env.assertTrue(len(first) > 0,
                message="no GRAPH.EFFECT reached the replica: EFFECTS_THRESHOLD "
                        "or EFFECTS_VERSION is not in force, so this test "
                        "compared two empty lists")

        self.env.assertEqual(len(first), len(second),
                message=f"the same write produced {len(first)} payloads then "
                        f"{len(second)}")

        for n, (a, b) in enumerate(zip(first, second)):
            self.env.assertTrue(a == b,
                    message=f"payload {n} of the same write serialized "
                            f"differently: record order, bitmap construction or "
                            f"the collapse rule is not a function of the ids. "
                            f"{self._first_diff(a, b)}")

    def test02_a_dense_delete_is_deterministic(self):
        # The collapse rule's own case. A dense ascending id list is what
        # crosses the range-versus-bitmap boundary, and the rule is normative
        # precisely because it must depend on the ids rather than on when the
        # encoder looked at them.
        setup = "UNWIND range(0, 499) AS i CREATE (:D {v: i})"
        wipe  = "MATCH (n:D) DELETE n"

        one, two = f"{GRAPH_ID}_d1", f"{GRAPH_ID}_d2"

        self._payloads_for(one, setup)
        self._payloads_for(two, setup)

        first  = self._payloads_for(one, wipe)
        second = self._payloads_for(two, wipe)

        self.env.assertTrue(len(first) > 0,
                message="the delete produced no effect payload")
        self.env.assertEqual(len(first), len(second))

        for n, (a, b) in enumerate(zip(first, second)):
            self.env.assertTrue(a == b,
                    message=f"delete payload {n} differed: the collapse rule or "
                            f"the bitmap build is not a function of the ids. "
                            f"{self._first_diff(a, b)}")

    def test03_a_supernode_fanout_is_deterministic(self):
        # Repeat's case. Every edge out of one node carries the same source id,
        # which is one segment rather than one per edge -- and a Repeat that was
        # built from a counter rather than from the ids would still replicate
        # correctly while emitting different bytes.
        one, two = f"{GRAPH_ID}_h1", f"{GRAPH_ID}_h2"

        for k in (one, two):
            self._payloads_for(k, "CREATE (:Hub {name: 'hub'})")

        fan = """
            MATCH (h:Hub {name: 'hub'})
            UNWIND range(0, 499) AS i
            CREATE (h)-[:OUT {i: i}]->(:Leaf {i: i})
        """

        first  = self._payloads_for(one, fan)
        second = self._payloads_for(two, fan)

        self.env.assertTrue(len(first) > 0,
                message="the fanout produced no effect payload")
        self.env.assertEqual(len(first), len(second))

        for n, (a, b) in enumerate(zip(first, second)):
            self.env.assertTrue(a == b,
                    message=f"fanout payload {n} differed: a Repeat segment is "
                            f"not being built from the ids. "
                            f"{self._first_diff(a, b)}")
