"""Shared fixture for the effects flow suites.

`test_effects_*.py` each drive a real primary/replica pair and assert on both
sides. This module holds what they share: the `Env` setup, the waiting helpers,
the MONITOR window machinery, and the per-version wire table.

THE DIFFERENCE FROM THE RUST SUITE THIS IS MODELLED ON. Rust emits v3 and only
v3, so its `effects_common.py` can hardcode `\\x03\\x00` as the header and carry
one opcode table. C emits **v2 by default and v3 on request**, and has to keep
both correct, so the version is a parameter here rather than a constant.

`EFFECTS_VERSION` is settable at RUNTIME and takes effect on the next query --
measured, not assumed: setting it to 3 and writing produces a payload opening
`\\x03\\x00\\x03`, setting it back to 2 produces `\\x02...`, on one server with no
restart. So a single class can assert the same behaviour at both versions with
`for_each_version`, and the suites do not need duplicating per version.

    test_effects_opcodes.py      does every opcode replicate at all?
    test_effects_values.py       does every value survive the wire?
    test_effects_shapes.py       what must one buffer survive -- merges,
                                 multi-entity statements, id reuse?
    test_effects_batch.py        does it hold at scale, and under random ops?
    test_effects_ddl.py          index and constraint DDL as effects
    test_effects_wire.py         framing, version selection, compression,
                                 byte determinism
    test_effects_topology.py     divergence, forced resync, promotion, AOF
    test_effects_commands.py     the commands that still replicate verbatim,
                                 and the threshold that only v2 honours
    test_effects_transition.py   a graph written at v2 and continued at v3

Class prefixes number 01..N WITHIN each file, and test methods 01..N within each
class. They are not a global sequence -- keeping one across a split leaves a
numbered family spread over several files and tells the reader nothing about
where a class lives.

Run order WITHIN a file is the sorted order of the class prefixes, not the order
they are written: RLTest iterates `dir(module)` and `dir()` sorts. Order ACROSS
files is not guaranteed and nothing here depends on it.

RLTest hands consecutive classes with identical `Env(...)` parameters the SAME
server, so classes grouped in one file share server state. That is why the files
are grouped by question rather than by size.
"""

import time
import threading
import itertools

from common import *
from graph_utils import graph_eq

MONITOR_MARK_KEY = "__effects_mark__"

# What each payload version puts on the wire, and what it can do.
#
# The header length is the part that bites when porting a v3-only helper: v2 is
# a BARE version byte, v3 is version + flags. A grep or an offset written for
# one is wrong for the other by one byte, silently.
#
# Measured on a live pair rather than read off the spec:
#   EFFECTS_VERSION 2 -> payload begins \x02 <record>...
#   EFFECTS_VERSION 3 -> payload begins \x03 \x00 <u32 opcode>...
WIRE = {
    2: {
        "header":      b"\x02",
        "header_len":  1,
        "has_flags":   False,
        # v2 has no flags byte, so no bit to carry "this body is compressed"
        "compresses":  False,
        # only v2 consults EFFECTS_THRESHOLD; v3 short-circuits above it in
        # _should_replicate_effects, so raising the threshold does NOT disable
        # effects under v3
        "threshold_disables": True,
    },
    3: {
        "header":      b"\x03\x00",
        "header_len":  2,
        "has_flags":   True,
        "compresses":  True,
        "threshold_disables": False,
    },
}

ALL_VERSIONS = (2, 3)


class _EffectsBase():
    """A primary/replica pair, the waiting helpers, and a version-aware MONITOR.

    Everything here is C-to-C: `Env(env='oss', useSlaves=True)` builds RLTest's
    own primary and replica, both running this module. Cross-engine agreement is
    pinned by the conformance corpus, not by these suites.
    """

    GRAPH_ID = "effects"

    # ------------------------------------------------------------------ setup

    def _setup(self, version=2, threshold=0):
        self.env, self.db = Env(env='oss', useSlaves=True)
        self._after_env(version=version, threshold=threshold)

    def _after_env(self, version=None, threshold=None):
        """Everything _setup does once `self.env` and `self.db` exist.

        Split out so a class that needs its own Env parameters -- moduleArgs for
        a version-pinned destructive test, say -- can build the Env itself and
        still get the rest of the fixture.
        """
        self.master  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()

        self._graph_serial = itertools.count()
        self._marks        = itertools.count()
        self.monitor       = []
        self.monitor_attached = False
        self.monitor_filter   = ()
        self.monitor_stopped  = False
        self.monitor_error    = None

        self.new_graph(self.GRAPH_ID)
        self.wait_for_replica_online()
        self.version = getattr(self, "version", 2)
        self.set_effects_config(version=version, threshold=threshold)

    def new_graph(self, name):
        """Point master and replica at a fresh graph key.

        Used between versions so a v2-written graph does not bleed into the v3
        assertions -- continuing one graph across a version change is a real
        case, but it is `test_effects_transition.py`'s case and not an accident
        every other suite should inherit.
        """
        self.graph_id     = name
        self.master_graph  = Graph(self.master,  name)
        self.replica_graph = Graph(self.replica, name)
        return self.master_graph

    def set_effects_config(self, version=None, threshold=None, compression=None):
        if threshold is not None:
            self.db.config_set("EFFECTS_THRESHOLD", threshold)
        if version is not None:
            self.db.config_set("EFFECTS_VERSION", version)
            self.version = version
        if compression is not None:
            self.db.config_set("EFFECTS_COMPRESSION", compression)

    @property
    def wire(self):
        """The wire table for the version currently configured."""
        return WIRE[self.version]

    # --------------------------------------------------------------- versions

    def for_each_version(self, body, versions=ALL_VERSIONS):
        """Run `body(version)` once per payload version, on a fresh graph each
        time, with the version left as it was found.

        This is the mechanism that keeps the suite from doubling. A behaviour
        that is version-independent -- every opcode replicating, every value
        surviving, the graphs agreeing -- is asserted once and run twice.

        It is NOT for behaviour that only one version has. EFFECTS_THRESHOLD
        disabling effects is v2-only; compression is v3-only. Those belong in a
        version-pinned class, because a loop over them would assert something
        false of one member.

        AND IT IS NOT FOR TESTS THAT DESTROY THE FIXTURE. Promotion is the
        example: `REPLICAOF NO ONE` ends the pair, so the loop's second
        iteration waits forever for an ack from something that is no longer a
        replica -- the test does not fail, it hangs. A test that leaves the pair
        unusable must be version-pinned and differentiated by `moduleArgs`, so
        RLTest gives each version its own SERVER rather than handing both the
        same one. Runtime `config_set` cannot do that: identical Env parameters
        mean a shared server by design.
        """
        was = getattr(self, "version", None)
        try:
            for v in versions:
                self.new_graph("%s_v%d_%d" % (self.GRAPH_ID, v,
                                              next(self._graph_serial)))
                self.set_effects_config(version=v)
                # DRAIN, do not assert quiescence. Fencing here stops the
                # previous version's effects being counted as this one's; it
                # must not also claim the buffer was empty, because a test that
                # legitimately emits hundreds (random ops, bulk writes) is not
                # in an error state for having done so.
                #
                # Only when a reader exists. Classes that assert on replication
                # STATE rather than on the feed -- divergence, promotion -- use
                # this loop without ever calling start_monitor, and requiring
                # one would make the loop fail for a reason unrelated to what
                # they test.
                if self.monitor_attached:
                    self.drain()
                body(v)
        finally:
            if was is not None:
                self.set_effects_config(version=was)

    # ---------------------------------------------------------------- waiting

    def wait_for_replica_online(self, timeout=60):
        """Wait until the replica is ONLINE, not merely linked.

        A write issued before the initial sync completes is folded into the
        full-sync RDB rather than propagated as a command, so it never appears
        in the replica's MONITOR feed at all. WAIT does not protect against
        this: it returns once the replica acks, which is after the fold.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            slave0 = self.master.info("replication").get("slave0")
            state  = (slave0.get("state") if isinstance(slave0, dict)
                      else str(slave0 or ""))
            if state == "online":
                return
            time.sleep(0.1)
        raise AssertionError("replica never reached state=online")

    def query_and_sync(self, q, params=None):
        """Run a write on the primary and wait for the replica to ack it."""
        res = self.master_graph.query(q, params) if params else \
              self.master_graph.query(q)
        self.master.execute_command("WAIT", "1", "0")
        return res

    def assert_graph_eq(self):
        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    # ---------------------------------------------------------------- monitor

    def start_monitor(self, *interesting):
        """Attach MONITOR to the replica, keeping commands whose text contains
        one of `interesting`. The fence key is always kept."""
        self.monitor_filter = tuple(interesting or
                                    ("GRAPH.EFFECT", "GRAPH.QUERY")) + \
                              (MONITOR_MARK_KEY,)
        # daemon=True so a stuck listen() cannot outlive the test process
        self._monitor_thread = threading.Thread(target=self._monitor_loop,
                                                daemon=True)
        self._monitor_thread.start()
        deadline = time.time() + 30
        while not self.monitor_attached:
            if time.time() > deadline:
                raise AssertionError("MONITOR did not attach within 30s")
            time.sleep(0.05)

    def _monitor_loop(self):
        # The exception is RECORDED, not swallowed. A MONITOR feed that dies
        # part way through leaves every later fence timing out, and "fence never
        # appeared" points at replication when the cause is a dead reader. This
        # keeps the real reason so monitor_mark can report it.
        try:
            with self.replica.monitor() as m:
                self.monitor_attached = True
                for cmd in m.listen():
                    if any(f in cmd['command'] for f in self.monitor_filter):
                        self.monitor.append(cmd['command'])
        except Exception as e:
            self.monitor_error = "%s: %s" % (type(e).__name__, e)
        finally:
            self.monitor_stopped = True

    def monitor_mark(self, timeout=60):
        """Fence the replica's feed and return everything recorded before the
        fence, dropping it from the buffer.

        MONITOR is asynchronous with respect to replication: the replica applies
        a replicated command and only then writes the feed line. An offset-based
        wait can therefore return before the line has reached this process, and
        clearing the buffer at that point drops nothing -- the line lands in the
        NEXT window and reads as that test's effect. Bracketing with a fence
        removes the race: when the fence's own line is visible, everything the
        replica produced before it is too.

        The fence is a plain SET on the primary; it replicates verbatim and so
        carries a token this process chose.
        """
        if not self.monitor_attached:
            raise AssertionError(
                "monitor_mark() called but start_monitor() was never called. "
                "Without a reader the fence can never appear, so this would "
                "otherwise time out after %ds and read as a replication "
                "failure." % timeout)
        token = "mark-%d" % next(self._marks)
        self.master.set(MONITOR_MARK_KEY, token)
        deadline = time.time() + timeout
        while time.time() < deadline:
            for i, cmd in enumerate(self.monitor):
                if token in cmd:
                    window = list(self.monitor[:i])
                    del self.monitor[:i + 1]
                    return window
            time.sleep(0.02)
        if getattr(self, "monitor_stopped", False):
            raise AssertionError(
                "MONITOR fence %s never appeared because the MONITOR reader "
                "stopped: %s. Every fence after this point would fail the same "
                "way, so the first one is the real failure."
                % (token, getattr(self, "monitor_error", "no exception recorded")))
        raise AssertionError("MONITOR fence %s never appeared" % token)

    @staticmethod
    def count_in(window, needle):
        return sum(1 for c in window if needle in c)

    # ------------------------------------------------------------- assertions

    def assert_effect_emitted(self, count=None):
        """Fence the feed and assert on the effects since the last fence.

        `count=None` means "at least one" -- the claim the per-opcode tests
        actually make. An exact count is for the classes that are about
        counting. `count=0` is the strict case: nothing in flight.
        """
        window = self.monitor_mark()
        n = self.count_in(window, "GRAPH.EFFECT")
        if count is None:
            self.env.assertTrue(
                n > 0, message="expected a GRAPH.EFFECT, saw none in the window")
        else:
            self.env.assertEquals(n, count)
        return window

    def assert_effect_count(self, count):
        return self.assert_effect_emitted(count=count)

    def drain(self):
        """Fence the feed and discard the window, asserting nothing.

        For use between phases, where the point is that the next window starts
        clean -- not that the last one was empty.
        """
        return self.monitor_mark()

    def assert_replicated_verbatim(self):
        """Assert the last window carried a GRAPH.QUERY and no GRAPH.EFFECT.

        This is the assertion `expect_effect=False` should make. Skipping the
        effect check instead -- only asserting the graphs agree -- passes
        whether or not effects were used, so it cannot tell "the threshold
        disabled effects" from "the threshold was ignored".
        """
        window = self.monitor_mark()
        self.env.assertEquals(self.count_in(window, "GRAPH.EFFECT"), 0)
        self.env.assertTrue(self.count_in(window, "GRAPH.QUERY") > 0)
        return window

    # ------------------------------------------------------------------- wire

    def payload_of(self, command):
        """The raw payload of a GRAPH.EFFECT line, as bytes.

        MONITOR renders the payload with escapes, and 0x20 appears in it freely,
        so the payload is everything after the graph key rather than "the third
        token" -- splitting on whitespace truncates it.
        """
        marker = '"%s"' % self.graph_id
        i = command.find(marker)
        if i < 0:
            i = command.find(self.graph_id)
            if i < 0:
                return None
            i += len(self.graph_id)
        else:
            i += len(marker)
        tail = command[i:].strip()
        if tail.startswith('"'):
            tail = tail[1:]
        if tail.endswith('"'):
            tail = tail[:-1]
        return tail.encode().decode('unicode_escape').encode('latin-1')

    def assert_wire_version(self, window=None, version=None):
        """Assert every effect payload in the window opens with the header for
        `version` -- the byte that says which format actually shipped.

        Worth asserting separately from behaviour: a test that only checks the
        graphs agree cannot tell which format carried them, and the whole point
        of running twice is that two formats did.
        """
        version = self.version if version is None else version
        header  = WIRE[version]["header"]
        if window is None:
            window = self.monitor_mark()
        effects = [c for c in window if "GRAPH.EFFECT" in c]
        self.env.assertTrue(len(effects) > 0,
                            message="no GRAPH.EFFECT to check the header of")
        for c in effects:
            payload = self.payload_of(c)
            self.env.assertTrue(
                payload is not None and payload.startswith(header),
                message="payload does not open with the v%d header %r: %r"
                        % (version, header, (payload or b"")[:8]))
        return window
