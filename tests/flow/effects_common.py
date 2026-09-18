"""Shared fixture for the effects flow suites.

`test_effects*.py` each drive a real primary/replica pair and assert on
**both** sides. This module holds what they share: the `Env` setup, the waiting
helpers, the MONITOR window machinery, and the log readers.

The files, and the question each one asks:

  test_effects.py            does every opcode replicate at all?
  test_effects_shapes.py     what must one buffer survive -- partitioning,
                             supernodes, id reuse, compound statements?
  test_effects_wire.py       framing, refusal of unreadable payloads,
                             compression, byte determinism
  test_effects_ddl.py        index and constraint DDL as effects, including
                             indexes a query never named
  test_effects_topology.py   divergence, forced resync, promotion
  test_effects_commands.py   GRAPH.RECORD, and the commands that still
                             replicate verbatim

They are split rather than one file so CI shards them -- the flow matrix cells
on `test_file`, so each file runs in its own cell against its own pair of
service containers. Three things that bought:

  - `FAIL_FAST=1` stops a *file*, so one class's failure no longer hides every
    class alphabetically after it.
  - the cells run in parallel, and these suites are slow -- million-node
    builds, 100k-node batches, 120-second settle timeouts.
  - RLTest hands consecutive classes with identical `Env(...)` parameters the
    *same* server, so sharing a file means sharing server state. That coupling
    is now bounded to the classes grouped together on purpose.

Run order WITHIN a file is the sorted order of the class prefixes, not the
order they are written: `RLTest/loader.py:124` is `for symbol in dir(module)`,
and `dir()` sorts. Order ACROSS files is not guaranteed and nothing here
depends on it -- each file gets its own pair.

Everything here is Rust-to-Rust. `Env(env='oss', useSlaves=True)` builds
RLTest's own primary and replica, both running this module; no C engine is
involved, so nothing in these files is a cross-engine claim.

A TRAP THE SPLIT INTRODUCED, for whoever edits `_setup` below. CI decides
whether a flow file needs a private container by scanning **that file** for
`Env(...)` flags -- `tests/flow/test_matrix_split.py`, and `files_for_entry`
resolves an entry to the one file: **it does not follow imports.** The `Env(...)`
call lives here, not in the test files, so the classifier sees no `Env()` in
any of them and routes them all to the shared-services bucket by default.

That is the correct destination today -- `useSlaves` and `enableDebugCommand`
are explicitly not spawn-forcing, because the services job supplies a replica
container and enables debug regardless. But it is the right answer for the
wrong reason. **If a spawn-forcing flag is ever added to the `Env(...)` below
-- `shardsCount`, `oss-cluster`, or `moduleArgs` with an immutable key -- the
classifier will not see it, every one of these files will stay in the services
bucket where it cannot work, and nothing will say so.** Change the classifier
to follow this import, or move that `Env(...)` back into the files that need it.

A skip here is a skip EVERYWHERE, for the same reason. All of these files are
in `services_files` and none is in `spawn_files`, so a guard of the form
`if os.getenv("FALKORDB_USE_SERVICE"): Environment.skip(None)` does not mean
"runs in the other mode" -- it means the class never runs in CI at all. The
`SANITIZER` guard in `_setup` is the one deliberate exception: replication
under sanitizer is unreliable (`test_replication.py` says the same), so the
asan lane runs none of these, and its green checks are not evidence about
this suite.
"""

import itertools
import re
import threading
import time

import codecs

from common import *
from constraint_utils import (create_mandatory_node_constraint,
                              create_unique_node_constraint,
                              drop_unique_node_constraint,
                              get_constraint, list_constraints)
from graph_utils import graph_eq
from index_utils import (create_edge_range_index, create_node_fulltext_index,
                         create_node_range_index, drop_node_range_index,
                         list_indicies, wait_for_indices_to_sync)

# A plain Redis key the tests SET on the primary purely so its replicated form
# shows up in the replica's MONITOR feed as a fence post. See `monitor_mark`.
MONITOR_MARK_KEY = "__effects_mark__"



class _EffectsBase():
    """A primary/replica pair with v3 selected, plus the waiting helpers.

    Not discovered as a test: RLTest only collects module-level names starting
    with "test".
    """

    # Overridden per class so no two classes share a graph key. RLTest may hand
    # them the same server.
    GRAPH_ID = "effects_base"

    #-------------------------------------------------------------------------
    # setup
    #-------------------------------------------------------------------------

    def _setup(self, compression=0):
        # replication under sanitizer is unreliable, as test_replication.py notes
        if SANITIZER:
            Environment.skip(None)

        # No `enableDebugCommand`: every class here runs against the shared
        # services container under CI (`test_matrix_split.py` puts all six
        # files in `services_files`), and `common.py:564` documents the flag as
        # a no-op in that mode anyway. Nothing here needs DEBUG.
        self.env, self.db = Env(env='oss', useSlaves=True)
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

    def replica_log_count(self, needle):
        """How many times `needle` appears across the master and replica logs.

        Counting a *specific* refusal is what separates asserting the mechanism
        from asserting the outcome. Thirteen different checks end in a resync,
        so `sync_full` moving proves only that one of them fired — a test that
        names a mechanism in its comment and then watches `sync_full` passes
        for any of the other twelve.
        """
        path = self.env.log_path
        if not path:
            # `None`, not `0`. CI's services mode sets `env.log_path = None`
            # (`common.py`), because the servers are Docker containers whose
            # logs are not on a path this process can read. Returning 0 there
            # made every log assertion *vacuous* rather than failing:
            # `assertEqual(diverged(), before)` is `0 == 0` whatever happened.
            # Callers must branch on `None` and assert something else.
            return None
        slave = path.replace("master-1", "slave-2")
        total = 0
        for p in (path, slave):
            if os.path.exists(p):
                with open(p, "rb") as f:
                    total += f.read().count(needle)
        return total

    def replica_log_diverged(self):
        """How many divergence refusals the replica has logged so far.

        `sync_full` alone would miss a refusal on a graph the guard could not
        find a master address for — that arm exits instead of resyncing, so it
        would show up as a dead replica rather than as a resync, and counting
        the log lines catches both.
        """
        return self.replica_log_count(b"diverged")

    def effect_failures(self):
        """How many `GRAPH.EFFECT` calls the replica has refused.

        The observable that survives every mode, because it comes back over the
        connection rather than off disk: `INFO commandstats` carries
        `cmdstat_graph.EFFECT` with a `failed_calls` field. Weaker than the log
        — it says a buffer was refused, not *which* check refused it — so it is
        the fallback where the log is unreadable, not the preferred assertion.
        """
        stats = self.replica.execute_command("INFO", "commandstats")
        if not isinstance(stats, dict):
            stats = {
                k.strip(): v.strip()
                for k, _, v in (l.partition(":") for l in str(stats).splitlines())
                if k.strip()
            }
        entry = stats.get("cmdstat_graph.EFFECT")
        if entry is None:
            return 0
        # redis-py parses a `cmdstat_*` value into a dict
        # (`{'calls': 1, ..., 'failed_calls': 1}`); a raw connection leaves it as
        # the wire's `calls=1,...,failed_calls=1`. Both shapes reach here.
        if isinstance(entry, dict):
            return int(entry.get("failed_calls", 0))
        m = re.search(r"failed_calls=(\d+)", str(entry))
        return int(m.group(1)) if m else 0

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

    def assert_effect_emitted(self, count=None):
        """Fence the replica's feed and assert on the effects since the last
        fence. Returns the window.

        `count=None` (the default) means "at least one". That is deliberately
        the weaker claim, because it is the one the per-opcode tests actually
        make: they write and then ask whether the write shipped as an effect,
        and several of them write more than once before asking. An exact
        `count` is available and the classes that are *about* counting pass
        one.

        `count=0` is the useful strict case — nothing is still in flight —
        which is what a test wants at its start when it shares a server with
        the class's earlier tests.

        Either way the fence drains the window, so a stale effect can satisfy
        at most one call. Counting inside a fenced window rather than polling
        for a line is what makes that true: MONITOR is asynchronous with
        respect to replication, so a poll can return before the line lands and
        the line then shows up inside a later test's window, where it reads as
        that test's effect. See `monitor_mark`.
        """
        window = self.monitor_mark()
        seen = self.count_in(window, 'GRAPH.EFFECT')
        if count is None:
            self.env.assertGreaterEqual(seen, 1)
        else:
            self.env.assertEqual(seen, count)
        return window

    # A v3 buffer opens with `u8 version · u8 flags`, and bit 0 of the flags
    # byte is FLAG_COMPRESSED. MONITOR renders the two as escape sequences.
    HEADER_PLAIN      = r'\x03\x00'
    HEADER_COMPRESSED = r'\x03\x01'

    # v3 opcodes, from `graph/src/effects/v3/mod.rs`. Named here so a failure
    # reads as "got CREATE_NODE, wanted CREATE_CONSTRAINT" rather than "got 3,
    # wanted 13".
    OPCODES = {
        1: 'UPDATE_NODE',      2: 'UPDATE_EDGE',
        3: 'CREATE_NODE',      4: 'CREATE_EDGE',
        5: 'DELETE_NODE',      6: 'DELETE_EDGE',
        7: 'SET_LABELS',       8: 'REMOVE_LABELS',
        9: 'ADD_SCHEMA',      10: 'ADD_ATTRIBUTE',
        11: 'CREATE_INDEX',   12: 'DROP_INDEX',
        13: 'CREATE_CONSTRAINT', 14: 'DROP_CONSTRAINT',
    }

    # The wire tags a CREATE_CONSTRAINT record carries, from
    # `graph/src/effects/v3/mod.rs`. Written out here rather than derived,
    # because the status numbering is C's and is *reversed* against Rust's enum
    # — C is `CT_ACTIVE = 0, CT_PENDING = 1`, Rust reads `UnderConstruction,
    # Operational`. A test that read the tag through the Rust enum would agree
    # with a regression that replaced `constraint_status_tag` with `as u32`;
    # this one disagrees, which is the point of asserting on bytes.
    CONSTRAINT_STATUS = {0: 'OPERATIONAL', 1: 'UNDER_CONSTRUCTION', 2: 'FAILED'}
    CONSTRAINT_TYPE   = {0: 'UNIQUE', 1: 'MANDATORY'}
    ENTITY_TAG        = {1: 'NODE', 2: 'RELATIONSHIP'}

    @staticmethod
    def payload_bytes(payload):
        """The buffer MONITOR rendered, back as bytes.

        MONITOR uses `sdscatrepr`, whose escapes are a subset of Python's, so
        `unicode_escape` reverses it and `latin-1` maps the code points back to
        bytes one for one.
        """
        return codecs.decode(payload, 'unicode_escape').encode('latin-1')

    @classmethod
    def constraint_announcements(cls, window, key):
        """`(type, entity, status)` for every CREATE_CONSTRAINT in `window`.

        Reading the status off the wire is the only way to tell the two
        announcements of an asynchronously validated constraint apart. Counting
        payloads says two things crossed; reading the opcode says both were
        constraints; only this says the first said UNDER CONSTRUCTION and the
        second said OPERATIONAL, which is the property the record exists for.

        The layout is fixed up to the status, so no record walking is needed:
        `u8 version · u8 flags · u32 opcode · u32 type · u32 entity · u32
        status`. Anything that is not a CREATE_CONSTRAINT is skipped — a UNIQUE
        constraint puts its supporting index on the wire first.
        """
        out = []
        for payload in cls.effect_payloads(window, key):
            raw = cls.payload_bytes(payload)
            if len(raw) < 18 or int.from_bytes(raw[2:6], 'little') != 13:
                continue
            out.append((
                cls.CONSTRAINT_TYPE.get(int.from_bytes(raw[6:10], 'little')),
                cls.ENTITY_TAG.get(int.from_bytes(raw[10:14], 'little')),
                cls.CONSTRAINT_STATUS.get(int.from_bytes(raw[14:18], 'little')),
            ))
        return out

    @classmethod
    def payload_opcodes(cls, payload):
        """Every record opcode in one MONITOR-rendered payload, named.

        Counting `GRAPH.EFFECT` lines says how many payloads crossed the wire
        and nothing about what was in them — a constraint announcement and a
        node create are both "one effect". Reading the opcodes is what makes an
        assertion about constraints an assertion about constraints.

        MONITOR renders the buffer with `sdscatrepr`, whose escapes are a
        subset of Python's, so `unicode_escape` reverses it and `latin-1`
        recovers the bytes one-for-one. Then the payload is
        `u8 version · u8 flags`, and after it a run of records, each opening
        with a `u32` opcode.

        Only the opcode and count are read, and the blocks between them are
        skipped by *this* function knowing each record's shape — which it
        cannot, so it stops after the first record. That is enough: a payload
        is built by one commit, and the tests that use this care about which
        kind of record a payload leads with.
        """
        raw = cls.payload_bytes(payload)
        if len(raw) < 6:
            return []
        if raw[1] & 1:
            # FLAG_COMPRESSED — the records are inside a zstd frame, and these
            # tests run uncompressed, so this is a setup error rather than a
            # case to handle.
            raise AssertionError("payload is compressed; opcodes are not readable")
        return [cls.OPCODES.get(int.from_bytes(raw[2:6], 'little'), 'UNKNOWN')]

    @classmethod
    def leading_opcodes(cls, window, key):
        """The first record's opcode of every effect payload in `window`."""
        return [op
                for payload in cls.effect_payloads(window, key)
                for op in cls.payload_opcodes(payload)]

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
# Built by hand rather than taken from a library, because every library option
# is absent where it matters. `compression.zstd` is Python 3.14+ and the flow
# CI container is `debian:trixie-slim` (build/Dockerfile:53), whose python3 is
# 3.13; `zstandard` is in neither tests/requirements.txt nor the image's pip
# line (build/Dockerfile:141). The previous shim degraded to `None` and the one
# test needing a valid frame skipped — locally it ran, in CI it printed [SKIP]
# and the checksum-refusal path was asserted nowhere. Adding the package would
# mean rebuilding the multi-arch toolchain image for one test.
#
# A single raw (uncompressed) block is a legal zstd frame, and nothing here
# needs compression to actually happen — the reader has to accept the frame,
# and the test then corrupts the checksum around it.


def zstd_raw_frame(data):
    """`data` wrapped in a valid single-block zstd frame, no library needed.

    Magic `0xFD2FB528`, then a Frame_Header_Descriptor with Single_Segment_flag
    (bit 5) set and Frame_Content_Size_flag 0, so exactly one byte of content
    size follows and no window descriptor does. Then one Block_Header — 24-bit
    little-endian `(size << 3) | (type << 1) | last`, with type 0 meaning raw —
    and the bytes themselves.
    """
    if len(data) > 255:
        # A longer payload only needs a wider FCS field; no caller wants one,
        # and failing loudly beats emitting a frame that decodes to garbage.
        raise ValueError("single-byte frame content size holds at most 255 bytes")
    return (b"\x28\xb5\x2f\xfd"
            + b"\x20"
            + bytes([len(data)])
            + (((len(data) << 3) | 1)).to_bytes(3, "little")
            + data)
