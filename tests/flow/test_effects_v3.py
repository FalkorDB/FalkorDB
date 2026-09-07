import struct
from common import *

GRAPH_ID = "effects_v3"

# GRAPH.EFFECT v3, end to end against a live server.
#
# The unit-level decode tests live outside the repo; what these pin is the part
# only a real server can show: that a v3 payload built by a peer reaches
# EffectsV3_Apply and lands the right graph state. Until C can EMIT v3 there is
# no round trip available here, so the payloads are hand-built to the format -
# which is also what makes them a useful cross-check on the decoder, since a
# decoder that reads a field one width narrow will not agree with bytes written
# from the spec rather than from itself.
#
# The instance reads v3 (EFFECTS_VERSION 3) and still EMITS v2
# (EFFECTS_VERSION_EMIT 2), so nothing here depends on the write path.
#
# Note on shape: a refused effect means "this instance has diverged", and the
# guard responds by forcing a resync or - with no master to resync from -
# shutting the server down. So every payload that must be REFUSED needs its own
# Env, and therefore its own class. Payloads that apply cleanly can share one.

EFFECTS_VERSION = 3

EFFECT_UPDATE_NODE   = 1
EFFECT_UPDATE_EDGE   = 2
EFFECT_CREATE_NODE   = 3
EFFECT_CREATE_EDGE   = 4
EFFECT_DELETE_NODE   = 5
EFFECT_DELETE_EDGE   = 6
EFFECT_SET_LABELS    = 7
EFFECT_REMOVE_LABELS = 8
EFFECT_ADD_SCHEMA    = 9
EFFECT_ADD_ATTRIBUTE = 10

SCHEMA_NODE = 0
SCHEMA_EDGE = 1

# src/value.h - SIType is a bitmask, not an ordinal
T_STRING = 1 << 11
T_BOOL   = 1 << 12
T_INT64  = 1 << 13
T_DOUBLE = 1 << 14
T_NULL   = 1 << 15

# segment header bits
SEG_RANGE      = 0
SEG_ASCENDING  = 1
SEG_REPEAT     = 2
SEG_DESCENDING = 0x40   # bit 6
SEG_RESERVED   = 0x80   # bit 7


def _u8(v):  return struct.pack('<B', v)
def _u16(v): return struct.pack('<H', v)
def _u32(v): return struct.pack('<I', v)
def _i32(v): return struct.pack('<i', v)
def _u64(v): return struct.pack('<Q', v)
def _i64(v): return struct.pack('<q', v)
def _f64(v): return struct.pack('<d', v)


def _string(s):
    raw = s.encode() + b'\x00'
    return _u64(len(raw)) + raw


def _width_code(v):
    """narrowest width that holds v - the encoder's rule, so the decoder's
    width handling is exercised rather than always taking the 8-byte path"""
    if v <= 0xFF:               return 0, _u8(v)
    if v <= 0xFFFF:             return 1, _u16(v)
    if v <= 0xFFFFFFFF:         return 2, _u32(v)
    return 3, _u64(v)


def seg_range(base, length, descending=False):
    vw, vb = _width_code(base)
    cw, cb = _width_code(length)
    hdr = SEG_RANGE | (vw << 2) | (cw << 4) | (SEG_DESCENDING if descending else 0)
    return _u8(hdr) + vb + cb


def seg_repeat(entity_id, count):
    vw, vb = _width_code(entity_id)
    cw, cb = _width_code(count)
    hdr = SEG_REPEAT | (vw << 2) | (cw << 4)
    return _u8(hdr) + vb + cb


def id_list(*segments):
    return _u32(len(segments)) + b''.join(segments)


def label_set(*labels):
    return _u16(len(labels)) + b''.join(_i32(l) for l in labels)


def attr_ids(*ids):
    return _u16(len(ids)) + b''.join(_u16(i) for i in ids)


def rel_type(r):
    return _i32(r)


def v_int(v):    return _u32(T_INT64)  + _i64(v)
def v_double(v): return _u32(T_DOUBLE) + _f64(v)
def v_bool(v):   return _u32(T_BOOL)   + _u8(1 if v else 0)
def v_null():    return _u32(T_NULL)
def v_string(s): return _u32(T_STRING) + _string(s)


def payload(*records, flags=0):
    return _u8(EFFECTS_VERSION) + _u8(flags) + b''.join(records)


# ---- record builders -------------------------------------------------------
# every batchable record is: opcode . count . <shape> . <attr ids> . IdList(s)
# . values -- the shape PRECEDES the rows

def rec_add_schema(schema_type, schema_id, name):
    return _u32(EFFECT_ADD_SCHEMA) + _u32(schema_type) + _i32(schema_id) \
        + _string(name)


def rec_add_attribute(attr_id, name):
    return _u32(EFFECT_ADD_ATTRIBUTE) + _u16(attr_id) + _string(name)


def rec_create_node(count, labels, attrs, ids, values):
    return _u32(EFFECT_CREATE_NODE) + _u32(count) + label_set(*labels) \
        + attr_ids(*attrs) + ids + b''.join(values)


def rec_update_node(count, labels, attrs, ids, values):
    return _u32(EFFECT_UPDATE_NODE) + _u32(count) + label_set(*labels) \
        + attr_ids(*attrs) + ids + b''.join(values)


def rec_delete_node(count, labels, ids):
    return _u32(EFFECT_DELETE_NODE) + _u32(count) + label_set(*labels) + ids


def rec_set_labels(count, labels, ids):
    return _u32(EFFECT_SET_LABELS) + _u32(count) + label_set(*labels) + ids


def rec_remove_labels(count, labels, ids):
    return _u32(EFFECT_REMOVE_LABELS) + _u32(count) + label_set(*labels) + ids


def rec_create_edge(count, r, attrs, ids, src, dst, values):
    return _u32(EFFECT_CREATE_EDGE) + _u32(count) + rel_type(r) \
        + attr_ids(*attrs) + ids + src + dst + b''.join(values)


def rec_delete_edge(count, r, ids, src, dst):
    return _u32(EFFECT_DELETE_EDGE) + _u32(count) + rel_type(r) + ids + src + dst


def rec_update_edge(count, r, attrs, ids, values):
    return _u32(EFFECT_UPDATE_EDGE) + _u32(count) + rel_type(r) \
        + attr_ids(*attrs) + ids + b''.join(values)


class testEffectsV3Apply():
    """Valid v3 payloads, applied to a live graph, sharing one Env.

    Ordered: the graph state each test leaves is the next one's input, which is
    also what lets the id checks be exact. CREATE_NODE verifies that the id the
    replica allocates equals the id on the wire, so the tests have to know
    exactly which ids are next - that agreement IS the check.
    """

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env()
        self.conn  = self.env.getConnection()
        self.graph = Graph(self.conn, GRAPH_ID)

        # the graph must exist before GRAPH.EFFECT will target it.
        # this also establishes label :L = 0 and attribute 'v' = 0, so the
        # first schema/attribute an effect adds gets id 1
        self.graph.query("CREATE (:L {v: 1})")

    def _send(self, buf):
        self.conn.execute_command("GRAPH.EFFECT", GRAPH_ID, buf)

    def test01_add_schema_and_attribute(self):
        # records 9 and 10 establish the id spaces every later record uses.
        # The replica computes the id it WOULD assign and refuses on mismatch,
        # so these ids are predictions that must come true.
        self._send(payload(
            rec_add_schema(SCHEMA_NODE, 1, "Person"),
            rec_add_attribute(1, "name"),
        ))

        # nothing observable changed yet, but the schema is now usable
        res = self.graph.query("MATCH (n:Person) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 0)

    def test02_create_node_batch(self):
        # one record, three nodes, one Range segment, two attributes each.
        # node 0 is taken by the constructor, so these are 1,2,3
        self._send(payload(rec_create_node(
            count  = 3,
            labels = [1],                      # :Person
            attrs  = [1, 0],                   # name, v
            ids    = id_list(seg_range(1, 3)),
            values = [
                v_string("alice"), v_int(30),
                v_string("bob"),   v_int(40),
                v_string("carol"), v_int(50),
            ],
        )))

        res = self.graph.query(
            "MATCH (n:Person) RETURN n.name, n.v ORDER BY n.v")
        self.env.assertEquals(res.result_set,
                              [["alice", 30], ["bob", 40], ["carol", 50]])

    def test03_row_k_binds_to_the_kth_id(self):
        # the ordering invariant: row k belongs to the k-th id AS WRITTEN.
        # If the decoder paired rows with ids in any other order this passes
        # only by luck, so the values are deliberately not symmetric.
        res = self.graph.query("MATCH (n:Person) RETURN id(n), n.name ORDER BY id(n)")
        self.env.assertEquals(res.result_set,
                              [[1, "alice"], [2, "bob"], [3, "carol"]])

    def test04_update_node_and_null_removes(self):
        # T_NULL in a value slot means REMOVE THIS ATTRIBUTE. FalkorDB never
        # stores a null property, so this is how a removal replicates - a
        # reader that filtered nulls would turn it into a no-op.
        self._send(payload(rec_update_node(
            count  = 2,
            labels = [1],
            attrs  = [1],                      # name
            ids    = id_list(seg_range(1, 2)),
            values = [v_string("alice2"), v_null()],
        )))

        res = self.graph.query(
            "MATCH (n:Person) RETURN id(n), n.name ORDER BY id(n)")
        self.env.assertEquals(res.result_set,
                              [[1, "alice2"], [2, None], [3, "carol"]])

    def test05_set_and_remove_labels(self):
        self._send(payload(rec_add_schema(SCHEMA_NODE, 2, "Young")))

        self._send(payload(rec_set_labels(
            count  = 2,
            labels = [2],
            ids    = id_list(seg_range(1, 2)),
        )))
        res = self.graph.query("MATCH (n:Young) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 2)

        self._send(payload(rec_remove_labels(
            count  = 1,
            labels = [2],
            ids    = id_list(seg_range(1, 1)),
        )))
        res = self.graph.query("MATCH (n:Young) RETURN id(n)")
        self.env.assertEquals(res.result_set, [[2]])

    def test06_create_edge_with_repeat_source(self):
        # the shape Repeat exists for: every edge out of one node shares a
        # source, so the source column is one segment rather than one per edge
        self._send(payload(rec_add_schema(SCHEMA_EDGE, 0, "KNOWS")))

        self._send(payload(rec_create_edge(
            count  = 2,
            r      = 0,
            attrs  = [0],                      # v
            ids    = id_list(seg_range(0, 2)),   # first edges: ids 0,1
            src    = id_list(seg_repeat(1, 2)),  # both out of node 1
            dst    = id_list(seg_range(2, 2)),   # to nodes 2 and 3
            values = [v_int(7), v_int(8)],
        )))

        res = self.graph.query(
            "MATCH (a)-[e:KNOWS]->(b) RETURN id(a), id(b), e.v ORDER BY id(b)")
        self.env.assertEquals(res.result_set, [[1, 2, 7], [1, 3, 8]])

    def test07_update_edge_recovers_its_endpoints(self):
        # UPDATE_EDGE carries its relationship type and deliberately NOT its
        # endpoints - apply recovers them by scanning the relationship tensor.
        # If that recovery is wrong the index is keyed under the wrong
        # src/dst, so this asserts the edge is still reachable by pattern
        # afterwards rather than only that the property changed.
        self._send(payload(rec_update_edge(
            count  = 1,
            r      = 0,
            attrs  = [0],
            ids    = id_list(seg_range(0, 1)),
            values = [v_int(99)],
        )))

        res = self.graph.query(
            "MATCH (a)-[e:KNOWS]->(b) WHERE e.v = 99 RETURN id(a), id(b)")
        self.env.assertEquals(res.result_set, [[1, 2]])

    def test08_delete_edge(self):
        self._send(payload(rec_delete_edge(
            count = 1,
            r     = 0,
            ids   = id_list(seg_range(1, 1)),
            src   = id_list(seg_range(1, 1)),
            dst   = id_list(seg_range(3, 1)),
        )))

        res = self.graph.query("MATCH ()-[e:KNOWS]->() RETURN count(e)")
        self.env.assertEquals(res.result_set[0][0], 1)

    def test09_delete_node_descending_range(self):
        # segment header bit 6. A descending Range's base is its FIRST and
        # HIGHEST id, so this names 3 then 2 - and 3 must be deleted, not 4.
        #
        # No fixture in the conformance corpus exercises bit 6: it landed after
        # the corpus was cut. This is the only place descending is checked
        # against a real server rather than against a unit-level decode.
        res = self.graph.query("MATCH (n:Person) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 3)

        self._send(payload(rec_delete_node(
            count  = 2,
            labels = [1],
            ids    = id_list(seg_range(3, 2, descending=True)),
        )))

        res = self.graph.query("MATCH (n:Person) RETURN id(n)")
        self.env.assertEquals(res.result_set, [[1]])

    def test10_multi_segment_mixed_direction(self):
        # rule 3: a reversal ends a run rather than continuing it the other
        # way, so 6,5,7 is descending Range(6,2) then ascending Range(7,1).
        # Three fresh nodes at 4,5,6... the ids just freed by test09 are
        # reusable, so this asserts what the graph holds rather than assuming.
        self._send(payload(rec_create_node(
            count  = 3,
            labels = [1],
            attrs  = [0],
            ids    = id_list(seg_range(2, 3)),
            values = [v_int(1), v_int(2), v_int(3)],
        )))

        res = self.graph.query("MATCH (n:Person) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 4)

        # now delete 3,2 (descending) and 4 (ascending) in one IdList
        self._send(payload(rec_delete_node(
            count  = 3,
            labels = [1],
            ids    = id_list(seg_range(3, 2, descending=True),
                             seg_range(4, 1)),
        )))

        res = self.graph.query("MATCH (n:Person) RETURN id(n)")
        self.env.assertEquals(res.result_set, [[1]])


class testRefusalControls():
    """The other half of the refusal tests below.

    Three of those refusals surface in the log only as "v3 payload refused:
    malformed" - the status does not say WHICH check fired. So a test asserting
    "this was refused" would pass even if it was refused for some unrelated
    reason, which is the failure mode where a green test means nothing.

    These are the paired controls: the SAME payload shapes with the offending
    mutation removed. If a control is accepted and applies correctly, then the
    only difference between it and the refused version is the mutation, so the
    mutation is what caused the refusal. Accepted payloads leave the server up,
    so all three share one Env.
    """

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env()
        self.conn  = self.env.getConnection()
        self.graph = Graph(self.conn, GRAPH_ID)

        # three nodes, ids 0,1,2 - one per control
        self.graph.query("CREATE (:L {v: 1}), (:L {v: 2}), (:L {v: 3})")

    def _send(self, buf):
        self.conn.execute_command("GRAPH.EFFECT", GRAPH_ID, buf)

    def test01_range_without_the_reserved_bit(self):
        # testReservedSegmentBitRefused sends exactly this with bit 7 set
        self._send(payload(rec_delete_node(1, [0], id_list(seg_range(0, 1)))))
        res = self.graph.query("MATCH (n:L) RETURN id(n) ORDER BY id(n)")
        self.env.assertEquals(res.result_set, [[1], [2]])

    def test02_repeat_without_the_descending_bit(self):
        # testDescendingRepeatRefused sends exactly this with bit 6 set
        self._send(payload(rec_delete_node(1, [0], id_list(seg_repeat(1, 1)))))
        res = self.graph.query("MATCH (n:L) RETURN id(n)")
        self.env.assertEquals(res.result_set, [[2]])

    def test03_count_matching_its_segments(self):
        # testCardinalityMismatchRefused sends count=3 over the same segment
        self._send(payload(rec_delete_node(1, [0], id_list(seg_range(2, 1)))))
        res = self.graph.query("MATCH (n:L) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 0)


class testExpansionAtTheExtremes():
    """Range expansion at the ends of the id space.

    The bug class is `base + len` where the intent is `base + len - 1`: it
    gives an id one too far, and at the extremes it wraps to something far
    away rather than trapping. C's unsigned overflow is defined, so nothing
    diagnoses it - there is no debug-mode tell the way Rust had a panic. Only
    extreme-value tests find it, and expansion is a different code path from
    the decode-side bound, so pinning it there is not enough.
    """

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env()
        self.conn  = self.env.getConnection()
        self.graph = Graph(self.conn, GRAPH_ID)
        self.graph.query("CREATE (:L {v: 1})")   # node 0

    def test01_descending_from_zero_names_zero(self):
        # Range{base: 0, len: 1} descending names exactly [0]. A `base - len`
        # would wrap to UINT64_MAX and the node would survive, so the deletion
        # landing is the assertion.
        self.conn.execute_command("GRAPH.EFFECT", GRAPH_ID,
                payload(rec_delete_node(1, [0],
                        id_list(seg_range(0, 1, descending=True)))))

        res = self.graph.query("MATCH (n:L) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 0)


class _RefusedCase():
    """Send one payload that must be REFUSED, in its own Env.

    A refusal takes the instance down, so these cannot share.
    """

    def _refuse(self, buf, what, expect_log=None):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        env, _ = Env()
        conn  = env.getConnection()
        graph = Graph(conn, GRAPH_ID)
        graph.query("CREATE (:L {v: 1})")

        try:
            conn.execute_command("GRAPH.EFFECT", GRAPH_ID, buf)
            accepted = True
        except Exception:
            # an error reply, or the connection dropping as the divergence
            # guard takes the instance down - both mean refused
            accepted = False

        env.assertFalse(accepted, message=f"payload was accepted: {what}")

        if expect_log is None:
            return

        # "it was refused" is a weak assertion on its own - it holds even if
        # the refusal came from an unrelated check. Where the log names the
        # value the decoder computed, assert on that instead.
        import os
        log_name = env.envRunner._getFileName("master", ".log")
        try:
            with open(os.path.join(env.logDir, log_name)) as f:
                log = f.read()
        except FileNotFoundError:
            return

        env.assertContains(expect_log, log)
        for fault in ("REDIS BUG REPORT", "Segmentation fault"):
            env.assertNotContains(fault, log)


class testMaxIdAscendingNamesMaxId(_RefusedCase):
    def test_refused(self):
        # Range{base: UINT64_MAX, len: 1} is a LEGAL segment - the decode
        # bound accepts it, `len - 1 <= UINT64_MAX - base` being `0 <= 0` -
        # and it names one real id. Expansion must produce UINT64_MAX.
        #
        # No such node exists, so this is refused; the point is WHICH id the
        # log names. `base + len` would wrap to 0, and node 0 does exist, so
        # that bug would show up as a SUCCESSFUL deletion of the wrong node.
        self._refuse(
            payload(rec_delete_node(1, [0],
                    id_list(seg_range(0xFFFFFFFFFFFFFFFF, 1)))),
            "ascending Range at the top of the id space",
            expect_log="references node 18446744073709551615")


class testMaxIdDescendingNamesMaxId(_RefusedCase):
    def test_refused(self):
        # the same segment read the other way names the same single id
        self._refuse(
            payload(rec_delete_node(1, [0],
                    id_list(seg_range(0xFFFFFFFFFFFFFFFF, 1,
                            descending=True)))),
            "descending Range at the top of the id space",
            expect_log="references node 18446744073709551615")


class testSchemaIdDivergenceRefused(_RefusedCase):
    def test_refused(self):
        # ADD_SCHEMA states the id the master assigned. Locally :L took 0, so
        # 'Person' must become 1 - claiming 7 is a numbering disagreement, and
        # catching it here is the whole reason v3 puts the id on the wire.
        self._refuse(payload(rec_add_schema(SCHEMA_NODE, 7, "Person")),
                     "schema id 7 where the replica would assign 1")


class testReservedSegmentBitRefused(_RefusedCase):
    def test_refused(self):
        # header bit 7 is reserved and must be REJECTED, not masked off
        seg = bytearray(seg_range(1, 1))
        seg[0] |= SEG_RESERVED
        self._refuse(payload(rec_delete_node(1, [0], id_list(bytes(seg)))),
                     "reserved segment header bit 7 set")


class testDescendingRepeatRefused(_RefusedCase):
    def test_refused(self):
        # a Repeat has no direction, so bit 6 there cannot mean anything -
        # a peer that set it meant something this build does not know
        seg = bytearray(seg_repeat(0, 1))
        seg[0] |= SEG_DESCENDING
        self._refuse(payload(rec_delete_node(1, [0], id_list(bytes(seg)))),
                     "descending bit on a Repeat segment")


class testCardinalityMismatchRefused(_RefusedCase):
    def test_refused(self):
        # the segment list must total the record's count. One id short would
        # land every later row on the wrong entity, so it fails instead.
        self._refuse(payload(rec_delete_node(3, [0], id_list(seg_range(0, 1)))),
                     "record count 3 with segments totalling 1")


class testUnknownLabelRefused(_RefusedCase):
    def test_refused(self):
        # label ids are RESOLVED against local schema, not range-checked: an
        # id inside the schema count can still map to nothing
        self._refuse(payload(rec_delete_node(1, [99], id_list(seg_range(0, 1)))),
                     "label schema 99 which does not exist locally")


class testCountExceedingGraphRefused(_RefusedCase):
    def test_refused(self):
        # being told to delete more nodes than exist locally is divergence,
        # and refusing on a bound taken from local graph state means a
        # wire-declared count never sizes an allocation
        self._refuse(
            payload(rec_delete_node(1000000, [0], id_list(seg_range(0, 1000000)))),
            "count far exceeding the local node count")


class testUnimplementedDDLRefused(_RefusedCase):
    def test_refused(self):
        # records 11-14 are a separate PR. They are refused as UNIMPLEMENTED
        # rather than MALFORMED - the bytes are not corrupt - but refused all
        # the same, because silently skipping a record we cannot apply is
        # data loss.
        EFFECT_CREATE_INDEX = 11
        rec = _u32(EFFECT_CREATE_INDEX) + _u32(SCHEMA_NODE) + _i32(0) \
            + _string("L") + _u16(0) + _string("v") + _u32(0x0E)
        self._refuse(payload(rec), "CREATE_INDEX, not implemented yet")
