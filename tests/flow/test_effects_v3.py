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

# a version ABOVE this build's read ceiling (src/effects/effects.h
# EFFECTS_VERSION, currently 3). Must be raised whenever that rises.
FUTURE_EFFECTS_VERSION = 4

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


class testEntitiesWithoutProperties():
    """The commonest write in the language: create with no properties.

    `CREATE (:Person)` and `CREATE (a)-[:R]->(b)` produce a record whose
    attribute set is EMPTY - n_attrs 0, no value rows. An earlier decoder
    refused that as malformed, reasoning that a record with entities but no
    attribute ids "states nothing about them". It has it backwards: the empty
    shape states precisely that these entities have no properties.

    No fixture in the conformance corpus exercises it - every create case there
    carries attributes - so a full corpus run stayed green while the most
    ordinary write there is was being rejected.
    """

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env()
        self.conn  = self.env.getConnection()
        self.graph = Graph(self.conn, GRAPH_ID)
        self.graph.query("CREATE (:L {v: 1})")   # node 0, label :L = 0

    def _send(self, buf):
        self.conn.execute_command("GRAPH.EFFECT", GRAPH_ID, buf)

    def test01_create_node_with_no_properties(self):
        self._send(payload(rec_create_node(
            count  = 2,
            labels = [0],
            attrs  = [],                       # no properties at all
            ids    = id_list(seg_range(1, 2)),
            values = [],
        )))

        res = self.graph.query("MATCH (n:L) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 3)

    def test02_create_edge_with_no_properties(self):
        self._send(payload(rec_add_schema(SCHEMA_EDGE, 0, "R")))
        self._send(payload(rec_create_edge(
            count  = 2,
            r      = 0,
            attrs  = [],
            ids    = id_list(seg_range(0, 2)),
            src    = id_list(seg_repeat(0, 2)),
            dst    = id_list(seg_range(1, 2)),
            values = [],
        )))

        res = self.graph.query(
            "MATCH (a)-[e:R]->(b) RETURN id(a), id(b) ORDER BY id(b)")
        self.env.assertEquals(res.result_set, [[0, 1], [0, 2]])


class testLabelChangeWithNoLabels():
    """A label record naming no labels must be a NO-OP, not a refusal.

    A Rust master emits this today: `MATCH (n) SET n:Foo REMOVE n:Foo` empties
    the label vector while leaving the entry, and their emitter has no guard
    against digesting it.

    Refusing it meant divergence, and the same buffer is refused identically on
    every retry - a forced-resync LOOP against a live peer, which presents as a
    flapping replica rather than a wrong answer. Decode already accepted the
    same bytes, so the two halves of the C implementation also disagreed.
    """

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env()
        self.conn  = self.env.getConnection()
        self.graph = Graph(self.conn, GRAPH_ID)
        self.graph.query("CREATE (:L {v: 1}), (:L {v: 2})")   # nodes 0, 1

    def _send(self, buf):
        self.conn.execute_command("GRAPH.EFFECT", GRAPH_ID, buf)

    def test01_set_labels_with_empty_label_set(self):
        self._send(payload(rec_set_labels(
            count  = 2,
            labels = [],                       # the record Rust emits
            ids    = id_list(seg_range(0, 2)),
        )))

        # accepted, and nothing changed
        res = self.graph.query("MATCH (n:L) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 2)

    def test02_remove_labels_with_empty_label_set(self):
        # the same guard covers REMOVE_LABELS. Rust's remove path cannot go
        # empty today, but that is an accident of their having two code paths,
        # and one guard is safer than a rule depending on it.
        self._send(payload(rec_remove_labels(
            count  = 2,
            labels = [],
            ids    = id_list(seg_range(0, 2)),
        )))

        res = self.graph.query("MATCH (n:L) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 2)

    def test03_the_server_is_still_usable_afterwards(self):
        # the point of the bug was a resync loop, so what matters is that the
        # instance is still alive and applying after those records - a refusal
        # would have taken it down before this ran.
        #
        # A NEW label deliberately, not one the nodes already carry. An earlier
        # version re-set an existing label and failed, which turned out not to
        # be a fault in the no-op at all.
        #
        # Applying a label a node ALREADY HAS drifts the label STATISTIC, not
        # the matrix: Graph_LabelNode's two Delta_Matrix_setElement_BOOL calls
        # are idempotent, but GraphStatistics_IncNodeCount after them is
        # unconditional, and Graph_LabeledNodeCount is a straight stats read
        # rather than a matrix scan. So over two nodes `count(n)` answers 3
        # while `id(n)` correctly returns two rows - one counter wrong, no
        # duplicate anywhere.
        #
        # (An earlier note here said the matrix held a duplicate. It does not,
        # and that would be the more serious bug - worth stating plainly so
        # nobody goes looking in the wrong structure.)
        #
        # The query path filters to genuinely-new labels before calling in, so
        # it never triggers it; the effects path passes the record through.
        # Reported, and not worked around here - the non-idempotent function is
        # Graph_LabelNode, which is shared, and the spec's "label add is an
        # idempotent set operation" is a claim about exactly that function.
        self._send(payload(rec_add_schema(SCHEMA_NODE, 1, "Fresh")))
        self._send(payload(rec_set_labels(
            count  = 1,
            labels = [1],                      # a label neither node has
            ids    = id_list(seg_range(0, 1)),
        )))

        res = self.graph.query("MATCH (n:Fresh) RETURN id(n)")
        self.env.assertEquals(res.result_set, [[0]])

        # and the pre-existing labels are untouched
        res = self.graph.query("MATCH (n:L) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 2)


class testEdgeBatchBoundary():
    """Cross the bulk-flush boundary.

    Edge creation goes through GraphHub_CreateEdges one batch at a time rather
    than one edge at a time - the singular call measured 16x slower on a
    payload 3.5x smaller. Every other edge test here creates two edges, so none
    of them reaches the flush inside the loop.

    4097 is deliberate: one full batch plus a remainder of one, which is where
    an off-by-one in the flush or in the wire-id alignment would show.
    """

    BATCH = 4096

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env()
        self.conn  = self.env.getConnection()
        self.graph = Graph(self.conn, GRAPH_ID)
        self.graph.query("CREATE (), ()")      # nodes 0 and 1

    def test01_one_record_spanning_two_batches(self):
        n = self.BATCH + 1

        self.conn.execute_command("GRAPH.EFFECT", GRAPH_ID,
                payload(rec_add_schema(SCHEMA_EDGE, 0, "R")))

        # every edge out of node 0 into node 1: the supernode shape, so both
        # endpoint columns are a single Repeat and the whole record is tiny
        self.conn.execute_command("GRAPH.EFFECT", GRAPH_ID,
                payload(rec_create_edge(
                    count  = n,
                    r      = 0,
                    attrs  = [],
                    ids    = id_list(seg_range(0, n)),
                    src    = id_list(seg_repeat(0, n)),
                    dst    = id_list(seg_repeat(1, n)),
                    values = [],
                )))

        res = self.graph.query("MATCH ()-[e:R]->() RETURN count(e)")
        self.env.assertEquals(res.result_set[0][0], n)

        # the ids must be exactly 0..n-1 with none lost or duplicated across
        # the flush - a wire-id misalignment would have been refused, but a
        # dropped remainder would not
        res = self.graph.query(
            "MATCH ()-[e:R]->() RETURN min(id(e)), max(id(e)), count(e)")
        self.env.assertEquals(res.result_set, [[0, n - 1, n]])


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

class testFutureVersionRefused(_RefusedCase):
    """A version above this build's read ceiling must be refused.

    Carried over from test_effects_malformed.py, which was removed with the v2
    reader-hardening work. Eight of its nine cases went with that topic; this
    one did not belong to it - it tests the version dispatch, which is what
    raising EFFECTS_VERSION to 3 changed, so it matters MORE here than it did
    there.

    FUTURE_EFFECTS_VERSION is deliberately a separate constant rather than
    EFFECTS_VERSION + 1. Deriving it coupled the test to the ceiling: when C
    learned to read v3, "one above what this file writes" silently became a
    version the build supports, and the payload was refused for the right
    reason under the wrong name. Raise this whenever the ceiling rises.
    """

    def test_refused(self):
        self._refuse(_u8(FUTURE_EFFECTS_VERSION) + _u32(EFFECT_ADD_ATTRIBUTE),
                     "a version above this build's read ceiling",
                     expect_log="version mismatch")

class testRemoveAllKeepsIndexesConsistent():
    """`SET n = {}` must leave the indexes in the same state whichever path
    applied it.

    v3 has no remove-all sentinel: the ruling is that the emitter resolves to
    end state and states every removed attribute explicitly as a T_NULL. That
    means a remove-all never reaches v2's `attr_id == ATTRIBUTE_ID_ALL` branch,
    which calls Schema_RemoveNodeFromIndex - it takes the per-attribute path
    instead, which RE-INDEXES. Re-index and remove are not obviously the same
    operation, and this codebase has been bitten by exactly that distinction
    before: a deletion path that was never reached left entries orphaned in
    HNSW, so a KNN query kept returning them.

    So this compares the two paths on index CONTENTS rather than graph
    contents, which is what a test asserting node counts would miss. Both a
    range index and a vector index, because the earlier bug was vector-only.

    Measured: they agree. The per-attribute re-index does remove the entry.
    """

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def _range_graph(self, name):
        g = Graph(self.conn, name)
        g.query("CREATE INDEX FOR (n:P) ON (n.v)")
        g.query("CREATE (:P {v: 5})")
        return g

    def _vec_graph(self, name):
        g = Graph(self.conn, name)
        g.create_node_vector_index("P", "emb", dim=2,
                                   similarity_function="euclidean")
        g.query("CREATE (:P {emb: vecf32([1.0, 1.0])})")
        return g

    def _remove_all_via_effect(self, name):
        # one UPDATE_NODE naming the attribute with a T_NULL value - the shape
        # the ruling says a remove-all arrives as
        self.conn.execute_command("GRAPH.EFFECT", name, payload(
            rec_update_node(count=1, labels=[0], attrs=[0],
                            ids=id_list(seg_range(0, 1)),
                            values=[v_null()])))

    def test01_range_index(self):
        # control first: the lookup must actually USE the index, or the test
        # proves nothing. GRAPH.EXPLAIN shows "Node By Index Scan".
        q = self._range_graph("ra_query")
        plan = self.conn.execute_command("GRAPH.EXPLAIN", "ra_query",
                "MATCH (n:P) WHERE n.v = 5 RETURN count(n)")
        self.env.assertContains("Index Scan", "\n".join(plan))
        self.env.assertEquals(
            q.query("MATCH (n:P) WHERE n.v = 5 RETURN count(n)").result_set[0][0], 1)

        q.query("MATCH (n:P) SET n = {}")
        via_query = q.query("MATCH (n:P) WHERE n.v = 5 RETURN count(n)").result_set[0][0]

        e = self._range_graph("ra_effect")
        self._remove_all_via_effect("ra_effect")
        via_effect = e.query("MATCH (n:P) WHERE n.v = 5 RETURN count(n)").result_set[0][0]

        self.env.assertEquals(via_effect, via_query)
        self.env.assertEquals(via_effect, 0)
        # the node itself survives - only its attributes went
        self.env.assertEquals(
            e.query("MATCH (n:P) RETURN count(n)").result_set[0][0], 1)

    def test02_vector_index(self):
        knn = ("CALL db.idx.vector.queryNodes('P', 'emb', 3, vecf32([1.0,1.0])) "
               "YIELD node RETURN count(node)")

        q = self._vec_graph("va_query")
        self.env.assertEquals(q.query(knn).result_set[0][0], 1)
        q.query("MATCH (n:P) SET n = {}")
        via_query = q.query(knn).result_set[0][0]

        e = self._vec_graph("va_effect")
        self.env.assertEquals(e.query(knn).result_set[0][0], 1)
        self._remove_all_via_effect("va_effect")
        via_effect = e.query(knn).result_set[0][0]

        # a stale HNSW entry would show up here as a non-zero count
        self.env.assertEquals(via_effect, via_query)
        self.env.assertEquals(via_effect, 0)
        self.env.assertEquals(
            e.query("MATCH (n:P) RETURN count(n)").result_set[0][0], 1)

