import time
import struct
from common import *

GRAPH_ID = "effects_v3_promotion"

# A promoted replica has to allocate its own ids, and they must not collide with
# the ones it was handed.
#
# WHY THIS SHAPE RATHER THAN RUST'S. Their testEffectsV3_07_PromotedReplica is
# about a constraint caught UNDER CONSTRUCTION: a replica installs the status the
# primary announced, which is right while it is a replica and wrong the moment it
# is promoted. That is a real case and it is not this one. The intention behind
# putting promotion on C's list was id allocation -- a node that has spent its
# life accepting the master's ids and then has to mint its own -- and that is a
# different mechanism reached through the same event.
#
# It is also where the known failure lives. A node id is reserved before a write
# commits and the reservation is released if the write is cancelled, but a
# cancelled reservation is never replicated: the replica's allocator does not
# learn the id was handed back. Promote that replica and its next allocation can
# name an id the graph is already using, which shows up as two nodes fused into
# one rather than as an error.
#
# WHAT THIS DOES NOT COVER, so nobody reads it as more than it is. Rust's
# constraint-settling race is not ported: it needs a million nodes and a ~400ms
# validation window to promote into, and C is separately known to abort during
# async constraint validation when the topology is disturbed. Attempting it here
# would confuse a crash in that area with a promotion defect. The
# emit-after-promotion half is also only partly covered -- see test03.


class testEffectsV3Promotion():
    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env(env='oss', useSlaves=True,
                moduleArgs='EFFECTS_THRESHOLD 0 EFFECTS_VERSION 3')

        self.master  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()

        # remember the link so it can be put back: a later test, or a rerun in a
        # reused env, expects a replica
        info = self.replica.info("replication")
        self.master_host = info.get("master_host")
        self.master_port = info.get("master_port")

        self.replica.config_set("slave-read-only", "no")

    def _wait_link(self, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.replica.info("replication").get("master_link_status") == "up":
                return
            time.sleep(0.2)
        raise Exception("replica never reported master_link_status: up")

    def _sync(self):
        # WAIT is the only thing that says the replica ACKED this write. Reading
        # the replica without it is the documented way to see an empty graph and
        # conclude replication is broken.
        self.master.execute_command("WAIT", "1", "0")

    def _ids(self, con, label):
        g = Graph(con, GRAPH_ID)
        return sorted(r[0] for r in
                g.ro_query(f"MATCH (n:{label}) RETURN id(n)").result_set)

    def _restore(self):
        try:
            if self.master_host:
                self.replica.execute_command(
                        "REPLICAOF", self.master_host, self.master_port)
                self._wait_link()
        except Exception:
            pass

    def test01_a_promoted_replica_allocates_ids_that_do_not_collide(self):
        try:
            self._promote_and_write()
        finally:
            self._restore()

    def _promote_and_write(self):
        self._wait_link()

        g = Graph(self.master, GRAPH_ID)

        # the replica spends this phase accepting ids it did not allocate
        g.query("UNWIND range(1, 500) AS i CREATE (:P {i: i})")
        self._sync()

        before = self._ids(self.replica, "P")
        self.env.assertEqual(len(before), 500,
                message=f"the replica holds {len(before)} of 500 nodes before "
                        f"promotion, so it never applied the effects and "
                        f"nothing below is about promotion")

        # a cancelled write, which is where the reservation asymmetry lives: the
        # master reserves an id, the write fails, the master releases it, and
        # the release is not replicated because a failed write emits nothing
        try:
            g.query("CREATE (:P {i: 1}) WITH 1 AS x "
                    "UNWIND [1, 0] AS d RETURN 1/d")
        except Exception:
            pass
        self._sync()

        # PROMOTE
        self.replica.execute_command("REPLICAOF", "NO", "ONE")

        deadline = time.time() + 30
        while time.time() < deadline:
            if self.replica.info("replication").get("role") == "master":
                break
            time.sleep(0.2)
        self.env.assertEqual(
                self.replica.info("replication").get("role"), "master",
                message="the replica never became a master")

        # now it has to mint its own
        pg = Graph(self.replica, GRAPH_ID)
        pg.query("UNWIND range(1, 200) AS i CREATE (:Q {i: i})")

        after_p = self._ids(self.replica, "P")
        after_q = self._ids(self.replica, "Q")

        # THE ASSERTION THAT MATTERS. A reused id does not surface as an error;
        # it surfaces as two entities occupying one slot, so the count of
        # DISTINCT ids is what catches it rather than any exception.
        overlap = set(after_p) & set(after_q)
        self.env.assertEqual(len(overlap), 0,
                message=f"the promoted node allocated {len(overlap)} id(s) that "
                        f"its inherited nodes already hold: {sorted(overlap)[:8]}"
                        f" -- two entities now share a slot")

        self.env.assertEqual(len(after_p), 500,
                message=f"promotion cost inherited nodes: {len(after_p)} of 500 "
                        f"remain, so an allocation overwrote one")
        self.env.assertEqual(len(after_q), 200,
                message=f"{len(after_q)} of 200 new nodes are addressable")
        self.env.assertEqual(len(set(after_p) | set(after_q)), 700,
                message="700 nodes do not occupy 700 distinct ids")

    def test02_the_promoted_node_still_answers_reads_consistently(self):
        # A fused pair is invisible to a count and visible to a property read:
        # both entities answer, and one of them answers with the other's
        # properties. So this checks the values rather than the cardinality.
        try:
            self._wait_link()
            g = Graph(self.master, GRAPH_ID)
            g.query("UNWIND range(1, 300) AS i CREATE (:R {i: i, s: 'r' + i})")
            self._sync()

            self.replica.execute_command("REPLICAOF", "NO", "ONE")
            time.sleep(1)

            pg = Graph(self.replica, GRAPH_ID)
            pg.query("UNWIND range(1, 300) AS i CREATE (:S {i: i, s: 's' + i})")

            for label, pfx in (("R", "r"), ("S", "s")):
                rows = pg.ro_query(
                        f"MATCH (n:{label}) RETURN n.i, n.s ORDER BY n.i"
                        ).result_set
                self.env.assertEqual(len(rows), 300,
                        message=f"{label}: {len(rows)} of 300 rows")
                bad = [r for r in rows if r[1] != f"{pfx}{r[0]}"]
                self.env.assertEqual(len(bad), 0,
                        message=f"{label}: {len(bad)} node(s) answer with "
                                f"another entity's properties, e.g. {bad[:4]} "
                                f"-- the signature of a reused id")
        finally:
            self._restore()

    def test03_what_is_not_covered_here(self):
        # Deliberately a statement rather than a test, because an absent case
        # that nobody wrote down gets read as a passing one.
        #
        # NOT COVERED: that a promoted node EMITS correctly. Checking that needs
        # a third node to receive from it, or the old master re-attached as its
        # replica -- a role swap this Env's two nodes can express but which
        # interacts with the resync that promotion already triggered, so it
        # deserves its own class rather than a tail on this one.
        #
        # NOT COVERED: Rust's constraint-settling race. A replica promoted while
        # holding a constraint UNDER CONSTRUCTION must finish validating it
        # itself, and until it does, writes that should be rejected are not.
        # That needs a validation window wide enough to promote into -- theirs
        # uses a million nodes for ~400ms -- and C is separately known to abort
        # during async constraint validation when the topology is disturbed, so
        # a failure here would be ambiguous between the two.
        #
        # NOT COVERED, AND THIS ONE IS SUBTLER, because test01 looks like it
        # covers it. test01 issues a cancelled write before promoting, aiming at
        # the reservation asymmetry: an id is reserved, the write fails, the
        # master releases it, and the release is not replicated. Measured on
        # this build, the promoted node's first minted id was exactly
        # max(inherited) + 1 -- 500 after ids 0..499. So it derived its
        # allocator from the graph rather than from any inherited reservation
        # state, which is collision-free whatever the reservation history was.
        #
        # That means test01 establishes that promotion allocates without
        # collision on this build, and does NOT establish that the asymmetry is
        # absent: graph-derived allocation would mask it. Probing it needs the
        # master's allocator left AHEAD of the graph's max id at the moment of
        # promotion, and the cancelled write here does not achieve that -- the
        # minted range shows no gap. Written down because the step is in the
        # test and a later reader would reasonably assume it does something.
        self.env.assertTrue(True)


#-------------------------------------------------------------------------------
# a constraint inherited UNDER CONSTRUCTION is settled on promotion
#-------------------------------------------------------------------------------

# Hand-built v3 records, kept to the minimum this class needs.
#
# Building wire bytes in Python is the pattern that was removed from
# test_effects_v3.py, and for a good reason: a Python encoder written from the
# spec can agree with a C decoder that reads the spec the same wrong way, and
# both pass. That objection does not apply HERE, because these bytes are not the
# thing under test - they are how the class reaches a state, and the very next
# assertion checks the state was reached. A malformed payload is refused, the
# constraint never appears, and the class fails at _inherit_under_construction
# rather than reporting a false result about promotion.
_EFFECTS_VERSION          = 3
_EFFECT_CREATE_CONSTRAINT = 13

def _u8(v):  return struct.pack('<B', v)
def _u16(v): return struct.pack('<H', v)
def _u32(v): return struct.pack('<I', v)
def _i32(v): return struct.pack('<i', v)
def _u64(v): return struct.pack('<Q', v)

def _string(s):
    # length is a BYTE COUNT and includes the terminator
    raw = s.encode() + b'\x00'
    return _u64(len(raw)) + raw

def _payload(*records, flags=0):
    return _u8(_EFFECTS_VERSION) + _u8(flags) + b''.join(records)

def _rec_create_constraint(ct, et, status, label_id, label, props):
    # the property count is a u8 here, not the u16 the index field list uses
    body = _u32(ct) + _u32(et) + _u32(status) + _i32(label_id) \
        + _string(label) + _u8(len(props))
    for aid, name in props:
        body += _u16(aid) + _string(name)
    return _u32(_EFFECT_CREATE_CONSTRAINT) + body


class testPromotionSettlesInheritedConstraint():
    """A constraint inherited UNDER CONSTRUCTION is settled on promotion.

    A v3 replica installs the constraint status off the wire and never
    validates. That is correct while it is a replica and wrong the instant it is
    promoted: the old primary was still building this constraint and now nobody
    is. Nothing else settles it either -- the primary re-announces only on
    CT_ACTIVE (indexer.c:230, which is already on origin/master, so the stuck
    window predates v3), and the constraint enforces while it sits there,
    rejecting writes against a rule it never checked.

    THE PAIRING IS THE TEST. Both cases promote identically and differ only in
    whether a row violates the constraint: a violating row must settle FAILED,
    clean data must settle OPERATIONAL. A handler that stamped a status rather
    than validating -- everything FAILED, or everything ACTIVE -- passes exactly
    one of them. Neither case alone distinguishes validating from stamping, so
    deleting either one silently guts the class.
    """

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env()
        self.conn  = self.env.getConnection()
        self.graph = Graph(self.conn, GRAPH_ID + "_constraint")

    def _status(self):
        res = self.graph.query(
            "CALL db.constraints() YIELD status RETURN status")
        return [row[0] for row in res.result_set]

    def _inherit_under_construction(self, data):
        # the state the primary's FIRST announcement leaves behind: status
        # PENDING (1), which the replica adopts without validating
        self.graph.query(data)
        self.conn.execute_command("GRAPH.EFFECT", self.graph.name, _payload(
            _rec_create_constraint(ct=1, et=1, status=1,
                label_id=0, label="Q", props=[(0, "title")])))

        # this assertion is also what makes the hand-built bytes safe: if the
        # payload were malformed it is refused here, not mistaken for a result
        self.env.assertEquals(self._status(), ["UNDER CONSTRUCTION"])

    def _promote(self):
        # a role change to master needs only this instance: REPLICAOF at a port
        # nothing is listening on makes it a replica with no link, and NO ONE
        # promotes it straight back, firing NOW_MASTER
        port = self.env.envRunner.port
        self.conn.execute_command("REPLICAOF", "127.0.0.1", str(port + 1))
        self.conn.execute_command("REPLICAOF", "NO", "ONE")

        # settling is queued to the indexer pool rather than done on the event
        # thread, so the status changes after the command returns
        for _ in range(100):
            s = self._status()
            if s and s[0] != "UNDER CONSTRUCTION":
                break
            time.sleep(0.1)

    def test01_violating_row_settles_failed(self):
        self._inherit_under_construction(
            "CREATE (:Q {title: 'x'}), (:Q {other: 1})")
        self._promote()
        self.env.assertEquals(self._status(), ["FAILED"])

    def test02_clean_data_settles_operational(self):
        self.graph.delete()
        self._inherit_under_construction(
            "CREATE (:Q {title: 'x'}), (:Q {title: 'y'})")
        self._promote()
        self.env.assertEquals(self._status(), ["OPERATIONAL"])
