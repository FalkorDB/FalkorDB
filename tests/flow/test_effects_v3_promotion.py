import time
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
