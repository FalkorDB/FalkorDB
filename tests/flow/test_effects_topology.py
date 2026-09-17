"""Effects v3 -- the classes that take the primary/replica pair APART and put it back: a forced resync, and a promotion.

This file is the reason the others do not need to worry about topology.

See `effects_common.py` for the shared fixture and why these are split.
"""

import time


from common import *
from constraint_utils import (create_unique_node_constraint)
from index_utils import (create_node_range_index, wait_for_indices_to_sync)

from effects_common import _EffectsBase


class testEffects_06c_DivergenceForcesResync(_EffectsBase):
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

    GRAPH_ID = "effects_divergence"

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

        # The refusal this test is named for, counted before and after. Without
        # this the test passes with `refuse_recycled` deleted outright: the
        # id-space counter cross-check catches the same divergence independently
        # and resyncs, so `sync_full` moves either way and the assertion below
        # cannot tell which check fired. Measured — with `refuse_recycled`
        # stubbed to `Ok(())` the whole 73-test flow file stayed green, and only
        # a unit test noticed. Defence in depth is good; a test that documents
        # one mechanism and pins another is not.
        recycled_before = self.replica_log_count(b"already in the recycle bin")
        failures_before = self.effect_failures()

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

        # Assert the strongest thing this mode can see. With a readable log that
        # is the MECHANISM — which check refused the buffer — and it is the
        # assertion this test exists for. CI's services mode has no readable
        # log (`env.log_path` is None), so there it falls back to the outcome:
        # a GRAPH.EFFECT was refused at all. That is weaker and says so, rather
        # than passing vacuously, which is what returning 0 used to do.
        recycled_after = self.replica_log_count(b"already in the recycle bin")
        if recycled_after is None:
            self.env.assertGreater(
                self.effect_failures(), failures_before,
                message="no GRAPH.EFFECT was refused on the replica, so nothing "
                        "detected the divergence. (Server logs are unreadable in "
                        "this mode, so which check refused it is unchecked here — "
                        "run locally or in spawn mode to pin the mechanism.)")
        else:
            self.env.assertGreater(
                recycled_after, recycled_before,
                message="the resync happened, but not for the reason this test "
                        "names: no DELETE_NODE was refused for an id in the "
                        "recycle bin. Some other divergence check fired instead")

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


class testEffects_07_PromotedReplica(_EffectsBase):
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

    GRAPH_ID = "effects_promotion"

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
