"""Effects v3 -- schema DDL as effects: constraints and indexes announced, converged, dropped, and settled under races.

See `effects_v3_common.py` for the shared fixture and why these are split.
"""

import time


from common import *
from constraint_utils import (create_mandatory_node_constraint, create_unique_node_constraint, drop_unique_node_constraint, list_constraints)
from index_utils import (create_node_range_index, drop_node_range_index, list_indicies, wait_for_indices_to_sync)

from effects_v3_common import _EffectsV3Base


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

        # Not just "one effect" — one effect that is the constraint. Counting
        # GRAPH.EFFECT lines cannot tell a constraint announcement from a node
        # create, so the opcode is what makes this an assertion about
        # constraints at all.
        # One announcement, and it already carries the settled status — there
        # is no UNDER CONSTRUCTION phase to observe when validation ran inline.
        self.env.assertEqual(
            self.constraint_announcements(window, self.GRAPH_ID),
            [('MANDATORY', 'NODE', 'OPERATIONAL')])
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

        self.env.assertEqual(
            self.leading_opcodes(window, self.GRAPH_ID), ['DROP_CONSTRAINT'])
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
        # UNIQUE builds its supporting index first, so the window opens with a
        # CREATE_INDEX. What matters is that the failure is on the wire — the
        # replica installs FAILED because it was told, not because it re-ran a
        # scan and reached the same conclusion.
        self.env.assertEqual(
            self.constraint_announcements(window, self.GRAPH_ID),
            [('UNIQUE', 'NODE', 'FAILED')])

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

        # The whole property, read off the wire: two announcements, the first
        # saying the constraint is still being built and the second saying it
        # is enforcing. Counting payloads would pass on any two effects;
        # checking the opcode would pass if both said OPERATIONAL, which is the
        # bug where the command announces a status it has not reached yet.
        #
        # MANDATORY needs no supporting index, so these two are the whole
        # window.
        self.env.assertEqual(
            self.constraint_announcements(window, self.GRAPH_ID),
            [('MANDATORY', 'NODE', 'UNDER_CONSTRUCTION'),
             ('MANDATORY', 'NODE', 'OPERATIONAL')])
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
# 5. an index, a constraint that depends on it, and the drops
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
# 5b. *how* index DDL reaches the replica, not just whether it did
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
        # The 2 is the x=1 node from `_build` plus the one just created.
        #
        # This assertion has a recorded oddity, deliberately left in place. It
        # has been observed failing as `[[1]] == [[2]]`, and every observation
        # is on a build with `IdSpace::refuse_recycled` stubbed out: 2 sightings
        # across 4 stubbed runs, against 0 across 8 serial runs on a clean
        # build. It also never fires when this class runs alone, only after the
        # earlier classes have.
        #
        # What that means is NOT established -- nobody has reproduced it on
        # demand, and no mechanism linking this count to that guard has been
        # shown. Recorded as an observation on purpose, because guessing at the
        # mechanism here and writing the guess down as a reason would send the
        # next person after the wrong thing.
        #
        # So: if this fails, check what build you are on before you touch the
        # test. And do not "fix" the fact that it depends on the classes before
        # it -- that dependency is the only condition under which it has ever
        # been seen to fail, so it is evidence, not a defect.
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
