"""Effects v3 -- the commands around a write: `GRAPH.RECORD`, and the ones that still replicate verbatim.

See `effects_common.py` for the shared fixture and why these are split.
"""

import time


from common import *
from constraint_utils import (create_unique_node_constraint)
from graph_utils import graph_eq
from index_utils import (create_node_range_index, list_indicies, wait_for_indices_to_sync)

from effects_common import _EffectsBase


class testEffects_04d_RecordCommand(_EffectsBase):
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

    GRAPH_ID = "effects_record"

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


class testEffects_04e_VerbatimCommands(_EffectsBase):
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

    GRAPH_ID = "effects_verbatim"

    def __init__(self):
        self._setup()

    def _nothing_was_refused(self, full_before, diverged_before, failures_before):
        self.env.assertEqual(
            self.master.info()["sync_full"], full_before,
            message="an effect was refused after a verbatim command, forcing a resync")
        # `replica_log_diverged` is None where the server logs are unreadable
        # (CI's services mode). Comparing None to None asserts nothing, so the
        # refusal count carries the check in that mode — it comes back over the
        # connection and is available everywhere.
        self.env.assertEqual(self.effect_failures(), failures_before,
            message="a GRAPH.EFFECT was refused after a verbatim command")
        diverged_now = self.replica_log_diverged()
        if diverged_now is not None:
            self.env.assertEqual(diverged_now, diverged_before)

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
        failures_before = self.effect_failures()

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
        self._nothing_was_refused(full_before, diverged_before, failures_before)

    def test02_delete_reaches_the_replica_and_the_rebuild_replays(self):
        # A graph key deleted on the master must go on the replica too, and the
        # writes that recreate it must land in the *new* graph rather than
        # against the dictionaries of the old one.
        self.set_effects_config()
        key = self.GRAPH_ID + "_copy"
        full_before = self.master.info()["sync_full"]
        diverged_before = self.replica_log_diverged()
        failures_before = self.effect_failures()

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
        self._nothing_was_refused(full_before, diverged_before, failures_before)

    def test03_udf_libraries_reach_the_replica(self):
        # UDFs are process state rather than graph state, so they have no
        # effect record and never will. They must still replicate, because a
        # promoted replica has to be able to answer a query that calls one.
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]
        diverged_before = self.replica_log_diverged()
        failures_before = self.effect_failures()

        script = """
        function Doubled (x) { return x * 2; }
        falkor.register ('Doubled', Doubled);
        """
        self.db.udf_load("EffectsV3Udf", script, True)
        self.wait_for_replica_offset()

        self.env.assertContains("EffectsV3Udf", self.udf_names(self.master))
        # `GRAPH.UDF` carries Redis's `write` flag for every subcommand, LIST
        # included, so reading the replica's libraries needs the read-only flag
        # lifted — the same dance `testEffects_05b_IndexDDLMechanism` does for
        # GRAPH.EXPLAIN, and lifted only for the read.
        self.env.assertContains("EffectsV3Udf", self.replica_udf_names())
        # and it is callable there, not merely listed
        self.assert_agree("RETURN EffectsV3Udf.Doubled(21)", [[42]])

        self.db.udf_delete("EffectsV3Udf")
        self.wait_for_replica_offset()
        self.env.assertEqual(self.replica_udf_names(),
                             self.udf_names(self.master))
        self.env.assertTrue("EffectsV3Udf" not in self.replica_udf_names())
        self._nothing_was_refused(full_before, diverged_before, failures_before)
