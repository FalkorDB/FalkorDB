"""Effects v3 -- one query, several kinds of record in one buffer.

Every other class in this suite drives one kind of change per query. This one is
about what a *commit* carries when a single statement produces more than one
record kind, and whether the replica applies them in an order that works.

**Index and constraint DDL cannot share a COMMIT with data**, and that is a
parser fact rather than an omission: `Parser::parse` tries
`parse_index_ops()` first and, if it matches, that *is* the whole query --
`expect_end_of_input()` then rejects anything after it, and a query carrying
more than one statement is refused outright ("query with more than one statement
is not supported"). `GRAPH.CONSTRAINT` is a command, not a clause. So there is
no `CREATE INDEX ... CREATE (n)` to test. `GRAPH.RECORD` is not a way round it
either -- it runs *a* query through the same parser.

That bounds what can share a buffer. It does **not** bound the sequence: DDL and
data land in consecutive commits all the time, and the replica has to apply them
in an order that works. Those are the `test06`-`test09` cases below, and they are
the ones where an ordering bug would actually show up in production.

What *is* compound, and what this file covers:

  - schema and data together: a label, relationship type or attribute that did
    not exist yet arrives in the same buffer as the rows that reference it, and
    the ordering is load-bearing -- a replica that applied `CREATE_NODE` before
    `ADD_SCHEMA` would be resolving a label id it does not hold
  - several mutation kinds at once: create, update, label change and delete from
    one `MATCH`
  - data mutation with an index present, so index maintenance rides the same
    commit as the rows it indexes
  - data mutation with a constraint present, violating and not

A note on what these can assert. `payload_opcodes` reads the opcode a payload
*leads with* and stops -- it cannot walk a multi-record buffer without knowing
each record's wire shape, which would duplicate the codec. So an ordering claim
here is "the buffer leads with the schema", plus convergence for the rest. That
is the strongest available statement without a second decoder in the test suite,
and it is the half that catches the failure that matters: schema arriving after
the rows that need it.
"""

import time

from common import *
from index_utils import (create_node_range_index, list_indicies,
                         wait_for_indices_to_sync)
from constraint_utils import create_unique_node_constraint, list_constraints

from effects_v3_common import _EffectsV3Base


class testEffectsV3_04g_CompoundSequences(_EffectsV3Base):
    """Several record kinds from one statement, applied in an order that works."""

    GRAPH_ID = "effects_v3_compound"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT')

    def test01_schema_and_data_ride_one_buffer(self):
        # A label, a relationship type and three attributes that do not exist
        # yet, all introduced by the statement that first uses them. The replica
        # has to intern every name before it can resolve the ids the node and
        # edge records carry, so the schema records must lead.
        self.set_effects_config()
        self.monitor_mark()
        self.query_and_sync(
            "CREATE (:Fresh {alpha: 1, beta: 'two'})"
            "-[:FRESHLY {gamma: 3.5}]->(:Fresh {alpha: 2, beta: 'three'})")
        window = self.monitor_mark()

        # One commit, so one payload -- and it leads with schema, not with the
        # node it describes. That is the assertion: a buffer that led with
        # CREATE_NODE would name a label id the replica has not interned.
        leading = self.leading_opcodes(window, self.GRAPH_ID)
        self.env.assertTrue(
            leading and leading[0] in ('ADD_SCHEMA', 'ADD_ATTRIBUTE'),
            message=f"a buffer introducing new schema must lead with it, led with {leading}")

        self.assert_agree("MATCH (n:Fresh) RETURN count(n)", [[2]])
        self.assert_agree(
            "MATCH (:Fresh)-[e:FRESHLY]->(:Fresh) RETURN count(e), sum(e.gamma)",
            [[1, 3.5]])
        self.assert_agree(
            "MATCH (n:Fresh) RETURN count(n.alpha), count(n.beta)", [[2, 2]])
        self.assert_graph_eq()

    def test02_create_update_label_and_delete_in_one_statement(self):
        # Four mutation kinds from one MATCH: a node created, an existing one
        # updated, a label added to a third, a fourth deleted. Whatever order the
        # writer partitions them into, both sides must end up agreeing -- and the
        # deleted node must not come back as a side effect of the create.
        self.set_effects_config()
        self.query_and_sync(
            "UNWIND range(1, 4) AS i CREATE (:Multi {i: i, tag: 'start'})")

        self.query_and_sync("""
            MATCH (a:Multi {i: 1}), (b:Multi {i: 2}), (c:Multi {i: 3})
            CREATE (:Multi {i: 5, tag: 'added'})
            SET a.tag = 'updated'
            SET b:Extra
            DELETE c
        """)

        self.assert_agree("MATCH (n:Multi) RETURN count(n)", [[4]])
        self.assert_agree("MATCH (n:Multi {tag: 'updated'}) RETURN count(n)", [[1]])
        self.assert_agree("MATCH (n:Extra) RETURN count(n)", [[1]])
        self.assert_agree("MATCH (n:Multi {i: 3}) RETURN count(n)", [[0]])
        self.assert_agree("MATCH (n:Multi {i: 5}) RETURN count(n)", [[1]])
        self.assert_graph_eq()

    def test03_index_maintenance_rides_the_same_commit(self):
        # With an index already present, a statement that creates, updates and
        # deletes indexed rows carries all of that in one commit. The replica's
        # index must answer for the rows as they finally are, not as any
        # intermediate step left them.
        self.set_effects_config()
        create_node_range_index(self.master_graph, 'Indexed', 'k', sync=True)
        self.wait_for_replica_offset()
        self.query_and_sync(
            "UNWIND range(1, 6) AS i CREATE (:Indexed {k: i, keep: true})")

        self.query_and_sync("""
            MATCH (n:Indexed) WHERE n.k <= 3
            SET n.k = n.k + 100
        """)
        self.query_and_sync("MATCH (n:Indexed) WHERE n.k = 6 DELETE n")

        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)

        # Read through the index on both sides. A stale replica index answers
        # these with the pre-update values and is the failure this catches.
        self.assert_agree("MATCH (n:Indexed) WHERE n.k = 101 RETURN count(n)", [[1]])
        self.assert_agree("MATCH (n:Indexed) WHERE n.k = 1 RETURN count(n)", [[0]])
        self.assert_agree("MATCH (n:Indexed) WHERE n.k = 6 RETURN count(n)", [[0]])
        self.assert_agree(
            "MATCH (n:Indexed) WHERE n.k > 100 RETURN count(n)", [[3]])
        self.assert_agree("MATCH (n:Indexed) RETURN count(n)", [[5]])
        self.env.assertEqual(list_indicies(self.replica_graph).result_set,
                             list_indicies(self.master_graph).result_set)
        self.assert_graph_eq()

    def test04_a_violating_write_beside_a_valid_one_commits_neither(self):
        # A UNIQUE constraint, then one statement whose rows are individually
        # fine and collectively are not. The write is refused whole, so the
        # replica must not have been told about the half that would have
        # succeeded -- a partial buffer here is divergence, not a partial write.
        self.set_effects_config()
        self.query_and_sync("CREATE (:Uniq {u: 1})")
        create_unique_node_constraint(self.master_graph, 'Uniq', 'u', sync=True)
        self.wait_for_replica_offset()
        self.wait_for_constraint_settled(self.master_graph, 'Uniq')

        full_before = self.master.info()["sync_full"]
        failures_before = self.effect_failures()

        rejected = None
        try:
            self.master_graph.query(
                "CREATE (:Uniq {u: 2}), (:Uniq {u: 1})")
        except Exception as e:
            rejected = str(e).lower()
        self.env.assertTrue(
            rejected is not None and "unique constraint violation" in rejected,
            message=f"the duplicate must be refused, got {rejected!r}")

        self.wait_for_replica_offset()
        # Neither row survives: the valid one was in the same statement.
        self.assert_agree("MATCH (n:Uniq) RETURN count(n)", [[1]])
        self.assert_agree("MATCH (n:Uniq {u: 2}) RETURN count(n)", [[0]])
        # And nothing was refused on the replica -- a refused write should send
        # nothing, rather than sending something the replica then rejects.
        self.env.assertEqual(self.effect_failures(), failures_before,
            message="a refused write still put a buffer on the wire")
        self.env.assertEqual(self.master.info()["sync_full"], full_before,
            message="a refused write forced a resync")
        self.assert_graph_eq()

    def test05_new_schema_and_a_delete_of_the_same_label_in_one_statement(self):
        # The shape `digest_cancelled` exists for, one level up: a label
        # introduced and then emptied inside one statement. The schema addition
        # is real and must survive even though no row that used it does.
        self.set_effects_config()
        self.query_and_sync(
            "CREATE (:Doomed {only: 1}) WITH 1 AS x "
            "MATCH (d:Doomed) DELETE d")

        self.assert_agree("MATCH (n:Doomed) RETURN count(n)", [[0]])
        # The label is interned on both sides even with no rows carrying it, so
        # a later create resolves the same id rather than minting a second.
        self.query_and_sync("CREATE (:Doomed {only: 2})")
        self.assert_agree("MATCH (n:Doomed) RETURN count(n), sum(n.only)", [[1, 2]])
        self.assert_graph_eq()

    # ── DDL and data in consecutive commits ───────────────────────────────
    #
    # Not one buffer -- the parser forbids that -- but one *sequence*, which is
    # what a replica actually sees. Each of these is two or more commits whose
    # order is load-bearing on the far side.

    def test06_an_index_created_then_populated_answers_on_the_replica(self):
        # DDL first, data second. The replica applies CREATE_INDEX, then the
        # CREATE_NODE records for rows the index must contain. If the index
        # arrived but the rows were indexed against the pre-index state, the
        # replica's index answers short.
        self.set_effects_config()
        self.query_and_sync("CREATE INDEX FOR (n:Seq) ON (n.k)")
        self.query_and_sync(
            "UNWIND range(1, 20) AS i CREATE (:Seq {k: i, pad: 'x'})")
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)

        self.assert_agree("MATCH (n:Seq) WHERE n.k = 7 RETURN count(n)", [[1]])
        self.assert_agree("MATCH (n:Seq) WHERE n.k > 15 RETURN count(n)", [[5]])
        self.env.assertEqual(list_indicies(self.replica_graph).result_set,
                             list_indicies(self.master_graph).result_set)
        self.assert_graph_eq()

    def test07_rows_created_then_indexed_are_in_the_replica_index(self):
        # The other order, and the harder one: the rows exist first and the
        # index has to be built over committed state on BOTH sides. A replica
        # that creates an empty index and never backfills answers zero.
        self.set_effects_config()
        self.query_and_sync(
            "UNWIND range(1, 20) AS i CREATE (:Backfill {k: i})")
        self.query_and_sync("CREATE INDEX FOR (n:Backfill) ON (n.k)")
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)

        self.assert_agree("MATCH (n:Backfill) WHERE n.k = 3 RETURN count(n)", [[1]])
        self.assert_agree(
            "MATCH (n:Backfill) WHERE n.k >= 10 RETURN count(n)", [[11]])
        self.env.assertEqual(list_indicies(self.replica_graph).result_set,
                             list_indicies(self.master_graph).result_set)
        self.assert_graph_eq()

    def test08_a_dropped_index_stops_answering_on_both_sides(self):
        # DROP_INDEX carries no options and is the one DDL record with nothing
        # in it but the target. Writes after the drop must still land, and
        # neither side may keep answering from an index it no longer has.
        self.set_effects_config()
        self.query_and_sync("CREATE INDEX FOR (n:Dropped) ON (n.k)")
        self.query_and_sync(
            "UNWIND range(1, 10) AS i CREATE (:Dropped {k: i})")
        wait_for_indices_to_sync(self.master_graph)
        self.query_and_sync("DROP INDEX FOR (n:Dropped) ON (n.k)")
        self.query_and_sync("CREATE (:Dropped {k: 99})")

        self.env.assertEqual(list_indicies(self.replica_graph).result_set,
                             list_indicies(self.master_graph).result_set)
        self.assert_agree("MATCH (n:Dropped) WHERE n.k = 99 RETURN count(n)", [[1]])
        self.assert_agree("MATCH (n:Dropped) RETURN count(n)", [[11]])
        self.assert_graph_eq()

    def test09_a_constraint_then_writes_it_governs(self):
        # CREATE_CONSTRAINT, then writes the replica must evaluate against it:
        # one that passes and one that the master refuses. The replica must end
        # up holding the constraint AND exactly the rows that survived it.
        self.set_effects_config()
        self.query_and_sync("CREATE (:Gov {u: 1})")
        create_unique_node_constraint(self.master_graph, 'Gov', 'u', sync=True)
        self.wait_for_replica_offset()
        self.wait_for_constraint_settled(self.master_graph, 'Gov')

        self.query_and_sync("CREATE (:Gov {u: 2})")

        rejected = None
        try:
            self.master_graph.query("CREATE (:Gov {u: 2})")
        except Exception as e:
            rejected = str(e).lower()
        self.env.assertTrue(
            rejected is not None and "unique constraint violation" in rejected,
            message=f"the duplicate must be refused, got {rejected!r}")
        self.wait_for_replica_offset()

        self.assert_agree("MATCH (n:Gov) RETURN count(n)", [[2]])
        # `list_constraints` returns a list, not a result-set wrapper -- unlike
        # `list_indicies` right above it, which is exactly the kind of asymmetry
        # that bites once per test suite.
        self.env.assertEqual(len(list_constraints(self.replica_graph)),
                             len(list_constraints(self.master_graph)))
        self.assert_graph_eq()
