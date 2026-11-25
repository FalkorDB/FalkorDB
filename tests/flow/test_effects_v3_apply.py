import time
from common import *
from graph_utils import graph_eq

# DOES THE REPLICA APPLY WHAT THE MASTER SENT, or does it refuse and recover?
#
# A v3 payload is a LIST of records applied in order, and the emitter decides
# that order. These assert that the order it picks is one a replica can apply,
# for the shapes where the question is open: a query that both creates and
# deletes (C's DataBlock hands a freed id straight back, so one payload names
# one id for two entities), the structural pairs a clause, MERGE or FOREACH can
# break, and a label or a staged update written either side of a reversal.
#
# COMPARING THE TWO GRAPHS IS NOT ENOUGH, and that is the design here. A
# replica that refuses an effect resyncs and ends up holding the master's data
# anyway, so a state comparison passes either way. The discriminating assertion
# is the master's 'sync_full' counter -- except for the update and label cases,
# which apply cleanly and leave the replica quietly wrong, so those compare the
# data instead.
#
# NOT ASSERTED: which ids the query hands out. C and Rust may disagree on
# id(n); what has to agree is a master and its own replica.


# ONE CLASS PER SCENARIO, and that is not tidiness. RLTest gives a class one
# master/replica pair, and a refusal does not leave the pair as it found it: the
# replica resyncs, and after that a later write in the same class can replicate
# by a path that never exercises the claim. A scenario that can be masked by the
# one before it is not coverage, so each gets its own pair.


class EffectsV3ApplyBase():
    # not collected: RLTest takes classes whose name begins with 'test'

    GRAPH = None

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        # EFFECTS_THRESHOLD 0 forces every write onto the effects path rather
        # than letting a small one replicate verbatim; EFFECTS_VERSION 3
        # because C still emits v2 by default.
        self.env, self.db = Env(env='oss', useSlaves=True,
                moduleArgs='EFFECTS_THRESHOLD 0 EFFECTS_VERSION 3')

        self.master  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()

        self.master_graph  = Graph(self.master,  self.GRAPH)
        self.replica_graph = Graph(self.replica, self.GRAPH)

        self.master.wait(1, 0)

    # run 'q' on the master and assert the replica took it as an effect
    def _applied_without_resync(self, q, what):
        before = self.master.info()["sync_full"]

        self.master_graph.query(q)
        self.master.wait(1, 400)

        # A refusal is not synchronous with the write, so the wait ends on a
        # positive signal instead of the deadline: sync_full is incremented
        # when the master STARTS a full sync, so a replica already holding the
        # new data while the counter has not moved can only have applied the
        # effect.
        deadline = time.time() + 5
        while time.time() < deadline:
            if self.master.info()["sync_full"] > before:
                break
            if graph_eq(self.master_graph, self.replica_graph):
                break
            time.sleep(0.1)

        after = self.master.info()["sync_full"]

        self.env.assertEqual(after, before,
                message=f"{what}: the replica forced a full resync instead of "
                        f"applying the effect (sync_full {before} -> {after}). "
                        f"It recovers the data either way, so the graphs will "
                        f"still match -- the resync is the only evidence the "
                        f"record was refused")

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph),
                message=f"{what}: replica and master disagree")

    # the ids a label actually holds, in order, from both sides
    def _ids(self, label):
        q = f"MATCH (n:{label}) RETURN id(n) ORDER BY id(n)"
        m = [r[0] for r in self.master_graph.ro_query(q).result_set]
        r = [r[0] for r in self.replica_graph.ro_query(q).result_set]
        return m, r

    # the ids held under a relationship type, from both sides
    def _edge_ids(self, rel):
        q = f"MATCH ()-[e:{rel}]->() RETURN id(e) ORDER BY id(e)"
        m = [r[0] for r in self.master_graph.ro_query(q).result_set]
        r = [r[0] for r in self.replica_graph.ro_query(q).result_set]
        return m, r

    # the ids really were handed back inside the one query -- stated, because
    # if the allocator stops reusing them the payload no longer names one id
    # twice and the scenario silently tests nothing
    def _assert_reused(self, freed, created, what):
        self.env.assertTrue(len(set(freed) & set(created)) > 0,
                message=f"{what}: the query did not reuse any of the ids it "
                        f"freed ({sorted(freed)} freed, {sorted(created)} "
                        f"created), so this payload never named one id twice "
                        f"and the ordering it tests was not exercised")


class testDeleteAndCreateInOneQuery(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_delete_create"

    def test01_a_delete_and_a_create_in_one_query_apply_without_a_resync(self):
        # The new nodes land on the ids the delete released, so the payload
        # names each of them twice. Creates-first ships CREATE for an id the
        # replica still has live, and apply stops at the first failure.
        self._applied_without_resync(
                "UNWIND range(0, 49) AS i CREATE (:Before {v: i})",
                "delete-and-create setup")

        freed = sorted(r[0] for r in self.master_graph.ro_query(
            "MATCH (n:Before) WHERE n.v % 2 = 0 RETURN id(n)").result_set)
        self.env.assertEqual(len(freed), 25)

        # one query doing both halves, which is the whole point: split across
        # two queries these are two payloads and nothing can be reordered past
        # anything
        self._applied_without_resync("""
            MATCH (n:Before) WHERE n.v % 2 = 0
            DELETE n
            CREATE (:After)
        """, "a delete and a create in one query")

        created, replica_created = self._ids("After")
        self.env.assertEqual(created, replica_created,
                message="the ids under :After differ between master and replica")

        self._assert_reused(freed, created, "a delete and a create in one query")


class testCreateAndDeleteTheSameEntity(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_create_delete_same"

    def test01_an_entity_created_and_deleted_in_one_query(self):
        # THE OTHER HALF OF THE ARGUMENT. Every other class deletes first and
        # creates afterwards, so a deletes-first emitter would carry them all.
        # Here the entity is created and then destroyed, so the CREATE has to
        # reach the replica first.
        self._applied_without_resync(
                "UNWIND range(0, 9) AS i CREATE (n:Gone {v: i}) DELETE n",
                "an entity created and deleted in one query")

        for side, graph in (("master", self.master_graph),
                            ("replica", self.replica_graph)):
            self.env.assertEqual(
                    graph.ro_query(
                        "MATCH (n:Gone) RETURN count(n)").result_set[0][0],
                    0, message=f"the :Gone nodes survived on the {side}")

        # The bare shape leaves an empty graph, which a replica that applied
        # nothing also leaves -- so this one creates, deletes what it created,
        # and creates again onto the freed ids, leaving survivors to compare.
        self._applied_without_resync(
                "UNWIND range(0, 9) AS i CREATE (:Keep {v: i})",
                "create-delete-create setup")

        self._applied_without_resync("""
            MATCH (k:Keep)
            CREATE (t:Temp {v: k.v})
            DELETE t
            CREATE (s:Survivor {v: k.v})
        """, "an entity created, deleted and its id created over")

        survivors, replica_survivors = self._ids("Survivor")
        self.env.assertEqual(survivors, replica_survivors,
                message="the ids under :Survivor differ between master and "
                        "replica")
        self.env.assertEqual(len(survivors), 10)

        for side, graph in (("master", self.master_graph),
                            ("replica", self.replica_graph)):
            self.env.assertEqual(
                    graph.ro_query(
                        "MATCH (n:Temp) RETURN count(n)").result_set[0][0],
                    0, message=f"the :Temp nodes survived on the {side}")

        # twenty nodes over ten :Keep ids, so survivors below 20 are the ids
        # :Temp held -- which is what makes the payload name one id in a
        # CREATE, a DELETE and a CREATE again
        self.env.assertTrue(max(survivors) < 20,
                message=f"the :Survivor nodes did not take the ids :Temp "
                        f"freed ({survivors}), so this payload never named "
                        f"one id in both directions")


class testTwoCreateClausesEitherSideOfADelete(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_create_delete_create"

    def test01_two_create_clauses_either_side_of_a_delete(self):
        # Both CREATE clauses share a shape, so they would be one record --
        # whose first entity predates the delete, while it holds the ids the
        # delete freed. Group-level appearance order does not cover this.
        self._applied_without_resync(
                "UNWIND range(0, 29) AS i CREATE (:Victim {v: i})",
                "create-delete-create setup")

        freed = sorted(r[0] for r in self.master_graph.ro_query(
            "MATCH (n:Victim) RETURN id(n)").result_set)
        self.env.assertEqual(len(freed), 30)

        self._applied_without_resync("""
            MATCH (n:Victim)
            CREATE (:Fresh {tag: 'first'})
            DELETE n
            CREATE (:Fresh {tag: 'second'})
        """, "a create either side of a delete")

        created, replica_created = self._ids("Fresh")
        self.env.assertEqual(created, replica_created,
                message="the ids under :Fresh differ between master and replica")

        self._assert_reused(freed, created, "a create either side of a delete")


class testDeleteWithIncidentEdges(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_incident_edges"

    def test01_a_detach_delete_and_a_create_in_one_query(self):
        # Edges before nodes on the way out, with id reuse on top: the edges
        # are implicit, collected from the nodes being deleted, and the creates
        # afterwards take the freed node ids.
        self._applied_without_resync("""
            UNWIND range(0, 19) AS i
            CREATE (:Hub {v: i})-[:LINK {v: i}]->(:Spoke {v: i})
        """, "incident-edge setup")

        freed_nodes = sorted(r[0] for r in self.master_graph.ro_query(
            "MATCH (n:Hub) RETURN id(n)").result_set)
        self.env.assertEqual(len(freed_nodes), 20)

        self._applied_without_resync("""
            MATCH (n:Hub)
            DELETE n
            CREATE (:Rebuilt {v: n.v})
        """, "a detach delete and a create in one query")

        created, replica_created = self._ids("Rebuilt")
        self.env.assertEqual(created, replica_created,
                message="the ids under :Rebuilt differ")

        self.env.assertEqual(
                self.master_graph.ro_query(
                    "MATCH ()-[e:LINK]->() RETURN count(e)").result_set[0][0],
                0, message="the implicit edges were not deleted")

        self._assert_reused(freed_nodes, created,
                "a detach delete and a create in one query")


class testCreateEdgesAcrossTwoClauses(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_two_edge_clauses"

    def test01_two_create_clauses_sharing_a_relationship_type(self):
        # Nodes before edges on the way in, across clauses: both clauses
        # create the same relationship type, so the second edge wants the
        # record the first clause opened -- which is ahead of the record
        # creating its endpoints.
        #
        # A GUARD, green on the emitter this change replaces: nothing here
        # switches direction. It rules out emitting groups in opening order.
        self._applied_without_resync("""
            UNWIND range(0, 19) AS i
            CREATE (:L {v: i})-[:E {v: i}]->(:L {v: i})
            CREATE (:M {v: i})-[:E {v: i}]->(:M {v: i})
        """, "two create clauses sharing a relationship type")

        m, r = self._edge_ids("E")
        self.env.assertEqual(m, r, message="the ids under :E differ")
        self.env.assertEqual(len(m), 40,
                message=f"expected forty edges, got {len(m)}")


class testMerge(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_merge"

    def test01_a_merge_and_a_delete_in_one_query(self):
        # MERGE commits through the same path as CREATE but decides per row
        # whether to create at all, so the payload interleaves what the match
        # found with what the create made.
        self._applied_without_resync(
                "UNWIND range(0, 19) AS i CREATE (:Seed {v: i})",
                "merge setup")

        freed = sorted(r[0] for r in self.master_graph.ro_query(
            "MATCH (n:Seed) WHERE n.v % 2 = 0 RETURN id(n)").result_set)
        self.env.assertEqual(len(freed), 10)

        self._applied_without_resync("""
            MATCH (n:Seed) WHERE n.v % 2 = 0
            DELETE n
            MERGE (:Merged {v: n.v})-[:M]->(:MergedTo {v: n.v})
        """, "a merge and a delete in one query")

        created, replica_created = self._ids("Merged")
        self.env.assertEqual(created, replica_created,
                message="the ids under :Merged differ")

        m, r = self._edge_ids("M")
        self.env.assertEqual(m, r, message="the ids under :M differ")

        self._assert_reused(freed, created, "a merge and a delete in one query")


class testForeach(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_foreach"

    def test01_a_foreach_that_deletes_and_creates(self):
        # FOREACH drains its body once over the whole record list, so its
        # write ops commit the way a top-level clause does -- but the body's
        # clauses still interleave.
        self._applied_without_resync(
                "UNWIND range(0, 19) AS i CREATE (:Loop {v: i})",
                "foreach setup")

        freed = sorted(r[0] for r in self.master_graph.ro_query(
            "MATCH (n:Loop) RETURN id(n)").result_set)
        self.env.assertEqual(len(freed), 20)

        self._applied_without_resync("""
            MATCH (n:Loop)
            WITH collect(n) AS nodes
            FOREACH (n IN nodes |
                DELETE n
                CREATE (:LoopOut {v: 1})
            )
        """, "a foreach that deletes and creates")

        created, replica_created = self._ids("LoopOut")
        self.env.assertEqual(created, replica_created,
                message="the ids under :LoopOut differ")

        self._assert_reused(freed, created,
                "a foreach that deletes and creates")


class testSeveralSwitchesInOnePayload(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_switches"

    def test01_two_delete_create_pairs_in_one_query(self):
        # MORE THAN ONE SWITCH IN A PAYLOAD: each clause pair deletes a label
        # and creates another on the ids it just freed. What this catches that
        # the single-switch classes do not is a rule that reverses once and
        # then stops -- a flag rather than a counter.
        #
        # It is NOT about two creates sharing a record: :B and :D are different
        # shapes and could never be one group. That case is
        # testTwoCreateClausesEitherSideOfADelete.
        #
        # WRITTEN AS CHAINED CLAUSES AND NOT `CALL {}`, deliberately: on C a
        # writing subquery acts on the first outer row only -- Delete, Merge
        # and MergeCreate register no reset callback (op_delete.c:239,
        # op_merge.c:94, op_merge_create.c:77) and OpBase_PropagateReset skips
        # an op that has none (op.c:233). Twenty :Outer nodes through
        # `CALL { WITH n DELETE n CREATE (:Inner) }` leave nineteen and one,
        # where Rust gives none and twenty. Pre-existing, unrelated to the
        # emitter, and a test built on it would prove nothing. Do not add it.
        self._applied_without_resync(
                "UNWIND range(0, 9) AS i CREATE (:A {v: i})",
                "switches setup, first label")
        self._applied_without_resync(
                "UNWIND range(0, 9) AS i CREATE (:C {v: i})",
                "switches setup, second label")

        freed_a = sorted(r[0] for r in self.master_graph.ro_query(
            "MATCH (n:A) RETURN id(n)").result_set)
        freed_c = sorted(r[0] for r in self.master_graph.ro_query(
            "MATCH (n:C) RETURN id(n)").result_set)
        self.env.assertEqual(len(freed_a), 10)
        self.env.assertEqual(len(freed_c), 10)

        self._applied_without_resync("""
            MATCH (a:A)
            DELETE a
            CREATE (:B)
            WITH count(*) AS x
            MATCH (c:C)
            DELETE c
            CREATE (:D)
        """, "two delete-create pairs in one query")

        for label, freed in (("B", freed_a), ("D", freed_c)):
            m, r = self._ids(label)
            self.env.assertEqual(m, r,
                    message=f"the ids under :{label} differ between master "
                            f"and replica")
            self._assert_reused(freed, m,
                    f"the clause pair that created :{label}")

        for label in ("A", "C"):
            self.env.assertEqual(
                    self.master_graph.ro_query(
                        f"MATCH (n:{label}) RETURN count(n)").result_set[0][0],
                    0, message=f"the :{label} nodes were not all deleted, so "
                               f"this query made fewer switches than it looks")


class testUpdateAcrossAReusedId(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_update_reuse"

    def test01_an_update_either_side_of_a_reused_id(self):
        # THE ONE FAILURE HERE THAT IS NOT A REFUSAL. Both SETs name the same
        # id -- the second node took the id the delete freed -- for two
        # different entities. Merged, the second node's property is written to
        # the node about to be deleted and never reaches the one it belongs to.
        # Nothing refuses that payload, so the property comparison catches it,
        # not the resync counter.
        self._applied_without_resync(
                "UNWIND range(0, 19) AS i CREATE (:Old {v: i})",
                "update-across-reuse setup")

        freed = sorted(r[0] for r in self.master_graph.ro_query(
            "MATCH (n:Old) RETURN id(n)").result_set)
        self.env.assertEqual(len(freed), 20)

        # 'v' is carried across the delete in a WITH: reading n.v afterwards
        # gives null and the second update would not be in the payload at all
        self._applied_without_resync("""
            MATCH (n:Old)
            WITH n, n.v AS v
            SET n.touched = true
            DELETE n
            CREATE (m:New)
            SET m.w = v * 10
        """, "an update either side of a reused id")

        created, replica_created = self._ids("New")
        self.env.assertEqual(created, replica_created,
                message="the ids under :New differ")

        self._assert_reused(freed, created,
                "an update either side of a reused id")

        # THE PROPERTY, from both sides. A replica that applied the merged
        # record holds :New nodes with no properties at all while the master's
        # carry 'w'
        q = "MATCH (n:New) RETURN id(n), n.w ORDER BY id(n)"
        m = [tuple(r) for r in self.master_graph.ro_query(q).result_set]
        r = [tuple(r) for r in self.replica_graph.ro_query(q).result_set]
        self.env.assertEqual(m, r,
                message="the ':New' nodes carry different properties on the "
                        "replica: the update was applied to the entity that "
                        "used to hold the id")
        self.env.assertTrue(all(row[1] is not None for row in m),
                message=f"the master did not set 'w' on every :New node "
                        f"({m}), so this write never produced the pair of "
                        f"updates the test is for")


class testALabelTakenOffAndPutBack(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_label_reversal"

    def test01_a_label_removed_and_set_again_in_one_query(self):
        # THE SAME FAULT AS A REUSED ID, ONE LAYER UP: SET_LABELS is stage 4
        # and REMOVE_LABELS is 5, so a run sorted by stage emits
        # `REMOVE n:Z SET n:Z` backwards and the replica loses a label.
        #
        # The unit test aLabelPutBackOpensANewRun is the deterministic guard --
        # the records came out inverted every time on the old emitter, but the
        # replica diverged in only four of six measured instances. Hence five
        # passes here, so one agreement cannot hide the fault.
        self._applied_without_resync(
                "CREATE (:S:Z {k: 1})", "label reversal setup")

        for i in range(5):
            self._applied_without_resync(
                    "MATCH (n {k: 1}) REMOVE n:Z SET n:Z",
                    f"a label removed and set again, pass {i}")

            m = self.master_graph.ro_query(
                    "MATCH (n {k: 1}) RETURN labels(n)").result_set[0][0]
            r = self.replica_graph.ro_query(
                    "MATCH (n {k: 1}) RETURN labels(n)").result_set[0][0]
            self.env.assertEqual(sorted(m), sorted(r),
                    message=f"pass {i}: the labels differ -- master {m}, "
                            f"replica {r}")
            self.env.assertTrue("Z" in m,
                    message=f"pass {i}: the master lost :Z, so the query no "
                            f"longer ends with the label set and this shape "
                            f"tests nothing")


class testUpdateOfADeletedNode(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_update_delete"

    def test01_an_update_of_a_node_the_query_deletes(self):
        # An update joins a record only at the flush, after every create and
        # delete has been filed; appended there it would follow the delete of
        # the node it names.
        #
        # A GUARD: the old emitter sorts updates (2) ahead of deletes (7) and
        # gets this right. It is here so the run mechanism does not break it.
        self._applied_without_resync(
                "UNWIND range(0, 19) AS i CREATE (:Doomed {v: i})",
                "update-then-delete setup")

        self._applied_without_resync("""
            MATCH (n:Doomed)
            SET n.w = n.v * 2
            DELETE n
        """, "an update of a node the query deletes")

        self.env.assertEqual(
                self.master_graph.ro_query(
                    "MATCH (n:Doomed) RETURN count(n)").result_set[0][0],
                0, message="the nodes were not deleted")
