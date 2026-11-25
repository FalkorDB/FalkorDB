import time
from common import *
from graph_utils import graph_eq

# DOES THE REPLICA APPLY WHAT THE MASTER SENT, or does it refuse and recover?
#
# A v3 payload is a LIST of records applied in order, and the emitter decides
# that order. These assert that the order it picks is one a replica can apply,
# for the query shapes where the question is open:
#
#   * a query that deletes and creates in one statement. C's DataBlock hands a
#     freed id straight back, so one payload names the same id in a DELETE and
#     in a CREATE, for two different entities. Ordering by opcode cannot carry
#     that - creates-first ships the create of an id the replica still has, and
#     deletes-first breaks the same pair written the other way round.
#
#   * the structural pairs: a node before the edge hanging off it on the way
#     in, the edge before the node on the way out. Those hold within a run, and
#     the shapes below are the ones where a clause, a MERGE or a FOREACH can
#     file an entity into a record its endpoints are not in yet.
#
#   * a staged property update on either side of a reused id. That one does
#     not refuse: the payload applies cleanly and leaves the replica a
#     property short, so it is caught by comparing properties rather than by
#     the resync counter.
#
# COMPARING THE TWO GRAPHS IS NOT ENOUGH, and that is the whole design here. A
# replica that refuses an effect logs the divergence and forces a full resync,
# after which it holds the master's data anyway -- so a state comparison passes
# either way, just more slowly. It is the recovery that makes the bug invisible.
# The assertion that discriminates is the master's 'sync_full' counter: a
# legitimate write has to be APPLIED, not recovered from.
#
# WHAT IS DELIBERATELY NOT ASSERTED: which ids the query hands out. C reuses an
# id inside the query that freed it and Rust does not, and the two engines are
# allowed to disagree about id(n). What has to agree is the master and its own
# replica, which is what the id comparisons below check.


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

        # A refusal is not synchronous with the write: the replica logs it,
        # drops the link and reconnects, so the counter moves some time after
        # the write returns and racing it would pass on a broken build.
        #
        # THE WAIT ENDS ON A POSITIVE SIGNAL, not on the deadline. sync_full
        # is incremented when the master STARTS a full sync, before it sends
        # anything -- so a replica already holding the new data while the
        # counter has not moved can only have got there by applying the
        # effect. Waiting out the five seconds on every green call cost this
        # file a minute and a half of pure sleeping.
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

    # the master handed the freed ids back inside the one query, which is the
    # precondition the ordering has to survive
    #
    # STATED RATHER THAN ASSUMED: if the allocator stops reusing ids within a
    # query the payload no longer names one id twice, and the scenario silently
    # becomes a test of nothing.
    def _assert_reused(self, freed, created, what):
        self.env.assertTrue(len(set(freed) & set(created)) > 0,
                message=f"{what}: the query did not reuse any of the ids it "
                        f"freed ({sorted(freed)} freed, {sorted(created)} "
                        f"created), so this payload never named one id twice "
                        f"and the ordering it tests was not exercised")


class testDeleteAndCreateInOneQuery(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_delete_create"

    def test01_a_delete_and_a_create_in_one_query_apply_without_a_resync(self):
        # THE SHAPE AN OPCODE ORDER CANNOT CARRY. C's DataBlock hands a query
        # back the ids it just freed, so the new nodes land on the ids the
        # delete released and the payload names each of them twice. With the
        # creates staged ahead of the deletes the replica meets CREATE_NODE for
        # an id it still has live, refuses the payload and resyncs -- and apply
        # stops at the first failure, so the delete behind it is never reached.
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


class testTwoCreateClausesEitherSideOfADelete(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_create_delete_create"

    def test01_two_create_clauses_either_side_of_a_delete(self):
        # BOTH CREATE CLAUSES SHARE A SHAPE, so they would be one record -- and
        # that record's first entity predates the delete, so it is emitted
        # ahead of it while holding the ids the delete freed. Appearance order
        # at the group level does not cover this; the emitter has to stop the
        # second clause joining the first clause's record at all.
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
        # EDGES BEFORE NODES ON THE WAY OUT, with the id reuse on top: the
        # edges are implicit (collected from the nodes being deleted) and the
        # creates afterwards take the freed node ids. A replica that meets
        # DELETE_NODE before the DELETE_EDGE that unhooks it refuses, and so
        # does one that meets the creates first.
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
        # NODES BEFORE EDGES ON THE WAY IN, across clauses. Both clauses create
        # the same relationship type, so both edges want one record -- and that
        # record was opened by the first clause, which is ahead of the record
        # creating the second clause's endpoints. _ApplyCreateEdge checks that
        # both endpoints exist before it claims anything, so the payload is
        # refused.
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
        # MERGE creates through the same commit path as CREATE -- nodes, then
        # edges -- but it decides per row whether to create at all, so the
        # payload interleaves whatever the match found with whatever the
        # create made. Run alongside a delete of the same label, it is the
        # create half that lands on the freed ids.
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
        # FOREACH drains its body ONCE over the whole record list, so its write
        # ops commit the way a top-level clause does -- but the body's clauses
        # still interleave, and the delete inside it frees ids the create
        # inside it is handed back.
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
        # MORE THAN ONE SWITCH IN A PAYLOAD. Each clause pair deletes a label
        # and creates another on the ids it just freed, so one payload carries
        # four stretches and names two sets of ids twice over. A rule that
        # separated the first delete from the first create but then let the
        # second create rejoin the first one's record would pass the
        # single-switch cases above and fail here.
        #
        # WRITTEN AS CHAINED CLAUSES AND NOT AS `CALL {}`, deliberately. A
        # writing subquery looks like the natural way to get many switches,
        # but on C it acts on the first outer row only: Delete, Merge and
        # MergeCreate register no reset callback (op_delete.c:239,
        # op_merge.c:94, op_merge_create.c:77) and OpBase_PropagateReset skips
        # an op that has none (op.c:233), so rows 2..N find DeleteConsume
        # already in handoff mode. Measured: twenty :Outer nodes through
        # `CALL { WITH n DELETE n CREATE (:Inner) }` leave nineteen :Outer and
        # one :Inner, where Rust gives 0 and 20. That is a pre-existing engine
        # defect with nothing to do with the emitter, and a test built on it
        # would either fail for the wrong reason or pass without exercising
        # anything. Do not put it back.
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
        # THE ONE FAILURE IN THIS FILE THAT IS NOT A REFUSAL. Both SETs name
        # the same id -- the second node was handed the id the delete freed --
        # for two different entities. Staged under the id alone they merge into
        # one record carrying the FIRST entity's labels and position, so the
        # second node's property is written to the node about to be deleted and
        # never reaches the node it belongs to.
        #
        # Nothing refuses that payload. It is well formed, every record
        # applies, sync_full does not move and no divergence is logged: the
        # replica is simply a property short. So the assertion that catches it
        # is the property comparison, not the resync counter.
        self._applied_without_resync(
                "UNWIND range(0, 19) AS i CREATE (:Old {v: i})",
                "update-across-reuse setup")

        freed = sorted(r[0] for r in self.master_graph.ro_query(
            "MATCH (n:Old) RETURN id(n)").result_set)
        self.env.assertEqual(len(freed), 20)

        # 'v' is carried across the delete in a WITH: reading n.v after the
        # DELETE gives null and sets nothing, which would leave the second
        # update out of the payload altogether
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


class testUpdateOfADeletedNode(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_update_delete"

    def test01_an_update_of_a_node_the_query_deletes(self):
        # AN UPDATE IS STAGED, not filed on arrival: its shape is the entity's
        # whole attribute set, which is not known until the query stops writing
        # to it, so it joins a record at the very end. Appended there it would
        # follow the delete of the node it names.
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
