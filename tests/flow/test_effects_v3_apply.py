import time
from common import *
from graph_utils import graph_eq

GRAPH_ID = "effects_v3_apply"

# DOES THE REPLICA APPLY WHAT THE MASTER SENT, or does it refuse and recover?
#
# test_effects_v3_determinism already writes the shape that matters -- several
# labels in one statement, so one query's ids are split across records and each
# record's id list is sparse in the id space. What that file asserts is that the
# MASTER emits the same bytes twice, which a replica refusing every one of those
# records satisfies perfectly. This asserts the other half, and the two together
# are what "the write replicated" means.
#
# COMPARING THE TWO GRAPHS IS NOT ENOUGH, and that is the whole design here. A
# replica that refuses an effect logs the divergence and forces a full resync,
# after which it holds the master's data anyway -- so a state comparison passes
# either way, just more slowly. It is the recovery that makes the bug invisible.
# The assertion that discriminates is the master's 'sync_full' counter: a
# legitimate write has to be APPLIED, not recovered from.
#
# Why the sparse shape is the case to cover: v3 sends one record per
# (opcode, shape) with the groups sorted by key, so CREATE (:A), (:B), (:A)
# allocates 0,1,2 and travels as ':A' holding {0, 2} and ':B' holding {1}.
# Neither record's ids are contiguous, and a replica that assumed they were --
# that a record creating n entities could only name n consecutive ids -- refuses
# the first one.


# ONE CLASS PER SCENARIO, and that is not tidiness. RLTest gives a class one
# master/replica pair, and a refusal does not leave the pair as it found it: the
# replica resyncs, and after that a later write in the same class can replicate
# by a path that never exercises the claim. Run together under a deliberately
# broken build, the edge case below passed while the two either side of it
# failed -- it was being shielded by the recovery from the first failure. A
# scenario that can be masked by the one before it is not coverage, so each gets
# its own pair.


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

        # a refusal is not synchronous with the write: the replica logs it,
        # drops the link and reconnects, so give it room to have happened
        # rather than racing it
        deadline = time.time() + 5
        while time.time() < deadline:
            if self.master.info()["sync_full"] > before:
                break
            time.sleep(0.25)

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

    # a record whose ids are contiguous would not exercise the sparse case, so
    # every scenario states which label it expects to be scattered
    def _assert_sparse(self, label):
        m, _ = self._ids(label)
        self.env.assertTrue(len(m) > 0 and max(m) - min(m) + 1 > len(m),
                message=f"the ':{label}' ids came out contiguous, so this write "
                        f"did not produce the sparse record the test is for")


class testMultiShapeCreate(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_multishape"

    def test01_a_multi_shape_create_applies_without_a_resync(self):
        # three shapes in one statement, so the ids round-robin across three
        # records and none of the three is contiguous
        self._applied_without_resync("""
            UNWIND range(0, 99) AS i
            CREATE (:A {v: i}),
                   (:B {v: i}),
                   (:A:B {v: i})
        """, "multi-shape create")

        # the ids themselves, not just the counts: a replica that re-derived
        # them would hold the right number of nodes under the wrong ids
        for label in ("A", "B"):
            m, r = self._ids(label)
            self.env.assertEqual(m, r,
                    message=f"the ids under :{label} differ between master and "
                            f"replica")

        self._assert_sparse("A")


class testInterleavedEdgeCreate(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_edges"

    def test01_interleaved_edge_ids_apply_without_a_resync(self):
        # the edge-shaped version: endpoints alternate, so the :S and :T node
        # records each hold every other id
        self._applied_without_resync("""
            UNWIND range(0, 99) AS i
            CREATE (:S {v: i})-[:R {v: i}]->(:T {v: i})
        """, "interleaved edge create")

        m, r = self._ids("S")
        self.env.assertEqual(m, r, message="the ids under :S differ")
        self._assert_sparse("S")


class testReuseAfterDelete(EffectsV3ApplyBase):
    GRAPH = "effects_v3_apply_churn"

    def test01_reuse_after_delete_applies_without_a_resync(self):
        # claiming ids off the free list rather than past the high-water mark,
        # which is the other half of the claim and the ordinary case for a
        # replica: C reuses the most recently freed id, Rust the smallest
        self._applied_without_resync(
                "UNWIND range(0, 49) AS i CREATE (:Churn {v: i})",
                "churn setup")

        self._applied_without_resync(
                "MATCH (n:Churn) WHERE n.v % 2 = 0 DELETE n",
                "churn delete")

        self._applied_without_resync("""
            UNWIND range(0, 49) AS i
            CREATE (:Churn2 {v: i}), (:Churn3 {v: i})
        """, "churn recreate over freed ids")

        m, r = self._ids("Churn2")
        self.env.assertEqual(m, r, message="the ids under :Churn2 differ")
