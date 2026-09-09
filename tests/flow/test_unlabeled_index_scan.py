from common import *
from index_utils import *

GRAPH_ID = "unlabeled_index_scan"


class testUnlabeledIndexScan():
    """An unlabelled pattern may borrow a label that covers the whole graph.

    `MATCH (n) WHERE n.p = v` cannot normally use an index, because the index is
    per label and an unlabelled pattern matches every node. When some label is
    carried by *every* live node, though, `MATCH (n)` and `MATCH (n:L)` select
    the same set, so the optimizer may borrow L and reach its index.

    The proof is about the data, not the schema. Creating one unlabelled node
    falsifies it and does not bump `schema_version`, which is what query plans
    are cached against — so the runtime re-checks it and widens to an all-node
    scan when it no longer holds. These tests pin both halves: that the rewrite
    happens, and that it never changes an answer.
    """

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def _seed(self, n=100, label='Person'):
        self.graph.query(
            f"UNWIND range(0, {n - 1}) AS i CREATE (:{label} {{v: i}})")
        self.graph.create_node_range_index(label, 'v')
        wait_for_indices_to_sync(self.graph)

    def test01_unlabeled_lookup_uses_the_index(self):
        self._seed()
        q = "MATCH (n) WHERE n.v = 42 RETURN n.v"
        plan = str(self.graph.explain(q))
        self.env.assertIn("Node By Index Scan", plan)
        self.env.assertIn("[universal label]", plan)
        self.env.assertEquals(self.graph.query(q).result_set, [[42]])

    def test02_no_rewrite_when_a_node_is_unlabeled(self):
        """One unlabelled node defeats the proof, so the plan must not use it."""
        self._seed()
        self.graph.query("CREATE ({v: 42})")
        plan = str(self.graph.explain("MATCH (n) WHERE n.v = 42 RETURN n.v"))
        self.env.assertIn("All Node Scan", plan)
        self.env.assertNotIn("[universal label]", plan)

    def test03_unlabeled_node_is_still_found(self):
        """The whole point: borrowing a label must not lose a node.

        A node carrying `v` without the label is invisible to the label's
        index, so if the rewrite fired here the answer would be wrong.
        """
        self._seed()
        self.graph.query("CREATE ({v: 42})")
        res = self.graph.query("MATCH (n) WHERE n.v = 42 RETURN count(n)")
        self.env.assertEquals(res.result_set, [[2]])

    def test04_cached_plan_widens_when_the_proof_lapses(self):
        """The decisive case for the runtime re-check.

        The plan is compiled and cached while the label is universal, then an
        unlabelled node with the same property is added. `schema_version` does
        not change, so the *same cached plan* runs again and must still find
        both nodes.
        """
        self._seed()
        q = "MATCH (n) WHERE n.v = 7 RETURN count(n)"
        self.env.assertEquals(self.graph.query(q).result_set, [[1]])
        plan = str(self.graph.explain(q))
        self.env.assertIn("[universal label]", plan)

        self.graph.query("CREATE ({v: 7})")
        res = self.graph.query(q)
        self.env.assertTrue(res.cached_execution)
        self.env.assertEquals(res.result_set, [[2]])

    def test05_borrowed_label_survives_a_deleted_node(self):
        """Deleting the unlabelled node restores the proof, and the answer."""
        self._seed()
        self.graph.query("CREATE ({v: 7})")
        self.env.assertEquals(
            self.graph.query("MATCH (n) WHERE n.v = 7 RETURN count(n)").result_set,
            [[2]])
        self.graph.query("MATCH (n) WHERE n.v = 7 AND labels(n) = [] DELETE n")
        self.env.assertEquals(
            self.graph.query("MATCH (n) WHERE n.v = 7 RETURN count(n)").result_set,
            [[1]])

    def test06_second_label_does_not_defeat_the_proof(self):
        """A universal label may coexist with a partial one.

        Every node is `:Person`; some are additionally `:Admin`. `:Person` still
        covers the graph, so the rewrite is valid — and must pick a label that
        covers, not merely the first one it finds.
        """
        self._seed()
        self.graph.query("MATCH (n:Person) WHERE n.v < 10 SET n:Admin")
        q = "MATCH (n) WHERE n.v = 42 RETURN n.v"
        plan = str(self.graph.explain(q))
        self.env.assertIn("Node By Index Scan", plan)
        self.env.assertEquals(self.graph.query(q).result_set, [[42]])
        self.env.assertEquals(
            self.graph.query("MATCH (n) WHERE n.v = 3 RETURN count(n)").result_set,
            [[1]])

    def test07_range_and_inline_forms(self):
        """The rewrite must be correct for a range predicate and inline attrs."""
        self._seed()
        self.env.assertEquals(
            self.graph.query("MATCH (n) WHERE n.v >= 98 RETURN count(n)").result_set,
            [[2]])
        self.env.assertEquals(
            self.graph.query("MATCH (n {v: 5}) RETURN n.v").result_set, [[5]])
        self.graph.query("CREATE ({v: 99})")
        self.env.assertEquals(
            self.graph.query("MATCH (n) WHERE n.v >= 98 RETURN count(n)").result_set,
            [[3]])
        self.env.assertEquals(
            self.graph.query("MATCH (n {v: 99}) RETURN count(n)").result_set, [[2]])

    def test08_traversal_endpoint_lookup(self):
        """The shape the gap was measured on: an unlabelled traversal endpoint."""
        self._seed()
        self.graph.query(
            "MATCH (a:Person), (b:Person) WHERE a.v = 1 AND b.v = 2 "
            "CREATE (a)-[:KNOWS]->(b)")
        q = "MATCH (a)-[:KNOWS]->(b) WHERE b.v = 2 RETURN a.v"
        plan = str(self.graph.explain(q))
        self.env.assertIn("Node By Index Scan", plan)
        self.env.assertEquals(self.graph.query(q).result_set, [[1]])

    def test09_empty_graph_is_not_universal(self):
        """An empty graph has no universal label, so nothing to borrow."""
        self.graph.create_node_range_index('Person', 'v')
        wait_for_indices_to_sync(self.graph)
        plan = str(self.graph.explain("MATCH (n) WHERE n.v = 1 RETURN n"))
        self.env.assertNotIn("[universal label]", plan)
        self.env.assertEquals(
            self.graph.query("MATCH (n) WHERE n.v = 1 RETURN count(n)").result_set,
            [[0]])
