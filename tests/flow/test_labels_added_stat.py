from common import *


# Regression for #2655: the `Labels added` query statistic must count label
# ASSIGNMENTS to nodes (one per new (node, label) pair), not newly-registered
# label TYPES in the schema. The four cases below are lifted from the issue
# report and match what the C engine and Neo4j return.
class testLabelsAddedStat():
    def __init__(self):
        self.env, self.db = Env()

    def _fresh(self, name):
        g = self.db.select_graph(name)
        try:
            g.delete()
        except Exception:
            pass
        return g

    def test01_create_two_nodes_same_label(self):
        # First registration of `:Node` on a fresh graph — two assignments.
        g = self._fresh("labels_added_case1")
        res = g.query("CREATE (:Node),(:Node)")
        self.env.assertEqual(res.labels_added, 2)

    def test02_create_one_node_two_labels(self):
        # One node, two labels — two assignments.
        g = self._fresh("labels_added_case2")
        res = g.query("CREATE (:A:B)")
        self.env.assertEqual(res.labels_added, 2)

    def test03_create_more_nodes_with_known_label(self):
        # `:Node` is already registered from case 1's shape. A subsequent
        # multi-node CREATE must still report every assignment.
        g = self._fresh("labels_added_case3")
        g.query("CREATE (:Node),(:Node)")
        res = g.query("CREATE (:Node),(:Node),(:Node)")
        self.env.assertEqual(res.labels_added, 3)

    def test04_set_label_on_many_matched_nodes(self):
        # `SET n:Extra` on five matched nodes — five assignments, even though
        # `:Extra` is only registered once. This is the case that the previous
        # (buggy) counter reported as 1.
        g = self._fresh("labels_added_case4")
        g.query("UNWIND range(1, 5) AS i CREATE (:Node)")
        res = g.query("MATCH (n:Node) SET n:Extra")
        self.env.assertEqual(res.labels_added, 5)

    def test05_set_already_present_label_is_zero(self):
        # Re-setting a label the node already carries must not count.
        g = self._fresh("labels_added_case5")
        g.query("CREATE (:A)")
        res = g.query("MATCH (n:A) SET n:A")
        self.env.assertEqual(res.labels_added, 0)

    def test06_repeated_set_within_one_query_counts_once(self):
        # Two SET clauses touching the same (node, label) in one query must
        # count exactly one assignment — the earlier stage is visible to the
        # later one via pending state.
        g = self._fresh("labels_added_case6")
        g.query("CREATE (:A)")
        res = g.query("MATCH (n:A) SET n:L WITH n SET n:L RETURN n")
        self.env.assertEqual(res.labels_added, 1)
