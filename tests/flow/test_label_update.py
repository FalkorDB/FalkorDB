from common import *
from graph_utils import graph_eq
from constraint_utils import create_unique_node_constraint, wait_on_constraint

GRAPH_ID = "label_update"

class testLabelUpdate():
    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True)
        self.master = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        self.master_graph = Graph(self.master, GRAPH_ID)
        self.replica_graph = Graph(self.replica, GRAPH_ID)

        # enable effects replication
        self.db.config_set("EFFECTS_THRESHOLD", 0)

    def tearDown(self):
        self.master_graph.delete()
        self.replica_graph = Graph(self.replica, GRAPH_ID)
        self.master.execute_command("WAIT", "1", "0")

    def query_master_and_wait(self, query):
        res = self.master_graph.query(query)
        self.master.execute_command("WAIT", "1", "0")
        return res

    def test_large_update_set(self):
        # Exercise the "batch" path in staged_updates.c _RemoveRedundancies.
        # REDUNDANCY_ITER_THRESHOLD is 512, so creating 10x that many nodes
        # forces _RemoveRedundancies to take the bulk-iteration branch for
        # both the add-label and remove-label directions.
        REDUNDANCY_ITER_THRESHOLD = 512
        node_count = REDUNDANCY_ITER_THRESHOLD * 10 + 1  # range is inclusive

        # Create nodes (fix: removed stray leading quote from original)
        query = f"UNWIND range(0, {REDUNDANCY_ITER_THRESHOLD * 10}) AS i CREATE ()"
        res = self.query_master_and_wait(query)
        self.env.assertEqual(res.nodes_created, node_count)

        # --- bulk ADD path ---
        res = self.query_master_and_wait("MATCH (n) SET n:L RETURN n")
        self.env.assertEqual(res.labels_added, node_count)
        # Every node must now carry label L
        res = self.query_master_and_wait("MATCH (n:L) RETURN count(n) AS c")
        self.env.assertEqual(res.result_set[0][0], node_count)

        # --- bulk REMOVE path ---
        res = self.query_master_and_wait("MATCH (n:L) REMOVE n:L")
        self.env.assertEqual(res.labels_removed, node_count)

        # No node should carry label L any longer
        res = self.query_master_and_wait("MATCH (n:L) RETURN count(n) AS c")
        self.env.assertEqual(res.result_set[0][0], 0)

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    def test_same_label_multiple_set_remove(self):
        # A single query that touches the same label in multiple SET / REMOVE
        self.query_master_and_wait("CREATE (:A {v: 1})")
        
        # --- repeated SET of the same label ---
        # Three SET n:L clauses; the label must be recorded only once.
        res = self.query_master_and_wait(
            "MATCH (n:A) SET n:L WITH n SET n:L WITH n SET n:L RETURN n"
        )
        self.env.assertEqual(res.labels_added, 1)

        res = self.query_master_and_wait("MATCH (n:A:L) RETURN count(n) AS c")
        self.env.assertEqual(res.result_set[0][0], 1)
        
        # --- repeated REMOVE of the same label ---
        res = self.query_master_and_wait(
            "MATCH (n:A:L) REMOVE n:L WITH n REMOVE n:L RETURN n"
        )
        self.env.assertEqual(res.labels_removed, 1)

        res = self.query_master_and_wait("MATCH (n:A:L) RETURN count(n) AS c")
        self.env.assertEqual(res.result_set[0][0], 0)
        
        # --- SET then REMOVE the same label within one query ---
        # Net effect on label M must be zero: it is added and immediately
        # removed
        self.query_master_and_wait("MATCH (n:A) SET n:L")   # give node L so query matches
        res = self.query_master_and_wait("MATCH (n:A:L) SET n:M WITH n REMOVE n:M RETURN n")
        self.env.assertEqual(res.labels_added, 1)
        self.env.assertEqual(res.labels_removed, 1)

        res = self.query_master_and_wait("MATCH (n:A:M) RETURN count(n) AS c")
        self.env.assertEqual(res.result_set[0][0], 0)

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    def test_remove_then_set_same_label(self):
        # Regression for #2777: a label removed and re-added in the same query
        # must survive. What this pins is the replicated end state — master and
        # replica agreeing that the node still carries L, which a stale
        # RemoveLabels record would break on the replica alone.
        #
        # It does not pin the record set on the wire: a zero-label SetLabels
        # applies as a no-op, so both sides agree whether or not one was sent.
        # That is pinned on the emitter and the staging maps, by
        # graph::runtime::pending::label_effect_tests.
        self.query_master_and_wait("CREATE (:A {v: 1})")
        self.query_master_and_wait("MATCH (n:A) SET n:L")

        # --- REMOVE then SET: the label is kept ---
        # no WITH between the clauses: both land in the same segment, so the
        # add has to cancel the staged removal rather than commit after it
        # the removal and the re-add are both counted, as C counts them: the
        # statistic is what each clause did, not the net change at commit
        res = self.query_master_and_wait("MATCH (n:A:L) REMOVE n:L SET n:L RETURN n")
        self.env.assertEqual(res.labels_removed, 1)
        self.env.assertEqual(res.labels_added, 1)

        res = self.query_master_and_wait("MATCH (n:A:L) RETURN count(n) AS c")
        self.env.assertEqual(res.result_set[0][0], 1)

        # the replica must still see the label, both in labels() and via a
        # label scan — a stale RemoveLabels record would strip it there only
        res = Graph(self.replica, GRAPH_ID).ro_query("MATCH (n:A:L) RETURN labels(n)")
        self.env.assertEqual(len(res.result_set), 1)
        self.env.assertContains("L", res.result_set[0][0])

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

        # --- SET then REMOVE: the label is dropped ---
        res = self.query_master_and_wait("MATCH (n:A:L) SET n:L REMOVE n:L RETURN n")
        self.env.assertEqual(res.labels_removed, 1)

        res = self.query_master_and_wait("MATCH (n:A:L) RETURN count(n) AS c")
        self.env.assertEqual(res.result_set[0][0], 0)

        # the node itself survives on the replica, it only lost the label
        res = Graph(self.replica, GRAPH_ID).ro_query("MATCH (n:A) RETURN labels(n)")
        self.env.assertEqual(len(res.result_set), 1)
        self.env.assertNotContains("L", res.result_set[0][0])

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    def test_redundant_label_set_remove(self):
        # A single query that touches the same label in multiple SET / REMOVE
        self.query_master_and_wait("CREATE (:A {v: 1})")
        
        # --- repeated SET of the same label ---
        # Three SET n:L clauses; the label must be recorded only once.
        res = self.query_master_and_wait("MATCH (n:A) SET n:L:L:L RETURN n")
        self.env.assertEqual(res.labels_added, 1)

        res = self.query_master_and_wait("MATCH (n:A:L) RETURN count(n) AS c")
        self.env.assertEqual(res.result_set[0][0], 1)
        
        # --- repeated REMOVE of the same label ---
        res = self.query_master_and_wait("MATCH (n:A:L) REMOVE n:L:L:L RETURN n")
        self.env.assertEqual(res.labels_removed, 1)

        res = self.query_master_and_wait("MATCH (n:A:L) RETURN count(n) AS c")
        self.env.assertEqual(res.result_set[0][0], 0)
        
        # --- SET then REMOVE the same label within one query ---
        # Net effect on label M must be zero: it is added and immediately
        # removed
        self.query_master_and_wait("MATCH (n:A) SET n:L")   # give node L so query matches

        res = self.query_master_and_wait( "MATCH (n:A:L) SET n:M:M:M WITH n REMOVE n:M:M:M RETURN n")
        self.env.assertEqual(res.labels_added, 1)
        self.env.assertEqual(res.labels_removed, 1)

        res = self.query_master_and_wait("MATCH (n:A:M) RETURN count(n) AS c")
        self.env.assertEqual(res.result_set[0][0], 0)

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    # Regression for #2655. `Labels added` / `Labels removed` count label
    # assignments (one per (node, label) pair a clause changes), not the label
    # types registered in the schema, and not the net change at commit. The
    # expected values are what the C engine reports.
    def test_labels_added_counts_assignments(self):
        # one label type, two nodes: two assignments
        res = self.master_graph.query("CREATE (:Node), (:Node)")
        self.env.assertEqual(res.labels_added, 2)

        # a known label still counts on every node
        res = self.master_graph.query("CREATE (:Node), (:Node), (:Node)")
        self.env.assertEqual(res.labels_added, 3)

        # one node, two labels
        res = self.master_graph.query("CREATE (:A:B)")
        self.env.assertEqual(res.labels_added, 2)

        # SET on many matched nodes: one per node
        res = self.master_graph.query("MATCH (n:Node) SET n:Extra")
        self.env.assertEqual(res.labels_added, 5)

        # a label the node already carries is not an assignment
        res = self.master_graph.query("MATCH (n:Node) SET n:Extra")
        self.env.assertEqual(res.labels_added, 0)

        # the same label twice in one query counts once per node
        res = self.master_graph.query("MATCH (n:Node) SET n:D:D")
        self.env.assertEqual(res.labels_added, 5)
        res = self.master_graph.query("MATCH (n:A) SET n:L WITH n SET n:L")
        self.env.assertEqual(res.labels_added, 1)

        # MERGE that creates
        res = self.master_graph.query("MERGE (:M:N)")
        self.env.assertEqual(res.labels_added, 2)

    def test_labels_removed_counts_assignments(self):
        self.master_graph.query("UNWIND range(1, 5) AS i CREATE (:Node)")
        self.master_graph.query("MATCH (n:Node) SET n:Extra")

        res = self.master_graph.query("MATCH (n:Node) REMOVE n:Extra")
        self.env.assertEqual(res.labels_removed, 5)

        # a label the node does not carry is not a removal
        res = self.master_graph.query("MATCH (n:Node) REMOVE n:Extra")
        self.env.assertEqual(res.labels_removed, 0)
        res = self.master_graph.query("MATCH (n:Node) REMOVE n:Unknown")
        self.env.assertEqual(res.labels_removed, 0)

        # removed twice in one query: the second finds nothing to remove
        self.master_graph.query("MATCH (n:Node) SET n:Extra")
        res = self.master_graph.query("MATCH (n:Node) REMOVE n:Extra REMOVE n:Extra")
        self.env.assertEqual(res.labels_removed, 5)

        # two labels on one node
        self.master_graph.query("CREATE (:L1:L2)")
        res = self.master_graph.query("MATCH (n:L1) REMOVE n:L1:L2")
        self.env.assertEqual(res.labels_removed, 2)

    def test_labels_stats_remove_and_set_in_one_query(self):
        self.master_graph.query("CREATE (:A)")

        # REMOVE then SET: both happened, even though the label survives
        res = self.master_graph.query("MATCH (n:A) REMOVE n:A SET n:A")
        self.env.assertEqual(res.labels_removed, 1)
        self.env.assertEqual(res.labels_added, 1)

        # SET then REMOVE
        res = self.master_graph.query("MATCH (n:A) SET n:B REMOVE n:B")
        self.env.assertEqual(res.labels_added, 1)
        self.env.assertEqual(res.labels_removed, 1)

        # SET, REMOVE, SET again: two additions and one removal
        res = self.master_graph.query("MATCH (n:A) SET n:W REMOVE n:W SET n:W")
        self.env.assertEqual(res.labels_added, 2)
        self.env.assertEqual(res.labels_removed, 1)

        # a node created and stripped of a label in the same query
        res = self.master_graph.query("CREATE (n:X:Y) REMOVE n:X")
        self.env.assertEqual(res.labels_added, 2)
        self.env.assertEqual(res.labels_removed, 1)

    def test_remove_nonexisting_label(self):
        # Try to remove a non existing label
        self.query_master_and_wait("CREATE (:A {v: 1})")
        
        res = self.query_master_and_wait("MATCH (n:A) REMOVE n:X RETURN n")
        self.env.assertEqual(res.labels_removed, 0)

        # make sure label 'X' wasn't added to the graph's schema
        labels = self.query_master_and_wait("CALL db.labels()").result_set[0][0]
        self.env.assertNotContains("X", labels)

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    def test_constraint_violation_rollback(self):
        # A unique-constraint violation raised inside _LabelNodes_Single must
        # cause a full rollback: no labels are permanently added to any node
        # that was processed before the violating node was reached.
        
        # Unique constraint on label C, property p.
        #self.master_graph.create_node_unique_constraint("X", "p")
        create_unique_node_constraint(self.master_graph, "X", "name", sync=True)
        wait_on_constraint(self.master_graph, "UNIQUE", "NODE", "X", "name")
        # wait for replica to activate constraint
        
        # Seed one node that already satisfies the constraint.
        self.query_master_and_wait("CREATE (:X {name: 1})")
        
        # Create several unlabelled nodes; one of them shares p=1 so that
        # when we try to bulk-label them all as C the constraint fires.
        self.query_master_and_wait("UNWIND [{name:2},{name:3},{name:1},{name:4}] AS d CREATE ({name: d.name})")
        
        try:
            # Attempt to assign label X to every unlabelled node.
            # The node with p=1 duplicates the existing :X node and must
            # trigger a constraint violation.
            self.query_master_and_wait("MATCH (n) WHERE NOT n:X SET n:X")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertContains("constraint", str(e).lower())
        
        # Rollback must restore the graph to exactly one :X node.
        res = self.query_master_and_wait("MATCH (n:X) RETURN count(n)")
        self.env.assertEqual(res.result_set[0][0], 1)
        
        # The original seed node must still be intact.
        res = self.query_master_and_wait("MATCH (n:X {name: 1}) RETURN count(n)")
        self.env.assertEqual(res.result_set[0][0], 1)

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    #def test_failed_update_on_first_attempt(self):

