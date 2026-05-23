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
        self.env.assertEquals(res.nodes_created, node_count)

        # --- bulk ADD path ---
        res = self.query_master_and_wait("MATCH (n) SET n:L RETURN n")
        self.env.assertEquals(res.labels_added, node_count)
        # Every node must now carry label L
        res = self.query_master_and_wait("MATCH (n:L) RETURN count(n) AS c")
        self.env.assertEquals(res.result_set[0][0], node_count)

        # --- bulk REMOVE path ---
        res = self.query_master_and_wait("MATCH (n:L) REMOVE n:L")
        self.env.assertEquals(res.labels_removed, node_count)

        # No node should carry label L any longer
        res = self.query_master_and_wait("MATCH (n:L) RETURN count(n) AS c")
        self.env.assertEquals(res.result_set[0][0], 0)

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    def test_same_label_multiple_set_remove(self):
        # A single query that touches the same label in multiple SET / REMOVE
        self.query_master_and_wait("CREATE (:A {v: 1})")
        
        # --- repeated SET of the same label ---
        # Three SET n:L clauses; the label must be recorded only once.
        res = self.query_master_and_wait(
            "MATCH (n:A) SET n:L WITH n SET n:L WITH n SET n:L RETURN n"
        )
        self.env.assertEquals(res.labels_added, 1)

        res = self.query_master_and_wait("MATCH (n:A:L) RETURN count(n) AS c")
        self.env.assertEquals(res.result_set[0][0], 1)
        
        # --- repeated REMOVE of the same label ---
        res = self.query_master_and_wait(
            "MATCH (n:A:L) REMOVE n:L WITH n REMOVE n:L RETURN n"
        )
        self.env.assertEquals(res.labels_removed, 1)

        res = self.query_master_and_wait("MATCH (n:A:L) RETURN count(n) AS c")
        self.env.assertEquals(res.result_set[0][0], 0)
        
        # --- SET then REMOVE the same label within one query ---
        # Net effect on label M must be zero: it is added and immediately
        # removed
        self.query_master_and_wait("MATCH (n:A) SET n:L")   # give node L so query matches
        res = self.query_master_and_wait("MATCH (n:A:L) SET n:M WITH n REMOVE n:M RETURN n")
        self.env.assertEquals(res.labels_added, 1)
        self.env.assertEquals(res.labels_removed, 1)

        res = self.query_master_and_wait("MATCH (n:A:M) RETURN count(n) AS c")
        self.env.assertEquals(res.result_set[0][0], 0)

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    def test_redundant_label_set_remove(self):
        # A single query that touches the same label in multiple SET / REMOVE
        self.query_master_and_wait("CREATE (:A {v: 1})")
        
        # --- repeated SET of the same label ---
        # Three SET n:L clauses; the label must be recorded only once.
        res = self.query_master_and_wait("MATCH (n:A) SET n:L:L:L RETURN n")
        self.env.assertEquals(res.labels_added, 1)

        res = self.query_master_and_wait("MATCH (n:A:L) RETURN count(n) AS c")
        self.env.assertEquals(res.result_set[0][0], 1)
        
        # --- repeated REMOVE of the same label ---
        res = self.query_master_and_wait("MATCH (n:A:L) REMOVE n:L:L:L RETURN n")
        self.env.assertEquals(res.labels_removed, 1)

        res = self.query_master_and_wait("MATCH (n:A:L) RETURN count(n) AS c")
        self.env.assertEquals(res.result_set[0][0], 0)
        
        # --- SET then REMOVE the same label within one query ---
        # Net effect on label M must be zero: it is added and immediately
        # removed
        self.query_master_and_wait("MATCH (n:A) SET n:L")   # give node L so query matches

        res = self.query_master_and_wait( "MATCH (n:A:L) SET n:M:M:M WITH n REMOVE n:M:M:M RETURN n")
        self.env.assertEquals(res.labels_added, 1)
        self.env.assertEquals(res.labels_removed, 1)

        res = self.query_master_and_wait("MATCH (n:A:M) RETURN count(n) AS c")
        self.env.assertEquals(res.result_set[0][0], 0)

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    def test_remove_nonexisting_label(self):
        # Try to remove a non existing label
        self.query_master_and_wait("CREATE (:A {v: 1})")
        
        res = self.query_master_and_wait("MATCH (n:A) REMOVE n:X RETURN n")
        self.env.assertEquals(res.labels_removed, 0)

        # make sure label 'X' wasn't added to the graph's schema
        labels = self.query_master_and_wait("CALL db.labels()").result_set[0][0]
        self.env.assertNotIn("X", labels)

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
            self.env.assertIn("constraint", str(e).lower())
        
        # Rollback must restore the graph to exactly one :X node.
        res = self.query_master_and_wait("MATCH (n:X) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 1)
        
        # The original seed node must still be intact.
        res = self.query_master_and_wait("MATCH (n:X {name: 1}) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 1)

        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    #def test_failed_update_on_first_attempt(self):

