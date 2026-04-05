# Recreation test for SIGSEGV in EvalEntityUpdates at update_functions.c:493
#
# Root cause:
#   1. NodeByLabelScan._UpdateRecord calls Graph_GetNode() but ignores the
#      return value.  If the DataBlock item for a label-matrix node-ID has been
#      deleted, Graph_GetNode returns false and sets n->attributes = NULL.
#   2. Graph_EntityIsDeleted() treats attributes == NULL as "reserved but not
#      yet created" and returns false, so EvalEntityUpdates proceeds.
#   3. EvalEntityUpdates dereferences *entity->attributes (line 493) → SIGSEGV.
#
# Fix: add an explicit NULL-attributes guard in EvalEntityUpdates:
#   if (unlikely(Graph_EntityIsDeleted(entity) || entity->attributes == NULL))
#       return;
#
# The tests below reconstruct the scenarios where the label matrix can yield
# a node-ID whose DataBlock slot is already deleted:
#   * DELETE committed in a prior write-query, then MATCH (n:L) SET in the
#     next write-query.  The delta-matrix DM tracking must correctly suppress
#     the deleted ID; the guard is an additional safety net.
#   * Bulk-delete (many nodes, forcing a DM→M synchronization flush) followed
#     by MATCH (n:L) SET.
#   * MATCH (n:L) DELETE n SET n.x within a single query, where the DELETE op
#     collects and commits entities before the UPDATE op reads the records.
#   * UNWIND-driven MATCH+DELETE then label-scan SET.

from common import *


GRAPH_ID = "test_update_label_scan_crash"


class testUpdateLabelScanCrash():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.conn = self.env.getConnection()

    def setUp(self):
        # delete graph before each test to start with an empty slate
        try:
            self.graph.delete()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def _create_labeled_nodes(self, label, count, prop=None):
        """Create *count* nodes with *label* and optional property."""
        if prop:
            q = f"UNWIND range(1, {count}) AS i CREATE (:{label} {{v: i}})"
        else:
            q = f"UNWIND range(1, {count}) AS i CREATE (:{label})"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, count)

    def _delete_all_labeled(self, label):
        """Delete every node carrying *label*."""
        res = self.graph.query(f"MATCH (n:{label}) DELETE n")
        return res.nodes_deleted

    # -------------------------------------------------------------------------
    # test 1 – DELETE in one query, SET via label scan in a subsequent query
    # -------------------------------------------------------------------------
    def test01_set_after_delete_single_node(self):
        """
        SIGSEGV path: committed DELETE leaves DataBlock item deleted but label
        matrix may still carry the entry.  The next write-query's
        NodeByLabelScan must not dereference a NULL attributes pointer.
        """
        # create and immediately delete a labeled node
        self._create_labeled_nodes("A", 1, prop=True)
        deleted = self._delete_all_labeled("A")
        self.env.assertEquals(deleted, 1)

        # run SET via label scan after the delete; must not crash
        res = self.graph.query("MATCH (n:A) SET n.x = 99 RETURN n.x")
        # no node survives deletion – result set must be empty
        self.env.assertEquals(res.result_set, [])
        self.env.assertEquals(res.properties_set, 0)

    # -------------------------------------------------------------------------
    # test 2 – DELETE multiple nodes then SET via label scan
    # -------------------------------------------------------------------------
    def test02_set_after_delete_multiple_nodes(self):
        """Same scenario with several nodes."""
        self._create_labeled_nodes("B", 10, prop=True)
        deleted = self._delete_all_labeled("B")
        self.env.assertEquals(deleted, 10)

        res = self.graph.query("MATCH (n:B) SET n.v = 0 RETURN count(n)")
        self.env.assertEquals(res.result_set, [[0]])
        self.env.assertEquals(res.properties_set, 0)

    # -------------------------------------------------------------------------
    # test 3 – DELETE crossing the DELTA_MAX_PENDING_CHANGES flush boundary
    # -------------------------------------------------------------------------
    def test03_flush_boundary_delete_then_set(self):
        """
        Lower DELTA_MAX_PENDING_CHANGES to 5 and delete 10 nodes so that
        the DeltaMatrix is forced to flush DM into M mid-deletion.  After the
        flush, M no longer contains the deleted entries and the iterator yields
        nothing; a subsequent label-scan SET must not crash.
        """
        # Lower threshold so the flush happens during the 10-node delete
        self.conn.execute_command(
            "GRAPH.CONFIG", "SET", "DELTA_MAX_PENDING_CHANGES", 5
        )
        try:
            n = 10  # > 5 threshold → flush triggered during deletion
            self._create_labeled_nodes("C", n, prop=True)
            deleted = self._delete_all_labeled("C")
            self.env.assertEquals(deleted, n)

            # SET on the now-empty label must not crash
            res = self.graph.query("MATCH (n:C) SET n.v = 0 RETURN count(n)")
            self.env.assertEquals(res.result_set, [[0]])
            self.env.assertEquals(res.properties_set, 0)
        finally:
            # Always restore the default so other tests are unaffected
            self.conn.execute_command(
                "GRAPH.CONFIG", "SET", "DELTA_MAX_PENDING_CHANGES", 10000
            )

    # -------------------------------------------------------------------------
    # test 4 – DELETE and SET in the same query (pipeline)
    # -------------------------------------------------------------------------
    def test04_delete_and_set_same_query(self):
        """
        MATCH (n:D) DELETE n SET n.x = 1
        The Delete op collects and commits all entities before the Update op
        sees the records.  After deletion, entity->attributes points to a
        deleted DataBlock slot; Graph_EntityIsDeleted should return true and
        EvalEntityUpdates should bail out without touching attributes.
        """
        self._create_labeled_nodes("D", 5, prop=True)

        res = self.graph.query(
            "MATCH (n:D) DELETE n SET n.x = 1 RETURN n.x"
        )
        self.env.assertEquals(res.nodes_deleted, 5)
        # no property write should survive a delete of the same entity
        self.env.assertEquals(res.properties_set, 0)

    # -------------------------------------------------------------------------
    # test 5 – UNWIND + label scan + DELETE then SET
    # -------------------------------------------------------------------------
    def test05_unwind_delete_then_set_via_label_scan(self):
        """
        UNWIND drives a NodeByLabelScanConsumeFromChild path.  For every UNWIND
        value, the scan re-attaches its iterator.  After all iterations a SET
        on the now-empty label must not crash.
        """
        self._create_labeled_nodes("E", 5, prop=True)

        # delete all E nodes using UNWIND to iterate multiple times
        res = self.graph.query(
            "UNWIND range(1,5) AS i MATCH (n:E {v: i}) DELETE n"
        )
        self.env.assertEquals(res.nodes_deleted, 5)

        # now try to SET on the deleted label
        res = self.graph.query("MATCH (n:E) SET n.v = 0 RETURN count(n)")
        self.env.assertEquals(res.result_set, [[0]])
        self.env.assertEquals(res.properties_set, 0)

    # -------------------------------------------------------------------------
    # test 6 – mix: some nodes survive, some do not
    # -------------------------------------------------------------------------
    def test06_partial_delete_then_set(self):
        """
        Delete a subset of labeled nodes; SET must update only surviving ones.
        Exercises the label-scan path where some IDs in the matrix are deleted
        (in DM) and some are live.
        """
        self._create_labeled_nodes("F", 10, prop=True)

        # delete the first 5
        res = self.graph.query(
            "MATCH (n:F) WHERE n.v <= 5 DELETE n"
        )
        self.env.assertEquals(res.nodes_deleted, 5)

        # SET on the label: only the 5 surviving nodes should be updated
        res = self.graph.query("MATCH (n:F) SET n.x = 1 RETURN count(n)")
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0], 5)
        self.env.assertEquals(res.properties_set, 5)

    # -------------------------------------------------------------------------
    # test 7 – label-scan SET on a label that never existed
    # -------------------------------------------------------------------------
    def test07_set_on_nonexistent_label(self):
        """
        NodeByLabelScan with GRAPH_UNKNOWN_LABEL switches to the NOP consume
        function and returns NULL immediately.  No attributes are accessed.
        This baseline ensures the label-scan guard paths are tested.
        """
        res = self.graph.query("MATCH (n:Ghost) SET n.x = 1 RETURN count(n)")
        self.env.assertEquals(res.result_set, [[0]])
        self.env.assertEquals(res.properties_set, 0)

    # -------------------------------------------------------------------------
    # test 8 – NodeByLabelScanConsumeFromChild after deletion
    # -------------------------------------------------------------------------
    def test08_label_scan_from_child_after_delete(self):
        """
        NodeByLabelScanConsumeFromChild (label scan with a parent feeding
        records) re-attaches its iterator for each child record.  After nodes
        are deleted the re-attached iterator must not yield deleted node IDs.
        """
        # create nodes with two labels so the child can drive the scan
        self.graph.query("CREATE (:X {v:1}), (:X {v:2}), (:Y {w:10})")

        # delete all X nodes in a prior query
        res = self.graph.query("MATCH (n:X) DELETE n")
        self.env.assertEquals(res.nodes_deleted, 2)

        # now drive a label scan of :X via a child (Y nodes)
        # NodeByLabelScanConsumeFromChild is chosen when there is a child op
        res = self.graph.query(
            "MATCH (y:Y) "
            "MATCH (x:X) "        # produces NodeByLabelScanConsumeFromChild
            "SET x.v = y.w "
            "RETURN y.w, x.v"
        )
        # no X nodes exist; no updates, no crash
        self.env.assertEquals(res.result_set, [])
        self.env.assertEquals(res.properties_set, 0)

    # -------------------------------------------------------------------------
    # test 9 – edge-relation scan after edge deletion
    # -------------------------------------------------------------------------
    def test09_set_edge_after_delete(self):
        """
        The same NULL-dereference risk exists for edges via the relation-matrix
        scan.  After deleting all edges of a type, SET on that type must not
        crash.
        """
        self.graph.query(
            "CREATE (:A)-[:R {v:1}]->(:B), (:A)-[:R {v:2}]->(:B)"
        )
        res = self.graph.query("MATCH ()-[e:R]->() DELETE e")
        self.env.assertEquals(res.relationships_deleted, 2)

        res = self.graph.query(
            "MATCH ()-[e:R]->() SET e.v = 99 RETURN count(e)"
        )
        self.env.assertEquals(res.result_set, [[0]])
        self.env.assertEquals(res.properties_set, 0)

    # -------------------------------------------------------------------------
    # test 10 – repeated delete+create cycle to exercise DM flush path
    # -------------------------------------------------------------------------
    def test10_repeated_delete_create_cycle(self):
        """
        Multiple rounds of create-then-delete force the DeltaMatrix to cycle
        through its pending-changes tracking.  A subsequent label-scan SET must
        be correct in every round.
        """
        for round_n in range(1, 6):
            self._create_labeled_nodes("G", 20, prop=True)
            deleted = self._delete_all_labeled("G")
            self.env.assertEquals(deleted, 20)

            res = self.graph.query(
                "MATCH (n:G) SET n.v = 0 RETURN count(n)"
            )
            self.env.assertEquals(res.result_set, [[0]])
            self.env.assertEquals(res.properties_set, 0)
