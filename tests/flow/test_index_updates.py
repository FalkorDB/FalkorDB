import string
import random
from common import *
from index_utils import *

GRAPH_ID = "G"
labels = ["label_a", "label_b"]
fields = ['unique', 'group', 'doubleval', 'intval', 'stringval']
groups = ["Group A", "Group B", "Group C","Group D", "Group E"]
node_ctr = 0


class testIndexUpdatesFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()
        self.build_indices()

    def new_node(self):
        return Node(labels = labels[node_ctr % 2],
                    properties = {'unique': node_ctr,
                                  'group': random.choice(groups),
                                  'doubleval': round(random.uniform(-1, 1), 2),
                                  'intval': random.randint(1, 10000),
                                  'stringval': ''.join(random.choice(string.ascii_lowercase) for x in range(6))})

    def populate_graph(self):
        global node_ctr
        nodes = []
        for i in range(1000):
            nodes.append(self.new_node())
            node_ctr += 1
        nodes_str = [str(node) for node in nodes]
        self.graph.query(f"CREATE {','.join(nodes_str)}")

    def build_indices(self):
        for field in fields:
            self.graph.create_node_range_index('label_a', field)
            self.graph.create_node_range_index('label_b', field)
        wait_for_indices_to_sync(self.graph)

    # Validate that all properties are indexed
    def validate_indexed(self):
        for field in fields:
            resp = str(self.graph.explain(f"MATCH (a:label_a) WHERE a.{field} > 0 RETURN a"))
            self.env.assertContains('Node By Index Scan', resp)

    # So long as 'unique' is not modified, label_a.unique will always be even and label_b.unique will always be odd
    def validate_unique(self):
        result = self.graph.query("MATCH (a:label_a) RETURN a.unique")
        # Remove the header
        result.result_set.pop(0)
        for val in result.result_set:
            self.env.assertEqual(int(float(val[0])) % 2, 0)

        result = self.graph.query("MATCH (b:label_b) RETURN b.unique")
        # Remove the header
        result.result_set.pop(0)
        for val in result.result_set:
            self.env.assertEqual(int(float(val[0])) % 2, 1)

    # The index scan ought to return identical results to a label scan over the same range of values.
    def validate_doubleval(self):
        for label in labels:
            resp = str(self.graph.explain(f"MATCH (a:{label}) WHERE a.doubleval < 100 RETURN a.doubleval ORDER BY a.doubleval"))
            self.env.assertContains('Node By Index Scan', resp)
            indexed_result = self.graph.query("""MATCH (a:%s) WHERE a.doubleval < 100 RETURN a.doubleval ORDER BY a.doubleval""" % (label))
            scan_result = self.graph.query("""MATCH (a:%s) RETURN a.doubleval ORDER BY a.doubleval""" % (label))

            self.env.assertEqual(len(indexed_result.result_set), len(scan_result.result_set))
            # Collect any elements between the two result sets that fail a string comparison
            # so that we may compare them as doubles (specifically, -0 and 0 should be considered equal)
            differences = [[i[0], j[0]] for i, j in zip(indexed_result.result_set, scan_result.result_set) if i != j]
            for pair in differences:
                self.env.assertEqual(float(pair[0]), float(pair[1]))

    # The intval property can be assessed similar to doubleval, but the result sets should be identical
    def validate_intval(self):
        for label in labels:
            resp = str(self.graph.explain(f"MATCH (a:{label}) WHERE a.intval > 0 RETURN a.intval ORDER BY a.intval"))
            self.env.assertContains('Node By Index Scan', resp)
            indexed_result = self.graph.query(f"MATCH (a:{label}) WHERE a.intval > 0 RETURN a.intval ORDER BY a.intval")
            scan_result = self.graph.query(f"MATCH (a:{label}) RETURN a.intval ORDER BY a.intval")

            self.env.assertEqual(indexed_result.result_set, scan_result.result_set)

    # Validate a series of premises to ensure that the graph has not been modified unexpectedly
    def validate_state(self):
        self.validate_unique()
        self.validate_indexed()
        self.validate_doubleval()
        self.validate_intval()

    # Modify a property, triggering updates to all nodes in two indices
    def test01_full_property_update(self):
        result = self.graph.query("MATCH (a) SET a.doubleval = a.doubleval + 1.1")
        self.env.assertEqual(result.properties_set, 1000)
        # Verify that index scans still function and return correctly
        self.validate_state()

    # Modify a property, triggering updates to a subset of nodes in two indices
    def test02_partial_property_update(self):
        self.graph.query("MATCH (a) WHERE a.doubleval > 0 SET a.doubleval = a.doubleval + 1.1")
        # Verify that index scans still function and return correctly
        self.validate_state()

    #  Add 100 randomized nodes and validate indices
    def test03_node_creation(self):
        global node_ctr
        nodes = []
        for i in range(100):
            nodes.append(self.new_node())
            node_ctr += 1
        nodes_str = [str(node) for node in nodes]
        self.graph.query(f"CREATE {','.join(nodes_str)}")
        self.validate_state()

    # Delete every other node in first 100 and validate indices
    def test04_node_deletion(self):
        global node_ctr
        # Delete nodes one at a time
        for i in range(0, 100, 2):
            result = self.graph.query("MATCH (a) WHERE ID(a) = %d DELETE a" % (i))
            self.env.assertEqual(result.nodes_deleted, 1)
            node_ctr -= 1
        self.validate_state()

        # Delete all nodes matching a filter
        result = self.graph.query("MATCH (a:label_a) WHERE a.group = 'Group A' DELETE a")
        self.env.assertGreater(result.nodes_deleted, 0)
        self.validate_state()

    def test05_unindexed_property_update(self):
        # Add an unindexed property to all nodes.
        self.graph.query("MATCH (a) SET a.unindexed = 'unindexed'")

        # Retrieve a single node
        result = self.graph.query("MATCH (a) RETURN a.unique LIMIT 1")
        unique_prop = result.result_set[0][0]
        query = f"""MATCH (a {{unique: {unique_prop} }}) SET a.unindexed = 5, a.unique = {unique_prop} RETURN a.unindexed, a.unique"""
        result = self.graph.query(query)
        expected_result = [[5, unique_prop]]
        self.env.assertEqual(result.result_set, expected_result)
        self.env.assertEqual(result.properties_set, 1)

    # Validate that after deleting an indexed property, that property can no longer be found in the index.
    def test06_remove_indexed_prop(self):
        # Create a new node with a single indexed property
        query = """CREATE (:NEW {v: 5})"""
        result = self.graph.query(query)
        self.env.assertEqual(result.properties_set, 1)
        self.env.assertEqual(result.labels_added, 1)
        create_node_range_index(self.graph, 'NEW', 'v', sync=True)

        # Delete the entity's property
        query = """MATCH (a:NEW {v: 5}) SET a.v = NULL"""
        result = self.graph.query(query)
        self.env.assertEqual(result.properties_set, 0)
        self.env.assertEqual(result.properties_removed, 1)

        # Query the index for the entity
        query = """MATCH (a:NEW {v: 5}) RETURN a"""
        plan = str(self.graph.explain(query))
        self.env.assertContains("Node By Index Scan", plan)
        result = self.graph.query(query)
        # No entities should be returned
        expected_result = []
        self.env.assertEqual(result.result_set, expected_result)

    # Validate that when a label has both exact-match and full-text indexes
    # on different properties, an update operation checks all indexes to
    # determine whether they must be updated.
    # This is necessary because either one of the indexes may not track the
    # property being updated, but that does not guarantee that the other
    # index does not track the property.
    def test07_update_property_only_on_fulltext_index(self):
        # Remove the exact-match index on a property
        drop_node_range_index(self.graph, 'label_a', 'group')

        # Add a full-text index on the property
        result = create_node_fulltext_index(self.graph, 'label_a', 'group', sync=True)
        self.env.assertEqual(result.indices_created, 1)

        # Modify the values of the property
        result = self.graph.query("MATCH (a:label_a) WHERE a.group = 'Group C' SET a.group = 'Group NEW'")
        modified_count = result.properties_set
        self.env.assertGreater(modified_count, 0)

        # Validate that the full-text index reflects the update
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('label_a', 'Group NEW')")
        self.env.assertEqual(len(result.result_set), modified_count)

        # Validate that the previous value has been removed
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('label_a', 'Group C')")
        self.env.assertEqual(len(result.result_set), 0)

    # A query that fails *after* an inner Commit already published index documents
    # must leave nothing behind: the rollback discards the entities those documents
    # point at, and `get_indexed_nodes` does not validate ids against the reader's
    # snapshot, so a leftover document surfaces as a phantom row.
    def test08_failed_create_leaves_no_index_entry(self):
        create_node_range_index(self.graph, 'RB', 'v', sync=True)

        # Fails on the division, i.e. after the CREATE has been committed and its
        # index document published by the nested Commit.
        try:
            self.graph.query(
                "CREATE (n:RB {v: 1}) WITH n UNWIND [1, 0] AS d RETURN n.v / d")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Division by zero", str(e))

        # The node itself is rolled back ...
        result = self.graph.query("MATCH (n:RB) RETURN count(n)")
        self.env.assertEqual(result.result_set[0][0], 0)

        # ... and so is its index document (an index scan must not find a phantom).
        result = self.graph.query("MATCH (n:RB) WHERE n.v = 1 RETURN count(n)")
        self.env.assertEqual(result.result_set[0][0], 0)

    # The undo for a failed *update* must restore the previous document, not delete
    # it: the entity still exists with its old value, so deleting would drop a live
    # entity out of the index and cause silent false negatives.
    def test09_failed_update_restores_previous_index_entry(self):
        create_node_range_index(self.graph, 'RBU', 'v', sync=True)
        self.graph.query("CREATE (:RBU {v: 10})")

        result = self.graph.query("MATCH (n:RBU) WHERE n.v = 10 RETURN count(n)")
        self.env.assertEqual(result.result_set[0][0], 1)

        try:
            self.graph.query(
                """MATCH (n:RBU) WHERE n.v = 10 SET n.v = 20
                   WITH n UNWIND [1, 0] AS d RETURN n.v / d""")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Division by zero", str(e))

        # The old value is still indexed ...
        result = self.graph.query("MATCH (n:RBU) WHERE n.v = 10 RETURN count(n)")
        self.env.assertEqual(result.result_set[0][0], 1)

        # ... the rolled-back new value is not ...
        result = self.graph.query("MATCH (n:RBU) WHERE n.v = 20 RETURN count(n)")
        self.env.assertEqual(result.result_set[0][0], 0)

        # ... and the graph agrees with the index.
        result = self.graph.query("MATCH (n:RBU) RETURN n.v")
        self.env.assertEqual(result.result_set, [[10]])

    # An index scan above a nested Commit must see what an earlier part of the same
    # query wrote (read-your-own-writes across a write -> read boundary).
    def test10_index_scan_sees_earlier_subquery_writes(self):
        create_node_range_index(self.graph, 'RYOW', 'v', sync=True)

        # The MATCH is served by a Node By Index Scan above the Commit that created
        # the node, so it only yields a row if that write is visible to the index.
        result = self.graph.query(
            """CREATE (n:RYOW {v: 77}) WITH n
               MATCH (m:RYOW) WHERE m.v = 77 CREATE (:RYOW {v: 78})""")
        self.env.assertEqual(result.nodes_created, 2)

        result = self.graph.query("MATCH (n:RYOW) WHERE n.v = 78 RETURN count(n)")
        self.env.assertEqual(result.result_set[0][0], 1)
