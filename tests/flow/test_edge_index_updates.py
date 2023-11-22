import random
import string
from common import *
from index_utils import *

GRAPH_ID = "G"
labels = ["label_a", "label_b"]
types = ["type_a", "type_b"]
fields = ['unique', 'group', 'doubleval', 'intval', 'stringval']
groups = ["Group A", "Group B", "Group C","Group D", "Group E"]
node_ctr = 0
edge_ctr = 0


class testEdgeIndexUpdatesFlow():
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

    def new_edge(self, from_node, to_node):
        return Edge(from_node, types[edge_ctr % 2], to_node,
                    properties={'unique': edge_ctr,
                                  'group': random.choice(groups),
                                  'doubleval': round(random.uniform(-1, 1), 2),
                                  'intval': random.randint(1, 10000),
                                  'stringval': ''.join(random.choice(string.ascii_lowercase) for x in range(6))})
    def populate_graph(self):
        global node_ctr
        global edge_ctr
        for i in range(1000):
            from_node = self.new_node()
            node_ctr += 1
            self.graph.add_node(from_node)
            to_node = self.new_node()
            node_ctr += 1
            self.graph.add_node(to_node)
            edge = self.new_edge(from_node, to_node)
            edge_ctr += 1
            self.graph.add_edge(edge)
        self.graph.commit()

    def build_indices(self):
        for field in fields:
            self.graph.query("CREATE INDEX FOR ()-[r:type_a]-() ON (r.%s)" % (field))
            self.graph.query("CREATE INDEX FOR ()-[r:type_b]-() ON (r.%s)" % (field))
        wait_for_indices_to_sync(self.graph)

    # Validate that all properties are indexed
    def validate_indexed(self):
        for field in fields:
            resp = str(self.graph.explain("""MATCH ()-[a:type_a]->() WHERE a.%s > 0 RETURN a""" % (field)))
            self.env.assertIn('Edge By Index Scan', resp)

    # So long as 'unique' is not modified, label_a.unique will always be even and label_b.unique will always be odd
    def validate_unique(self):
        result = self.graph.query("MATCH ()-[r:type_a]->() RETURN r.unique")
        # Remove the header
        result.result_set.pop(0)
        for val in result.result_set:
            self.env.assertEquals(int(float(val[0])) % 2, 0)

        result = self.graph.query("MATCH ()-[r:type_b]->() RETURN r.unique")
        # Remove the header
        result.result_set.pop(0)
        for val in result.result_set:
            self.env.assertEquals(int(float(val[0])) % 2, 1)

    # The index scan ought to return identical results to a label scan over the same range of values.
    def validate_doubleval(self):
        for type in types:
            resp = str(self.graph.explain("""MATCH ()-[a:%s]->() WHERE a.doubleval < 100 RETURN a.doubleval ORDER BY a.doubleval""" % (type)))
            self.env.assertIn('Edge By Index Scan', resp)
            indexed_result = self.graph.query("""MATCH ()-[a:%s]->() WHERE a.doubleval < 100 RETURN a.doubleval ORDER BY a.doubleval""" % (type))
            scan_result = self.graph.query("""MATCH ()-[a:%s]->() RETURN a.doubleval ORDER BY a.doubleval""" % (type))

            self.env.assertEqual(len(indexed_result.result_set), len(scan_result.result_set))
            # Collect any elements between the two result sets that fail a string comparison
            # so that we may compare them as doubles (specifically, -0 and 0 should be considered equal)
            differences = [[i[0], j[0]] for i, j in zip(indexed_result.result_set, scan_result.result_set) if i != j]
            for pair in differences:
                self.env.assertEqual(float(pair[0]), float(pair[1]))

    # The intval property can be assessed similar to doubleval, but the result sets should be identical
    def validate_intval(self):
        for type in types:
            resp = str(self.graph.explain("""MATCH ()-[a:%s]->() WHERE a.intval > 0 RETURN a.intval ORDER BY a.intval""" % (type)))
            self.env.assertIn('Edge By Index Scan', resp)
            indexed_result = self.graph.query("""MATCH ()-[a:%s]->() WHERE a.intval > 0 RETURN a.intval ORDER BY a.intval""" % (type))
            scan_result = self.graph.query("""MATCH ()-[a:%s]->() RETURN a.intval ORDER BY a.intval""" % (type))

            self.env.assertEqual(indexed_result.result_set, scan_result.result_set)

    # Validate a series of premises to ensure that the graph has not been modified unexpectedly
    def validate_state(self):
        self.validate_unique()
        self.validate_indexed()
        self.validate_doubleval()
        self.validate_intval()

    # Modify a property, triggering updates to all edges in two indices
    def test01_full_property_update(self):
        result = self.graph.query("MATCH ()-[a]->() SET a.doubleval = a.doubleval + 1.1")
        self.env.assertEquals(result.properties_set, 1000)
        # Verify that index scans still function and return correctly
        self.validate_state()

    # Modify a property, triggering updates to a subset of edges in two indices
    def test02_partial_property_update(self):
        self.graph.query("MATCH ()-[a]->() WHERE a.doubleval > 0 SET a.doubleval = a.doubleval + 1.1")
        # Verify that index scans still function and return correctly
        self.validate_state()

    #  Add 100 randomized edges and validate indices
    def test03_edge_creation(self):
        # Reset nodes in the Graph object so that we won't double-commit the originals
        self.graph.nodes = {}
        self.graph.edges = []
        global node_ctr
        global edge_ctr
        for i in range(100):
            from_node = self.new_node()
            self.graph.add_node(from_node)
            node_ctr += 1
            to_node = self.new_node()
            self.graph.add_node(to_node)
            node_ctr += 1
            edge = self.new_edge(from_node, to_node)
            self.graph.add_edge(edge)
            edge_ctr += 1
        self.graph.commit()
        self.validate_state()

    # Delete every other edge in first 100 and validate indices
    def test04_edge_deletion(self):
        # Reset nodes in the Graph object so that we won't double-commit the originals
        self.graph.nodes = {}
        self.graph.edges = []
        global node_ctr
        global edge_ctr
        # Delete edges one at a time
        for i in range(0, 100, 2):
            result = self.graph.query("MATCH ()-[a]->() WHERE ID(a) = %d DELETE a" % (i))
            self.env.assertEquals(result.relationships_deleted, 1)
            edge_ctr -= 1
        self.validate_state()

        # Delete all edges matching a filter
        result = self.graph.query("MATCH ()-[a:type_a]->() WHERE a.group = 'Group A' DELETE a")
        self.env.assertGreater(result.relationships_deleted, 0)
        self.validate_state()

    def test05_unindexed_property_update(self):
        # Add an unindexed property to all edges.
        self.graph.query("MATCH ()-[a]->() SET a.unindexed = 'unindexed'")

        # Retrieve a single edge
        result = self.graph.query("MATCH ()-[a]->() RETURN a.unique LIMIT 1")
        unique_prop = result.result_set[0][0]
        query = """MATCH ()-[a {unique: %s }]->() SET a.unindexed = 5, a.unique = %s RETURN a.unindexed, a.unique""" % (unique_prop, unique_prop)
        result = self.graph.query(query)
        expected_result = [[5, unique_prop]]
        self.env.assertEquals(result.result_set, expected_result)
        self.env.assertEquals(result.properties_set, 1)

    # Validate that after deleting an indexed property, that property can no longer be found in the index.
    def test06_remove_indexed_prop(self):
        # Create a new edge with a single indexed property
        query = """CREATE ()-[:NEW {v: 5}]->()"""
        result = self.graph.query(query)
        self.env.assertEquals(result.properties_set, 1)
        create_edge_range_index(self.graph, 'NEW', 'v', sync=True)

        # Delete the entity's property
        query = """MATCH ()-[a:NEW {v: 5}]->() SET a.v = NULL"""
        result = self.graph.query(query)
        self.env.assertEquals(result.properties_set, 0)
        self.env.assertEquals(result.properties_removed, 1)

        # Query the index for the entity
        query = """MATCH ()-[a:NEW {v: 5}]->() RETURN a"""
        plan = str(self.graph.explain(query))
        self.env.assertIn("Edge By Index Scan", plan)
        result = self.graph.query(query)
        # No entities should be returned
        expected_result = []
        self.env.assertEquals(result.result_set, expected_result)
