from common import *
from index_utils import *

GRAPH_ID = "update"

class testEntityUpdate():
    def __init__(self):
        self.env, self.db = Env()
        # create a graph with a single node with attribute 'v'
        self.graph = self.db.select_graph(GRAPH_ID)
        self.graph.query("CREATE ({v:1})")

        # create a graph with a two nodes connected by an edge
        self.multiple_entity_graph = self.db.select_graph('multiple_entity_update')
        self.multiple_entity_graph.query("CREATE (:L {v1: 1})-[:R {v1: 3}]->(:L {v2: 2})")
        create_node_range_index(self.multiple_entity_graph, 'L', 'v1', 'v2', sync=True)

    def test01_update_attribute(self):
        # update existing attribute 'v'
        result = self.graph.query("MATCH (n) SET n.v = 2")
        self.env.assertEqual(result.properties_set, 1)

    def test02_update_none_existing_attr(self):
        # introduce a new attribute 'x'
        result = self.graph.query("MATCH (n) SET n.x = 1")
        self.env.assertEqual(result.properties_set, 1)

    def test03_update_no_change(self):
        # setting 'x' to its current value
        result = self.graph.query("MATCH (n) SET n.x = 1")
        self.env.assertEqual(result.properties_set, 0)

        # setting both 'v' and 'x' to their current values
        result = self.graph.query("MATCH (n) SET n.v = 2, n.x = 1")
        self.env.assertEqual(result.properties_set, 0)

        # update 'v' to a new value, 'x' remains the same
        result = self.graph.query("MATCH (n) SET n.v = 1, n.x = 1")
        self.env.assertEqual(result.properties_set, 1)

        # update 'x' to a new value, 'v' remains the same
        result = self.graph.query("MATCH (n) SET n.v = 1, n.x = 2")
        self.env.assertEqual(result.properties_set, 1)

    def test04_update_remove_attribute(self):
        # remove the 'x' attribute
        result = self.graph.query("MATCH (n) SET n.x = NULL")
        self.env.assertEqual(result.properties_set, 0)
        self.env.assertEqual(result.properties_removed, 1)

        # remove null attribute using MERGE...ON CREATE SET
        result = self.graph.query("UNWIND [{id: 1, aField: 'withValue', andOneWithout: null}] AS item MERGE (m:X{id: item.id}) ON CREATE SET m += item RETURN properties(m)")
        self.env.assertEqual(result.labels_added, 1)
        self.env.assertEqual(result.nodes_created, 1)
        self.env.assertEqual(result.properties_set, 2)
        expected_result = [[{'id': 1, 'aField': 'withValue'}]]
        self.env.assertEqual(result.result_set, expected_result)

        result = self.graph.query("MATCH (m:X) DELETE(m)")
        self.env.assertEqual(result.nodes_deleted, 1)

        # remove the 'x' attribute using MERGE...ON MATCH SET
        result = self.graph.query("CREATE (n:N {x:5})")
        result = self.graph.query("MERGE (n:N) ON MATCH SET n.x=null RETURN n")
        self.env.assertEqual(result.properties_set, 0)
        self.env.assertEqual(result.properties_removed, 1)
        result = self.graph.query("MATCH (n:N) DELETE(n)")
        self.env.assertEqual(result.nodes_deleted, 1)

    def test05_update_from_projection(self):
        result = self.graph.query("MATCH (n) UNWIND ['Calgary'] as city_name SET n.name = city_name RETURN n.v, n.name")
        expected_result = [[1, 'Calgary']]
        self.env.assertEqual(result.properties_set, 1)
        self.env.assertEqual(result.result_set, expected_result)

    # Set the entity's properties to an empty map
    def test06_replace_property_map(self):
        empty_node = Node()
        result = self.graph.query("MATCH (n) SET n = {} RETURN n")
        expected_result = [[empty_node]]
        # The node originally had 2 properties, 'name' and 'city_name'
        self.env.assertEqual(result.properties_set, 0)
        self.env.assertEqual(result.properties_removed, 2)
        self.env.assertEqual(result.result_set, expected_result)

        # similarly updateing an "empty" node with a map containing only nulls
        result = self.graph.query("CREATE (n) SET n = {v:null} DELETE n RETURN n")
        n = result.result_set[0][0]
        self.env.assertEqual(result.properties_set, 0)
        self.env.assertEqual(result.properties_removed, 0)
        self.env.assertEqual(len(n.properties), 0)

    # Update the entity's properties by setting a specific property and merging property maps
    def test07_update_property_map(self):
        node = Node(properties={"v": 1, "v2": 2})
        result = self.graph.query("MATCH (n) SET n.v = 1, n += {v2: 2} RETURN n")
        expected_result = [[node]]
        self.env.assertEqual(result.properties_set, 2)
        self.env.assertEqual(result.result_set, expected_result)

    # Replacement maps overwrite existing properties and previous SETs but do not modify subsequent non-replacement SETs
    def test08_multiple_updates_to_property_map(self):
        node = Node(properties={"v": 1, "v2": 2, "v4": 4})
        result = self.graph.query("MATCH (n) SET n.v3 = 3, n = {v: 1}, n += {v2: 2}, n.v4 = 4 RETURN n")
        expected_result = [[node]]
        self.env.assertEqual(result.result_set, expected_result)

    # MERGE updates should support the same operations as SET updates
    def test09_merge_update_map(self):
        node = Node(properties={"v": 5})
        result = self.graph.query("MERGE (n {v: 1}) ON MATCH SET n = {}, n.v = 5 RETURN n")
        expected_result = [[node]]
        self.env.assertEqual(result.result_set, expected_result)

    # Update properties with a map retrieved by alias
    def test10_property_map_from_identifier(self):
        # Overwrite existing properties
        node = Node(properties={"v2": 10})
        result = self.graph.query("WITH {v2: 10} as props MATCH (n) SET n = props RETURN n")
        expected_result = [[node]]
        self.env.assertEqual(result.result_set, expected_result)

        # Merge property maps
        node = Node(properties={"v1": True, "v2": 10})
        result = self.graph.query("WITH {v1: True} as props MATCH (n) SET n += props RETURN n")
        expected_result = [[node]]
        self.env.assertEqual(result.result_set, expected_result)

    # Update properties with a map retrieved from a parameter
    def test11_property_map_from_parameter(self):
        # Overwrite existing properties
        node = Node(properties={"v2": 10})
        result = self.graph.query("CYPHER props={v2: 10} MATCH (n) SET n = $props RETURN n")
        expected_result = [[node]]
        self.env.assertEqual(result.result_set, expected_result)

        # Merge property maps
        node = Node(properties={"v1": True, "v2": 10})
        result = self.graph.query("CYPHER props={v1: true} MATCH (n) SET n += $props RETURN n")
        expected_result = [[node]]
        self.env.assertEqual(result.result_set, expected_result)

    # Fail update an entity property when left hand side is not alias
    def test12_fail_update_property_of_non_alias_entity(self):
        try:
            self.graph.query("MATCH P=() SET nodes(P).prop = 1 RETURN nodes(P)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("FalkorDB does not currently support non-alias references on the left-hand side of SET expressions", str(e))

        try:
            self.graph.query("MERGE (n:N) ON CREATE SET n.a.b=3 RETURN n")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("FalkorDB does not currently support non-alias references on the left-hand side of SET expressions", str(e))

        try:
            self.graph.query("MERGE (n:N) ON CREATE SET n = {v: 1}, n.a.b=3 RETURN n")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("FalkorDB does not currently support non-alias references on the left-hand side of SET expressions", str(e))

        try:
            self.graph.query("MERGE (n:N) ON MATCH SET n.a.b=3 RETURN n")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("FalkorDB does not currently support non-alias references on the left-hand side of SET expressions", str(e))

        try:
            self.graph.query("MERGE (n:N) ON MATCH SET n = {v: 1}, n.a.b=3 RETURN n")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("FalkorDB does not currently support non-alias references on the left-hand side of SET expressions", str(e))

    # Fail when a property is a complex type nested within an array type
    def test13_invalid_complex_type_in_array(self):
        # Test combinations of invalid types with nested and top-level arrays
        # Invalid types are NULL, maps, nodes, edges, and paths
        queries = ["MATCH (a) SET a.v = [a]",
                   "MATCH (a) SET a = {v: ['str', [1, NULL]]}",
                   "MATCH (a) SET a += [[{k: 'v'}]]",
                   "CREATE (a:L)-[e:R]->(:L) SET a.v = [e]"]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Property values can only be of primitive types or arrays of primitive types", str(e))

    # fail when attempting to perform invalid map assignment
    def test14_invalid_map_assignment(self):
        try:
            self.graph.query("MATCH (a) SET a.v = {f: true}")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Property values can only be of primitive types or arrays of primitive types", str(e))

    # update properties by attribute set reassignment
    def test15_assign_entity_properties(self):
        # merge attribute set of a node with existing properties
        node = Node(labels="L", properties={"v1": 1, "v2": 2})
        result = self.multiple_entity_graph.query("MATCH (n1 {v1: 1}), (n2 {v2: 2}) SET n1 += n2 RETURN n1")
        expected_result = [[node]]
        self.env.assertEqual(result.result_set, expected_result)
        # validate index updates
        result = self.multiple_entity_graph.query("MATCH (n:L) WHERE n.v1 > 0 RETURN n.v1 ORDER BY n.v1")
        expected_result = [[1]]
        self.env.assertEqual(result.result_set, expected_result)
        result = self.multiple_entity_graph.query("MATCH (n:L) WHERE n.v2 > 0 RETURN n.v2 ORDER BY n.v2")
        expected_result = [[2],
                           [2]]
        self.env.assertEqual(result.result_set, expected_result)

        # overwrite attribute set of node with attribute set of edge
        node = Node(labels="L", properties={"v1": 3})
        result = self.multiple_entity_graph.query("MATCH (n {v1: 1})-[e]->() SET n = e RETURN n")
        expected_result = [[node]]
        self.env.assertEqual(result.result_set, expected_result)
        # validate index updates
        result = self.multiple_entity_graph.query("MATCH (n:L) WHERE n.v1 > 0 RETURN n.v1 ORDER BY n.v1")
        expected_result = [[3]]
        self.env.assertEqual(result.result_set, expected_result)
        result = self.multiple_entity_graph.query("MATCH (n:L) WHERE n.v2 > 0 RETURN n.v2 ORDER BY n.v2")
        expected_result = [[2]]
        self.env.assertEqual(result.result_set, expected_result)

    # repeated attribute set reassignment
    def test16_assign_entity_properties(self):
        # repeated merges to the attribute set of a node
        node = Node(labels="L", properties={"v1": 3, "v2": 2})
        result = self.multiple_entity_graph.query("MATCH (n), (x) WHERE ID(n) = 0 WITH n, x ORDER BY ID(x) SET n += x RETURN n")
        expected_result = [[node],
                           [node]]
        self.env.assertEqual(result.result_set, expected_result)
        # validate index updates
        result = self.multiple_entity_graph.query("MATCH (n:L) WHERE n.v1 > 0 RETURN n.v1 ORDER BY n.v1")
        expected_result = [[3]]
        self.env.assertEqual(result.result_set, expected_result)
        result = self.multiple_entity_graph.query("MATCH (n:L) WHERE n.v2 > 0 RETURN n.v2 ORDER BY n.v2")
        expected_result = [[2],
                           [2]]
        self.env.assertEqual(result.result_set, expected_result)

        # repeated updates to the attribute set of a node
        node = Node(labels="L", properties={"v2": 2})
        result = self.multiple_entity_graph.query("MATCH (n), (x) WHERE ID(n) = 0 WITH n, x ORDER BY ID(x) SET n = x RETURN n")
        expected_result = [[node],
                           [node]]
        self.env.assertEqual(result.result_set, expected_result)
        # validate index updates
        result = self.multiple_entity_graph.query("MATCH (n:L) WHERE n.v1 > 0 RETURN n.v1 ORDER BY n.v1")
        expected_result = []
        self.env.assertEqual(result.result_set, expected_result)
        result = self.multiple_entity_graph.query("MATCH (n:L) WHERE n.v2 > 0 RETURN n.v2 ORDER BY n.v2")
        expected_result = [[2],
                           [2]]
        self.env.assertEqual(result.result_set, expected_result)

        # repeated multiple updates to the attribute set of a node
        node = Node(labels="L", properties={"v2": 2})
        result = self.multiple_entity_graph.query("MATCH (n), (x) WHERE ID(n) = 0 WITH n, x ORDER BY ID(x) SET n = x, n += x RETURN n")
        expected_result = [[node],
                           [node]]
        self.env.assertEqual(result.result_set, expected_result)
        # validate index updates
        result = self.multiple_entity_graph.query("MATCH (n:L) WHERE n.v1 > 0 RETURN n.v1 ORDER BY n.v1")
        expected_result = []
        self.env.assertEqual(result.result_set, expected_result)
        result = self.multiple_entity_graph.query("MATCH (n:L) WHERE n.v2 > 0 RETURN n.v2 ORDER BY n.v2")
        expected_result = [[2],
                           [2]]
        self.env.assertEqual(result.result_set, expected_result)

    # fail when attempting to perform invalid entity assignment
    def test17_invalid_entity_assignment(self):
        queries = ["MATCH (a) SET a.v = [a]",
                   "MATCH (a) SET a = a.v",
                   "MATCH (a) SET a = NULL"]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Property values can only be of primitive types or arrays of primitive types", str(e))


    def validate_node_labels(self, graph, labels, expected_count):
        for label in labels:
            result = graph.query(f"MATCH (n:{label}) RETURN n")
            self.env.assertEqual(len(result.result_set), expected_count)
            if expected_count > 0:
                for record in result.result_set:
                    self.env.assertTrue(label in record[0].labels)


    def test18_update_node_label(self):
        labels = ["TestLabel"]
        
        self.validate_node_labels(self.graph, labels, 0)
        result = self.graph.query(f"MATCH (n) SET n:{labels[0]}")
        self.env.assertEqual(result.labels_added, 1)
        self.validate_node_labels(self.graph, labels, 1)

        # multiple node updates
        self.validate_node_labels(self.multiple_entity_graph, labels, 0)
        result = self.multiple_entity_graph.query(f"MATCH (n) SET n:{labels[0]}")
        self.env.assertEqual(result.labels_added, 2)
        self.validate_node_labels(self.multiple_entity_graph, labels, 2)


    def test19_update_node_multiple_label(self):
        labels = ["TestLabel2", "TestLabel3"]

        self.validate_node_labels(self.graph, labels, 0)   
        result = self.graph.query(f"MATCH (n) SET n:{':'.join(labels)}")
        self.env.assertEqual(result.labels_added, 2)
        self.validate_node_labels(self.graph, labels, 1)

        # multiple node updates
        self.validate_node_labels(self.multiple_entity_graph, labels, 0)   
        result = self.multiple_entity_graph.query(f"MATCH (n) SET n:{':'.join(labels)}")
        self.env.assertEqual(result.labels_added, 4)
        self.validate_node_labels(self.multiple_entity_graph, labels, 2)
    

    def test20_update_node_comma_separated_labels(self):
        labels = ["TestLabel4", "TestLabel5"]

        self.validate_node_labels(self.graph, labels, 0)
        result = self.graph.query(f"MATCH (n) SET n:{labels[0]}, n:{labels[1]}")
        self.env.assertEqual(result.labels_added, 2)
        self.validate_node_labels(self.graph, labels, 1)

        # multiple node updates
        self.validate_node_labels(self.multiple_entity_graph, labels, 0)
        result = self.multiple_entity_graph.query(f"MATCH (n) SET n:{labels[0]}, n:{labels[1]}")
        self.env.assertEqual(result.labels_added, 4)
        self.validate_node_labels(self.multiple_entity_graph, labels, 2)


    def test21_update_node_label_and_property(self):
        labels = ["TestLabel6"]
       
        self.validate_node_labels(self.graph, labels, 0)
        result = self.graph.query("MATCH (n {testprop:'testvalue'}) RETURN n")
        self.env.assertEqual(len(result.result_set), 0)
        result = self.graph.query(f"MATCH (n) SET n:{labels[0]}, n.testprop='testvalue'")
        self.env.assertEqual(result.labels_added, 1)
        self.env.assertEqual(result.properties_set, 1)
        self.validate_node_labels(self.graph, labels, 1)
        result = self.graph.query("MATCH (n {testprop:'testvalue'}) RETURN n")
        self.env.assertEqual(len(result.result_set), 1)

        # multiple node updates
        self.validate_node_labels(self.multiple_entity_graph, labels, 0)
        result = self.multiple_entity_graph.query("MATCH (n {testprop:'testvalue'}) RETURN n")
        self.env.assertEqual(len(result.result_set), 0)
        result = self.multiple_entity_graph.query(f"MATCH (n) SET n:{labels[0]}, n.testprop='testvalue'")
        self.env.assertEqual(result.labels_added, 2)
        self.env.assertEqual(result.properties_set, 2)
        self.validate_node_labels(self.multiple_entity_graph, labels, 2)
        result = self.multiple_entity_graph.query("MATCH (n {testprop:'testvalue'}) RETURN n")
        self.env.assertEqual(len(result.result_set), 2)
    

    def test22_update_cp_nodes_labels_and_properties(self):
        labels = ["TestLabel7", "TestLabel8"]
        self.validate_node_labels(self.multiple_entity_graph, labels, 0)
        result = self.multiple_entity_graph.query("MATCH (n {testprop2:'testvalue'}) RETURN n")
        self.env.assertEqual(len(result.result_set), 0)

        result = self.multiple_entity_graph.query(f"MATCH (n), (m) SET n:{labels[0]}, n.testprop2='testvalue', m:{labels[1]}")
        self.env.assertEqual(result.labels_added, 4)
        self.env.assertEqual(result.properties_set, 2)
        self.validate_node_labels(self.multiple_entity_graph, labels, 2)
        result = self.multiple_entity_graph.query("MATCH (n {testprop2:'testvalue'}) RETURN n")
        self.env.assertEqual(len(result.result_set), 2)


    def test23_update_connected_nodes_labels_and_properties(self):
        labels = ["TestLabel9", "TestLabel10"]
        self.validate_node_labels(self.multiple_entity_graph, labels, 0)
        result = self.multiple_entity_graph.query("MATCH (n {testprop3:'testvalue'}) RETURN n")
        self.env.assertEqual(len(result.result_set), 0)

        result = self.multiple_entity_graph.query(f"MATCH (n)-->(m) SET n:{labels[0]}, n.testprop3='testvalue', m:{labels[1]}")
        self.env.assertEqual(result.labels_added, 2)
        self.env.assertEqual(result.properties_set, 1)
        self.validate_node_labels(self.multiple_entity_graph, labels, 1)
        result = self.multiple_entity_graph.query("MATCH (n {testprop3:'testvalue'}) RETURN n")
        self.env.assertEqual(len(result.result_set), 1)


    def test_24_fail_update_non_matched_nodes(self):
        queries = ["MATCH (n) SET x:L", "MATCH (n) SET x:L:L:L"]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("'x' not defined", str(e))


    def test_25_fail_update_labels_for_edge(self):
        queries = ["MATCH ()-[r]->() SET r:L", "MATCH (n)-[r]->(m) WITH n, r, m UNWIND [n, r, m] AS x SET x:L"]
        for query in queries:
            try:
                self.multiple_entity_graph.query(query)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Type mismatch: expected Node but was Relationship", str(e))
    

    def test_26_fail_update_label_for_constant(self):
        queries = ["WITH 1 AS x SET x:L"]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Update error: alias 'x' did not resolve to a graph entity", str(e))
    

    def test_27_set_label_on_merge(self):
        # on match
        labels = ["Trigger", "TestLabel11", "TestLabel12"]
        self.validate_node_labels(self.graph, labels, 0)
        # This will create a node with Trigger label, and set the label TestLabel11
        result = self.graph.query(f"MERGE(n:{labels[0]}) ON CREATE SET n:{labels[1]} ON MATCH SET n:{labels[2]}")
        self.env.assertEqual(result.labels_added, 2)
        self.validate_node_labels(self.graph, labels[0:1], 1)
        # This will find a node with Trigger label and set the label TestLabel12
        result = self.graph.query(f"MERGE(n:{labels[0]}) ON CREATE SET n:{labels[1]} ON MATCH SET n:{labels[2]}")
        self.env.assertEqual(result.labels_added, 1)
        self.validate_node_labels(self.graph, labels, 1)

    
    def test_28_remove_node_labels(self):
        self.graph.delete()
        self.graph.query("CREATE ()")
        labels = ["Foo", "Bar"]
        self.validate_node_labels(self.graph, labels, 0)

        result = self.graph.query(f"MATCH (n) SET n:{':'.join(labels)}")
        self.env.assertEqual(result.labels_added, 2)
        self.validate_node_labels(self.graph, labels, 1)
        for label in labels:
            self.graph.query(f"MATCH (n:{label}) REMOVE n:{label} RETURN 1")
            self.validate_node_labels(self.graph, [label], 0)
        self.validate_node_labels(self.graph, labels, 0)

    def test_29_mix_add_and_remove_node_labels(self):
        self.graph.delete()
        self.graph.query("CREATE (:Foo)")
        labels_to_add = ["Bar"]
        labels_to_remove = ["Foo"]
        self.validate_node_labels(self.graph, labels_to_remove, 1)

        # call set prior to remove
        result = self.graph.query(f"MATCH (n:Foo) SET n:{':'.join(labels_to_add)} REMOVE n:{':'.join(labels_to_remove)} RETURN 1")
        self.env.assertEqual(result.labels_added, 1)
        self.validate_node_labels(self.graph, labels_to_remove, 0)
        self.validate_node_labels(self.graph, labels_to_add, 1)

        self.graph.delete()
        self.graph.query("CREATE (:Foo)")
        self.validate_node_labels(self.graph, labels_to_remove, 1)

        # call remove prior to set
        result = self.graph.query(f"MATCH (n:Foo) REMOVE n:{':'.join(labels_to_remove)} SET n:{':'.join(labels_to_add)} RETURN 1")
        self.env.assertEqual(result.labels_added, 1)
        self.validate_node_labels(self.graph, labels_to_remove, 0)
        self.validate_node_labels(self.graph, labels_to_add, 1)

    def test_30_mix_add_and_remove_same_labels(self):
        self.graph.delete()
        self.graph.query("CREATE ()")
        labels = ["Foo"]
        self.validate_node_labels(self.graph, labels, 0)

        # call set prior to remove
        result = self.graph.query(f"MATCH (n) SET n:{':'.join(labels)} REMOVE n:{':'.join(labels)} RETURN 1")
        self.env.assertEqual(result.labels_added, 1)
        self.env.assertEqual(result.labels_removed, 1)
        self.validate_node_labels(self.graph, labels, 0)

        self.graph.delete()
        self.graph.query("CREATE ()")
        self.validate_node_labels(self.graph, labels, 0)

        # call remove prior to set
        result = self.graph.query(f"MATCH (n) REMOVE n:{':'.join(labels)} SET n:{':'.join(labels)} RETURN 1")
        self.env.assertEqual(result.labels_added, 1)
        self.env.assertEqual(result.labels_removed, 0)
        self.validate_node_labels(self.graph, labels, 1)

    def test_32_mix_merge_and_remove_node_labels(self):
        self.graph.delete()
        labels_to_remove = ["Foo"]
        self.validate_node_labels(self.graph, labels_to_remove, 0)

        result = self.graph.query(f"MERGE (n:{':'.join(labels_to_remove)})  REMOVE n:{':'.join(labels_to_remove)} RETURN 1")
        self.env.assertEqual(result.labels_added, 1)
        self.env.assertEqual(result.labels_removed, 1)
        self.validate_node_labels(self.graph, labels_to_remove, 0)

    def test_33_syntax_error_remove_labels_on_match_on_create(self):
        queries = ["MERGE (n) ON MATCH REMOVE n:Foo RETURN 1", "MERGE (n) ON CREATE REMOVE n:Foo RETURN 1"]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Invalid input 'R':", str(e))

    def test_34_fail_remove_labels_for_edge(self):
        queries = ["MATCH ()-[r]->() REMOVE r:L RETURN 1", "MATCH (n)-[r]->(m) WITH n, r, m UNWIND [n, r, m] AS x REMOVE x:L RETURN 1"]
        for query in queries:
            try:
                self.multiple_entity_graph.query(query)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Type mismatch: expected Node but was Relationship", str(e))
    
    def test_35_fail_remove_label_for_constant(self):
        queries = ["WITH 1 AS x REMOVE x:L RETURN x"]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Update error: alias 'x' did not resolve to a graph entity", str(e))

        queries = ["REMOVE NULL.v",
                   "REMOVE 1.v",
                   "REMOVE 'a'.v",
                   "REMOVE f(1).v"]
        for q in queries:
            try:
                self.graph.query(q)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("REMOVE operates on either a node, relationship or a map", str(e))

    def test_36_mix_add_and_remove_node_properties(self):
        self.graph.delete()
        self.graph.query("CREATE ({v:1})")
        result = self.graph.query("MATCH (n {v:1}) REMOVE n.v SET n.v=1")
        self.env.assertEqual(result.properties_set, 1)
        self.env.assertEqual(result.properties_removed, 1)

    def test_37_set_property_null(self):
        self.graph.delete()
        self.graph.query("CREATE ()")
        result = self.graph.query("MATCH (v) SET v.p1 = v.p8, v.p1 = v.p5, v.p2 = v.p4")
        result = self.graph.query("MATCH (v) RETURN v")
        self.env.assertEqual(result.header, [[1, 'v']])

    def test38_accumulating_updates(self):
        """Tests that updates are performed relative to the latest update"""
        self.graph.delete()

        # create a node with property `v` with value 1
        self.graph.query("CREATE ({v: 1})")

        res = self.graph.query("MATCH (n) UNWIND [0, 1, 2, 3] AS x SET n.v = n.v + x RETURN n")

        # assert results
        self.env.assertEquals(res.result_set[0][0], Node(properties={'v': 7}))

    # Set the entity's properties to itself
    def test39_assign_self(self):
        self.graph.delete()

        empty_node  = self.graph.query("CREATE (n) RETURN n").result_set[0][0]

        queries = ["MATCH (n) SET n = n RETURN n",
                   "MATCH (n) SET n += n RETURN n"]

        for q in queries:
            result = self.graph.query(q)
            actual_node = result.result_set[0][0]

            self.env.assertEqual(result.properties_set, 0)
            self.env.assertEqual(result.properties_removed, 0)
            self.env.assertEqual(empty_node, actual_node)

        #-----------------------------------------------------------------------

        empty_edge = self.graph.query("CREATE ()-[e:R]->() RETURN e").result_set[0][0]

        queries = ["MATCH ()-[e]->() SET e = e RETURN e",
                   "MATCH ()-[e]->() SET e += e RETURN e"]

        for q in queries:
            result = self.graph.query(q)
            actual_edge = result.result_set[0][0]

            self.env.assertEqual(result.properties_set, 0)
            self.env.assertEqual(result.properties_removed, 0)
            self.env.assertEqual(empty_edge, actual_edge)

        self.graph.delete()

        #-----------------------------------------------------------------------

        node = self.graph.query("CREATE (n {a:1, b:'str', c:vecf32([1,2,3])}) RETURN n").result_set[0][0]

        queries = ["MATCH (n) SET n = n RETURN n",
                   "MATCH (n) SET n += n RETURN n"]

        for q in queries:
            result = self.graph.query(q)
            actual_node = result.result_set[0][0]

            self.env.assertEqual(result.properties_set, 0)
            self.env.assertEqual(result.properties_removed, 0)
            self.env.assertEqual(node, actual_node)

        #-----------------------------------------------------------------------

        edge = self.graph.query("CREATE ()-[e:R {a:1, b:'str', c:vecf32([1,2,3])}]->() RETURN e").result_set[0][0]

        queries = ["MATCH ()-[e]->() SET e = e RETURN e",
                   "MATCH ()-[e]->() SET e += e RETURN e"]

        for q in queries:
            result = self.graph.query(q)
            actual_edge = result.result_set[0][0]

            self.env.assertEqual(result.properties_set, 0)
            self.env.assertEqual(result.properties_removed, 0)
            self.env.assertEqual(edge, actual_edge)

    # Clear attributes via map
    def test40_remove_by_map(self):
        self.graph.delete()

        self.graph.query("CREATE (n {a:1, b:2, c: 'str'}) RETURN n").result_set[0][0]

        update_map = {'a': 2, 'b':None, 'c':'str'}
        q = "MATCH (n) SET n = $map RETURN n"

        result = self.graph.query(q, {'map': update_map})
        actual_node = result.result_set[0][0]

        self.env.assertEqual(result.properties_set, 2)
        self.env.assertEqual(result.properties_removed, 3)

        self.env.assertEqual(len(actual_node.properties), 2)
        self.env.assertEqual(actual_node.properties['a'], 2)
        self.env.assertEqual(actual_node.properties['c'], 'str')

        #-----------------------------------------------------------------------

        self.graph.delete()

        self.graph.query("CREATE (n {a:1, b:2, c: 'str'}) RETURN n").result_set[0][0]

        update_map = {'a': 2, 'b':None, 'c':'str'}
        q = "MATCH (n) SET n += $map RETURN n"

        result = self.graph.query(q, {'map': update_map})
        actual_node = result.result_set[0][0]

        self.env.assertEqual(result.properties_set, 1)
        self.env.assertEqual(result.properties_removed, 2)

        self.env.assertEqual(len(actual_node.properties), 2)
        self.env.assertEqual(actual_node.properties['a'], 2)
        self.env.assertEqual(actual_node.properties['c'], 'str')

    # multiple updates to the same entity
    def test41_last_update_persists(self):
        self.graph.delete()

        v = self.graph.query("""CREATE (n)
                                WITH n AS n, n AS m
                                SET n.v = 1,
                                    m.v = 3,
                                    n.v = 2,
                                    m.v = null,
                                    m.v = 2,
                                    n.v = null,
                                    m.v = 1,
                                    n.v = 3,
                                    m.v = 4,
                                    n.v = 4
                                RETURN n.v""").result_set[0][0]
        self.env.assertEqual(v, 4)

        #-----------------------------------------------------------------------

        self.graph.delete()

        self.graph.query("CREATE (n) RETURN n").result_set[0][0]

        update_map_1 = {'a': 2, 'b':None, 'c':'str'}
        update_map_2 = {'a': None, 'b':None, 'c':'str1'}
        update_map_3 = {'a': 4, 'b':None, 'c':'str'}
        q = "MATCH (n) SET n = $map1, n = $map2, n = $map3 RETURN n"

        result = self.graph.query(q, {'map1': update_map_1,
                                      'map2': update_map_2,
                                      'map3': update_map_3})
        actual_node = result.result_set[0][0]

        self.env.assertEqual(result.properties_set, 2)
        self.env.assertEqual(result.properties_removed, 0)

        self.env.assertEqual(len(actual_node.properties), 2)
        self.env.assertEqual(actual_node.properties['a'], 4)
        self.env.assertEqual(actual_node.properties['c'], 'str')

        #-----------------------------------------------------------------------

        self.graph.delete()

        self.graph.query("CREATE (n) RETURN n").result_set[0][0]

        update_map_1 = {'a': 2, 'b':None, 'c':'str'}
        update_map_2 = {'a': None, 'b':None, 'c':'str1'}
        update_map_3 = {'a': 4, 'b':None, 'c':'str'}
        q = "MATCH (n) SET n += $map1, n += $map2, n += $map3 RETURN n"

        result = self.graph.query(q, {'map1': update_map_1,
                                      'map2': update_map_2,
                                      'map3': update_map_3})
        actual_node = result.result_set[0][0]

        self.env.assertEqual(len(actual_node.properties), 2)
        self.env.assertEqual(actual_node.properties['a'], 4)
        self.env.assertEqual(actual_node.properties['c'], 'str')

        #-----------------------------------------------------------------------

        self.graph.delete()

        self.graph.query("CREATE (n {a:1, b:2, c: 'str'}) RETURN n").result_set[0][0]

        update_map_1 = {'a': 2, 'b':None, 'c':'str'}
        update_map_2 = {'a': None, 'b':None, 'c':'str1'}
        update_map_3 = {'a': 4, 'b':None, 'c':'str'}
        q = "MATCH (n) SET n = $map1, n = $map2, n = $map3 RETURN n"

        result = self.graph.query(q, {'map1': update_map_1,
                                      'map2': update_map_2,
                                      'map3': update_map_3})
        actual_node = result.result_set[0][0]

        self.env.assertEqual(result.properties_set, 2)
        self.env.assertEqual(result.properties_removed, 3)

        self.env.assertEqual(len(actual_node.properties), 2)
        self.env.assertEqual(actual_node.properties['a'], 4)
        self.env.assertEqual(actual_node.properties['c'], 'str')

        #-----------------------------------------------------------------------

        self.graph.delete()

        self.graph.query("CREATE (n {a:1, b:2, c: 'str'}) RETURN n").result_set[0][0]

        update_map_1 = {'a': 2, 'b':None, 'c':'str'}
        update_map_2 = {'a': None, 'b':None, 'c':'str1'}
        update_map_3 = {'a': 4, 'b':None, 'c':'str'}
        q = "MATCH (n) SET n += $map1, n += $map2, n += $map3 RETURN n"

        result = self.graph.query(q, {'map1': update_map_1,
                                      'map2': update_map_2,
                                      'map3': update_map_3})
        actual_node = result.result_set[0][0]

        self.env.assertEqual(len(actual_node.properties), 2)
        self.env.assertEqual(actual_node.properties['a'], 4)
        self.env.assertEqual(actual_node.properties['c'], 'str')

class testEntityUpdateReplication():
    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True)
        self.master = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        self.master_graph = Graph(self.master, GRAPH_ID)
        self.replica_graph = Graph(self.replica, GRAPH_ID)

    # verify label matrix is resized and updated
    # when it is dimensions are lagging behind
    def test01_update_node_labels(self):
        """scenario to test:
        1. Introduce label X
           X is of size MxM
        2. Create enough nodes to cause X to require a resize
        3. Update node n with ID > M to be associated with label X
        4. Verify n's labels
        5. Verify labels statistics
        """

        M = self.db.config_get("NODE_CREATION_BUFFER")

        # introduce label X
        q = "CREATE (:X)"
        res = self.master_graph.query(q)
        self.env.assertEqual(res.labels_added,  1)
        self.env.assertEqual(res.nodes_created, 1)

        # create enough nodes to cause X matrix dimensions to lag behind the graph
        q = f"UNWIND range(1, {M*3}) AS x CREATE ()"
        res = self.master_graph.query(q)
        self.env.assertEqual(res.nodes_created, M*3)

        # update node with internal ID 3M to be associated with label X
        q = f"MATCH (n) WHERE ID(n) = {M*3} SET n:X RETURN n, labels(n)"
        res = self.master_graph.query(q).result_set

        # wait for replica
        self.master.wait(1, 0)

        n = res[0][0]
        self.env.assertEqual(n.labels, ["X"])

        n_lbls = res[0][1]
        self.env.assertEqual(n_lbls, ["X"])

        # verify number of nodes associated with the label X
        queries = [
                "MATCH (n:X) RETURN count(n)",  # uses graph internal stas
                "MATCH (n:X) RETURN count(1)"   # perform actual counting
        ]

        for q in queries:
            x_count = self.master_graph.query(q).result_set[0][0]
            self.env.assertEqual(x_count, 2)

            x_count = self.replica_graph.ro_query(q).result_set[0][0]
            self.env.assertEqual(x_count, 2)

    # verify label matrix is resized and updated
    # when it is dimensions are lagging behind
    def test02_remove_node_label(self):
        """scenario to test:
        1. Introduce label X
           X is of size MxM
        2. Create enough nodes to cause X to require a resize
        3. Remove label X from node n with ID > M
           although n isn't associated with X
        4. Verify n's labels
        5. Verify labels statistics
        """

        self.master_graph.delete()

        M = self.db.config_get("NODE_CREATION_BUFFER")

        # introduce label X
        q = "CREATE (:X)"
        res = self.master_graph.query(q)
        self.env.assertEqual(res.labels_added,  1)
        self.env.assertEqual(res.nodes_created, 1)

        # create enough nodes to cause X matrix dimensions to lag behind the graph
        q = f"UNWIND range(1, {M*3}) AS x CREATE ()"
        res = self.master_graph.query(q)
        self.env.assertEqual(res.nodes_created, M*3)

        # update node with internal ID 3M to be associated with label X
        q = f"MATCH (n) WHERE ID(n) = {M*3} REMOVE n:X RETURN n, labels(n)"
        res = self.master_graph.query(q).result_set

        # wait for replica
        self.master.wait(1, 0)

        n = res[0][0]
        self.env.assertEqual(n.labels, None)

        n_lbls = res[0][1]
        self.env.assertEqual(n_lbls, [])

        # verify number of nodes associated with the label X
        queries = [
                "MATCH (n:X) RETURN count(n)",  # uses graph internal stas
                "MATCH (n:X) RETURN count(1)"   # perform actual counting
        ]

        for q in queries:
            x_count = self.master_graph.query(q).result_set[0][0]
            self.env.assertEqual(x_count, 1)

            x_count = self.replica_graph.ro_query(q).result_set[0][0]
            self.env.assertEqual(x_count, 1)

    # verify label matrix is resized and updated
    # when it is dimensions are lagging behind
    def test03_merge_update_node_labels(self):
        """scenario to test:
        1. Introduce labels X & Z
           X is of size MxM
           Z is of size MxM
        2. Create enough nodes to cause X & Z matrices to require a resize
        3. Update node n with ID > M to be associated with label X
        4. Create a new node and associate it with label Z
        5. Verify nodes labels
        6. Verify labels statistics
        """

        self.master_graph.delete()

        M = self.db.config_get("NODE_CREATION_BUFFER")

        # introduce label X & Z
        q = "CREATE (:X), (:Z)"
        res = self.master_graph.query(q)
        self.env.assertEqual(res.labels_added,  2)
        self.env.assertEqual(res.nodes_created, 2)

        # create enough nodes to cause the X & Z matrices dimensions to lag behind the graph
        q = "UNWIND range(1, $end) AS x CREATE ({v:x})"
        res = self.master_graph.query(q, {'end': M*3})
        self.env.assertEqual(res.nodes_created, M*3)

        # update node with internal ID 3M to be associated with label X
        q = "MERGE (n {v:$v}) ON MATCH SET n:X RETURN n, labels(n)"
        res_x = self.master_graph.query(q, {'v': M*3}).result_set

        # merge a new node and associate it with label Z
        q = "MERGE (n {v:-2}) ON CREATE SET n:Z RETURN n, labels(n)"
        res_z = self.master_graph.query(q).result_set

        # wait for replica
        self.master.wait(1, 0)

        n = res_x[0][0]
        self.env.assertEqual(n.labels, ['X'])

        n_lbls = res_x[0][1]
        self.env.assertEqual(n_lbls, ['X'])

        n = res_z[0][0]
        self.env.assertEqual(n.labels, ['Z'])

        n_lbls = res_z[0][1]
        self.env.assertEqual(n_lbls, ['Z'])

        # verify number of nodes associated with the label X
        queries = [
                "MATCH (n:X) RETURN count(n)",  # uses graph internal stas
                "MATCH (n:X) RETURN count(1)",  # perform actual counting
                "MATCH (n:Z) RETURN count(n)",  # uses graph internal stas
                "MATCH (n:Z) RETURN count(1)"   # perform actual counting
        ]

        for q in queries:
            count = self.master_graph.query(q).result_set[0][0]
            self.env.assertEqual(count, 2)

            count = self.replica_graph.ro_query(q).result_set[0][0]
            self.env.assertEqual(count, 2)

