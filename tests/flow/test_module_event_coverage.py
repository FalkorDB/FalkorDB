from common import *

GRAPH_ID = "test_graph"

class TestModuleEventHandlers(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
    
    def test01_rename_preserves_graph_data(self):
        """Test that renaming a graph preserves all data and metadata"""
        # Create a graph with nodes and edges
        self.graph.query("CREATE (:Person {name: 'Alice', age: 30})")
        self.graph.query("CREATE (:Person {name: 'Bob', age: 25})")
        self.graph.query("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)")
        
        # Verify initial data
        result = self.graph.query("MATCH (n) RETURN count(n) as count")
        self.env.assertEquals(result.result_set[0][0], 2)
        
        result = self.graph.query("MATCH ()-[r]->() RETURN count(r) as count")
        self.env.assertEquals(result.result_set[0][0], 1)
        
        # Rename the graph
        new_name = "renamed_graph"
        self.redis_con.rename(GRAPH_ID, new_name)
        
        # Access renamed graph
        renamed_graph = self.db.select_graph(new_name)
        
        # Verify data is preserved
        result = renamed_graph.query("MATCH (n) RETURN count(n) as count")
        self.env.assertEquals(result.result_set[0][0], 2)
        
        result = renamed_graph.query("MATCH ()-[r]->() RETURN count(r) as count")
        self.env.assertEquals(result.result_set[0][0], 1)
        
        # Verify we can still modify the graph
        renamed_graph.query("CREATE (:Person {name: 'Charlie', age: 35})")
        result = renamed_graph.query("MATCH (n) RETURN count(n) as count")
        self.env.assertEquals(result.result_set[0][0], 3)
        
        # Cleanup
        renamed_graph.delete()
    
    def test02_rename_with_hash_tag(self):
        """Test renaming graphs with hash tags (cluster sharding)"""
        # Create graph with hash tag
        tagged_graph_id = "{shard1}graph_a"
        tagged_graph = self.db.select_graph(tagged_graph_id)
        tagged_graph.query("CREATE (:Node {id: 1})")
        
        # Verify data
        result = tagged_graph.query("MATCH (n) RETURN n.id as id")
        self.env.assertEquals(result.result_set[0][0], 1)
        
        # Rename to another name with same shard
        new_tagged_name = "{shard1}graph_b"
        self.redis_con.rename(tagged_graph_id, new_tagged_name)
        
        # Access renamed graph
        renamed_tagged = self.db.select_graph(new_tagged_name)
        result = renamed_tagged.query("MATCH (n) RETURN n.id as id")
        self.env.assertEquals(result.result_set[0][0], 1)
        
        # Cleanup
        renamed_tagged.delete()
    
    def test03_delete_removes_graph(self):
        """Test that DEL properly removes graph from tracking"""
        test_graph_id = "graph_to_delete"
        test_graph = self.db.select_graph(test_graph_id)
        test_graph.query("CREATE (:Node {id: 1})")
        
        # Verify graph exists
        result = test_graph.query("MATCH (n) RETURN count(n) as count")
        self.env.assertEquals(result.result_set[0][0], 1)
        
        # Delete the key using Redis DEL
        self.redis_con.delete(test_graph_id)
        
        # Create new graph with same name - should start fresh
        new_graph = self.db.select_graph(test_graph_id)
        result = new_graph.query("MATCH (n) RETURN count(n) as count")
        self.env.assertEquals(result.result_set[0][0], 0)
        
        # Cleanup
        new_graph.delete()
    
    def test04_verbose_output_format(self):
        """Test verbose output formatting for various data types"""
        test_graph_id = "verbose_test_graph"
        test_graph = self.db.select_graph(test_graph_id)
        
        # Create nodes with various property types
        test_graph.query("CREATE (:TestNode {string_prop: 'test', int_prop: 42, float_prop: 3.14, bool_prop: true})")
        
        # Query with verbose results
        result = test_graph.query("MATCH (n:TestNode) RETURN n")
        
        # Verify we got results
        self.env.assertEquals(len(result.result_set), 1)
        
        # Verify node has expected properties
        node = result.result_set[0][0]
        self.env.assertEquals(node.properties['string_prop'], 'test')
        self.env.assertEquals(node.properties['int_prop'], 42)
        self.env.assertAlmostEqual(node.properties['float_prop'], 3.14, places=2)
        self.env.assertEquals(node.properties['bool_prop'], True)
        
        # Cleanup
        test_graph.delete()
    
    def test05_empty_graph_stats(self):
        """Test statistics output for empty results"""
        test_graph_id = "empty_stats_graph"
        test_graph = self.db.select_graph(test_graph_id)
        
        # Query on empty graph
        result = test_graph.query("MATCH (n) RETURN n")
        
        # Verify empty result set
        self.env.assertEquals(len(result.result_set), 0)
        
        # Verify statistics are present
        self.env.assertTrue(hasattr(result, 'run_time_ms'))
        
        # Cleanup
        test_graph.delete()
    
    def test06_multiple_label_node(self):
        """Test verbose formatting of nodes with multiple labels"""
        test_graph_id = "multi_label_graph"
        test_graph = self.db.select_graph(test_graph_id)
        
        # Create node with multiple labels
        test_graph.query("CREATE (:Person:Employee {name: 'John'})")
        
        # Query the node
        result = test_graph.query("MATCH (n) RETURN n")
        
        # Verify node has multiple labels
        node = result.result_set[0][0]
        labels = node.labels
        self.env.assertIn('Person', labels)
        self.env.assertIn('Employee', labels)
        self.env.assertEquals(len(labels), 2)
        
        # Cleanup
        test_graph.delete()
    
    def test07_null_properties(self):
        """Test handling of null property values"""
        test_graph_id = "null_props_graph"
        test_graph = self.db.select_graph(test_graph_id)
        
        # Create node with null property
        test_graph.query("CREATE (:Node {name: 'test', nullable: null})")
        
        # Query the node
        result = test_graph.query("MATCH (n:Node) RETURN n.name, n.nullable")
        
        # Verify null is handled correctly
        self.env.assertEquals(result.result_set[0][0], 'test')
        self.env.assertEquals(result.result_set[0][1], None)
        
        # Cleanup
        test_graph.delete()
    
    def test08_large_result_stats(self):
        """Test statistics with various operations"""
        test_graph_id = "stats_graph"
        test_graph = self.db.select_graph(test_graph_id)
        
        # Create multiple nodes
        result = test_graph.query("UNWIND range(1, 10) AS i CREATE (:Node {id: i})")
        self.env.assertEquals(result.nodes_created, 10)
        self.env.assertEquals(result.properties_set, 10)
        
        # Create relationships
        result = test_graph.query("MATCH (a:Node), (b:Node) WHERE a.id < b.id CREATE (a)-[:CONNECTS]->(b)")
        self.env.assertTrue(result.relationships_created > 0)
        
        # Delete some nodes
        result = test_graph.query("MATCH (n:Node) WHERE n.id > 5 DELETE n")
        self.env.assertEquals(result.nodes_deleted, 5)
        
        # Verify remaining nodes
        result = test_graph.query("MATCH (n) RETURN count(n) as count")
        self.env.assertEquals(result.result_set[0][0], 5)
        
        # Cleanup
        test_graph.delete()
