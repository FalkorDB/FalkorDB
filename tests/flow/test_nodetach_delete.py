#!/usr/bin/env python3

from common import *

GRAPH_ID = "test_nodetach_delete"

class testNodetachDelete():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_nodetach_delete_node_without_relationships(self):
        """Test NODETACH DELETE on nodes with no relationships - should succeed"""
        # Create isolated nodes
        self.graph.query("CREATE (:Person {name: 'Alice'}), (:Person {name: 'Bob'})")
        
        # NODETACH DELETE should work on nodes without relationships
        result = self.graph.query("MATCH (p:Person {name: 'Alice'}) NODETACH DELETE p")
        self.env.assertEquals(result.nodes_deleted, 1)
        
        # Verify the node was deleted
        result = self.graph.query("MATCH (p:Person {name: 'Alice'}) RETURN p")
        self.env.assertEquals(len(result.result_set), 0)
        
        # Other node should still exist
        result = self.graph.query("MATCH (p:Person {name: 'Bob'}) RETURN p")
        self.env.assertEquals(len(result.result_set), 1)

    def test02_nodetach_delete_node_with_relationships_should_fail(self):
        """Test NODETACH DELETE on nodes with relationships - should fail"""
        # Clear any existing data
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        # Create nodes with relationships
        self.graph.query("""
            CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})
        """)
        
        # NODETACH DELETE should fail when the node has relationships
        try:
            self.graph.query("MATCH (p:Person {name: 'Alice'}) NODETACH DELETE p")
            self.env.assertTrue(False, "Expected NODETACH DELETE to fail on node with relationships")
        except Exception as e:
            # Should get an error about relationships existing
            self.env.assertContains("relationships", str(e).lower())

        # Both nodes should still exist
        result = self.graph.query("MATCH (p:Person) RETURN COUNT(p)")
        self.env.assertEquals(result.result_set[0][0], 2)
        
        # Relationship should still exist
        result = self.graph.query("MATCH ()-[r:KNOWS]->() RETURN COUNT(r)")
        self.env.assertEquals(result.result_set[0][0], 1)

    def test03_nodetach_delete_multiple_nodes(self):
        """Test NODETACH DELETE on multiple nodes"""
        # Clear graph
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        # Create mix of connected and isolated nodes
        self.graph.query("""
            CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'}),
                   (c:Person {name: 'Charlie'}),
                   (d:Person {name: 'David'})
        """)
        
        # Try to delete all nodes with NODETACH DELETE - should fail for connected ones
        try:
            self.graph.query("MATCH (p:Person) NODETACH DELETE p")
            self.env.assertTrue(False, "Expected NODETACH DELETE to fail on nodes with relationships")
        except Exception as e:
            self.env.assertContains("relationships", str(e).lower())

        # All nodes should still exist
        result = self.graph.query("MATCH (p:Person) RETURN COUNT(p)")
        self.env.assertEquals(result.result_set[0][0], 4)

    def test04_nodetach_delete_vs_regular_delete(self):
        """Test difference between NODETACH DELETE and regular DELETE"""
        # Clear graph
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        # Create isolated nodes for comparison
        self.graph.query("""
            CREATE (:Test {id: 1}), (:Test {id: 2})
        """)
        
        # Regular DELETE should work on isolated nodes
        result = self.graph.query("MATCH (t:Test {id: 1}) DELETE t")
        self.env.assertEquals(result.nodes_deleted, 1)
        
        # NODETACH DELETE should also work on isolated nodes
        result = self.graph.query("MATCH (t:Test {id: 2}) NODETACH DELETE t")
        self.env.assertEquals(result.nodes_deleted, 1)
        
        # Both should be deleted
        result = self.graph.query("MATCH (t:Test) RETURN COUNT(t)")
        self.env.assertEquals(result.result_set[0][0], 0)

    def test05_nodetach_delete_vs_detach_delete(self):
        """Test difference between NODETACH DELETE and DETACH DELETE"""
        # Clear graph
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        # Create connected nodes
        self.graph.query("""
            CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'}),
                   (c:Person {name: 'Charlie'})-[:KNOWS]->(d:Person {name: 'David'})
        """)
        
        # DETACH DELETE should work on connected nodes
        result = self.graph.query("MATCH (p:Person {name: 'Alice'}) DETACH DELETE p")
        self.env.assertEquals(result.nodes_deleted, 1)
        self.env.assertEquals(result.relationships_deleted, 1)
        
        # NODETACH DELETE should fail on connected nodes
        try:
            self.graph.query("MATCH (p:Person {name: 'Charlie'}) NODETACH DELETE p")
            self.env.assertTrue(False, "Expected NODETACH DELETE to fail")
        except Exception as e:
            self.env.assertContains("relationships", str(e).lower())

        # Charlie and David should still exist with their relationship
        result = self.graph.query("MATCH (p:Person) WHERE p.name IN ['Charlie', 'David'] RETURN COUNT(p)")
        self.env.assertEquals(result.result_set[0][0], 2)
        
        result = self.graph.query("MATCH ()-[r:KNOWS]->() RETURN COUNT(r)")
        self.env.assertEquals(result.result_set[0][0], 1)

    def test06_nodetach_delete_with_self_relationships(self):
        """Test NODETACH DELETE on nodes with self-relationships"""
        # Clear graph
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        # Create node with self-relationship
        self.graph.query("""
            CREATE (a:Person {name: 'Alice'})-[:LIKES]->(a)
        """)
        
        # NODETACH DELETE should fail
        try:
            self.graph.query("MATCH (p:Person {name: 'Alice'}) NODETACH DELETE p")
            self.env.assertTrue(False, "Expected NODETACH DELETE to fail on node with self-relationship")
        except Exception as e:
            self.env.assertContains("relationships", str(e).lower())

        # Node should still exist
        result = self.graph.query("MATCH (p:Person {name: 'Alice'}) RETURN COUNT(p)")
        self.env.assertEquals(result.result_set[0][0], 1)

    def test07_nodetach_delete_with_incoming_and_outgoing_relationships(self):
        """Test NODETACH DELETE on nodes with both incoming and outgoing relationships"""
        # Clear graph
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        # Create a hub node with multiple relationships
        self.graph.query("""
            CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(hub:Person {name: 'Hub'})-[:LIKES]->(b:Person {name: 'Bob'}),
                   (c:Person {name: 'Charlie'})-[:FOLLOWS]->(hub)
        """)
        
        # NODETACH DELETE should fail on the hub node
        try:
            self.graph.query("MATCH (p:Person {name: 'Hub'}) NODETACH DELETE p")
            self.env.assertTrue(False, "Expected NODETACH DELETE to fail on hub node")
        except Exception as e:
            self.env.assertContains("relationships", str(e).lower())

        # All nodes should still exist
        result = self.graph.query("MATCH (p:Person) RETURN COUNT(p)")
        self.env.assertEquals(result.result_set[0][0], 4)
        
        # All relationships should still exist
        result = self.graph.query("MATCH ()-[r]->() RETURN COUNT(r)")
        self.env.assertEquals(result.result_set[0][0], 3)

    def test08_nodetach_delete_after_relationship_removal(self):
        """Test NODETACH DELETE succeeds after manually removing relationships"""
        # Clear graph
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        # Create connected nodes
        self.graph.query("""
            CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})
        """)
        
        # First, delete the relationship
        result = self.graph.query("MATCH ()-[r:KNOWS]->() DELETE r")
        self.env.assertEquals(result.relationships_deleted, 1)
        
        # Now NODETACH DELETE should work
        result = self.graph.query("MATCH (p:Person {name: 'Alice'}) NODETACH DELETE p")
        self.env.assertEquals(result.nodes_deleted, 1)
        
        # Only Bob should remain
        result = self.graph.query("MATCH (p:Person) RETURN p.name")
        self.env.assertEquals(len(result.result_set), 1)
        self.env.assertEquals(result.result_set[0][0], 'Bob')

    def test09_nodetach_delete_with_match_where_clause(self):
        """Test NODETACH DELETE with complex WHERE clauses"""
        # Clear graph
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        # Create test data
        self.graph.query("""
            CREATE (:Item {type: 'A', value: 1}),
                   (:Item {type: 'A', value: 2}),
                   (:Item {type: 'B', value: 3})-[:RELATES_TO]->(:Item {type: 'C', value: 4})
        """)
        
        # Delete isolated nodes of type 'A'
        result = self.graph.query("MATCH (i:Item {type: 'A'}) NODETACH DELETE i")
        self.env.assertEquals(result.nodes_deleted, 2)
        
        # Try to delete connected node - should fail
        try:
            self.graph.query("MATCH (i:Item {type: 'B'}) NODETACH DELETE i")
            self.env.assertTrue(False, "Expected NODETACH DELETE to fail")
        except Exception as e:
            self.env.assertContains("relationships", str(e).lower())

        # Should have 2 remaining nodes (B and C)
        result = self.graph.query("MATCH (i:Item) RETURN COUNT(i)")
        self.env.assertEquals(result.result_set[0][0], 2)

    def test10_nodetach_delete_return_deleted_properties(self):
        """Test returning properties of nodes deleted with NODETACH DELETE"""
        # Clear graph
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        # Create node
        self.graph.query("CREATE (:Person {name: 'Alice', age: 30})")
        
        # Delete and return properties
        result = self.graph.query("MATCH (p:Person {name: 'Alice'}) NODETACH DELETE p RETURN p.name, p.age")
        self.env.assertEquals(len(result.result_set), 1)
        self.env.assertEquals(result.result_set[0][0], 'Alice')
        self.env.assertEquals(result.result_set[0][1], 30)
        
        # Node should be deleted
        result = self.graph.query("MATCH (p:Person) RETURN COUNT(p)")
        self.env.assertEquals(result.result_set[0][0], 0)

    def test11_nodetach_delete_transaction_rollback(self):
        """Test that failed NODETACH DELETE doesn't leave partial changes"""
        # Clear graph
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        # Create test data
        self.graph.query("""
            CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'}),
                   (c:Person {name: 'Charlie'})
        """)
        
        # Try a query that should fail - deleting connected and unconnected nodes
        try:
            self.graph.query("""
                MATCH (p:Person) 
                NODETACH DELETE p 
                RETURN COUNT(p)
            """)
            self.env.assertTrue(False, "Expected query to fail")
        except Exception as e:
            self.env.assertContains("relationships", str(e).lower())

        # All original data should still be intact (transaction rollback)
        result = self.graph.query("MATCH (p:Person) RETURN COUNT(p)")
        self.env.assertEquals(result.result_set[0][0], 3)
        
        result = self.graph.query("MATCH ()-[r:KNOWS]->() RETURN COUNT(r)")
        self.env.assertEquals(result.result_set[0][0], 1)

    def test12_nodetach_delete_with_labels_and_properties(self):
        """Test NODETACH DELETE with various label and property combinations"""
        # Clear graph
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        # Create diverse test data
        self.graph.query("""
            CREATE (:Person:Employee {name: 'Alice', dept: 'IT'}),
                   (:Person {name: 'Bob'}),
                   (:Employee {name: 'Charlie', dept: 'HR'}),
                   (:Person {name: 'David'})-[:WORKS_WITH]->(:Person {name: 'Eve'})
        """)
        
        # Delete isolated Person nodes
        result = self.graph.query("""
            MATCH (p:Person) 
            WHERE NOT EXISTS((p)-[]->()) AND NOT EXISTS(()-[]->(p))
            NODETACH DELETE p
        """)
        self.env.assertEquals(result.nodes_deleted, 2)  # Bob and Alice
        
        # Remaining nodes should be Charlie, David, and Eve
        result = self.graph.query("MATCH (n) RETURN COUNT(n)")
        self.env.assertEquals(result.result_set[0][0], 3)

    def tearDown(self):
        """Clean up after tests"""
        try:
            self.graph.delete()
        except:
            pass  # Graph might already be deleted
