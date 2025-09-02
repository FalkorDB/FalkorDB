#!/usr/bin/env python3

"""
Test for the EXISTS pattern issue fix - issue #1248

This test validates that EXISTS properly handles pattern expressions
and returns false when patterns don't match, instead of incorrectly
returning true.
"""

import sys
import os

# Add the tests directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(current_dir)
sys.path.append(tests_dir)

from flow.test_env import FlowTestsBase

class TestExistsPatternIssue1248(FlowTestsBase):
    """Test cases for EXISTS pattern issue #1248"""
    
    def test_exists_pattern_returns_false_when_no_match(self):
        """
        Test case for issue #1248: EXISTS on path returns True when it should return false.
        
        Reproduces the exact issue:
        - CREATE (user:User) creates a single user node
        - EXISTS((user)<-[:AUTHENTICATES]-(:Identity)) should return false
        - Previously returned true (bug), now should return false
        """
        
        # Create a graph for this test
        graph = self.db.select_graph("test_exists_issue_1248")
        
        # Step 1: Create a user node (matching the issue description)
        result = graph.query("CREATE (user:User)")
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.labels_added, 1)
        
        # Step 2: Test EXISTS with non-matching pattern
        # This is the exact query from the issue report
        query = "MATCH (user) RETURN EXISTS((user)<-[:AUTHENTICATES]-(:Identity)) AS result"
        result = graph.query(query)
        
        # Validate the result
        result_set = result.result_set
        self.env.assertEquals(len(result_set), 1)
        exists_result = result_set[0][0]
        
        # This should be False (was previously True due to the bug)
        self.env.assertEquals(exists_result, False,
            "EXISTS should return false when pattern doesn't match any nodes/relationships")
    
    def test_exists_pattern_returns_true_when_match_exists(self):
        """
        Test that EXISTS correctly returns true when the pattern actually matches.
        """
        
        # Create a graph for this test
        graph = self.db.select_graph("test_exists_true_1248")
        
        # Create the required nodes and relationships
        graph.query("CREATE (user:User)")
        graph.query("CREATE (identity:Identity)")
        graph.query("MATCH (user:User), (identity:Identity) CREATE (user)<-[:AUTHENTICATES]-(identity)")
        
        # Test EXISTS with matching pattern
        query = "MATCH (user) RETURN EXISTS((user)<-[:AUTHENTICATES]-(:Identity)) AS result"
        result = graph.query(query)
        
        # Validate the result
        result_set = result.result_set
        self.env.assertEquals(len(result_set), 1)
        exists_result = result_set[0][0]
        
        # This should be True
        self.env.assertEquals(exists_result, True,
            "EXISTS should return true when pattern matches existing nodes/relationships")
    
    def test_exists_property_functionality_preserved(self):
        """
        Test that EXISTS still works correctly for property existence checks.
        This ensures our fix doesn't break existing functionality.
        """
        
        # Create a graph for this test
        graph = self.db.select_graph("test_exists_property_1248")
        
        # Create nodes with and without properties
        graph.query("CREATE (n1:Node {name: 'test', age: 30})")
        graph.query("CREATE (n2:Node {name: 'test2'})")
        graph.query("CREATE (n3:Node)")
        
        # Test EXISTS with existing property
        result = graph.query("MATCH (n:Node) WHERE EXISTS(n.name) RETURN count(n)")
        result_set = result.result_set
        self.env.assertEquals(result_set[0][0], 2)
        
        # Test EXISTS with non-existing property
        result = graph.query("MATCH (n:Node) WHERE EXISTS(n.age) RETURN count(n)")
        result_set = result.result_set
        self.env.assertEquals(result_set[0][0], 1)
        
        # Test EXISTS with property that doesn't exist on any node
        result = graph.query("MATCH (n:Node) WHERE EXISTS(n.nonexistent) RETURN count(n)")
        result_set = result.result_set
        self.env.assertEquals(result_set[0][0], 0)

    def test_various_pattern_formats(self):
        """
        Test EXISTS with various pattern formats to ensure consistent behavior.
        """
        
        # Create a graph for this test
        graph = self.db.select_graph("test_exists_patterns_1248")
        
        # Create a simple graph structure
        graph.query("CREATE (a:A)-[:REL]->(b:B)")
        
        # Test various non-matching patterns - all should return false
        patterns_should_be_false = [
            "EXISTS((a)<-[:NONEXISTENT]-(:C))",
            "EXISTS((a)-[:NONEXISTENT]->(:C))",
            "EXISTS((:C)-[:REL]->(a))",
            "EXISTS((a)-[:REL]->(:C))",
        ]
        
        for pattern in patterns_should_be_false:
            query = f"MATCH (a:A) RETURN {pattern} AS result"
            result = graph.query(query)
            result_set = result.result_set
            exists_result = result_set[0][0]
            self.env.assertEquals(exists_result, False,
                f"Pattern {pattern} should return false when no match exists")
        
        # Test matching patterns - should return true
        patterns_should_be_true = [
            "EXISTS((a)-[:REL]->(:B))",
            "EXISTS((:B)<-[:REL]-(a))",
        ]
        
        for pattern in patterns_should_be_true:
            query = f"MATCH (a:A) RETURN {pattern} AS result"
            result = graph.query(query)
            result_set = result.result_set
            exists_result = result_set[0][0]
            self.env.assertEquals(exists_result, True,
                f"Pattern {pattern} should return true when match exists")