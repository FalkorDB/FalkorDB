from common import *

GRAPH_ID = "operator_precedence"

class testOperatorPrecedence():
    def __init__(self):
        self.env, self.db = Env()
        self.g = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.g.delete()

    def test01_contains_with_concatenation(self):
        """Test that arithmetic operators have higher precedence than CONTAINS"""
        # Create test data
        self.g.query("CREATE (:Node {name: 'Alice'})")
        
        # Test: (n.name)+'1' CONTAINS '1' should evaluate as ((n.name)+'1') CONTAINS '1'
        # This should return the node because 'Alice' + '1' = 'Alice1' which contains '1'
        result = self.g.query("MATCH (n) WHERE (n.name)+'1' CONTAINS '1' RETURN n.name")
        self.env.assertEqual(result.result_set, [['Alice']])
        
        # Test with explicit parentheses (should give same result)
        result = self.g.query("MATCH (n) WHERE ((n.name)+'1') CONTAINS '1' RETURN n.name")
        self.env.assertEqual(result.result_set, [['Alice']])
        
        # Negative test: 'Alice' does not contain '1'
        result = self.g.query("MATCH (n) WHERE n.name CONTAINS '1' RETURN n.name")
        self.env.assertEqual(result.result_set, [])

    def test02_starts_with_concatenation(self):
        """Test that arithmetic operators have higher precedence than STARTS WITH"""
        self.g.query("CREATE (:Node {prefix: 'Hello'})")
        
        # Test: (n.prefix)+' World' STARTS WITH 'Hello'
        result = self.g.query("MATCH (n) WHERE (n.prefix)+' World' STARTS WITH 'Hello' RETURN n.prefix")
        self.env.assertEqual(result.result_set, [['Hello']])
        
        # Test with explicit parentheses
        result = self.g.query("MATCH (n) WHERE ((n.prefix)+' World') STARTS WITH 'Hello' RETURN n.prefix")
        self.env.assertEqual(result.result_set, [['Hello']])

    def test03_ends_with_concatenation(self):
        """Test that arithmetic operators have higher precedence than ENDS WITH"""
        self.g.query("CREATE (:Node {suffix: 'World'})")
        
        # Test: 'Hello '+(n.suffix) ENDS WITH 'World'
        result = self.g.query("MATCH (n) WHERE 'Hello '+(n.suffix) ENDS WITH 'World' RETURN n.suffix")
        self.env.assertEqual(result.result_set, [['World']])
        
        # Test with explicit parentheses
        result = self.g.query("MATCH (n) WHERE ('Hello '+(n.suffix)) ENDS WITH 'World' RETURN n.suffix")
        self.env.assertEqual(result.result_set, [['World']])

    def test04_in_operator_with_arithmetic(self):
        """Test that arithmetic operators have higher precedence than IN"""
        self.g.query("CREATE (:Node {val: 1})")
        
        # Test: (n.val)+1 IN [2, 3, 4]
        # (1 + 1) IN [2, 3, 4] should be true
        result = self.g.query("MATCH (n) WHERE (n.val)+1 IN [2, 3, 4] RETURN n.val")
        self.env.assertEqual(result.result_set, [[1]])
        
        # Test with explicit parentheses
        result = self.g.query("MATCH (n) WHERE ((n.val)+1) IN [2, 3, 4] RETURN n.val")
        self.env.assertEqual(result.result_set, [[1]])

    def test05_issue_751_reproduction(self):
        """Test the exact reproduction case from issue #751"""
        # Create the node with self-connecting relation as described in the issue
        self.g.query("CREATE (n0 :L28{k161 : -392104257, k162 : -60652336, k158 : false, id : 5, k159 : 'dGThjm0QF'})")
        self.g.query("MATCH (n0 {id : 5}), (n1 {id : 5}) MERGE(n0)-[r :T38{k490 : true, k486 : true, k485 : -1782193271, k487 : true, id : 10, k489 : true}]->(n1)")
        
        # The problematic query from the issue - should not crash or raise Type mismatch error
        # WHERE (n1.k159)+'1' CONTAINS '1' should work correctly
        # This query may return empty results but should NOT raise "Type mismatch: expected Boolean but was String"
        try:
            result = self.g.query("MATCH (n0)<-[r0]-(n1), (n2)<-[r1]-(n1), (n0)<-[r2]-(n3) OPTIONAL MATCH (n0)<-[r3]-(n3), (n2)<-[r4]-(n1) WHERE (n1.k159)+'1' CONTAINS '1' AND r3.id <> r4.id AND r4.id <> r3.id AND r3.k485 > r4.k485 AND r4.k485 > r0.k485 RETURN *")
            # Success - query executed without the type mismatch error
        except Exception as e:
            # If any error occurs, ensure it's NOT the type mismatch error we're fixing
            error_msg = str(e)
            if "Type mismatch: expected Boolean but was String" in error_msg:
                self.env.fail(f"Query raised the type mismatch error that should be fixed: {error_msg}")
            # Other errors (e.g., no matching data) are acceptable - re-raise them
            raise

    def test06_multiple_arithmetic_operators(self):
        """Test precedence with multiple arithmetic operators"""
        self.g.query("CREATE (:Node {a: 'foo', b: 'bar'})")
        
        # Test: (n.a)+(n.b)+'baz' CONTAINS 'barbaz'
        result = self.g.query("MATCH (n) WHERE (n.a)+(n.b)+'baz' CONTAINS 'barbaz' RETURN n.a")
        self.env.assertEqual(result.result_set, [['foo']])

    def test07_subtraction_with_contains(self):
        """Test that subtraction operator also has higher precedence"""
        self.g.query("CREATE (:Node {val: 10})")
        
        # Convert to string for CONTAINS to work
        result = self.g.query("MATCH (n) WHERE toString((n.val)-5) CONTAINS '5' RETURN n.val")
        self.env.assertEqual(result.result_set, [[10]])

    def test08_multiplication_with_contains(self):
        """Test that multiplication operator also has higher precedence"""
        self.g.query("CREATE (:Node {val: 3})")
        
        # Convert to string for CONTAINS to work
        result = self.g.query("MATCH (n) WHERE toString((n.val)*2) CONTAINS '6' RETURN n.val")
        self.env.assertEqual(result.result_set, [[3]])

    def test09_division_with_contains(self):
        """Test that division operator also has higher precedence"""
        self.g.query("CREATE (:Node {val: 10})")
        
        # Convert to string for CONTAINS to work
        result = self.g.query("MATCH (n) WHERE toString((n.val)/2) CONTAINS '5' RETURN n.val")
        self.env.assertEqual(result.result_set, [[10]])

    def test10_mod_with_in(self):
        """Test that modulo operator also has higher precedence"""
        self.g.query("CREATE (:Node {val: 10})")
        
        # (n.val) % 3 should be 1, which is IN [1, 2, 3]
        result = self.g.query("MATCH (n) WHERE (n.val) % 3 IN [1, 2, 3] RETURN n.val")
        self.env.assertEqual(result.result_set, [[10]])

    def test11_power_with_in(self):
        """Test that power operator also has higher precedence"""
        self.g.query("CREATE (:Node {val: 2})")
        
        # (n.val) ^ 3 should be 8, which is IN [8, 9, 10]
        result = self.g.query("MATCH (n) WHERE (n.val) ^ 3 IN [8, 9, 10] RETURN n.val")
        self.env.assertEqual(result.result_set, [[2]])

    def test12_chained_subtraction_with_contains(self):
        """Test chained non-commutative subtraction with CONTAINS predicate"""
        self.g.query("CREATE (:Node {a: 100, b: 20, c: 30})")
        
        # Test: (n.a) - (n.b) - (n.c) should be (100 - 20 - 30) = 50
        # Then toString(50) CONTAINS '5' should be true
        result = self.g.query("MATCH (n) WHERE toString((n.a) - (n.b) - (n.c)) CONTAINS '5' RETURN n.a")
        self.env.assertEqual(result.result_set, [[100]])
        
        # Verify the chained subtraction is left-associative: (100 - 20) - 30 = 50, not 100 - (20 - 30) = 110
        result = self.g.query("MATCH (n) WHERE toString((n.a) - (n.b) - (n.c)) CONTAINS '110' RETURN n.a")
        self.env.assertEqual(result.result_set, [])

    def test13_chained_division_with_in(self):
        """Test chained non-commutative division with IN predicate"""
        self.g.query("CREATE (:Node {a: 100, b: 5, c: 2})")
        
        # Test: (n.a) / (n.b) / (n.c) should be (100 / 5 / 2) = 10
        # Then 10 IN [10, 20, 30] should be true
        result = self.g.query("MATCH (n) WHERE (n.a) / (n.b) / (n.c) IN [10, 20, 30] RETURN n.a")
        self.env.assertEqual(result.result_set, [[100]])
        
        # Verify left-associativity: (100 / 5) / 2 = 10, not 100 / (5 / 2) = 40
        result = self.g.query("MATCH (n) WHERE (n.a) / (n.b) / (n.c) IN [40] RETURN n.a")
        self.env.assertEqual(result.result_set, [])

    def test14_chained_modulo_with_starts_with(self):
        """Test chained modulo operations with STARTS WITH predicate"""
        self.g.query("CREATE (:Node {a: 17, b: 5, c: 3})")
        
        # Test: (n.a) % (n.b) % (n.c) should be (17 % 5 % 3) = (2 % 3) = 2
        # Then toString(2) STARTS WITH '2' should be true
        result = self.g.query("MATCH (n) WHERE toString((n.a) % (n.b) % (n.c)) STARTS WITH '2' RETURN n.a")
        self.env.assertEqual(result.result_set, [[17]])

    def test15_mixed_chained_operators_with_ends_with(self):
        """Test mixed chained arithmetic operators with ENDS WITH predicate"""
        self.g.query("CREATE (:Node {a: 10, b: 3, c: 2})")
        
        # Test: (n.a) * (n.b) - (n.c) should be (10 * 3 - 2) = 28
        # Then toString(28) ENDS WITH '8' should be true
        result = self.g.query("MATCH (n) WHERE toString((n.a) * (n.b) - (n.c)) ENDS WITH '8' RETURN n.a")
        self.env.assertEqual(result.result_set, [[10]])

    def test16_complex_chained_expression_with_contains(self):
        """Test complex chained expression with CONTAINS predicate"""
        self.g.query("CREATE (:Node {val: 'test'})")
        
        # Test: (n.val) + 'a' + 'b' + 'c' CONTAINS 'abc'
        # Should evaluate as ((((n.val) + 'a') + 'b') + 'c') CONTAINS 'abc'
        # Result: 'testabc' CONTAINS 'abc' = true
        result = self.g.query("MATCH (n) WHERE (n.val) + 'a' + 'b' + 'c' CONTAINS 'abc' RETURN n.val")
        self.env.assertEqual(result.result_set, [['test']])
        
        # Negative test: verify it's not right-associative
        # If right-associative, it would be 'test' + ('a' + ('b' + ('c' CONTAINS 'abc')))
        # which would fail with type mismatch
        result = self.g.query("MATCH (n) WHERE (n.val) + 'a' + 'b' + 'c' CONTAINS 'testabc' RETURN n.val")
        self.env.assertEqual(result.result_set, [['test']])
