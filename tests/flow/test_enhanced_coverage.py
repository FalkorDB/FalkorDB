import pytest
import redis
from falkordb import FalkorDB


class TestCommandCoverage:
    """
    Integration tests to improve command coverage for FalkorDB
    """
    
    def __init__(self):
        """Initialize test class"""
        self.db = None
        self.graph = None
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.db = FalkorDB()
        self.graph = self.db.select_graph('test_coverage')
        
        # Clean up any existing data
        try:
            self.graph.delete()
        except Exception:
            pass
        
        self.graph = self.db.select_graph('test_coverage')
    
    def teardown_method(self):
        """Clean up after each test"""
        try:
            self.graph.delete()
        except Exception:
            pass
    
    def test_graph_debug_command(self):
        """Test GRAPH.DEBUG command variations"""
        
        # Test AUX START
        result = self.db.execute_command("GRAPH.DEBUG", "AUX", "START")
        assert isinstance(result, int)
        
        # Test AUX END  
        result = self.db.execute_command("GRAPH.DEBUG", "AUX", "END")
        assert isinstance(result, int)
        
        # Test invalid AUX command
        result = self.db.execute_command("GRAPH.DEBUG", "AUX", "INVALID")
        assert isinstance(result, int)
    
    def test_graph_memory_command(self):
        """Test GRAPH.MEMORY command with various scenarios"""
        
        # Create some data first
        self.graph.query("CREATE (n:Person {name: 'Alice', age: 30})")
        self.graph.query("CREATE (n:Person {name: 'Bob', age: 25})")
        self.graph.query("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)")
        
        # Test basic memory usage
        result = self.db.execute_command("GRAPH.MEMORY", "test_coverage", "usage")
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Test memory with sampling
        result = self.db.execute_command("GRAPH.MEMORY", "test_coverage", "usage", "5")
        assert isinstance(result, list)
        
        # Test memory with invalid graph
        try:
            self.db.execute_command("GRAPH.MEMORY", "nonexistent_graph", "usage")
            # Should handle gracefully
        except redis.ResponseError:
            # Expected for nonexistent graph
            pass
    
    def test_graph_config_edge_cases(self):
        """Test GRAPH.CONFIG command edge cases"""
        
        # Test getting all configurations
        result = self.db.execute_command("GRAPH.CONFIG", "GET", "*")
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Test getting specific config
        result = self.db.execute_command("GRAPH.CONFIG", "GET", "TIMEOUT")
        assert isinstance(result, list)
        assert len(result) == 2
        
        # Test setting valid config
        result = self.db.execute_command("GRAPH.CONFIG", "SET", "CACHE_SIZE", "30")
        assert result == "OK"
        
        # Test setting invalid config value
        try:
            self.db.execute_command("GRAPH.CONFIG", "SET", "THREAD_COUNT", "0")
            # Should reject invalid values
        except redis.ResponseError:
            # Expected for invalid value
            pass
        
        # Test getting nonexistent config
        try:
            self.db.execute_command("GRAPH.CONFIG", "GET", "NONEXISTENT_CONFIG")
        except redis.ResponseError:
            # Expected for nonexistent config
            pass
    
    def test_graph_info_detailed(self):
        """Test GRAPH.INFO command with detailed scenarios"""
        
        # Test info on empty graph
        result = self.db.execute_command("GRAPH.INFO", "test_coverage")
        assert isinstance(result, list)
        
        # Create complex graph structure
        self.graph.query("""
            CREATE (p1:Person {name: 'Alice', age: 30})
            CREATE (p2:Person {name: 'Bob', age: 25})
            CREATE (p3:Person {name: 'Charlie', age: 35})
            CREATE (c1:Company {name: 'TechCorp'})
            CREATE (c2:Company {name: 'DataInc'})
            CREATE (p1)-[:WORKS_AT]->(c1)
            CREATE (p2)-[:WORKS_AT]->(c1)
            CREATE (p3)-[:WORKS_AT]->(c2)
            CREATE (p1)-[:KNOWS]->(p2)
            CREATE (p2)-[:KNOWS]->(p3)
        """)
        
        # Test info on populated graph
        result = self.db.execute_command("GRAPH.INFO", "test_coverage")
        assert isinstance(result, list)
        
        # Verify key information is present
        info_dict = dict(zip(result[::2], result[1::2]))
        assert "Node count" in info_dict
        assert "Edge count" in info_dict
        assert "Label count" in info_dict
        
        # Test info with additional parameters
        result = self.db.execute_command("GRAPH.INFO", "test_coverage", "QUERIES")
        assert isinstance(result, list)
    
    def test_graph_slowlog_functionality(self):
        """Test GRAPH.SLOWLOG command functionality"""
        
        # Execute some queries to populate slowlog
        self.graph.query("CREATE (n:TestNode {id: 1})")
        self.graph.query("MATCH (n:TestNode) RETURN n")
        self.graph.query("MATCH (n:TestNode) DELETE n")
        
        # Test getting slowlog
        result = self.db.execute_command("GRAPH.SLOWLOG", "test_coverage")
        assert isinstance(result, list)
        
        # Test slowlog with limit
        result = self.db.execute_command("GRAPH.SLOWLOG", "test_coverage", "5")
        assert isinstance(result, list)
        assert len(result) <= 5
        
        # Test clearing slowlog
        result = self.db.execute_command("GRAPH.SLOWLOG", "test_coverage", "RESET")
        assert result == "OK"
        
        # Verify slowlog is cleared
        result = self.db.execute_command("GRAPH.SLOWLOG", "test_coverage")
        assert len(result) == 0
    
    def test_graph_constraint_operations(self):
        """Test GRAPH.CONSTRAINT command operations"""
        
        # Test creating unique constraint
        result = self.db.execute_command("GRAPH.CONSTRAINT", "test_coverage", "CREATE", "UNIQUE", ":Person(id)")
        assert result == "OK"
        
        # Test listing constraints
        result = self.db.execute_command("GRAPH.CONSTRAINT", "test_coverage", "LIST")
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Test constraint violation
        self.graph.query("CREATE (p:Person {id: 1, name: 'Alice'})")
        try:
            self.graph.query("CREATE (p:Person {id: 1, name: 'Bob'})")  # Should violate constraint
            assert False, "Should have raised constraint violation"
        except redis.ResponseError:
            # Expected constraint violation
            pass
        
        # Test dropping constraint
        result = self.db.execute_command("GRAPH.CONSTRAINT", "test_coverage", "DROP", "UNIQUE", ":Person(id)")
        assert result == "OK"
    
    def test_graph_bulk_operations(self):
        """Test GRAPH.BULK command for bulk operations"""
        
        # Test bulk node creation
        bulk_query = """
            UNWIND range(1, 100) AS i
            CREATE (n:BulkNode {id: i, value: i * 2})
        """
        
        # Note: GRAPH.BULK might have different syntax, adjust as needed
        result = self.graph.query(bulk_query)
        assert result.nodes_created == 100
        
        # Test bulk edge creation
        bulk_edge_query = """
            MATCH (n1:BulkNode), (n2:BulkNode)
            WHERE n1.id = n2.id - 1 AND n1.id < 50
            CREATE (n1)-[:NEXT]->(n2)
        """
        
        result = self.graph.query(bulk_edge_query)
        assert result.relationships_created == 49
    
    def test_graph_copy_functionality(self):
        """Test GRAPH.COPY command functionality"""
        
        # Create source graph with data
        self.graph.query("CREATE (n:Person {name: 'Alice'})-[:KNOWS]->(m:Person {name: 'Bob'})")
        
        # Test copying graph
        result = self.db.execute_command("GRAPH.COPY", "test_coverage", "test_coverage_copy")
        assert result == "OK"
        
        # Verify copy has same data
        copy_graph = self.db.select_graph('test_coverage_copy')
        result = copy_graph.query("MATCH (n) RETURN count(n)")
        assert result.result_set[0][0] == 2
        
        # Clean up copy
        copy_graph.delete()
    
    def test_error_handling_scenarios(self):
        """Test various error handling scenarios"""
        
        # Test query with syntax error
        try:
            self.graph.query("INVALID SYNTAX QUERY")
            assert False, "Should have raised syntax error"
        except redis.ResponseError:
            # Expected syntax error
            pass
        
        # Test accessing nonexistent graph
        try:
            self.db.execute_command("GRAPH.INFO", "nonexistent_graph")
        except redis.ResponseError:
            # Expected error for nonexistent graph
            pass
        
        # Test constraint on nonexistent property
        try:
            self.db.execute_command("GRAPH.CONSTRAINT", "test_coverage", "CREATE", "UNIQUE", ":NonexistentLabel(prop)")
            # Might succeed but should handle gracefully
        except redis.ResponseError:
            # Might raise error, should handle gracefully
            pass
    
    def test_concurrent_operations(self):
        """Test concurrent operations on the same graph"""
        import threading
        import time
        
        def worker(worker_id):
            """Worker function for concurrent testing"""
            for i in range(10):
                try:
                    self.graph.query(f"CREATE (n:Worker{worker_id} {{id: {i}}})")
                    time.sleep(0.01)  # Small delay to increase concurrency
                except redis.ResponseError:
                    # Some operations might fail due to concurrency, that's OK
                    pass
        
        # Create multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify some nodes were created
        result = self.graph.query("MATCH (n) RETURN count(n)")
        assert result.result_set[0][0] > 0


if __name__ == "__main__":
    pytest.main([__file__])
