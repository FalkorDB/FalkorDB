from common import *

GRAPH_ID = "graph_list"


# tests the GRAPH.LIST command
class testGraphList(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    def tearDown(self):
        self.db.flushdb()

    def create_graph(self, graph_name, con):
        con.execute_command("GRAPH.QUERY", graph_name, "RETURN 1")

    # TODO: remove once FalkorDB-py extendes list_graphs to accept a pattern
    def list_graphs(self, pattern=None):
        if pattern is None:
            return self.db.connection.execute_command("GRAPH.LIST")
        else:
            return self.db.connection.execute_command("GRAPH.LIST", pattern)

    def test_graph_list(self):
        # no graphs, expecting an empty array
        con = self.env.getConnection()
        graphs = self.db.list_graphs()
        self.env.assertEquals(graphs, [])

        # create graph key GRAPH_ID
        self.create_graph(GRAPH_ID, con)
        graphs = self.db.list_graphs()
        self.env.assertEquals(graphs, [GRAPH_ID])

        # create a second graph key "X"
        self.create_graph("X", con)
        graphs = self.db.list_graphs()
        graphs.sort()
        self.env.assertEquals(graphs, ["X", GRAPH_ID])

        # create a string key "str", graph list shouldn't be effected
        con.set("str", "some string")
        graphs = self.db.list_graphs()
        graphs.sort()
        self.env.assertEquals(graphs, ["X", GRAPH_ID])

        # delete graph key GRAPH_ID
        con.delete(GRAPH_ID)
        graphs = self.db.list_graphs()
        self.env.assertEquals(graphs, ["X"])

        # rename graph key X to Z
        con.rename("X", "Z")
        graphs = self.db.list_graphs()
        self.env.assertEquals(graphs, ["Z"])

        # delete graph key "Z", no graph keys in the keyspace
        con.execute_command("GRAPH.DELETE", "Z")
        graphs = self.db.list_graphs()
        self.env.assertEquals(graphs, [])

    def test_graph_list_regex_patterns(self):
        """Extensively test GRAPH.LIST with regex pattern matching"""
        con = self.env.getConnection()

        # Test data: create diverse graph names to test various regex patterns
        test_graphs = [
            "Service",
            "service",
            "Services",
            "services",
            "ServiceManager",
            "UserService",
            "DataService",
            "TestService",
            "ServiceA",
            "ServiceB",
            "MyService123",
            "ProductCatalog",
            "OrderManagement",
            "CustomerData",
            "Analytics",
            "ReportGenerator",
            "UserProfile",
            "SessionManager",
            "ConfigService",
            "LogService",
            "EmailService",
            "NotificationService",
            "AuthenticationService",
            "graph_test_1",
            "graph_test_2",
            "GRAPH_MAIN",
            "temp_graph",
            "backup_graph_2024",
            "dev_environment_graph",
            "prod_api_graph"
        ]

        # Create all test graphs
        for graph_name in test_graphs:
            self.create_graph(graph_name, con)

        # Verify all graphs were created
        all_graphs = self.list_graphs()
        self.env.assertEquals(len(all_graphs), len(test_graphs))

        # Test 1: Case-insensitive "service" with optional "s"
        pattern = "^[sS]ervice[s]*$"
        expected = ["Service", "service", "Services", "services"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 2: Exact match (no regex special chars)
        pattern = "ProductCatalog"
        expected = ["ProductCatalog"]
        result = self.list_graphs(pattern)
        self.env.assertEquals(result, expected)

        # Test 3: Case-sensitive prefix match
        pattern = "^Service"
        expected = ["Service", "Services", "ServiceManager", "ServiceA", "ServiceB"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 4: Case-insensitive prefix match
        pattern = "^[sS]ervice"
        expected = ["Service", "service", "Services", "services", "ServiceManager", "ServiceA", "ServiceB"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 5: Suffix match
        pattern = ".*Service$"
        expected = ["Service", "UserService", "DataService", "TestService", "ConfigService", "LogService", "EmailService", "NotificationService", "AuthenticationService"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 6: Contains pattern (anywhere in string)
        pattern = ".*graph"
        expected = ["graph_test_1", "graph_test_2", "temp_graph", "backup_graph_2024", "dev_environment_graph", "prod_api_graph"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 7: Case-insensitive contains
        pattern = ".*[gG][rR][aA][pP][hH]"
        expected = ["graph_test_1", "graph_test_2", "GRAPH_MAIN", "temp_graph", "backup_graph_2024", "dev_environment_graph", "prod_api_graph"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 8: Digit matching
        pattern = ".*[0-9].*"
        expected = ["MyService123", "graph_test_1", "graph_test_2", "backup_graph_2024"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 9: Specific digit patterns
        pattern = ".*[1-3]$"
        expected = ["MyService123", "graph_test_1", "graph_test_2"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 10: Multiple character classes
        pattern = "^[A-Z][a-z]+[A-Z][a-z]+$"
        expected = ["ServiceManager", "ProductCatalog", "OrderManagement", "CustomerData", "ReportGenerator", "UserProfile", "SessionManager", "ConfigService", "LogService", "EmailService", "NotificationService", "AuthenticationService", "UserService" ,"DataService" ,"TestService"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 11: Underscore patterns
        pattern = ".*_.*"
        expected = ["graph_test_1" ,"graph_test_2" ,"GRAPH_MAIN" ,"temp_graph" ,"backup_graph_2024" ,"dev_environment_graph" ,"prod_api_graph"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 12: Optional character patterns
        pattern = "^[Tt]est[A-Za-z]*"
        expected = ["TestService"]
        result = self.list_graphs(pattern)
        self.env.assertEquals(result, expected)

        # Test 13: Alternative patterns (OR)
        pattern = "^(User|Data|Test).*"
        expected = ["UserService", "UserProfile", "DataService", "TestService"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 14: Negation - exclude patterns
        pattern = "^(?!.*Service).*graph.*"
        expected = ["graph_test_1", "graph_test_2", "temp_graph", "backup_graph_2024", "dev_environment_graph", "prod_api_graph"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 15: Length-based matching
        pattern = "^.{4,8}$"
        expected = ["Service", "service", "Services", "services", "ServiceA", "ServiceB"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 16: Empty pattern (should match all)
        pattern = ""
        result = self.list_graphs(pattern)
        self.env.assertEquals(len(result), len(test_graphs))

        # Test 17: Pattern that matches nothing
        pattern = "^NonExistentPattern$"
        result = self.list_graphs(pattern)
        self.env.assertEquals(result, [])

        # Test 18: Complex pattern combining multiple features
        pattern = "^[A-Za-z]+Service[A-Za-z0-9]*$"
        expected = ["UserService", "DataService", "TestService", "ConfigService", "LogService", "EmailService", "NotificationService", "AuthenticationService", "MyService123"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 19: Special regex characters escaping test
        # First create a graph with special characters
        special_graph = "test.graph[1]"
        self.create_graph(special_graph, con)

        # Test exact match with escaped special characters
        pattern = "^test\\.graph\\[1\\]$"
        result = self.list_graphs(pattern)
        self.env.assertEquals(result, [special_graph])

        # Test 20: Unicode/international characters (if supported)
        unicode_graph = "график_данных"
        self.create_graph(unicode_graph, con)
        pattern = "граф.*"
        result = self.list_graphs(pattern)
        self.env.assertEquals(result, [unicode_graph])

        # Test 21: Performance test with complex pattern
        pattern = "^(?=.*[aeiou])(?=.*[A-Z]).*Service.*$"
        expected = ["Service", "Services" ,"ServiceManager" ,"UserService" ,"DataService" ,"TestService" ,"ServiceA" ,"ServiceB" ,"MyService123" ,"ConfigService" ,"LogService" ,"EmailService" ,"NotificationService" ,"AuthenticationService"]
        result = self.list_graphs(pattern)
        result.sort()
        expected.sort()
        self.env.assertEquals(result, expected)

        # Test 22: Boundary conditions
        pattern = "^$"  # Empty string match
        result = self.list_graphs(pattern)
        self.env.assertEquals(result, [])

        # Test 23: Single character patterns
        pattern = "^.$"  # Single character
        result = self.list_graphs(pattern)
        self.env.assertEquals(result, [])

        # Test 24: Very long pattern
        pattern = "Service|graph|GRAPH|Analytics|Report|User|Data|Test|Config|Log|Email|Notification|Authentication|Product|Order|Customer|Session|Management|Manager|Profile|Generator|Catalog|temp|backup|dev|prod|api|environment"
        result = self.list_graphs(pattern)
        # Should match most graphs containing any of these words
        self.env.assertTrue(len(result) > 20)

        # Clean up special test graphs
        con.execute_command("GRAPH.DELETE", special_graph)
        con.execute_command("GRAPH.DELETE", unicode_graph)

        # Final verification: test original functionality still works
        all_graphs_final = self.list_graphs()
        self.env.assertEquals(len(all_graphs_final), len(test_graphs))

