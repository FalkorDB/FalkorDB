from common import *

GRAPH_ID = "optional_match_path_crash"

class testOptionalMatchPathCrash(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    # Test the crash scenario with multiple single-node path variables in OPTIONAL MATCH
    def test01_multiple_single_node_paths(self):
        # This query was causing a crash
        query = """OPTIONAL MATCH (n0)--(n1)
                   OPTIONAL MATCH p0 = (n1), p1 = (n0)
                   RETURN *"""
        
        # The query should not crash and should return valid results
        # When n0 and n1 don't exist (empty graph), all should be null
        actual_result = self.graph.query(query)
        
        # Should return one row with all nulls
        expected_result = [[None, None, None, None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Test with data in the graph
    def test02_multiple_single_node_paths_with_data(self):
        self.graph.delete()
        self.graph.query("CREATE (n0 {v: 'v0'})-[:E]->(n1 {v: 'v1'})")
        
        query = """OPTIONAL MATCH (n0)--(n1)
                   OPTIONAL MATCH p0 = (n1), p1 = (n0)
                   RETURN n0.v, n1.v, length(p0), length(p1)
                   ORDER BY n0.v, n1.v"""
        
        actual_result = self.graph.query(query)
        
        # Should return paths with length 0 (single node paths)
        # There should be multiple rows since (n0)--(n1) is bidirectional
        self.env.assertTrue(len(actual_result.result_set) > 0)
        
        # Each row should have non-null node values and path lengths of 0
        for row in actual_result.result_set:
            self.env.assertIsNotNone(row[0])  # n0.v
            self.env.assertIsNotNone(row[1])  # n1.v
            self.env.assertEquals(row[2], 0)   # length(p0) should be 0
            self.env.assertEquals(row[3], 0)   # length(p1) should be 0

    # Test with single named path in OPTIONAL MATCH
    def test03_single_node_path(self):
        self.graph.delete()
        self.graph.query("CREATE (n {v: 'value'})")
        
        query = """OPTIONAL MATCH (n)
                   OPTIONAL MATCH p = (n)
                   RETURN n.v, length(p)"""
        
        actual_result = self.graph.query(query)
        expected_result = [['value', 0]]
        self.env.assertEquals(actual_result.result_set, expected_result)
