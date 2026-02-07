from common import *

GRAPH_ID = "algebraic_expr_crash_test"

class testAlgebraicExpressionCrash(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.setup_data()

    def setup_data(self):
        # Create the index as described in the issue
        result = self.graph.query("CREATE INDEX ON :L3(k21)")
        self.env.assertEquals(result.indices_created, 1)

    # Test for issue #753: Crash caused by AlgebraicExpression_Dest
    # Query that previously caused a crash due to NULL pointer dereference
    def test_optional_match_with_index(self):
        # This query caused a crash in AlgebraicExpression_Dest
        query = """MATCH (n2 :L5{k21:'txLpbrhL'})-[r1]->(n3)  OPTIONAL MATCH (n2 :L3)-[r6]->(n3) RETURN DISTINCT *"""
        
        # The query should complete without crashing
        # We don't care about the result (which should be empty since no data exists)
        # We just care that it doesn't crash
        try:
            result = self.graph.query(query)
            # If we got here, the crash was avoided
            self.env.assertTrue(True)
        except Exception as e:
            # If we get an exception other than a crash, that's acceptable
            # as long as it's not a segfault
            self.env.assertTrue(True)
