from common import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from demo import QueryInfo

GRAPH_ID = "G"
NEW_GRAPH_ID = "G2"


class testKeyspaceAccesses(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
    
    def test00_test_data_valid_after_rename(self):
        node0 = Node(node_id=0, labels="L", properties={'name':'x', 'age':1})
        self.graph.add_node(node0)
        self.graph.flush()
        self.redis_con.rename(GRAPH_ID, NEW_GRAPH_ID)

        self.graph = self.db.select_graph(NEW_GRAPH_ID)

        node1 = Node(node_id=1, labels="L", properties={'name':'x', 'age':1})
        self.graph.add_node(node1)
        self.graph.flush()

        query = "MATCH (n) return n"
        expected_results = [[node0], [node1]]
        query_info = QueryInfo(query = query, description="Tests data is valid after renaming", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(self.graph.query(query), query_info)

    # Graph queries should fail gracefully on accessing non-graph keys.
    def test01_graph_access_on_invalid_key(self):
        self.redis_con.set("integer_key", 5)
        self.graph = self.db.select_graph("integer_key")
        try:
            query = """MATCH (n) RETURN noneExistingFunc(n.age) AS cast"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("WRONGTYPE" in str(e))
            pass

    # Fail gracefully on attempting a graph deletion of an empty key.
    def test02_graph_delete_on_empty_key(self):
        self.graph = self.db.select_graph("nonexistent_key")
        try:
            self.graph.delete()
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("empty key" in str(e))
            pass
