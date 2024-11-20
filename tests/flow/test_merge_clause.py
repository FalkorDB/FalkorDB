import redis
from common import *

GRAPH_ID = "merge_clause_test"
redis_con = None

class TestMergeClause:
    @classmethod
    def setup_class(cls):
        global redis_con
        redis_con = redis.Redis(host='localhost', port=6379)
        redis_con.flushall()

    @classmethod
    def teardown_class(cls):
        redis_con.flushall()
        redis_con.close()

    def test_merge_clause_behavior(self):
        # Test case 1: MERGE () UNION MERGE ()
        query = """
        MERGE ()
        RETURN 0 AS n0
        UNION
        MERGE ()
        RETURN 0 AS n0
        """
        result = redis_con.execute_command("GRAPH.QUERY", GRAPH_ID, query)
        assert len(result[1]) == 2, f"Expected 2 nodes, but got {len(result[1])}"

        # Test case 2: MERGE () UNION WITH * MERGE ()
        query = """
        MERGE ()
        RETURN 0 AS n0
        UNION
        WITH *
        MERGE ()
        RETURN 0 AS n0
        """
        result = redis_con.execute_command("GRAPH.QUERY", GRAPH_ID, query)
        assert len(result[1]) == 1, f"Expected 1 node, but got {len(result[1])}"
