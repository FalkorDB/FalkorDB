from common import *
import time

slave_con = None
master_con = None

def checkSlaveSynced(env, masterConn, slaveConn, graph_name):
    masterConn.execute_command("WAIT", "1", "0")
    res = slaveConn.execute_command("keys", graph_name)
    assert res == [graph_name]

class test_read_only_query(FlowTestsBase):
    @classmethod
    def setup_class(cls):
        if SANITIZER:
            pytest.skip("sanitizer is not working correctly with replication")

        cls.env, cls.db = Env(useSlaves=True)
        global master_con
        global slave_con
        master_con = cls.env.getConnection()
        slave_con = cls.env.getSlaveConnection()

    def test01_test_simple_read_only_command(self):
        # This test check graph.RO_QUERY to execute read only commands with success.
        graph_name = "Test_RO_QUERY_command"
        graph = Graph(master_con, graph_name)
        graph.query("UNWIND range(0,20) as i CREATE ()")
        result_set = graph.ro_query("MATCH (n) RETURN COUNT(n)").result_set
        assert 21 == result_set[0][0]
        # Try execute write commands with RO_QUERY
        try:
            graph.ro_query("CREATE()")
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            assert str(e) in "graph.RO_QUERY is to be executed only on read-only queries"
            pass
    
    def test02_test_RO_QUERY_fail_on_write_operations(self):
        # This test check graph.RO_QUERY to execute read only commands with success.
        graph_name = "Test_RO_QUERY_fail_on_write_command"
        graph = Graph(master_con, graph_name)
        # Create the graph
        graph.query("RETURN 1")

        queries = [
            "CREATE()",
            "MERGE()",
            "MATCH(n) DELETE n",
            "CREATE INDEX ON :person(age)",
            "DROP INDEX ON :Person(age)"
        ]

        # Try execute write commands with RO_QUERY
        for query in queries:
            try:
                graph.ro_query(query)
                assert(False)
            except redis.ResponseError as e:
                # Expecting an error.
                assert str(e) in "graph.RO_QUERY is to be executed only on read-only queries"
                pass

    def test03_test_replica_read_only(self):
        # This test checks that only RO_QUERY is valid on replicas.
        graph_name = "Test_RO_QUERY_command_on_replica"
        master_graph = Graph(master_con, graph_name)
        slave_graph = Graph(slave_con, graph_name)

        master_graph.query("UNWIND range(0,20) as i CREATE ()")
        checkSlaveSynced(self.env, master_con, slave_con, graph_name)
        result_set = slave_graph.ro_query("MATCH (n) RETURN COUNT(n)").result_set
        assert 21 == result_set[0][0]
        try:
            # Every GRAPH.QUERY command is a write command, see that replica connection throws an exception.
            slave_graph.query("MATCH (n) RETURN COUNT(n)")
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            assert str(e) in "You can't write against a read only replica."
            pass

    def test04_read_only_should_not_create_graph(self):
        graph_name = "Test_RO_QUERY_should_not_create_graph"
        graph = Graph(master_con, graph_name)
        try:
            graph.ro_query("MATCH (n) RETURN n")
            assert False
        except redis.ResponseError as e:
            # Expecting an error.
            assert str(e) in "Invalid graph operation on empty key"
            pass
