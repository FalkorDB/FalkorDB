from common import Env, FalkorDB, SANITIZER, VALGRIND
from redis import BusyLoadingError
from random_graph import create_random_schema, create_random_graph
from graph_utils import graph_eq
from constraint_utils import create_constraint
import time

GRAPH_ID = "graph_copy"

# tests the GRAPH.COPY command
class testGraphCopy():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

    def graph_copy(self, src, dest):
        # invokes the GRAPH.COPY command
        self.conn.execute_command("GRAPH.COPY", src, dest)

    # compare graphs
    def assert_graph_eq(self, A, B):
        max_iterations = 20
        # tests that the graphs are the same
        # limit retries to 20
        for _ in range(max_iterations):
            try:
                self.env.assertTrue(graph_eq(A, B))
                return
            except BusyLoadingError as e:
                print("Retry!")
                time.sleep(1)

        raise RuntimeError("Redis not loaded after 20 seconds")

    def test_01_invalid_invocation(self):
        # skip test if we're running under Sanitizer
        if SANITIZER:
            self.env.skip()

        # validate invalid invocations of the GRAPH.COPY command

        # missing src graph
        src = 'A'
        dest = 'Z'

        # wrong number of arguments
        try:
            self.conn.execute_command("GRAPH.COPY", src)
            self.env.assertTrue(False)
        except Exception:
            pass

        try:
            self.conn.execute_command("GRAPH.COPY", src, dest, 3)
            self.env.assertTrue(False)
        except Exception:
            pass

        # src graph doesn't exists
        try:
            self.graph_copy(src, dest)
            self.env.assertTrue(False)
        except Exception:
            pass

        # src key isn't a graph
        self.conn.set(src, 1)

        try:
            self.graph_copy(src, dest)
            self.env.assertTrue(False)
        except Exception:
            pass
        self.conn.delete(src)

        # create src graph
        src_graph = self.db.select_graph(src)
        src_graph.query("RETURN 1")

        # dest key exists
        # key type doesn't matter
        try:
            self.graph_copy(src, src)
            self.env.assertTrue(False)
        except Exception:
            pass

        # clean up
        self.conn.delete(src, dest)

    def test_02_copy_empty_graph(self):
        # skip test if we're running under Sanitizer
        if SANITIZER:
            self.env.skip()

        # src is an empty graph
        src = 'a'
        dest = 'z'
        src_graph = self.db.select_graph(src)

        # create empty src graph
        src_graph.query("RETURN 1")

        # make a copy of src graph
        self.graph_copy(src, dest)

        # validate that both src and dest graph exists and are the same
        self.env.assertTrue(self.conn.type(src)  == 'graphdata')
        self.env.assertTrue(self.conn.type(dest) == 'graphdata')

        dest_graph = self.db.select_graph(dest)
        
        # make sure both src and dest graph are empty
        src_node_count = src_graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        dest_node_count = dest_graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        self.env.assertEqual(src_node_count, 0)
        self.env.assertEqual(dest_node_count, 0)

        # clean up
        src_graph.delete()
        dest_graph.delete()

    def test_03_copy_random_graph(self):
        # skip test if we're running under Sanitizer
        if SANITIZER:
            self.env.skip()

        # make sure copying of a random graph is working as expected
        src = 'n'
        dest = 'm'

        src_graph = self.db.select_graph(src)
        nodes, edges = create_random_schema()
        create_random_graph(src_graph, nodes, edges)

        # copy src graph to dest graph
        self.graph_copy(src, dest)
        dest_graph = self.db.select_graph(dest)

        # validate src and dest graphs are the same
        self.assert_graph_eq(src_graph, dest_graph)

        # clean up
        src_graph.delete()
        dest_graph.delete()

    def test_04_copy_constraints(self):
        # skip test if we're running under Sanitizer
        if SANITIZER:
            self.env.skip()

        # make sure constrains and indexes are copied

        src_id = GRAPH_ID
        clone_id = GRAPH_ID + "_copy"

        src_graph = self.db.select_graph(src_id)
        clone_graph = self.db.select_graph(clone_id)

        # create graph with both indices and constrains
        src_graph.create_node_range_index("Person", "name", "age")

        create_constraint(src_graph, "UNIQUE", "NODE", "Person", "name", sync=True)

        # copy graph
        self.graph_copy(src_id, clone_id)

        # make sure src and cloned graph are the same
        self.assert_graph_eq(src_graph, clone_graph)

        # clean up
        src_graph.delete()
        clone_graph.delete()

    def test_05_chain_of_copies(self):
        # skip test if we're running under Sanitizer
        if SANITIZER:
            self.env.skip()

        # make multiple copies of essentially the same graph
        # start with graph A
        # copy A to B, copy B to C and C to D
        # A == B == C == D
        src = 'A'

        # create a random graph
        src_graph = self.db.select_graph(src)
        nodes, edges = create_random_schema()
        create_random_graph(src_graph, nodes, edges)

        # clone graph multiple times
        for key in range(ord('B'), ord('D')+1):
            dest = chr(key)
            self.graph_copy(src, dest)
            src = dest

        # validate src and dest graphs are the same
        src_graph = self.db.select_graph('A')
        dest_graph = self.db.select_graph('D')
        self.assert_graph_eq(src_graph, dest_graph)

        # clean up
        for key in range(ord('A'), ord('D')+1):
            i = chr(key)
            graph = self.db.select_graph(chr(key))
            graph.delete()

    def test_06_write_to_copy(self):
        # skip test if we're running under Sanitizer
        if SANITIZER:
            self.env.skip()

        # make sure copied graph is writeable and loadable
        src_graph_id = GRAPH_ID
        copy_graph_id = GRAPH_ID + "_copy"

        query = "CREATE (:A {v:1})-[:R {v:2}]->(:B {v:3})"
        src_graph = self.db.select_graph(src_graph_id)
        src_graph.query(query)

        # create a copy
        self.graph_copy(src_graph_id, copy_graph_id)
        copy_graph = self.db.select_graph(copy_graph_id)

        query = "MATCH (b:B {v:3}) CREATE (b)-[:R {v:4}]->(:C {v:5})"
        src_graph.query(query)
        copy_graph.query(query)

        # reload entire keyspace
        self.conn.execute_command("DEBUG", "RELOAD")

        # make sure both src and copy exists and functional
        self.assert_graph_eq(src_graph, copy_graph)

        # clean up
        src_graph.delete()
        copy_graph.delete()

    def test_07_copy_uneffected_by_vkey_size(self):
        # skip test if we're running under Sanitizer
        if SANITIZER:
            self.env.skip()

        # set size of virtual key to 1
        # i.e. number of entities per virtual key is 1.
        vkey_max_entity_count = self.db.config_get("VKEY_MAX_ENTITY_COUNT")
        self.db.config_set("VKEY_MAX_ENTITY_COUNT", 1)

        # make sure configuration chnaged
        self.env.assertEqual(self.db.config_get("VKEY_MAX_ENTITY_COUNT"), 1)

        src_graph_id  = GRAPH_ID
        copy_graph_id = GRAPH_ID + "_copy"

        # create graph
        src_graph = self.db.select_graph(src_graph_id)
        nodes, edges = create_random_schema()
        create_random_graph(src_graph, nodes, edges)

        # make a copy
        self.graph_copy(src_graph_id, copy_graph_id)
        copy_graph = self.db.select_graph(copy_graph_id)

        # restore original VKEY_MAX_ENTITY_COUNT
        self.db.config_set("VKEY_MAX_ENTITY_COUNT", vkey_max_entity_count)

        # validate src_graph and copy_graph are the same
        self.assert_graph_eq(src_graph, copy_graph)

        # clean up
        src_graph.delete()

    def test_08_replicated_copy(self):
        # skip test if we're running under Valgrind or sanitizer
        if VALGRIND or SANITIZER:
            self.env.skip() # valgrind is not working correctly with replication

        # make sure the GRAPH.COPY command is replicated

        # stop old environment
        self.env.stop()

        # start a new environment, one which have a master and a replica
        self.env, self.db = Env(env='oss', useSlaves=True)

        master_con = self.env.getConnection()

        # create a random graph
        src_graph_id  = GRAPH_ID
        copy_graph_id = "copy_" + GRAPH_ID

        src_graph = self.db.select_graph(src_graph_id)
        nodes, edges = create_random_schema()
        create_random_graph(src_graph, nodes, edges)

        # copy graph
        self.graph_copy(src_graph_id, copy_graph_id)

        # the WAIT command forces master slave sync to complete
        master_con.execute_command("WAIT", "1", "0")

        # make sure dest graph was replicated
        # assuming replica port is env port+1
        replica_db = FalkorDB("localhost", self.env.port+1)
        replica_cloned_graph = replica_db.select_graph(copy_graph_id)
        
        # make sure src graph on master is the same as cloned graph on replica
        self.assert_graph_eq(src_graph, replica_cloned_graph)

