from common import Env, FalkorDB
from random_graph import create_random_schema, create_random_graph
import time

GRAPH_ID = "graph_copy"

# tests the GRAPH.COPY command
class testGraphCopy():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

    def graph_copy(self, src, dest):
        # invokes the GRAPH.COPY command
        # handels exception when the command failed due to failure in creating
        # a fork, in which case the command is retried
        while True:
            try:
                # it is possible for GRAPH.COPY to fail in case FalkorDB was unable
                # to create a fork (Redis restricts us to a single fork at a time)
                self.conn.execute_command("GRAPH.COPY", src, dest)
                break
            except Exception as e:
                # retry if FalkorDB failed to fork
                # otherwise raise an exception
                if str(e) == "Graph copy failed, could not fork, please retry":
                    time.sleep(0.1)
                    continue
                else:
                    raise e

    # compare graphs
    def compare_graphs(self, A, B):
        # tests that the same set of nodes and edges exists in both graphs
        # compare nodes
        q = "MATCH (n) RETURN n ORDER BY ID(n)"
        A_nodes = A.ro_query(q).result_set
        B_nodes = B.ro_query(q).result_set
        self.env.assertEqual(A_nodes, B_nodes)

        # compare edges
        q = "MATCH ()-[e]->() RETURN e ORDER BY ID(e)"
        A_edges = A.ro_query(q).result_set
        B_edges = B.ro_query(q).result_set
        self.env.assertEqual(A_edges, B_edges)

    def test_01_invalid_invocation(self):
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
        # make sure copying of a random graph is working as expected
        src = 'a'
        dest = 'z'

        src_graph = self.db.select_graph(src)
        nodes, edges = create_random_schema()
        create_random_graph(src_graph, nodes, edges)

        # copy src graph to dest graph
        self.graph_copy(src, dest)
        dest_graph = self.db.select_graph(dest)

        # validate src and dest graphs are the same
        self.compare_graphs(src_graph, dest_graph)

        # clean up
        src_graph.delete()
        dest_graph.delete()

    def test_04_chain_of_copies(self):
        # make multiple copies of essentially the same graph
        # start with graph A
        # copy A to B, copy B to C and so on and so forth un to J
        # A == B == C == ... J
        src = 'A'

        # create a random graph
        src_graph = self.db.select_graph(src)
        nodes, edges = create_random_schema()
        create_random_graph(src_graph, nodes, edges)

        # clone graph multiple times
        for key in range(ord('B'), ord('J')+1):
            dest = chr(key)
            self.graph_copy(src, dest)
            src = dest

        # validate src and dest graphs are the same
        src_graph = self.db.select_graph('A')
        dest_graph = self.db.select_graph('J')
        self.compare_graphs(src_graph, dest_graph)

        # clean up
        for key in range(ord('A'), ord('J')+1):
            i = chr(key)
            graph = self.db.select_graph(chr(key))
            graph.delete()

    def test_05_write_to_copy(self):
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
        self.compare_graphs(src_graph, copy_graph)

        # clean up
        src_graph.delete()
        copy_graph.delete()

    def test_06_copy_uneffected_by_vkey_size(self):
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

        # restore original VKEY_MAX_ENTITY_COUNT
        self.db.config_set("VKEY_MAX_ENTITY_COUNT", vkey_max_entity_count)

        # make a copy
        self.graph_copy(src_graph_id, copy_graph_id)
        copy_graph = self.db.select_graph(copy_graph_id)

        # validate src_graph and copy_graph are the same
        self.compare_graphs(src_graph, copy_graph)

        # clean up
        src_graph.delete()

    def test_07_replicated_copy(self):
        # make sure the GRAPH.COPY command is replicated

        # stop old environment
        self.env.stop()

        # start a new environment, one which have a master and a replica
        self.env, self.db = Env(env='oss', useSlaves=True)

        source_con = self.env.getConnection()

        # create a random graph
        src_graph_id  = GRAPH_ID
        copy_graph_id = "copy_" + GRAPH_ID

        src_graph = self.db.select_graph(src_graph_id)
        nodes, edges = create_random_schema()
        create_random_graph(src_graph, nodes, edges)

        # copy graph
        self.graph_copy(src_graph_id, copy_graph_id)

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")

        # make sure dest graph was replicated
        # assuming replica port is env port+1
        replica_db = FalkorDB("localhost", self.env.port+1)
        replica_cloned_graph = replica_db.select_graph(copy_graph_id)
        
        # make sure src graph on master is the same as cloned graph on replica
        self.compare_graphs(src_graph, replica_cloned_graph)

        # clean up
        self.env.stop()

