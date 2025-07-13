import os
import threading
from common import *
from redis import Redis
from graph_utils import graph_eq
from random_graph import create_random_schema, create_random_graph

GRAPH_ID          = "dump_restore"
RESTORED_GRAPH_ID = "restored_empty_graph"

class testDumpRestore(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = Redis(port=self.env.port, decode_responses=False)
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.conn.flushall()
        self.graph = self.db.select_graph(GRAPH_ID)

        # delete file if exists
        if os.path.exists("./dump"):
            os.remove("./dump")

    def dump_graph_to_file(self, graph_name, file_path):
        # delete file if exists
        if os.path.exists(file_path):
            os.remove(file_path)

        with open(file_path, 'wb') as f:
            # Dump graph
            #dump = self.conn.execute_command("GRAPH.DUMP", graph_name)
            dump = self.conn.execute_command("DUMP", graph_name)

            # Write graph representation to file
            f.write(dump)

    def restore_graph_from_file(self, file_path, graph_name):
        # load dumped graph from disk
        with open(file_path, 'rb') as f:
            dump = f.read()

        #self.conn.execute_command("GRAPH.RESTORE", graph_name, dump)
        self.conn.execute_command("RESTORE", graph_name, 0, dump)

        return self.db.select_graph(graph_name)

    def test_dump_restore_empty_graph(self):
        """Test dumping and restoring an empty graph"""

        # Create an empty graph
        self.graph.query("RETURN 1")

        # Dump graph
        #dump = self.conn.execute_command("GRAPH.DUMP", GRAPH_ID)
        dump = self.conn.execute_command("DUMP", GRAPH_ID)

        # make sure graph was dumped
        self.env.assertTrue(dump is not None)
        self.env.assertGreater(len(dump), 0)

        # delete original graph
        self.graph.delete()
        graphs = self.db.list_graphs()
        self.env.assertNotIn(GRAPH_ID, graphs)

        # restore dumped graph
        #self.conn.execute_command("GRAPH.RESTORE", GRAPH_ID, dump)
        self.conn.execute_command("RESTORE", GRAPH_ID, 0, dump)

        # validate graph was restored correctly
        graphs = self.db.list_graphs()
        self.env.assertIn(GRAPH_ID, graphs)

        # validate restored graph is empty
        self.graph = self.db.select_graph(GRAPH_ID)
        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        self.env.assertEqual(0, node_count)

        #-----------------------------------------------------------------------

        # restore empty graph into a different key

        # Dump graph
        #dump = self.conn.execute_command("GRAPH.DUMP", GRAPH_ID)
        dump = self.conn.execute_command("DUMP", GRAPH_ID)

        # make sure graph was dumped
        self.env.assertTrue(dump is not None)
        self.env.assertGreater(len(dump), 0)

        # restore dumped graph along side the original graph
        #self.conn.execute_command("GRAPH.RESTORE", GRAPH_ID, dump)
        self.conn.execute_command("RESTORE", RESTORED_GRAPH_ID, 0, dump)

        # validate graph was restored correctly
        graphs = self.db.list_graphs()
        self.env.assertIn(GRAPH_ID, graphs)
        self.env.assertIn(RESTORED_GRAPH_ID, graphs)

        restored_graph = self.db.select_graph(RESTORED_GRAPH_ID)
        self.env.assertTrue(graph_eq(self.graph, restored_graph))

    def test_dump_restore_random_graph(self):
        """Test dumping and restoring a random graph"""

        # create a random graph
        nodes, edges = create_random_schema()
        create_random_graph(self.graph, nodes, edges)

        self.dump_graph_to_file(GRAPH_ID, "./dump")
        restored = self.restore_graph_from_file("./dump", "restored")

        self.env.assertTrue(graph_eq(self.graph, restored))

    def test_dump_restore_virtual_keys(self):
        """Test dump restore works as expected when graph size exceed virtual
        key max entity count"""

        # save vkey config
        vkey_max_entity_count = self.db.config_get("VKEY_MAX_ENTITY_COUNT")
        low_vkey_max_entity_count = 4

        # update vkey config to a relative small nunber
        self.db.config_set("VKEY_MAX_ENTITY_COUNT", low_vkey_max_entity_count)

        # create a graph with entities count greater than VKEY_MAX_ENTITY_COUNT
        q = "UNWIND range(0, $x) AS i CREATE (:A)-[:R]->(:B)"
        res = self.graph.query(q, {'x': low_vkey_max_entity_count * 2000})
        self.env.assertGreater(res.nodes_created, low_vkey_max_entity_count)
        self.env.assertGreater(res.relationships_created, low_vkey_max_entity_count)

        # dump graph to file
        self.dump_graph_to_file(GRAPH_ID, "./dump")

        # restore graph from file under a different key name
        restored = self.restore_graph_from_file("./dump", "restored")

        self.env.assertTrue(graph_eq(self.graph, restored))

        # restore original VKEY_MAX_ENTITY_COUNT
        self.db.config_set("VKEY_MAX_ENTITY_COUNT", vkey_max_entity_count)

    def test_dump_restore_chain(self):
        """dump restore multiple times:
        DUMP A, RESTORE A as B, DUMP B, RESTORE B as C ... Z
        finally compare Z to A"""

        # create a random graph
        nodes, edges = create_random_schema()
        create_random_graph(self.graph, nodes, edges)

        current  = GRAPH_ID
        restored = None

        for key in range(ord('a'), ord('e') + 1):
            key = chr(key)
            self.dump_graph_to_file(current, "./dump")
            restored = self.restore_graph_from_file("./dump", key)
            current = key

        self.env.assertTrue(graph_eq(self.graph, restored))

    def test_restore_same_source(self):
        """dump a given graph once, use the dump to restore the same graph
        multiple times under different names"""

        # create a random graph
        nodes, edges = create_random_schema()
        create_random_graph(self.graph, nodes, edges)

        # dump graph
        self.dump_graph_to_file(GRAPH_ID, "./dump")

        for key in range(ord('a'), ord('e') + 1):
            key = chr(key)
            restored = self.restore_graph_from_file("./dump", key)
            self.env.assertTrue(graph_eq(self.graph, restored))

    def test_dump_restore_existing_graph(self):
        """try to restoring into an already occupied key"""
        self.graph.query("RETURN 1")
        self.dump_graph_to_file(GRAPH_ID, "./dump")

        try:
            self.restore_graph_from_file("./dump", GRAPH_ID)
            self.env.assertFalse(True and "can't restore into existing key")
        except Exception as e:
            self.env.assertIn("Target key name already exists", str(e))

    def test_dump_restore_large_graph(self):
        node_count = 100000
        q = "UNWIND range(0, $n-1) AS i CREATE (:A)-[:R]->(:B)"
        res = self.graph.query(q, {'n': node_count})
        self.env.assertEqual(node_count * 2, res.nodes_created)

        # dump graph
        self.dump_graph_to_file(GRAPH_ID, "./dump")

        # TODO: query the graph as its being dumped

        restored = self.restore_graph_from_file("./dump", "restored")
        # TODO: query the DB as the graph being restored

        self.env.assertTrue(graph_eq(self.graph, restored))

    def test_parallel_restore(self):
        """test concurrent graph restores"""

        # create a random graph
        nodes, edges = create_random_schema()
        create_random_graph(self.graph, nodes, edges)

        self.dump_graph_to_file(GRAPH_ID, "./dump")

        threads = []
        for key in range(ord('a'), ord('e')):
            key = chr(key)
            thread = threading.Thread(
                target=self.restore_graph_from_file,
                args=("./dump", key)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        for key in range(ord('a'), ord('e')):
            key = chr(key)
            restored_graph = self.db.select_graph(key)
            self.env.assertTrue(graph_eq(self.graph, restored_graph))

