import os
from common import *
from random_graph import create_random_schema, create_random_graph, run_random_graph_ops, ALL_OPS

GRAPH_ID = "dump_restore"

def dump_graph_to_file(conn, graph_name, file_path):
    # delete file if exists
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, 'wb') as f:
        # Dump graph
        dump = conn.execute_command("GRAPH.DUMP", graph_name)

        # Write graph representation to file
        f.write(dump)

def restore_graph_from_file(conn, file_path, graph_name):
    # load dumped graph from disk
    with open(file_path, 'rb') as f:
        dump = f.read()

    #conn.restore(graph_name, 0, dump, replace=False)
    conn.execute_command("GRAPH.RESTORE", graph_name, dump)

class testDumpRestore(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.conn.flushall()

    def test_dump_restore_empty_graph(self):
        """Test dumping and restoring an empty graph"""

        # Create an empty graph
        self.graph.query("RETURN 1")

        # Dump graph
        dump = self.conn.execute_command("GRAPH.DUMP", GRAPH_ID)

        # make sure graph was dumped
        self.env.assertTrue(dump is not None)
        self.env.assertGreater(len(dump), 0)

        # delete original graph
        self.graph.delete()
        graphs = self.db.list_graphs()
        self.env.assertNotIn(GRAPH_ID, graphs)

        # restore dumped graph
        self.conn.execute_command("graph.restore", GRAPH_ID, dump)

        # validate graph was restored correctly
        graphs = self.db.list_graphs()
        self.env.assertIn(GRAPH_ID, graphs)

        self.graph = self.db.select_graph(GRAPH_ID)
        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        self.env.assertEqual(0, node_count)

        edge_count = self.graph.query("MATCH ()-[e]->() RETURN count(e)").result_set[0][0]
        self.env.assertEqual(0, edge_count)

    def _test_dump_restore_random_graph(self):
        """Test dumping and restoring a random graph"""

        # create a random graph
        nodes, edges = create_random_schema()
        create_random_graph(self.graph, nodes, edges)

        dump_graph_to_file(self.conn, GRAPH_ID, "./dump")

        restore_graph_from_file(self.conn, "./dump", "restored")
        restored = self.db.select_graph("restored")

        self.env.assertTrue(graph_eq(self.graph, restored))

