from common import Env
from random_graph import create_random_schema, create_random_graph
#nodes, edges = create_random_schema()
#res = create_random_graph(redis_graph, nodes, edges)

GRAPH_ID = "graph_copy"

# tests the GRAPH.LIST command
class testGraphCopy():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def graph_copy(self, src, dest):
        self.conn.execute_command("GRAPH.COPY", src, dest)

    def test_01_invalid_invocation(self):
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
        src = 'a'
        dest = 'z'

        src_graph = self.db.select_graph(src)
        nodes, edges = create_random_schema()
        res = create_random_graph(src_graph, nodes, edges)

        # copy src graph to dest graph
        self.graph_copy(src, dest)
        dest_graph = self.db.select_graph(dest)

        src_graph.delete()
        dest_graph.delete()

