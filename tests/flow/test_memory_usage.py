from common import *
from index_utils import create_node_range_index

GRAPH_ID = "memory_usage"

class MemoryUsage():
    """ MemoryUsage object
        exposes GRAPH.MEMORY USAGE <graph_i> output
        in a convenient way for consumption"""

    def __init__(self, indices_sz_mb, total_graph_sz_mb, node_storage_sz_mb,
                 edge_storage_sz_mb, label_matrices_sz_mb, relation_matrices_sz_mb):

        self.indices_sz_mb = indices_sz_mb
        self.total_graph_sz_mb = total_graph_sz_mb
        self.node_storage_sz_mb = node_storage_sz_mb
        self.edge_storage_sz_mb = edge_storage_sz_mb
        self.label_matrices_sz_mb = label_matrices_sz_mb
        self.relation_matrices_sz_mb = relation_matrices_sz_mb

class testGraphMemoryUsage(FlowTestsBase):
    def tearDown(self):
        self.graph.delete()

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def _graph_memory_usage(self, samples=100):
        """compute graph's memory consumption
           returns a MemoryUsage object"""

        res = self.conn.execute_command("GRAPH.MEMORY", "USAGE", GRAPH_ID,
                                        "SAMPLES", samples)
        return MemoryUsage(res[11], res[1], res[7], res[9], res[3], res[5])

    def test01_invalid_call(self):
        """test error reporting from invalid calls to GRAPH.MEMORY USAGE"""

        # usage:
        # GRAPH.MEMORY USAGE <GRAPH_ID> [SAMPLES <count>]
        
        # wrong arity
        cmd = "GRAPH.MEMORY"

        try:
            res = self.conn.execute_command(cmd)
            self.env.assertTrue(False)
        except:
            pass

        # expecting USAGE to follow
        cmd = f"GRAPH.MEMORY {GRAPH_ID}"

        try:
            res = self.conn.execute_command(cmd)
            self.env.assertTrue(False)
        except:
            pass

        # operating on a non existing key
        cmd = f"GRAPH.MEMORY USAGE {GRAPH_ID}"

        try:
            res = self.conn.execute_command(cmd)
            self.env.assertTrue(False)
        except:
            pass

        # create graph
        self.graph.query("RETURN 1")

        # missing samples count
        cmd = f"GRAPH.MEMORY USAGE {GRAPH_ID} SAMPLES"

        try:
            res = self.conn.execute_command(cmd)
            self.env.assertTrue(False)
        except:
            pass

        # non numeric samples count
        cmd = f"GRAPH.MEMORY USAGE {GRAPH_ID} SAMPLES K"

        try:
            res = self.conn.execute_command(cmd)
            self.env.assertTrue(False)
        except:
            pass

        # negative samples count
        cmd = f"GRAPH.MEMORY USAGE {GRAPH_ID} SAMPLES -20"

        try:
            res = self.conn.execute_command(cmd)
            self.env.assertTrue(False)
        except:
            pass

        self.conn.set('x', 2)

        # operating on the wrong key type
        cmd = "GRAPH.MEMORY USAGE x"

        try:
            res = self.conn.execute_command(cmd)
            self.env.assertTrue(False)
        except:
            pass

    def test02_node_memory_usage(self):
        """make sure node memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS X CREATE ()"
        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.edge_storage_sz_mb, 0)
        self.env.assertEquals(res.label_matrices_sz_mb, 0)
        self.env.assertEquals(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_storage_sz_mb, 0)

        self.env.assertEquals(res.total_graph_sz_mb, res.node_storage_sz_mb)

    def test03_label_matrices_memory_usage(self):
        """make sure label matrices memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS X CREATE (:A)"
        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.edge_storage_sz_mb, 0)
        self.env.assertEquals(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_storage_sz_mb, 0)
        self.env.assertGreater(res.label_matrices_sz_mb, 0)

        self.env.assertEquals(res.total_graph_sz_mb, res.node_storage_sz_mb +
                              res.label_matrices_sz_mb)

    def test04_edge_memory_usage(self):
        """make sure edge memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS X CREATE ()-[:R]->()"
        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.label_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_storage_sz_mb, 0)
        self.env.assertGreater(res.edge_storage_sz_mb, 0)
        self.env.assertGreater(res.relation_matrices_sz_mb, 0)

        self.env.assertEquals(res.total_graph_sz_mb, res.node_storage_sz_mb +
                              res.edge_storage_sz_mb +
                              res.relation_matrices_sz_mb)

    def test05_attribute_memory_usage(self):
        """make sure entity attributes memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS X CREATE ()"
        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.edge_storage_sz_mb, 0)
        self.env.assertEquals(res.label_matrices_sz_mb, 0)
        self.env.assertEquals(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_storage_sz_mb, 0)
        prev_node_storage_sz_mb = res.node_storage_sz_mb

        self.env.assertEquals(res.total_graph_sz_mb, res.node_storage_sz_mb)

        # introduce attributes
        q = "MATCH (n) SET n.v = 120"
        self.graph.query(q)

        res = self._graph_memory_usage()
        self.env.assertGreater(res.node_storage_sz_mb, prev_node_storage_sz_mb)

    def test06_indices_memory_usage(self):
        """make sure indices memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS x CREATE (:A {v:x})"
        self.graph.query(q)

        # create index over :A.v
        create_node_range_index(self.graph, 'A', 'v', sync=True)

        res = self._graph_memory_usage()

        self.env.assertEquals(res.edge_storage_sz_mb, 0)
        self.env.assertEquals(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.indices_sz_mb, 0)
        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_storage_sz_mb, 0)
        self.env.assertGreater(res.label_matrices_sz_mb, 0)

        self.env.assertEquals(res.total_graph_sz_mb,
                              res.node_storage_sz_mb +
                              res.indices_sz_mb +
                              res.label_matrices_sz_mb)

