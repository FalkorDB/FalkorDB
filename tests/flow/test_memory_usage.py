import itertools
from common import *
from index_utils import *

GRAPH_ID = "memory_usage"

class MemoryUsage():
    """ MemoryUsage object
        exposes GRAPH.MEMORY USAGE <graph_i> output
        in a convenient way for consumption"""

    def __init__(self, indices_sz_mb, total_graph_sz_mb,
                 node_block_storage_sz_mb, unlabeled_node_attributes_sz_mb,
                 node_attributes_by_label_storage_sz_mb,
                 edge_block_storage_sz_mb, edge_attributes_by_type_storage_sz_mb,
                 label_matrices_sz_mb, relation_matrices_sz_mb):

        self.indices_sz_mb                          = indices_sz_mb
        self.total_graph_sz_mb                      = total_graph_sz_mb
        self.label_matrices_sz_mb                   = label_matrices_sz_mb
        self.relation_matrices_sz_mb                = relation_matrices_sz_mb
        self.edge_block_storage_sz_mb               = edge_block_storage_sz_mb
        self.node_block_storage_sz_mb               = node_block_storage_sz_mb
        self.unlabeled_node_attributes_sz_mb        = unlabeled_node_attributes_sz_mb
        self.edge_attributes_by_type_storage_sz_mb  = edge_attributes_by_type_storage_sz_mb
        self.node_attributes_by_label_storage_sz_mb = node_attributes_by_label_storage_sz_mb

        # make sure total reported graph size is the sum of all components
        assert(total_graph_sz_mb == (indices_sz_mb                  +
                                    node_block_storage_sz_mb        +
                                    unlabeled_node_attributes_sz_mb +
                                    edge_block_storage_sz_mb        +
                                    label_matrices_sz_mb            +
                                    sum([x for i, x in enumerate(node_attributes_by_label_storage_sz_mb) if i % 2 == 1] ) +
                                    sum([x for i, x in enumerate(edge_attributes_by_type_storage_sz_mb) if i % 2 == 1])   +
                                    relation_matrices_sz_mb))

class testGraphMemoryUsage(FlowTestsBase):
    def tearDown(self):
        self.graph.delete()
        self.graph = self.db.select_graph(GRAPH_ID)

    def __init__(self):
        self.env, self.db = Env(env='oss-cluster')
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def _graph_memory_usage(self, samples=100):
        """compute graph's memory consumption
           returns a MemoryUsage object"""

        res = self.conn.execute_command("GRAPH.MEMORY", "USAGE", GRAPH_ID,
                                        "SAMPLES", samples)
        return MemoryUsage(res[17], res[1], res[7], res[11], res[9], res[13], res[15], res[3], res[5])

    def test_invalid_call(self):
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

    def test_node_memory_usage(self):
        """make sure node memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS x CREATE ()"
        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.edge_block_storage_sz_mb, 0)
        self.env.assertEquals(res.label_matrices_sz_mb, 0)
        self.env.assertEquals(res.unlabeled_node_attributes_sz_mb, 0)
        self.env.assertEquals(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_block_storage_sz_mb, 0)

        self.env.assertEquals(res.total_graph_sz_mb, res.node_block_storage_sz_mb)

    def test_label_matrices_memory_usage(self):
        """make sure label matrices memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS x CREATE (:A)"
        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.edge_block_storage_sz_mb, 0)
        self.env.assertEquals(res.unlabeled_node_attributes_sz_mb, 0)
        self.env.assertEquals(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_block_storage_sz_mb, 0)
        self.env.assertGreater(res.label_matrices_sz_mb, 0)
        self.env.assertContains("A", res.node_attributes_by_label_storage_sz_mb)

        self.env.assertEquals(res.total_graph_sz_mb, res.node_block_storage_sz_mb +
                              res.label_matrices_sz_mb)

    def test_edge_memory_usage(self):
        """make sure edge memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS x CREATE ()-[:R]->()"
        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.label_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_block_storage_sz_mb, 0)
        self.env.assertGreater(res.edge_block_storage_sz_mb, 0)
        self.env.assertGreater(res.relation_matrices_sz_mb, 0)

        self.env.assertEquals(res.total_graph_sz_mb, res.node_block_storage_sz_mb +
                              res.edge_block_storage_sz_mb +
                              res.relation_matrices_sz_mb)

    def test_attribute_memory_usage(self):
        """make sure entity attributes memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS x CREATE ()"
        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.edge_block_storage_sz_mb, 0)
        self.env.assertEquals(res.label_matrices_sz_mb, 0)
        self.env.assertEquals(res.unlabeled_node_attributes_sz_mb, 0)
        self.env.assertEquals(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_block_storage_sz_mb, 0)
        prev_node_storage_sz_mb = res.node_block_storage_sz_mb

        self.env.assertEquals(res.total_graph_sz_mb, res.node_block_storage_sz_mb)

        # introduce attributes
        q = "MATCH (n) SET n.v = 120"
        self.graph.query(q)

        res = self._graph_memory_usage()
        self.env.assertGreater(res.unlabeled_node_attributes_sz_mb, 0)
        self.env.assertEquals(res.node_block_storage_sz_mb, prev_node_storage_sz_mb)

    def test_indices_memory_usage(self):
        """make sure indices memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS x CREATE (:A {v:x})-[:R {v:-x}]->()"
        self.graph.query(q)

        # create index over :A.v
        create_node_range_index(self.graph,    'A', 'v')
        create_node_fulltext_index(self.graph, 'A', 'v')
        create_edge_range_index(self.graph,    'R', 'v')
        create_edge_fulltext_index(self.graph, 'R', 'v')
        create_node_vector_index(self.graph,   'A', 'v', dim=3)
        create_edge_vector_index(self.graph,   'R', 'v', dim=3, sync=True)

        res = self._graph_memory_usage()

        self.env.assertGreater(res.indices_sz_mb, 0)
        self.env.assertGreater(res.total_graph_sz_mb, 0)

    def test_different_attributes_memory_consumption(self):
        """ make sure we can compute memory consumption of each
            entity attribute type
        """

        q = """
                UNWIND range(0, 32000) AS x
                CREATE ({v: 1}),
                       ({v: -2}),
                       ({v: 3.14}),
                       ({v:'str'}),
                       ({v: true}),
                       ({v: point({latitude: 32.0705767, longitude: 34.8185946})}),
                       ({v:[1,'2',3, [4,5, [6]]]}),
                       ({v: vecf32([1,2.2,-3.1])})"""

        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.edge_block_storage_sz_mb, 0)
        self.env.assertEquals(res.label_matrices_sz_mb, 0)
        self.env.assertEquals(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_block_storage_sz_mb, 0)

    def test_restricted_samples_size(self):
        """make sure samples size is restricted"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS x CREATE ()"
        self.graph.query(q)

        # ask for a huge number of samples
        # if number of samples weren't restricted this test
        # would take forever to complete
        res = self._graph_memory_usage(samples=2**64-1)

        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.edge_block_storage_sz_mb, 0)
        self.env.assertEquals(res.label_matrices_sz_mb, 0)
        self.env.assertEquals(res.unlabeled_node_attributes_sz_mb, 0)
        self.env.assertEquals(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_block_storage_sz_mb, 0)

        self.env.assertEquals(res.total_graph_sz_mb, res.node_block_storage_sz_mb)

    def test_memory_usage_empty_graph(self):
        """test memory consumption of an empty graph"""

        q = "RETURN 1"
        self.graph.query(q)

        # ask for a huge number of samples
        # if number of samples weren't restricted this test
        # would take forever to complete
        res = self._graph_memory_usage()

        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.edge_block_storage_sz_mb, 0)
        self.env.assertEquals(res.label_matrices_sz_mb, 0)
        self.env.assertEquals(res.relation_matrices_sz_mb, 0)
        self.env.assertEquals(res.total_graph_sz_mb, 0)
        self.env.assertEquals(res.node_block_storage_sz_mb, 0)
        self.env.assertEquals(res.unlabeled_node_attributes_sz_mb, 0)

    def test_node_label_overlap(self):
        """test memory consumption of a graph containing multi label nodes"""

        # compute how much node_storage is required for 250000 nodes
        # with a single attribute
        q = "UNWIND range(0, 250000) AS x CREATE ({v:-x})"
        self.graph.query(q)

        res = self._graph_memory_usage()
        node_storage = res.node_block_storage_sz_mb

        # make sure node storage memory consumption if greater than 0
        self.env.assertGreater(node_storage, 0)

        # clear graph
        self.graph.delete()

        # create a graph of the same size only this time each node
        # has multiple labels A & B
        q = "UNWIND range(0, 250000) AS x CREATE (:A:B {v:-x})"
        self.graph.query(q)

        # expecting the exact same memory consumption as with the labeless graph
        res = self._graph_memory_usage()
        self.env.assertEquals(node_storage, res.node_block_storage_sz_mb)

        # clear graph
        self.graph.delete()

        queries = [
            "UNWIND range(0, 83333) AS x CREATE (:A {v:-x})",
            "UNWIND range(0, 83333) AS x CREATE (:B {v:-x})",
            "UNWIND range(0, 83333) AS x CREATE (:A:B {v:-x})"
        ]

        # Generate all 3! = 6 permutations
        permutations = list(itertools.permutations(queries))
        for i, perm in enumerate(permutations, 1):
            for q in perm:
                self.graph.query(q)

            # expecting the exact same memory consumption as with the labeless graph
            res = self._graph_memory_usage()
            self.env.assertEquals(node_storage, res.node_block_storage_sz_mb)

            # clear graph
            self.graph.delete()

        # create a graph where forth of the nodes are of type A,
        # forth of type B, forth of type A&B and forth do not have any labels
        queries = [
            "UNWIND range(0, 62500) AS x CREATE ({v:-x})",
            "UNWIND range(0, 62500) AS x CREATE (:A {v:-x})",
            "UNWIND range(0, 62500) AS x CREATE (:B {v:-x})",
            "UNWIND range(0, 62500) AS x CREATE (:A:B {v:-x})"
        ]

        # Generate all 4! = 24 permutations
        permutations = list(itertools.permutations(queries))
        for i, perm in enumerate(permutations, 1):
            for q in perm:
                self.graph.query(q)

            # expecting the exact same memory consumption as with the labeless graph
            res = self._graph_memory_usage()
            self.env.assertEquals(node_storage, res.node_block_storage_sz_mb)

            # clear graph
            self.graph.delete()

        self.graph.query("RETURN 1")

    def test_node_label_overlap_diff_sample_size(self):
        """test memory consumption of a graph containing multi label nodes
           using different sample sizes"""

        # compute how much node_storage is required for 250000 nodes
        # with a single attribute
        q = "UNWIND range(0, 250000) AS x CREATE ({v:-x})"
        self.graph.query(q)

        res = self._graph_memory_usage()
        node_storage = res.node_block_storage_sz_mb

        # make sure node storage memory consumption if greater than 0
        self.env.assertGreater(node_storage, 0)

        # clear graph
        self.graph.delete()

        sample_sizes = [10, 50, 100]
        for sample_size in sample_sizes:
            # create a graph of the same size only this time each node
            # has multiple labels A & B
            q = "UNWIND range(0, 250000) AS x CREATE (:A:B {v:-x})"
            self.graph.query(q)

            # expecting the exact same memory consumption as with the labeless graph
            res = self._graph_memory_usage(sample_size)
            self.env.assertEquals(node_storage, res.node_block_storage_sz_mb)

            # clear graph
            self.graph.delete()

            queries = [
                "UNWIND range(0, 83333) AS x CREATE (:A {v:-x})",
                "UNWIND range(0, 83333) AS x CREATE (:B {v:-x})",
                "UNWIND range(0, 83333) AS x CREATE (:A:B {v:-x})"
            ]

            for q in queries:
                self.graph.query(q)

            # expecting the exact same memory consumption as with the labeless graph
            res = self._graph_memory_usage(sample_size)
            self.env.assertEquals(node_storage, res.node_block_storage_sz_mb)

            # clear graph
            self.graph.delete()

            # create a graph where forth of the nodes are of type A,
            # forth of type B, forth of type A&B and forth do not have any labels
            queries = [
                "UNWIND range(0, 62500) AS x CREATE ({v:-x})",
                "UNWIND range(0, 62500) AS x CREATE (:A {v:-x})",
                "UNWIND range(0, 62500) AS x CREATE (:B {v:-x})",
                "UNWIND range(0, 62500) AS x CREATE (:A:B {v:-x})"
            ]

            for q in queries:
                self.graph.query(q)

            # expecting the exact same memory consumption as with the labeless graph
            res = self._graph_memory_usage(sample_size)
            self.env.assertEquals(node_storage, res.node_block_storage_sz_mb)

            # clear graph
            self.graph.delete()

        self.graph.query("RETURN 1")

    def test_node_count_smaller_than_sample_size(self):
        """test memory consumption report when graph size is smaller than
           number of entities in the graph"""

        # compute how much node_storage is required for 250000 nodes
        # with a single attribute
        long_string = 'A' * 1000
        q = "UNWIND range(0, 4000) AS x CREATE ({v:$long_string})"
        self.graph.query(q, {'long_string': long_string})

        res = self._graph_memory_usage(20)

        # make sure node attributes storage memory consumption if greater than 0
        self.env.assertGreater(res.unlabeled_node_attributes_sz_mb, 0)

    def test_graph_with_deleted_nodes(self):
        """test memory consumption of a graph containing deleted nodes"""

        # create a graph with deleted nodes
        q = "UNWIND range(0, 250000) AS x CREATE ({v:-x})"
        self.graph.query(q)

        res = self._graph_memory_usage()
        node_storage = res.node_block_storage_sz_mb

        # make sure node storage memory consumption if greater than 0
        self.env.assertGreater(node_storage, 0)

        # double the number of nodes
        q = "UNWIND range(0, 250000) AS x CREATE ({v:-x})"
        self.graph.query(q)

        # memory consumption should dobule
        res = self._graph_memory_usage()
        double_sized_graph_node_storage = res.node_block_storage_sz_mb
        self.env.assertGreater(double_sized_graph_node_storage, node_storage * 1.5)

        # delete half of the nodes
        q = "MATCH (n) WHERE ID(n) % 2 = 0 DELETE n"
        self.graph.query(q)

        # memory consumption should drop back to original
        res = self._graph_memory_usage()
        self.env.assertGreater(res.node_block_storage_sz_mb, node_storage)

        # datablock remaind the same, delete array index grow
        self.env.assertGreater(res.node_block_storage_sz_mb, double_sized_graph_node_storage)

    def test_graph_with_multi_edges(self):
        """test memory consumption of a graph containing multi-edges"""

        # create a graph with multi-edges
        q = """CREATE (a), (b)
               WITH a, b
               UNWIND range(0, 250000) AS x
               CREATE (a)-[:R {v:x}]->(b)"""
        self.graph.query(q)

        # delete a few edges
        q = """MATCH ()-[e:R]->()
               WITH e
               LIMIT 5
               DELETE e"""
        self.graph.query(q)

        res = self._graph_memory_usage()

        # validate graph's memory consumption
        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.edge_block_storage_sz_mb, 0)
        self.env.assertGreater(res.edge_attributes_by_type_storage_sz_mb[1], 0)

    def test_graph_with_empty_relationship_type(self):
        """test memory consumption of a graph containing an empty relationship-type"""

        # create a graph with an empty relationship-type
        q = "CREATE ()-[:R]->()"
        self.graph.query(q)

        # delete the only edge
        q = """MATCH ()-[e:R]->()
               DELETE e"""
        self.graph.query(q)

        # compute graph memory consumption
        res = self._graph_memory_usage()

        # validate graph's memory consumption
        self.env.assertEquals(res.indices_sz_mb, 0)
        self.env.assertEquals(res.edge_block_storage_sz_mb, 0)
        self.env.assertEquals(res.label_matrices_sz_mb, 0)
        self.env.assertEquals(res.relation_matrices_sz_mb, 0)
        self.env.assertEquals(res.total_graph_sz_mb, 0)
        self.env.assertEquals(res.node_block_storage_sz_mb, 0)
        self.env.assertEquals(res.unlabeled_node_attributes_sz_mb, 0)

