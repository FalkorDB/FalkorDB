import itertools

from common import *
from index_utils import *

GRAPH_ID = "memory_usage"

class MemoryUsage:
    """MemoryUsage object
    exposes GRAPH.MEMORY USAGE <graph_i> output
    in a convenient way for consumption"""

    def __init__(
        self,
        indices_sz_mb,
        total_graph_sz_mb,
        node_block_storage_sz_mb,
        unlabeled_node_attributes_sz_mb,
        node_attributes_by_label_storage_sz_mb,
        edge_block_storage_sz_mb,
        edge_attributes_by_type_storage_sz_mb,
        label_matrices_sz_mb,
        relation_matrices_sz_mb,
    ):

        self.indices_sz_mb = indices_sz_mb
        self.total_graph_sz_mb = total_graph_sz_mb
        self.label_matrices_sz_mb = label_matrices_sz_mb
        self.relation_matrices_sz_mb = relation_matrices_sz_mb
        self.edge_block_storage_sz_mb = edge_block_storage_sz_mb
        self.node_block_storage_sz_mb = node_block_storage_sz_mb
        self.unlabeled_node_attributes_sz_mb = unlabeled_node_attributes_sz_mb
        self.edge_attributes_by_type_storage_sz_mb = (
            edge_attributes_by_type_storage_sz_mb
        )
        self.node_attributes_by_label_storage_sz_mb = (
            node_attributes_by_label_storage_sz_mb
        )

        # make sure total reported graph size is the sum of all components
        expected = (indices_sz_mb
            + node_block_storage_sz_mb
            + unlabeled_node_attributes_sz_mb
            + edge_block_storage_sz_mb
            + label_matrices_sz_mb
            + sum([ x for i, x in enumerate(node_attributes_by_label_storage_sz_mb) if i % 2 == 1 ])
            + sum([ x for i, x in enumerate(edge_attributes_by_type_storage_sz_mb) if i % 2 == 1 ])
            + relation_matrices_sz_mb
        )

        assert (abs(total_graph_sz_mb - expected) < 20)

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

    def _assert_mb_close(self, actual, expected, tolerance_mb=1):
        self.env.assertLessEqual(abs(actual - expected), tolerance_mb)

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

        self.conn.set("x", 2)

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

        self.env.assertEqual(res.indices_sz_mb, 0)
        self.env.assertEqual(res.edge_block_storage_sz_mb, 0)
        self.env.assertEqual(res.label_matrices_sz_mb, 0)
        self.env.assertEqual(res.unlabeled_node_attributes_sz_mb, 0)
        self.env.assertEqual(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_block_storage_sz_mb, 0)

        self._assert_mb_close(res.total_graph_sz_mb, res.node_block_storage_sz_mb)

    def test_label_matrices_memory_usage(self):
        """make sure label matrices memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS x CREATE (:A)"
        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEqual(res.indices_sz_mb, 0)
        self.env.assertEqual(res.edge_block_storage_sz_mb, 0)
        self.env.assertEqual(res.unlabeled_node_attributes_sz_mb, 0)
        self.env.assertEqual(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_block_storage_sz_mb, 0)
        self.env.assertGreater(res.label_matrices_sz_mb, 0)
        self.env.assertContains("A", res.node_attributes_by_label_storage_sz_mb)

        self._assert_mb_close(
            res.total_graph_sz_mb,
            res.node_block_storage_sz_mb + res.label_matrices_sz_mb,
        )

    def test_edge_memory_usage(self):
        """make sure edge memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS x CREATE ()-[:R]->()"
        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEqual(res.indices_sz_mb, 0)
        self.env.assertEqual(res.label_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_block_storage_sz_mb, 0)
        self.env.assertGreater(res.edge_block_storage_sz_mb, 0)
        self.env.assertGreater(res.relation_matrices_sz_mb, 0)

        expected_total = (
            res.node_block_storage_sz_mb
            + res.edge_block_storage_sz_mb
            + res.relation_matrices_sz_mb
        )
        self._assert_mb_close(res.total_graph_sz_mb, expected_total)

    def test_attribute_memory_usage(self):
        """make sure entity attributes memory consumption is reported"""

        # create a graph with only nodes
        q = "UNWIND range(0, 250000) AS x CREATE ()"
        self.graph.query(q)

        res = self._graph_memory_usage()

        self.env.assertEqual(res.indices_sz_mb, 0)
        self.env.assertEqual(res.edge_block_storage_sz_mb, 0)
        self.env.assertEqual(res.label_matrices_sz_mb, 0)
        self.env.assertEqual(res.unlabeled_node_attributes_sz_mb, 0)
        self.env.assertEqual(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_block_storage_sz_mb, 0)
        prev_node_storage_sz_mb = res.node_block_storage_sz_mb

        self._assert_mb_close(res.total_graph_sz_mb, res.node_block_storage_sz_mb)

        # introduce attributes
        q = "MATCH (n) SET n.v = 120"
        self.graph.query(q)

        res = self._graph_memory_usage()
        self.env.assertGreater(res.unlabeled_node_attributes_sz_mb, 0)

        # C keeps node_block_storage flat here: its nodes DataBlock reserves an
        # AttributeSet pointer per node up front, so seeding properties only
        # grows the separately-reported attribute sizes. Rust allocates the
        # attribute store's slot table lazily, so the first property write also
        # grows block storage - the same memory, paid later.
        self.env.assertGreater(res.node_block_storage_sz_mb, prev_node_storage_sz_mb)

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
        """make sure we can compute memory consumption of each
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

        self.env.assertEqual(res.indices_sz_mb, 0)
        self.env.assertEqual(res.edge_block_storage_sz_mb, 0)
        self.env.assertEqual(res.label_matrices_sz_mb, 0)
        self.env.assertEqual(res.relation_matrices_sz_mb, 0)

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
        res = self._graph_memory_usage(samples=2**64 - 1)

        self.env.assertEqual(res.indices_sz_mb, 0)
        self.env.assertEqual(res.edge_block_storage_sz_mb, 0)
        self.env.assertEqual(res.label_matrices_sz_mb, 0)
        self.env.assertEqual(res.unlabeled_node_attributes_sz_mb, 0)
        self.env.assertEqual(res.relation_matrices_sz_mb, 0)

        self.env.assertGreater(res.total_graph_sz_mb, 0)
        self.env.assertGreater(res.node_block_storage_sz_mb, 0)

        self._assert_mb_close(res.total_graph_sz_mb, res.node_block_storage_sz_mb)

    def test_memory_usage_empty_graph(self):
        """test memory consumption of an empty graph"""

        q = "RETURN 1"
        self.graph.query(q)

        # ask for a huge number of samples
        # if number of samples weren't restricted this test
        # would take forever to complete
        res = self._graph_memory_usage()

        self.env.assertEqual(res.indices_sz_mb, 0)
        self.env.assertEqual(res.edge_block_storage_sz_mb, 0)
        self.env.assertEqual(res.label_matrices_sz_mb, 0)
        self.env.assertEqual(res.relation_matrices_sz_mb, 0)
        self.env.assertEqual(res.total_graph_sz_mb, 0)
        self.env.assertEqual(res.node_block_storage_sz_mb, 0)
        self.env.assertEqual(res.unlabeled_node_attributes_sz_mb, 0)

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

        # Expecting the same memory consumption as with the labeless graph: what
        # this test guards is that a node carrying N labels is still counted
        # once, not N times.
        #
        # Compared within a 1 MB tolerance rather than exactly, because
        # node_block_storage includes the node-existence matrix, and a
        # VersionedMatrix reports m + dp + dm - how much of it sits in the
        # deltas depends on how many write transactions built the graph, so the
        # same 250k nodes can land ~1 MB apart when created by one query versus
        # three. Double counting would show up as a multiple, not as ±1 MB.
        res = self._graph_memory_usage()
        self._assert_mb_close(node_storage, res.node_block_storage_sz_mb)

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
            self._assert_mb_close(node_storage, res.node_block_storage_sz_mb)

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
            self._assert_mb_close(node_storage, res.node_block_storage_sz_mb)

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
            self._assert_mb_close(node_storage, res.node_block_storage_sz_mb)

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
            self._assert_mb_close(node_storage, res.node_block_storage_sz_mb)

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
            self._assert_mb_close(node_storage, res.node_block_storage_sz_mb)

            # clear graph
            self.graph.delete()

        self.graph.query("RETURN 1")

    def test_node_count_smaller_than_sample_size(self):
        """test memory consumption report when graph size is smaller than
           number of entities in the graph"""

        # compute how much node_storage is required for 250000 nodes
        # with a single attribute
        #
        # each node gets a *distinct* long string: attribute sizes are reported
        # amortized, and a single shared string value is stored once behind an
        # Arc no matter how many nodes reference it, so seeding every node with
        # the same parameter would correctly report ~0 and measure nothing.
        long_string = 'A' * 1000
        q = "UNWIND range(0, 4000) AS x CREATE ({v:$long_string + x})"
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
        self.env.assertEqual(res.indices_sz_mb, 0)
        self.env.assertEqual(res.edge_block_storage_sz_mb, 0)
        self.env.assertEqual(res.label_matrices_sz_mb, 0)
        self.env.assertEqual(res.relation_matrices_sz_mb, 0)
        self.env.assertEqual(res.total_graph_sz_mb, 0)
        self.env.assertEqual(res.node_block_storage_sz_mb, 0)
        self.env.assertEqual(res.unlabeled_node_attributes_sz_mb, 0)

    def test_graph_recreate_memory_consumption(self):
        """test memory consumption of a graph which had a large set of deletions
           followed by entities reintroduction, the reconstructed graph memory
           consumption should be similar to the original state"""

        #-----------------------------------------------------------------------
        # create a graph with 250K nodes and 500K edges
        #-----------------------------------------------------------------------

        node_count = 250000
        edge_count = node_count * 2 - 2

        q = """UNWIND range (1, $node_count) AS x
               CREATE (a)"""

        res = self.graph.query(q, {'node_count': node_count})
        self.env.assertEqual(res.nodes_created, node_count)

        # create edges
        q = """MATCH (a)
               WITH a, ID(a) + 1 AS b_id
               MATCH (b)
               WHERE ID(b) = b_id
               CREATE (a)-[:R]->(a), (a)-[:R]->(b)"""

        res = self.graph.query(q, {'node_count': node_count})
        self.env.assertEqual(res.relationships_created, edge_count)

        # compute graph memory consumption
        original_memory_consumption = self._graph_memory_usage()
        self.env.assertGreater(original_memory_consumption.total_graph_sz_mb, 0)
        self.env.assertGreater(original_memory_consumption.node_block_storage_sz_mb, 0)
        self.env.assertGreater(original_memory_consumption.edge_block_storage_sz_mb, 0)

        #-----------------------------------------------------------------------
        # delete entities
        #-----------------------------------------------------------------------

        # delete all entities
        q = "MATCH (n) DELETE n"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_deleted, node_count)
        self.env.assertEqual(res.relationships_deleted, edge_count)

        # compute graph memory consumption
        deleted_memory_consumption = self._graph_memory_usage()
        self.env.assertGreater(deleted_memory_consumption.total_graph_sz_mb, 0)

        # C expects the datablocks to consume MORE space here, since they never
        # shrink and the deleted_idx array holds every deleted ID. The Rust
        # attribute store instead compacts a block once its abandoned entries
        # dominate, so deleting every entity releases the attribute arenas and
        # block storage drops - it must stay bounded by the original size.
        self.env.assertLessEqual(deleted_memory_consumption.node_block_storage_sz_mb,
                                 original_memory_consumption.node_block_storage_sz_mb)

        self.env.assertLessEqual(deleted_memory_consumption.edge_block_storage_sz_mb,
                                 original_memory_consumption.edge_block_storage_sz_mb)

        #-----------------------------------------------------------------------
        # restore entities
        #-----------------------------------------------------------------------

        q = "UNWIND range (1, $node_count) AS x CREATE (a)"

        res = self.graph.query(q, {'node_count': node_count})
        self.env.assertEqual(res.nodes_created, node_count)

        q = """MATCH (a)
               WITH a, ID(a) + 1 AS b_id
               MATCH (b)
               WHERE ID(b) = b_id
               CREATE (a)-[:R]->(a), (a)-[:R]->(b)"""

        res = self.graph.query(q, {'node_count': node_count})
        self.env.assertEqual(res.relationships_created, edge_count)

        # compute graph memory consumption
        #
        # Back to roughly the original footprint, but not to the byte: the
        # matrices this figure includes keep the capacity they grew to and carry
        # a different amount of pending delta state after a delete-everything
        # /recreate cycle than they did when first built, which is real memory
        # the report is right to show. Bounded rather than exact.
        reconstructed_memory_consumption = self._graph_memory_usage()
        self._assert_mb_close(reconstructed_memory_consumption.total_graph_sz_mb,
                              original_memory_consumption.total_graph_sz_mb,
                              tolerance_mb=5)

        # datablock memory consumption should return to its original size
        # now that the its deleted IDs array been cleared

        self._assert_mb_close(reconstructed_memory_consumption.node_block_storage_sz_mb,
                              original_memory_consumption.node_block_storage_sz_mb,
                              tolerance_mb=5)
        self._assert_mb_close(reconstructed_memory_consumption.edge_block_storage_sz_mb,
                              original_memory_consumption.edge_block_storage_sz_mb,
                              tolerance_mb=5)
