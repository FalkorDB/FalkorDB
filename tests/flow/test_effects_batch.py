"""Effects -- does it hold at scale, and under operations nobody chose?

Two ways of stressing the same path. The bulk cases put thousands of entities
through one commit, where the record partitioning and the id ranges are wide
enough to expose an off-by-one that a three-node query never would. The random
case does not choose its operations at all.

See `effects_common.py` for the shared fixture.
"""

import random

from common import *

from effects_common import _EffectsBase


class testEffects_01_Bulk(_EffectsBase):
    """Thousands of entities created and deleted in single statements.

    Each test starts with `env.flush()`. That is FLUSHALL against the server,
    not just this graph -- which is why this class is on its own file: RLTest
    hands consecutive classes with identical `Env(...)` parameters the *same*
    server, so a class that flushes is hostile to anything sharing it.
    """

    GRAPH_ID = "effects_bulk"

    def __init__(self):
        self._setup()

    def test01_multiple_nodes(self):
        """Test the creation & deletion of multiple nodes."""

        self.env.flush()  # clean slate

        # labels
        lbls = ["L0", "L1", "L2", "L3"]

        # create 2048 nodes with random labels: L0, L1, L2, L3
        q = "(:{})"
        nodes = [q.format(random.choice(lbls)) for _ in range(2048)]
        multi_create = "CREATE " + ",".join(nodes)
        res = self.query_and_sync(multi_create)

        self.env.assertEqual(res.nodes_created, 2048)
        self.assert_graph_eq()

        # delete nodes
        res = self.query_and_sync("MATCH (n) DELETE n")
        self.env.assertEqual(res.nodes_deleted, 2048)

        self.assert_graph_eq()

        q = "MATCH (n) RETURN count(n)"
        replica_node_count = self.replica_graph.ro_query(q).result_set[0][0]
        master_node_count = self.master_graph.query(q).result_set[0][0]
        self.env.assertEqual(master_node_count, 0)
        self.env.assertEqual(replica_node_count, master_node_count)

        for l in lbls:
            q = "MATCH (n:{}) RETURN count(n)".format(l)
            master_node_count = self.master_graph.query(q).result_set[0][0]
            replica_node_count = self.replica_graph.ro_query(q).result_set[0][0]
            self.env.assertEqual(master_node_count, 0)
            self.env.assertEqual(replica_node_count, master_node_count)

    def test02_multiple_edges(self):
        """Test the creation & deletion of multiple edges."""

        self.env.flush()  # clean slate

        # relation types
        types = ["R0", "R1", "R2", "R3"]

        # create 2048 edges of types: R0, R1, R2, R3
        q = "()-[:{}]->()"
        edges = [q.format(random.choice(types)) for _ in range(2048)]
        multi_create = "CREATE" + ",".join(edges)
        res = self.query_and_sync(multi_create)

        self.env.assertEqual(res.relationships_created, 2048)
        self.assert_graph_eq()

        # delete edges
        res = self.query_and_sync("MATCH ()-[e]->() DELETE e")

        self.env.assertEqual(res.relationships_deleted, 2048)
        self.assert_graph_eq()

        q = "MATCH ()-[e]->() RETURN count(e)"
        replica_edge_count = self.replica_graph.ro_query(q).result_set[0][0]
        master_edge_count = self.master_graph.query(q).result_set[0][0]
        self.env.assertEqual(master_edge_count, 0)
        self.env.assertEqual(replica_edge_count, master_edge_count)

        for t in types:
            q = "MATCH ()-[e:{}]->() RETURN count(e)".format(t)
            master_edge_count = self.master_graph.query(q).result_set[0][0]
            replica_edge_count = self.replica_graph.ro_query(q).result_set[0][0]
            self.env.assertEqual(master_edge_count, 0)
            self.env.assertEqual(replica_edge_count, master_edge_count)

    def test03_multiple_entities(self):
        """Test creation & deletion of multiple entities with a single randomized delete query."""

        self.env.flush()  # clean slate

        # labels and relation types
        lbls = ["L0", "L1", "L2", "L3"]
        types = ["R0", "R1", "R2", "R3"]

        edge_count = 2048
        node_count = edge_count * 2

        #-------------------------------------------------------------------
        # create nodes + edges in a single query
        #-------------------------------------------------------------------

        q_pattern = "(:{src_lbl})-[:{r_type}]->(:{dest_lbl})"
        patterns = [
            q_pattern.format(
                src_lbl=random.choice(lbls),
                r_type=random.choice(types),
                dest_lbl=random.choice(lbls)
            )
            for _ in range(edge_count)
        ]
        multi_create = "CREATE " + ",".join(patterns)
        res = self.query_and_sync(multi_create)

        self.env.assertEqual(res.nodes_created, node_count)
        self.env.assertEqual(res.relationships_created, edge_count)
        self.assert_graph_eq()

        #-------------------------------------------------------------------
        # assign random IDs to nodes and edges
        #-------------------------------------------------------------------

        node_ids = list(range(node_count))
        edge_ids = list(range(edge_count))
        random.shuffle(node_ids)
        random.shuffle(edge_ids)

        node_id_map = {l: [] for l in lbls}
        for nid in node_ids:
            label = random.choice(lbls)
            node_id_map[label].append(nid)

        edge_id_map = {t: [] for t in types}
        for eid in edge_ids:
            r_type = random.choice(types)
            edge_id_map[r_type].append(eid)

        #-------------------------------------------------------------------
        # build single delete query
        #-------------------------------------------------------------------

        delete_clauses = []

        # edges first
        for t, ids in edge_id_map.items():
            if ids:
                delete_clauses.append(
                    f"OPTIONAL MATCH ()-[e:{t}]->() WHERE ID(e) IN {ids} DELETE e WITH count(1) AS x"
                )

        # nodes per label
        for l, ids in node_id_map.items():
            if ids:
                delete_clauses.append(
                    f"OPTIONAL MATCH (n:{l}) WHERE ID(n) IN {ids} DELETE n WITH count(1) AS x"
                )

        # final catch-all for any remaining nodes
        delete_clauses.append("MATCH (n) DELETE n")

        # combine everything into a single query
        single_delete_query = "\n".join(delete_clauses)

        # execute the delete
        res = self.query_and_sync(single_delete_query)

        self.env.assertEqual(res.nodes_deleted, node_count)
        self.env.assertEqual(res.relationships_deleted, edge_count)
        self.assert_graph_eq()

        #-------------------------------------------------------------------
        # verification: master and replica must be empty
        #-------------------------------------------------------------------

        q = "MATCH (n) RETURN count(n)"
        replica_node_count = self.replica_graph.ro_query(q).result_set[0][0]
        master_node_count = self.master_graph.query(q).result_set[0][0]
        self.env.assertEqual(master_node_count, 0)
        self.env.assertEqual(replica_node_count, master_node_count)

        q = "MATCH ()-[e]->() RETURN count(e)"
        replica_edge_count = self.replica_graph.ro_query(q).result_set[0][0]
        master_edge_count = self.master_graph.query(q).result_set[0][0]
        self.env.assertEqual(master_edge_count, 0)
        self.env.assertEqual(replica_edge_count, master_edge_count)


class testEffects_02_RandomOps(_EffectsBase):
    """A randomly built graph put through a random sequence of operations.

    The only test here that does not know what it is asserting in advance: it
    builds a schema and a graph from `random_graph.py`, runs every operation
    kind against it, and requires the two sides to still agree. It has caught
    shapes no hand-written case covered.
    """

    GRAPH_ID = "effects_random"

    def __init__(self):
        self._setup()

    def test01_random_ops(self):
        from random_graph import create_random_schema, create_random_graph, run_random_graph_ops, ALL_OPS
        nodes, edges = create_random_schema()
        create_random_graph(self.master_graph, nodes, edges)

        # wait for replica and master to sync
        self.wait_for_replica_offset()
        self.assert_graph_eq()

        run_random_graph_ops(self.master_graph, nodes, edges, ALL_OPS)

        # wait for replica and master to sync
        self.wait_for_replica_offset()
        self.assert_graph_eq()
