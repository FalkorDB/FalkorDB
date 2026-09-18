import time
import random

from common import *
from index_utils import *

from effects_common import _EffectsBase


class testEffects(_EffectsBase):
    """Every effect opcode, end to end against a live primary and replica.

    One test per kind of mutation a query can produce -- schema, attribute,
    node, edge, labels, delete, merge -- plus the batch, value-type and
    non-deterministic cases. `test_effects_shapes.py` asks what a *buffer* has
    to survive; this file asks whether each opcode replicates at all.

    Effects are the only way a write reaches a replica, so nothing here turns
    them on or off -- there is no query-replay alternative left to compare
    against.

    The tests are a chain, deliberately: `test04` creates the `:A:B` node that
    `test08` and `test09` relabel and that `test12`/`test13` MERGE against.
    RLTest runs a class's tests in sorted order (`RLTest/loader.py:124` is
    `for symbol in dir(module)`, and `dir()` sorts), so the order holds -- but
    a test inserted between them has to leave that node alone.
    """

    GRAPH_ID = "effects"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT')

        # create indices
        create_node_range_index(self.master_graph, "L", "a", "b", "c")
        create_edge_range_index(self.master_graph, "R", "a", "b", "c")
        self.wait_for_replica_offset()

        # Index DDL ships as effects of its own. Discard them so the first
        # test's window starts empty and `assert_effect_emitted` counts only
        # what that test wrote.
        self.monitor_mark()

    def test02_add_schema_effect(self):
        # test the introduction of a schema by an effect

        # introduce a new label which in turn creates a new schema
        q = "CREATE (:L)"
        res = self.query_and_sync(q)
        self.env.assertEqual(res.nodes_created, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # introduce multiple labels
        q = "CREATE (:X:Y)"
        res = self.query_and_sync(q)
        self.env.assertEqual(res.labels_added, 2)
        self.env.assertEqual(res.nodes_created, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # introduce a new relationship-type which in turn creates a new schema
        q = "CREATE ()-[:R]->()"
        res = self.query_and_sync(q)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test03_add_attribute_effect(self):
        # test the introduction of an attribute by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        # set a new attribute for each supported attribute type
        q = """MATCH (n:L) WITH n
                LIMIT 1
                SET
                n.a = 1,
                n.b = 'str',
                n.c = True,
                n.d = [1, [2], '3'],
                n.v = vecf32([1.0, 2.0, 3.0])
            """

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 5)

        self.assert_effect_emitted()

        q = """MATCH ()-[e]->()
                WITH e
                LIMIT 1
                SET
                e.e = point({latitude: 51, longitude: 0}),
                e.f=3.14,
                e.empty_string = '',
                e.v = vecf32([1.0, 2.0, 3.0])
            """

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 4)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test04_create_node_effect(self):
        # test the introduction of a new node by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        # empty node
        q0 = """CREATE ()"""

        # label-less node with attributes
        q1 = """CREATE ({
                            i:1,
                            s:'str',
                            b:True,
                            a:[1, [2], '3'],
                            p:point({latitude: 51, longitude: 0}),
                            f:3.14,
                            empty_string: '',
                            v: vecf32([1.0, 2.0, 3.0])
                        })"""

        # labeled node without attributes
        q2 = """CREATE (:L)"""

        # node with multiple labels and attributes
        q3 = """CREATE (:A:B {
                            i:1,
                            s:'str',
                            b:True,
                            a:[1, [2], '3'],
                            p:point({latitude: 51, longitude: 0}),
                            f:3.14,
                            empty_string: '',
                            v: vecf32([1.0, 2.0, 3.0])
                        })"""

        queries = [q0, q1, q2, q3]
        for q in queries:
            res = self.query_and_sync(q)
            self.env.assertEqual(res.nodes_created, 1)

            self.assert_effect_emitted()

        self.assert_graph_eq()

    def test05_create_edge_effect(self):
        # tests the introduction of a new edge by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        # edge without attributes
        q1 = """CREATE ()-[:R]->()"""

        # edge with attributes
        q2 = """CREATE ()-[:CONNECT {
                                      ei:1,
                                      s:'str',
                                      eb:True,
                                      a:[1, [2], '3'],
                                      ep:point({latitude: 51, longitude: 0}),
                                      f:3.14,
                                      empty_string: '',
                                      v: vecf32([1.0, 2.0, 3.0])}
                            ]->()"""

        # edge between an existing node and a new node
        q3 = """MATCH (a) WITH a LIMIT 1 CREATE (a)-[:R]->()"""

        # edge between two existing nodes
        q4 = """MATCH (a), (b) WITH a, b LIMIT 1 CREATE (a)-[:R]->(b)"""

        queries = [q1, q2, q3, q4]
        for q in queries:
            res = self.query_and_sync(q)
            self.env.assertEqual(res.relationships_created, 1)

            self.assert_effect_emitted()

        self.assert_graph_eq()

    def test06_update_node_effect(self):
        # test an entity attribute set update by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               SET
                    n.xa = 2,
                    n.b = 'string',
                    n.xc = False,
                    n.d = [[2], 1, '3'],
                    n.xe = point({latitude: 41, longitude: 2}),
                    n.f=6.28,
                    n.xempty_string = '',
                    n.v = vecf32([-1.0, -2.0, -3.0])"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update the same attribute multiple times
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               UNWIND range(0, 10) AS i
               SET
                    n.xa = n.xa + 1"""

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update using map overwrite
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               SET n = {
                a:3,
                b:'_string_',
                c:True,
                d:[['3'], 2, 1],
                e:point({latitude: 2, longitude: 41}),
                f:2.68,
                empty_string:'',
                v: vecf32([-1.1, 2.2, -3.3])}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update using map addition
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               SET n += {
                a:4,
                b:'string_',
                c:False,
                d:[['1'], 3, 2.0],
                e:point({latitude: 3, longitude: 40}),
                f:8.26,
                empty_string:'',
                v: vecf32([-1.2, 2.4, -3.6])}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove attribute

        q = "MATCH (n:L) WITH n LIMIT 1 SET n.b = NULL"

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove all attributes

        q = "MATCH (n:L) WITH n LIMIT 1 SET n = {}"

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # add attribute, remove all attributes and add again
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               SET n.v = 'value'
               WITH n
               SET n = {}
               WITH n
               SET n.v = 'value2'"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove attribute via map addition
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               SET n += {x:1, v:NULL, y:2}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test07_update_edge_effect(self):

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        # test an edge attribute set update by an effect
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               SET
                    e.a = 2,
                    e.b = 'string',
                    e.c = False,
                    e.d = [[2], 1, '3'],
                    e.e = point({latitude: 41, longitude: 2}),
                    e.f=6.28,
                    e.empty_string = '',
                    e.v = vecf32([-1.0, -2.0, -3.0])"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update the same attribute multiple times
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               UNWIND range(0, 10) AS i
               SET
                    e.a = e.a + 1"""

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update using map overwrite
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               SET e = {
                a:3,
                b:'_string_',
                c:True,
                d:[['3'], 2, 1],
                e:point({latitude: 2, longitude: 41}),
                f:2.68,
                empty_string:'',
                v: vecf32([-1.1, 2.2, -3.3])}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update using map addition
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               SET e += {
                a:4,
                b:'string_',
                c:False,
                d:[['1'], 3, 2.0],
                e:point({latitude: 3, longitude: 40}),
                f:8.26,
                empty_string:'',
                v: vecf32([-1.2, 2.4, -3.6])}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove attribute

        q = "MATCH ()-[e]->() WITH e LIMIT 1 SET e.b = NULL"

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove all attributes

        q = "MATCH ()-[e]->() WITH e LIMIT 1 SET e = {}"

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # add attribute, remove all attributes and add again
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               SET e.v = 'value'
               WITH e
               SET e = {}
               WITH e
               SET e.v = 'value2'"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove attribute via map addition
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               SET e += {x:1, v:NULL, y:2}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test08_set_labels_effect(self):
        # test the addition of a new node label by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MATCH (n:A:B) SET n:C"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.labels_added, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # test the addition of an existing and anew node label by an effect
        q = """MATCH (n:A:B:C) SET n:C:D"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.labels_added, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test09_remove_labels_effect(self):
        # test the removal of a node label by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MATCH (n:C) REMOVE n:C RETURN n"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.labels_removed, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test10_delete_edge_effect(self):
        # test the deletion of an edge by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MATCH ()-[e]->() WITH e LIMIT 1 DELETE e"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.relationships_deleted, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test11_delete_node_effect(self):
        # test the deletion of a node by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        # using 'n' and 'x' to try and introduce "duplicated" deletions
        q = "MATCH (n) WITH n as n, n as x DELETE n, x"
        res = self.query_and_sync(q)
        self.env.assertGreater(res.nodes_deleted, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test12_merge_node(self):
        # test create and update of a node by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MERGE (n:A {v:'red'})
               ON MATCH SET n.v = 'green'
               ON CREATE SET n.v = 'blue'"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.nodes_created, 1)
        self.env.assertEqual(res.properties_set, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # this time MERGE will match
        q = """MERGE (n:A {v:'blue'})
               ON MATCH SET n.v = 'green'
               ON CREATE SET n.v = 'red'"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 1)
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test13_merge_edge(self):
        # test create and update of an edge by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MERGE (n:A {v:'red'})
               MERGE (n)-[e:R{v:'red'}]->(n)
               ON MATCH SET e.v = 'green'
               ON CREATE SET e.v = 'blue'"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 2)
        self.env.assertEqual(res.relationships_created, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # this time MERGE will match
        q = """MERGE (n:A {v:'red'})
               MERGE (n)-[e:R{v:'blue'}]->(n)
               ON MATCH SET e.v = 'green'
               ON CREATE SET e.v = 'red'"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 1)
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test14_empty_vector(self):
        # test creation of an empty vector

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = "CREATE ({v:vecf32([])})"
        res = self.query_and_sync(q)
        self.env.assertEqual(res.nodes_created, 1)
        self.env.assertEqual(res.properties_set, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test15_create_node_with_random_and_timestamp_effect(self):
        q = "CREATE ({r:rand(), t:timestamp()})"
        res = self.query_and_sync(q)
        self.env.assertEqual(res.nodes_created, 1)
        self.env.assertEqual(res.properties_set, 2)

        self.assert_effect_emitted()
        self.assert_graph_eq()

    def test17_random_ops(self):
        # A graph key of its own. The random ops below are unconstrained,
        # and the tests after this one stay on this key -- which is what the
        # module-global `GRAPH_ID` used to express by being reassigned here.
        self.GRAPH_ID = "random_graph"
        self.master_graph = Graph(self.master, self.GRAPH_ID)
        self.replica_graph = Graph(self.replica, self.GRAPH_ID)

        # enable effects replication

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

    def test18_multiple_nodes(self):
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

    def test19_multiple_edges(self):
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

    def test20_multiple_entities(self):
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

    def test21_mandatory_effects(self):
        """A non deterministic query still replicates correctly.

        It used to be the case that these had to *force* effects, because a
        cheap write would otherwise replay the query and the two sides would
        evaluate `rand()` or `date()` separately. Effects are now the only
        mechanism, so the hazard is gone — but the queries are still worth
        replicating and comparing.
        """

        self.env.flush()        # clean slate

        self.master_graph  = Graph(self.master, self.GRAPH_ID)
        self.replica_graph = Graph(self.replica, self.GRAPH_ID)

        # each of the following queries contains a non deterministic element
        queries = [
            "WITH date()                  AS x CREATE ()",
            "WITH rand()                  AS x CREATE ()",
            "WITH timestamp()             AS x CREATE ()",
            "WITH localtime()             AS x CREATE ()",
            "WITH randomuuid()            AS x CREATE ()",
            "WITH localdatetime()         AS x CREATE ()",
            "WITH date.transaction()      AS x CREATE ()",
            "WITH localtime.transaction() AS x CREATE ()",

            "CREATE ({v:date()})",
            "CREATE ({v:rand()})",
            "CREATE ({v:timestamp()})",
            "CREATE ({v:localtime()})",
            "CREATE ({v:randomuuid()})",
            "CREATE ({v:localdatetime()})",
            "CREATE ({v:date.transaction()})",
            "CREATE ({v:localtime.transaction()})",

            # duplicated query for DB internal execution-plan cache utilization
            "CREATE ({v:date()})",
            "CREATE ({v:rand()})",
            "CREATE ({v:timestamp()})",
            "CREATE ({v:localtime()})",
            "CREATE ({v:randomuuid()})",
            "CREATE ({v:localdatetime()})",
            "CREATE ({v:date.transaction()})",
            "CREATE ({v:localtime.transaction()})",
            ]

        for q in queries:
            self.master_graph.query(q)

            # although effects are disabled
            # we're still expecting replication to use effect
            self.assert_effect_emitted()

        # make sure graphs are the same!
        self.wait_for_replica_offset()
        self.assert_graph_eq()

    def test22_schema_replication(self):
        """
        Make sure a query which introduces a new schema
        but fails doesn't replicate the schema creation
        and removes the schema
        """

        # clean slate
        self.env.flush()

        # replicate via effects

        self.master_graph  = Graph(self.master, self.GRAPH_ID)
        self.replica_graph = Graph(self.replica, self.GRAPH_ID)

        # create a new node schame 'A' mapped to schema id 0
        q = "CREATE (a:A) RETURN a / 0"
        try:
            self.master_graph.query (q)
            # we shouldn't be here
            self.env.assertTrue(False)
        except Exception:
            # as expected
            pass

        # graph should remain empty
        q = "CALL db.labels()"
        res = self.master_graph.ro_query(q).result_set
        self.env.assertEqual(len(res), 0)

        # try to create a second label
        q = "CREATE (b:B)"
        #res = self.master_graph.query (q)
        res = self.query_and_sync(q)
        self.env.assertEqual(res.labels_added, 1)

        q = "CALL db.meta.stats()"
        master_stats  = self.master_graph.ro_query  (q).result_set
        replica_stats = self.replica_graph.ro_query (q).result_set

        self.env.assertEqual(master_stats, replica_stats)


    def test23_update_then_delete_in_one_transaction(self):
        # An entity updated and deleted by the same query. The update is not
        # part of the payload — the replica has no entity to apply it to, and
        # the DELETE record puts it in the same place anyway.
        #
        # Two things are asserted here that unit tests cannot both reach. The
        # cascade form (`DELETE n`, taking its edges with it) only removes the
        # edge inside `commit`, via `delete_implicit_edges`, so it needs a real
        # graph. And `Properties set` must stay 1: the SET did happen, and
        # suppressing the update at the source rather than in the payload would
        # silently change what the query reports — the node form has always
        # reported 1, and the edge forms have to agree with it.
        #
        # Without the payload filter, `digest_updates` reads the type of an
        # edge `commit` has already cleared, and the ordinary accessor panics:
        # the master goes down on an ordinary query.
        self.monitor_mark()

        # A label per case: an earlier case leaves its endpoint behind, and a
        # shared label would make the node case match that leftover too.
        cases = [
            ("UDa", "MATCH (:UDa)-[e:UDR]->() SET e.x = 1 DELETE e",         "explicit edge"),
            ("UDb", "MATCH (n:UDb)-[e:UDR]->() SET e.x = 1 DELETE n",        "cascade"),
            ("UDc", "MATCH (n:UDc)-[e:UDR]->() SET e.x = 1 DETACH DELETE n", "detach"),
            ("UDd", "MATCH (n:UDd) SET n.x = 1 DELETE n",                    "node"),
        ]

        for label, q, name in cases:
            self.query_and_sync(f"CREATE (:{label})-[:UDR]->(:UDB)")
            res = self.query_and_sync(q)

            # The SET ran. What the payload carries is a separate question.
            self.env.assertEqual(res.properties_set, 1, message=name)

            self.assert_effect_emitted()
            self.assert_graph_eq()

            # And the master is still here — this is the regression.
            self.env.assertEqual(self.master.ping(), True, message=name)

        # Neither engine kept a trace of the deleted entities.
        for g in (self.master_graph, self.replica_graph):
            self.env.assertEqual(
                g.ro_query("MATCH ()-[e:UDR]->() RETURN count(e)").result_set[0][0], 0)
            for label in ("UDb", "UDc", "UDd"):
                self.env.assertEqual(
                    g.ro_query(f"MATCH (n:{label}) RETURN count(n)").result_set[0][0], 0)
    def test24_value_edges_survive_the_round_trip(self):
        """Value shapes that otherwise only `graph/src/effects/v3/value.rs`
        covers.

        The `04b_ValueTypes` class was cut when this suite was split and its
        unique cases moved down into unit tests. Those pin the codec and say
        nothing about a real replica -- and two of them are cases a unit test
        structurally *cannot* fail. `Value`'s `PartialEq` compares floats with
        `compare_floats`, so `-0.0 == 0.0`: a round trip that drops the sign
        bit round-trips "successfully". Reading the sign back out through
        `toString`, on both sides, is what turns that into a claim.

        The string cases are here for the other reason. The wire carries a
        string's length as a *byte* count while `size()` reports characters --
        27 bytes against 11 characters below -- so a multi-byte string is
        exactly the shape where a length confusion shows up, and an ASCII one
        never would.
        """
        self.assert_effect_emitted(0)

        self.query_and_sync("""CREATE (:VEdge {
                                   nz:    -0.0,
                                   pz:     0.0,
                                   pinf:   1.0/0.0,
                                   ninf:  -1.0/0.0,
                                   empty:  '',
                                   multi:  'שלום 世界 🐦\u200d🔥'
                              })""")
        self.assert_effect_emitted()

        # Signed zero. `n.nz = n.pz` is true on both sides no matter what
        # happened, so the assertion has to go through the rendering.
        self.assert_agree(
            "MATCH (n:VEdge) RETURN toString(n.nz), toString(n.pz)",
            [['-0', '0']])

        # Infinities, which have no decimal representation to round-trip
        # through and so exercise the float encoding directly.
        self.assert_agree(
            "MATCH (n:VEdge) RETURN toString(n.pinf), toString(n.ninf)",
            [['inf', '-inf']])

        # Byte count against character count, and the empty string -- whose
        # encoded length is the case a "length includes the terminator" rule
        # gets wrong first.
        self.assert_agree("MATCH (n:VEdge) RETURN size(n.empty)", [[0]])
        self.assert_agree("MATCH (n:VEdge) RETURN size(n.multi)", [[11]])
        self.assert_agree("MATCH (n:VEdge) RETURN n.multi",
                          [['שלום 世界 🐦\u200d🔥']])

        # Temporal types, which the old suite only ever created and compared
        # whole-graph. Their values are non-deterministic, so the claim is that
        # the two sides hold the same one rather than any particular one.
        self.query_and_sync("""CREATE (:TEdge {
                                   d:  date(),
                                   t:  localtime(),
                                   dt: localdatetime(),
                                   ts: timestamp()
                              })""")
        self.assert_effect_emitted()
        m, r = self.probe("MATCH (n:TEdge) RETURN n.d, n.t, n.dt, n.ts")
        self.env.assertEqual(len(m), 1)
        self.env.assertEqual(len(m[0]), 4)
        self.env.assertEqual(r, m)

        self.assert_graph_eq()


#-----------------------------------------------------------------------------
# null is a removal, not a value
#
# Moved here from the former test_effects_v3_values.py: it is per-opcode
# coverage of UPDATE_NODE/UPDATE_EDGE, not a shape concern, so it belongs
# beside the update tests above rather than in its own file.
#-----------------------------------------------------------------------------


class testEffects_01_NullIsRemove(_EffectsBase):
    """`SET x = NULL` removes the property, and the replica must remove it too.

    Regression for a real v3 divergence: the apply side filtered nulls out of a
    record's value rows before merging them, so `MATCH (n) SET n.x = NULL`
    removed the property on the primary and did nothing at all on the replica.
    Nothing surfaced it — the link stayed healthy, no error anywhere, and the
    two only re-converged on the next full resync.

    FalkorDB never stores a null property value, which is what makes a null on
    the wire unambiguous: it can only mean "remove this attribute".
    """

    GRAPH_ID = "effects_null"

    def __init__(self):
        self._setup()

    def test01_node_property_set_to_null(self):
        self.query_and_sync("CREATE (:N {id: 1, x: 'gone', y: 'kept'})")
        # baseline — both sides can see the property before it is removed
        self.assert_agree("MATCH (n:N) RETURN count(n.x)", [[1]])

        res = self.query_and_sync("MATCH (n:N) SET n.x = NULL")
        self.env.assertEqual(res.properties_removed, 1)

        # count(expr) skips nulls, so a value surviving on either side is a 1
        self.assert_agree("MATCH (n:N) RETURN count(n.x)", [[0]])
        # ... and the property is *absent*, not merely reading as null
        self.assert_agree("MATCH (n:N) RETURN 'x' IN keys(n)", [[False]])
        # the untouched sibling is still there on both sides
        self.assert_agree("MATCH (n:N) RETURN n.y", [['kept']])
        # keys() enumerates registered attributes in a deterministic order, so
        # the two sides' whole property shape is comparable
        m, r = self.probe("MATCH (n:N) RETURN keys(n)")
        self.env.assertEqual(m, [[['id', 'y']]])
        self.env.assertEqual(r, m)

        self.assert_graph_eq()

    def test02_edge_property_set_to_null(self):
        self.query_and_sync("CREATE ()-[:R {id: 1, x: 'gone', y: 'kept'}]->()")
        self.assert_agree("MATCH ()-[e:R]->() RETURN count(e.x)", [[1]])

        res = self.query_and_sync("MATCH ()-[e:R]->() SET e.x = NULL")
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_agree("MATCH ()-[e:R]->() RETURN count(e.x)", [[0]])
        self.assert_agree("MATCH ()-[e:R]->() RETURN 'x' IN keys(e)", [[False]])
        self.assert_agree("MATCH ()-[e:R]->() RETURN e.y", [['kept']])

        self.assert_graph_eq()

    def test03_one_of_several_properties_set_to_null(self):
        # A null in the middle of a multi-assignment SET: the record carries
        # three attribute columns, one of which is a removal.
        self.query_and_sync("CREATE (:M {a: 1, b: 2, c: 3, d: 4})")

        res = self.query_and_sync("MATCH (n:M) SET n.a = 10, n.b = NULL, n.c = 30")
        # `a` and `c` are overwrites, which FalkorDB accounts for as a removal
        # plus a set; `b` is the only pure removal. So two sets, three removals.
        self.env.assertEqual(res.properties_set, 2)
        self.env.assertEqual(res.properties_removed, 3)

        self.assert_agree("MATCH (n:M) RETURN n.a, n.c, n.d", [[10, 30, 4]])
        self.assert_agree("MATCH (n:M) RETURN 'b' IN keys(n)", [[False]])
        self.assert_agree("MATCH (n:M) RETURN count(n.b)", [[0]])

        # ... and the same via map addition, which reaches the same record
        res = self.query_and_sync("MATCH (n:M) SET n += {a: NULL, e: 5}")
        self.env.assertEqual(res.properties_set, 1)
        self.env.assertEqual(res.properties_removed, 1)
        self.assert_agree("MATCH (n:M) RETURN n.c, n.d, n.e", [[30, 4, 5]])
        self.assert_agree("MATCH (n:M) RETURN 'a' IN keys(n)", [[False]])

        self.assert_graph_eq()

    def test04_whole_shape_null(self):
        # Every column of the record is a removal.
        self.query_and_sync("CREATE (:W {a: 1, b: 2, c: 3})")
        self.query_and_sync("CREATE ()-[:WR {a: 1, b: 2}]->()")

        res = self.query_and_sync("MATCH (n:W) SET n = {a: NULL, b: NULL, c: NULL}")
        self.env.assertEqual(res.properties_removed, 3)
        self.assert_agree("MATCH (n:W) RETURN keys(n)", [[[]]])

        res = self.query_and_sync("MATCH ()-[e:WR]->() SET e = {a: NULL, b: NULL}")
        self.env.assertEqual(res.properties_removed, 2)
        self.assert_agree("MATCH ()-[e:WR]->() RETURN keys(e)", [[[]]])

        # the empty-map form of the same thing
        self.query_and_sync("MATCH (n:W) SET n.z = 1")
        res = self.query_and_sync("MATCH (n:W) SET n = {}")
        self.env.assertEqual(res.properties_removed, 1)
        self.assert_agree("MATCH (n:W) RETURN keys(n)", [[[]]])

        self.assert_graph_eq()

    def test05_null_arriving_as_a_query_parameter(self):
        # The literal NULL and a null-valued parameter take different routes
        # through the planner; both must reach the wire as a removal.
        self.query_and_sync("CREATE (:Param {id: 7, v: 'here'})")
        res = self.master_graph.query("MATCH (n:Param) SET n.v = $new",
                                      {'new': None})
        self.wait_for_replica_offset()
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_agree("MATCH (n:Param) RETURN count(n.v)", [[0]])
        self.assert_agree("MATCH (n:Param) RETURN 'v' IN keys(n)", [[False]])
        self.assert_agree("MATCH (n:Param) RETURN n.id", [[7]])

        self.assert_graph_eq()

    def test06_null_in_only_some_rows_of_one_record(self):
        # The shape most likely to break: one record, many entities, and the
        # column is a real value for some rows and a removal for others. A
        # filtering apply path either drops the whole column or misaligns the
        # values against the ids.
        self.query_and_sync(
            "UNWIND range(1, 1000) AS i CREATE (:P {id: i, x: 'v'})")
        self.assert_agree("MATCH (p:P) RETURN count(p.x)", [[1000]])

        res = self.query_and_sync(
            "MATCH (p:P) SET p.x = CASE WHEN p.id % 2 = 0 THEN NULL ELSE 'v2' END")
        # 500 nulls are pure removals; the 500 overwrites are each accounted
        # for as a removal plus a set.
        self.env.assertEqual(res.properties_removed, 1000)
        self.env.assertEqual(res.properties_set, 500)

        self.assert_agree("MATCH (p:P) RETURN count(p)", [[1000]])
        self.assert_agree("MATCH (p:P) RETURN count(p.x)", [[500]])
        # the survivors are exactly the odd ids, on both sides — this is what
        # catches a column that landed against the wrong entities
        self.assert_agree(
            "MATCH (p:P) WHERE p.x IS NOT NULL RETURN count(p), sum(p.id % 2)",
            [[500, 500]])
        self.assert_agree(
            "MATCH (p:P) WHERE p.x IS NULL RETURN count(p), sum(p.id % 2)",
            [[500, 0]])
        self.assert_agree("MATCH (p:P) WHERE p.x IS NOT NULL RETURN DISTINCT p.x",
                          [['v2']])

        self.assert_graph_eq()

    def test07_null_removes_every_property_of_many_entities(self):
        self.query_and_sync(
            "UNWIND range(1, 500) AS i CREATE (:Q {id: i, x: i, y: i})")
        res = self.query_and_sync("MATCH (q:Q) SET q.x = NULL, q.y = NULL")
        self.env.assertEqual(res.properties_removed, 1000)

        self.assert_agree("MATCH (q:Q) RETURN count(q), count(q.x), count(q.y)",
                          [[500, 0, 0]])
        self.assert_agree("MATCH (q:Q) RETURN DISTINCT keys(q)", [[['id']]])

        self.assert_graph_eq()
