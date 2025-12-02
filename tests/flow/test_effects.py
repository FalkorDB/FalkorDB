import time
import random
import threading
from common import *
from index_utils import *
from graph_utils import graph_eq

GRAPH_ID = "effects"
MONITOR_ATTACHED = False

class testEffects():
    # enable effects replication
    def effects_enable(self):
       self.db.config_set("EFFECTS_THRESHOLD", 0)

    # disable effects replication
    def effects_disable(self):
        self.db.config_set("EFFECTS_THRESHOLD", 999999)

    # checks if effects replication is enabled
    def effects_enabled(self):
        threshold = self.db.config_get("EFFECTS_THRESHOLD")
        return (threshold == 0)

    # checks if effects replication is enabled
    def effects_disabled(self):
        return not self.effects_enabled()

    def monitor_thread(self):
        global MONITOR_ATTACHED
        try:
            with self.replica.monitor() as m:
                MONITOR_ATTACHED = True
                for cmd in m.listen():
                    if 'GRAPH.EFFECT' in cmd['command'] or 'GRAPH.QUERY' in cmd['command']:
                        self.monitor.append(cmd)
        except:
            pass

    def wait_for_command(self, cmd, timeout=500):
        # wait for monitor to receive cmd
        found = False
        interval = 0.2

        while not found and timeout > 0:
            while len(self.monitor) == 0:
                # wait for an item
                time.sleep(interval)
                timeout -= interval
            item = self.monitor.pop()
            found = cmd in item['command']

        if found is False:
            raise Exception(f"missing expected replicated command: {cmd}")

    def wait_for_effect(self):
        self.wait_for_command('GRAPH.EFFECT')

    def wait_for_query(self):
        self.wait_for_command('GRAPH.QUERY')

    def monitor_containt_effect(self):
        for item in self.monitor:
            if 'GRAPH.EFFECT' in item['command']:
                return True
        return False

    def clear_monitor(self):
        self.monitor = []

    # query master and wait for replica
    def query_master_and_wait(self, q):
        res = self.master_graph.query(q)

        # wait for replica to ack write
        self.master.wait(1, 400)

        return res

    # asserts that master and replica have the same view over the graph
    def assert_graph_eq(self):
        self.env.assertTrue(graph_eq(self.master_graph, self.replica_graph))

    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True)
        self.monitor = []
        self.master = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        self.master_graph = Graph(self.master, GRAPH_ID)
        self.replica_graph = Graph(self.replica, GRAPH_ID)

        # create indices
        create_node_range_index(self.master_graph, "L", "a", "b", "c")
        create_edge_range_index(self.master_graph, "R", "a", "b", "c")

        # wait for replica and master to sync
        self.master.wait(1, 0)

        self.effects_enable()

        self.monitor_thread = threading.Thread(target=self.monitor_thread)
        self.monitor_thread.start()
        # wait for monitor thread to attach
        while MONITOR_ATTACHED is False:
            time.sleep(0.2)

    def __del__(self):
        # all done, shutdown replica
        # stops monitor thread
        self.replica.shutdown()
    
    def test01_effect_default_config(self):
        # make sure effects are enabled by default
        self.env.assertTrue(self.effects_enabled())

    def test02_add_schema_effect(self, expect_effect=True):
        # test the introduction of a schema by an effect

        # introduce a new label which in turn creates a new schema
        q = "CREATE (:L)"
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.nodes_created, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # introduce multiple labels
        q = "CREATE (:X:Y)"
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.labels_added, 2)
        self.env.assertEquals(res.nodes_created, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # introduce a new relationship-type which in turn creates a new schema
        q = "CREATE ()-[:R]->()"
        res = self.query_master_and_wait(q)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

    def test03_add_attribute_effect(self, expect_effect=True):
        # test the introduction of an attribute by an effect

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

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

        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.properties_set, 5)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        q = """MATCH ()-[e]->()
                WITH e
                LIMIT 1
                SET
                e.e = point({latitude: 51, longitude: 0}),
                e.f=3.14,
                e.empty_string = '',
                e.v = vecf32([1.0, 2.0, 3.0])
            """

        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.properties_set, 4)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

    def test04_create_node_effect(self, expect_effect=True):
        # test the introduction of a new node by an effect

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

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
            res = self.query_master_and_wait(q)
            self.env.assertEquals(res.nodes_created, 1)

            if(expect_effect):
                self.wait_for_effect()
            else:
                self.wait_for_query()

        self.assert_graph_eq()

    def test05_create_edge_effect(self, expect_effect=True):
        # tests the introduction of a new edge by an effect

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

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
            res = self.query_master_and_wait(q)
            self.env.assertEquals(res.relationships_created, 1)

            if(expect_effect):
                self.wait_for_effect()
            else:
                self.wait_for_query()

        self.assert_graph_eq()

    def test06_update_node_effect(self, expect_effect=True):
        # test an entity attribute set update by an effect

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

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

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_set, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # update the same attribute multiple times
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               UNWIND range(0, 10) AS i
               SET
                    n.xa = n.xa + 1"""

        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.properties_set, 11)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

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

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

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

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_set, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # remove attribute

        q = "MATCH (n:L) WITH n LIMIT 1 SET n.b = NULL"

        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.properties_removed, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # remove all attributes

        q = "MATCH (n:L) WITH n LIMIT 1 SET n = {}"

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_removed, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

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

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_removed, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # remove attribute via map addition
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               SET n += {x:1, v:NULL, y:2}"""

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

    def test07_update_edge_effect(self, expect_effect=True):

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

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

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_set, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # update the same attribute multiple times
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               UNWIND range(0, 10) AS i
               SET
                    e.a = e.a + 1"""

        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.properties_set, 11)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

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

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

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

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_set, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # remove attribute

        q = "MATCH ()-[e]->() WITH e LIMIT 1 SET e.b = NULL"

        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.properties_removed, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # remove all attributes

        q = "MATCH ()-[e]->() WITH e LIMIT 1 SET e = {}"

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_removed, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

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

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_removed, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # remove attribute via map addition
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               SET e += {x:1, v:NULL, y:2}"""

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

    def test08_set_labels_effect(self, expect_effect=True):
        # test the addition of a new node label by an effect

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

        q = """MATCH (n:A:B) SET n:C"""
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.labels_added, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # test the addition of an existing and anew node label by an effect
        q = """MATCH (n:A:B:C) SET n:C:D"""
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.labels_added, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

    def test09_remove_labels_effect(self, expect_effect=True):
        # test the removal of a node label by an effect

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

        q = """MATCH (n:C) REMOVE n:C RETURN n"""
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.labels_removed, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

    def test10_delete_edge_effect(self, expect_effect=True):
        # test the deletion of an edge by an effect

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

        q = """MATCH ()-[e]->() WITH e LIMIT 1 DELETE e"""
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.relationships_deleted, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

    def test11_delete_node_effect(self, expect_effect=True):
        # test the deletion of a node by an effect

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

        # using 'n' and 'x' to try and introduce "duplicated" deletions
        q = "MATCH (n) WITH n as n, n as x DELETE n, x"
        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.nodes_deleted, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

    def test12_merge_node(self, expect_effect=True):
        # test create and update of a node by an effect

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

        q = """MERGE (n:A {v:'red'})
               ON MATCH SET n.v = 'green'
               ON CREATE SET n.v = 'blue'"""
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.properties_set, 2)
        self.env.assertEquals(res.properties_removed, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # this time MERGE will match
        q = """MERGE (n:A {v:'blue'})
               ON MATCH SET n.v = 'green'
               ON CREATE SET n.v = 'red'"""
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.properties_set, 1)
        self.env.assertEquals(res.properties_removed, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

    def test13_merge_edge(self, expect_effect=True):
        # test create and update of an edge by an effect

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

        q = """MERGE (n:A {v:'red'})
               MERGE (n)-[e:R{v:'red'}]->(n)
               ON MATCH SET e.v = 'green'
               ON CREATE SET e.v = 'blue'"""
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.properties_set, 3)
        self.env.assertEquals(res.relationships_created, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

        # this time MERGE will match
        q = """MERGE (n:A {v:'red'})
               MERGE (n)-[e:R{v:'blue'}]->(n)
               ON MATCH SET e.v = 'green'
               ON CREATE SET e.v = 'red'"""
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.properties_set, 1)
        self.env.assertEquals(res.properties_removed, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

    def test14_empty_vector(self, expect_effect=True):
        # test creation of an empty vector

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

        q = "CREATE ({v:vecf32([])})"
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.properties_set, 1)

        if(expect_effect):
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.assert_graph_eq()

    def test15_create_node_with_random_and_timestamp_effect(self, expect_effect=True):
        q = "CREATE ({r:rand(), t:timestamp()})"
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.properties_set, 2)

        if expect_effect:
            self.wait_for_effect()
            self.assert_graph_eq()
        else:
            self.wait_for_query()
            # graphs will likely differ.

    def test16_rerun_disable_effects(self):
        # test replication works when effects are disabled

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

        # update graph key
        global GRAPH_ID
        GRAPH_ID = "effects_disabled"

        # update graph objects to use new graph key
        self.master_graph  = Graph(self.master,  GRAPH_ID)
        self.replica_graph = Graph(self.replica, GRAPH_ID)

        # disable effects replication
        self.effects_disable()

        # re-run tests, this time effects is turned off
        # replication should be done via query replication
        self.test02_add_schema_effect(False)
        self.test03_add_attribute_effect(False)
        self.test04_create_node_effect(False)
        self.test05_create_edge_effect(False)
        self.test06_update_node_effect(False)
        self.test07_update_edge_effect(False)
        self.test08_set_labels_effect(False)
        self.test09_remove_labels_effect(False)
        self.test10_delete_edge_effect(False)
        self.test11_delete_node_effect(False)
        self.test12_merge_node(False)
        self.test13_merge_edge(False)
        self.test14_empty_vector(False)
        self.test15_create_node_with_random_and_timestamp_effect(True) # non deterministic

        # make sure no effects had been recieved
        self.env.assertFalse(self.monitor_containt_effect())

    def test17_random_ops(self):
        # update graph key
        global GRAPH_ID
        GRAPH_ID = "random_graph"

        # update graph objects to use new graph key
        self.master_graph = Graph(self.master, GRAPH_ID)
        self.replica_graph = Graph(self.replica, GRAPH_ID)

        # enable effects replication
        self.effects_enable()

        from random_graph import create_random_schema, create_random_graph, run_random_graph_ops, ALL_OPS
        nodes, edges = create_random_schema()
        create_random_graph(self.master_graph, nodes, edges)

        # wait for replica and master to sync
        self.master.wait(1, 0)
        self.assert_graph_eq()

        run_random_graph_ops(self.master_graph, nodes, edges, ALL_OPS)

        # wait for replica and master to sync
        self.master.wait(1, 0)
        self.assert_graph_eq()

    def test18_multiple_nodes(self):
        """Test the creation & deletion of multiple nodes."""

        self.env.flush()  # clean slate
        self.effects_enable()

        # labels
        lbls = ["L0", "L1", "L2", "L3"]

        # create 2048 nodes with random labels: L0, L1, L2, L3
        q = "(:{})"
        nodes = [q.format(random.choice(lbls)) for _ in range(2048)]
        multi_create = "CREATE " + ",".join(nodes)
        res = self.query_master_and_wait(multi_create)

        self.env.assertEquals(res.nodes_created, 2048)
        self.assert_graph_eq()

        # delete nodes
        res = self.query_master_and_wait("MATCH (n) DELETE n")
        self.env.assertEquals(res.nodes_deleted, 2048)

        self.assert_graph_eq()

        q = "MATCH (n) RETURN count(n)"
        replica_node_count = self.replica_graph.ro_query(q).result_set[0][0]
        master_node_count = self.master_graph.query(q).result_set[0][0]
        self.env.assertEquals(master_node_count, 0)
        self.env.assertEquals(replica_node_count, master_node_count)

        for l in lbls:
            q = "MATCH (n:{}) RETURN count(n)".format(l)
            master_node_count = self.master_graph.query(q).result_set[0][0]
            replica_node_count = self.replica_graph.ro_query(q).result_set[0][0]
            self.env.assertEquals(master_node_count, 0)
            self.env.assertEquals(replica_node_count, master_node_count)

    def test19_multiple_edges(self):
        """Test the creation & deletion of multiple edges."""

        self.env.flush()  # clean slate
        self.effects_enable()

        # relation types
        types = ["R0", "R1", "R2", "R3"]

        # create 2048 edges of types: R0, R1, R2, R3
        q = "()-[:{}]->()"
        edges = [q.format(random.choice(types)) for _ in range(2048)]
        multi_create = "CREATE" + ",".join(edges)
        res = self.query_master_and_wait(multi_create)

        self.env.assertEquals(res.relationships_created, 2048)
        self.assert_graph_eq()

        # delete edges
        res = self.query_master_and_wait("MATCH ()-[e]->() DELETE e")

        self.env.assertEquals(res.relationships_deleted, 2048)
        self.assert_graph_eq()

        q = "MATCH ()-[e]->() RETURN count(e)"
        replica_edge_count = self.replica_graph.ro_query(q).result_set[0][0]
        master_edge_count = self.master_graph.query(q).result_set[0][0]
        self.env.assertEquals(master_edge_count, 0)
        self.env.assertEquals(replica_edge_count, master_edge_count)

        for t in types:
            q = "MATCH ()-[e:{}]->() RETURN count(e)".format(t)
            master_edge_count = self.master_graph.query(q).result_set[0][0]
            replica_edge_count = self.replica_graph.ro_query(q).result_set[0][0]
            self.env.assertEquals(master_edge_count, 0)
            self.env.assertEquals(replica_edge_count, master_edge_count)

    def test20_multiple_entities(self):
        """Test creation & deletion of multiple entities with a single randomized delete query."""

        self.env.flush()  # clean slate
        self.effects_enable()

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
        res = self.query_master_and_wait(multi_create)

        self.env.assertEquals(res.nodes_created, node_count)
        self.env.assertEquals(res.relationships_created, edge_count)
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
        res = self.query_master_and_wait(single_delete_query)

        self.env.assertEqual(res.nodes_deleted, node_count)
        self.env.assertEqual(res.relationships_deleted, edge_count)
        self.assert_graph_eq()

        #-------------------------------------------------------------------
        # verification: master and replica must be empty
        #-------------------------------------------------------------------

        q = "MATCH (n) RETURN count(n)"
        replica_node_count = self.replica_graph.ro_query(q).result_set[0][0]
        master_node_count = self.master_graph.query(q).result_set[0][0]
        self.env.assertEquals(master_node_count, 0)
        self.env.assertEquals(replica_node_count, master_node_count)

        q = "MATCH ()-[e]->() RETURN count(e)"
        replica_edge_count = self.replica_graph.ro_query(q).result_set[0][0]
        master_edge_count = self.master_graph.query(q).result_set[0][0]
        self.env.assertEquals(master_edge_count, 0)
        self.env.assertEquals(replica_edge_count, master_edge_count)

    def test21_mandatory_effects(self):
        """Make sure non deterministic queries always uses effects"""

        self.env.flush()        # clean slate
        self.effects_disable()  # disable effects

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
            self.wait_for_effect()

        # make sure graphs are the same!
        self.assert_graph_eq()

