import time
import random
import threading
from common import *
from index_utils import *
from constraint_utils import *
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

        self.master_graph  = Graph(self.master, GRAPH_ID)
        self.replica_graph = Graph(self.replica, GRAPH_ID)

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
        self.master.execute_command("WAIT", 1, 0)
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
        self.effects_enable()

        self.master_graph  = Graph(self.master, GRAPH_ID)
        self.replica_graph = Graph(self.replica, GRAPH_ID)

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
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.labels_added, 1)

        q = "CALL db.meta.stats()"
        master_stats  = self.master_graph.ro_query  (q).result_set
        replica_stats = self.replica_graph.ro_query (q).result_set

        self.env.assertEquals(master_stats, replica_stats)

    def test23_create_index_effect(self, expect_effect=True):
        """Test the introduction of range/fulltext/vector indices by an effect"""

        self.env.flush()  # clean slate

        if expect_effect:
            self.effects_enable()

        global GRAPH_ID
        GRAPH_ID = "index_constraint_effects"
        self.master_graph  = Graph(self.master,  GRAPH_ID)
        self.replica_graph = Graph(self.replica, GRAPH_ID)

        # reset monitor state - a multi-index-creation sequence like this
        # one can legitimately have more than one command in flight, unlike
        # the single-effect-per-step tests above, so start from a clean slate
        # rather than asserting exact monitor-queue quiescence
        self.clear_monitor()

        # range index, single field, introduces a brand new schema+attribute
        create_node_range_index(self.master_graph, "IdxL", "a")
        if expect_effect:
            self.wait_for_effect()
        else:
            self.wait_for_query()

        # range index over a relationship-type, multiple fields
        create_edge_range_index(self.master_graph, "IdxR", "a", "b")
        if expect_effect:
            self.wait_for_effect()
        else:
            self.wait_for_query()

        # fulltext index with index-level language + stopwords options
        create_node_fulltext_index(self.master_graph, "IdxL", "c",
                                    language="english", stopwords=["the", "a"])
        if expect_effect:
            self.wait_for_effect()
        else:
            self.wait_for_query()

        # vector index with custom HNSW parameters
        create_node_vector_index(self.master_graph, "IdxL", "v", dim=4, m=8,
                                  efConstruction=100, efRuntime=5)
        if expect_effect:
            self.wait_for_effect()
        else:
            self.wait_for_query()

        # make sure the replica has actually applied everything above before
        # querying it - waiting on the monitor stream only confirms a
        # command was observed, not that the replica finished applying it
        self.master.wait(1, 400)

        # wait for population to complete on both sides
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)

        self.env.assertEquals(list_indicies(self.master_graph).result_set,
                               list_indicies(self.replica_graph).result_set)

    def test24_create_multi_field_fulltext_index_effect(self, expect_effect=True):
        """
        Multi-field fulltext index creation w/ stopwords - regression test for
        Index_SetStopwords being called more than once for the same statement,
        since each field of a multi-field CREATE INDEX emits its own
        EFFECT_CREATE_INDEX effect
        """

        # reset monitor state (see test23's comment on why)
        self.clear_monitor()

        q = """CREATE FULLTEXT INDEX FOR (e:MultiIdx) ON (e.f1, e.f2, e.f3)
               OPTIONS {stopwords: ['the', 'a'], language: 'english'}"""

        if expect_effect:
            self.query_master_and_wait(q)
            self.wait_for_effect()
        else:
            self.master_graph.query(q)
            self.wait_for_query()

        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)

        self.env.assertEquals(
            list_indicies(self.master_graph, "MultiIdx").result_set,
            list_indicies(self.replica_graph, "MultiIdx").result_set)

    def test25_drop_index_effect(self, expect_effect=True):
        """Test the deletion of an index by an effect"""

        # reset monitor state (see test23's comment on why)
        self.clear_monitor()

        drop_node_range_index(self.master_graph, "IdxL", "a")
        if expect_effect:
            self.wait_for_effect()
        else:
            self.wait_for_query()

        drop_edge_range_index(self.master_graph, "IdxR", "a")
        if expect_effect:
            self.wait_for_effect()
        else:
            self.wait_for_query()

        drop_node_fulltext_index(self.master_graph, "IdxL", "c")
        if expect_effect:
            self.wait_for_effect()
        else:
            self.wait_for_query()

        drop_node_vector_index(self.master_graph, "IdxL", "v")
        if expect_effect:
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.master.wait(1, 400)

        self.env.assertEquals(list_indicies(self.master_graph).result_set,
                               list_indicies(self.replica_graph).result_set)

    def test26_create_constraint_effect(self, expect_effect=True):
        """
        Test the introduction of unique/mandatory constraints by an effect,
        including the async re-announcement issued once enforcement completes
        (Constraint_Replicate) - the replica must converge to the same state
        without crashing/diverging on that second, idempotent announcement
        """

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

        create_unique_node_constraint(self.master_graph, "CtL", "a", sync=True)
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        create_mandatory_node_constraint(self.master_graph, "CtL", "b", sync=True)
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        create_unique_edge_constraint(self.master_graph, "CtR", "a", sync=True)
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        # give the replica time to converge, including any async
        # re-announcement fired once enforcement completed on the master
        self.master.wait(1, 400)
        time.sleep(1)
        self.clear_monitor()

        # replica must still be responsive (no crash/divergence-guard trip)
        master_constraints  = list_constraints(self.master_graph)
        replica_constraints = list_constraints(self.replica_graph)

        self.env.assertEquals(len(master_constraints), 3)
        self.env.assertEquals(master_constraints, replica_constraints)
        for c in master_constraints:
            self.env.assertEquals(c.status, "OPERATIONAL")

    def test27_drop_constraint_effect(self, expect_effect=True):
        """Test the deletion of a constraint by an effect"""

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

        drop_unique_node_constraint(self.master_graph, "CtL", "a")
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        drop_mandatory_node_constraint(self.master_graph, "CtL", "b")
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        drop_unique_edge_constraint(self.master_graph, "CtR", "a")
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        self.master.wait(1, 400)

        self.env.assertEquals(list_constraints(self.master_graph),
                               list_constraints(self.replica_graph))
        self.env.assertEquals(len(list_constraints(self.replica_graph)), 0)

    def test28_edge_fulltext_and_vector_index_effect(self, expect_effect=True):
        """
        Test23 only covers node fulltext/vector index creation - this fills
        the edge-entity gap for those two index types (edge range is already
        covered by test23's IdxR). The fulltext index deliberately reuses
        attribute 'a' as a stopword - the exact condition that triggered the
        GraphHub_AddConstraint use-after-free this suite now guards against,
        this time exercised on the edge/index path instead of the
        node/constraint path that originally surfaced it.
        """

        self.clear_monitor()

        # edge fulltext index with index-level language + stopwords options,
        # added onto IdxR - which already carries a range field ('b'
        # survives test25's drop of 'a') - merging field types onto one
        # underlying index
        create_edge_fulltext_index(self.master_graph, "IdxR", "c",
                                    language="english", stopwords=["the", "a"])
        if expect_effect:
            self.wait_for_effect()
        else:
            self.wait_for_query()

        # edge vector index
        create_edge_vector_index(self.master_graph, "IdxR", "v", dim=4, m=8,
                                  efConstruction=100, efRuntime=5)
        if expect_effect:
            self.wait_for_effect()
        else:
            self.wait_for_query()

        self.master.wait(1, 400)

        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)

        self.env.assertEquals(list_indicies(self.master_graph, "IdxR").result_set,
                               list_indicies(self.replica_graph, "IdxR").result_set)

    def test29_drop_remaining_indices_effect(self, expect_effect=True):
        """
        Complete the edge-index-deletion coverage test25 left open: drop
        IdxR's remaining range field ('b'), plus the fulltext/vector fields
        added by test28, fully emptying IdxR of indices.
        """

        self.clear_monitor()

        drop_edge_range_index(self.master_graph, "IdxR", "b")
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        drop_edge_fulltext_index(self.master_graph, "IdxR", "c")
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        drop_edge_vector_index(self.master_graph, "IdxR", "v")
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        self.master.wait(1, 400)

        master_indices = list_indicies(self.master_graph, "IdxR").result_set
        self.env.assertEquals(master_indices,
                               list_indicies(self.replica_graph, "IdxR").result_set)
        self.env.assertEquals(len(master_indices), 0)

    def test30_index_correctness_after_data_mutation(self, expect_effect=True):
        """
        Verify range/fulltext/vector indices stay correct on the replica not
        just at the db.indexes() metadata level, but for actual index-backed
        query results, across data creation, update and deletion - this
        codebase has a history of vector-index KNN divergence bugs that pure
        metadata comparison would never catch (see project memory
        project_replica_divergence_findings).
        """

        self.clear_monitor()

        create_node_range_index(self.master_graph, "Doc", "score")
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        create_node_fulltext_index(self.master_graph, "Doc", "body")
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        create_node_vector_index(self.master_graph, "Doc", "emb", dim=4)
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        self.master.wait(1, 400)
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)

        def assert_indices_agree():
            range_q = """MATCH (d:Doc) WHERE d.score > 5
                         RETURN d.score ORDER BY d.score"""
            m_range = self.master_graph.ro_query(range_q).result_set
            r_range = self.replica_graph.ro_query(range_q).result_set
            self.env.assertEquals(m_range, r_range)
            self.env.assertTrue(len(m_range) > 0)

            ft_q = """CALL db.idx.fulltext.queryNodes('Doc', 'fox')
                      YIELD node RETURN node.score ORDER BY node.score"""
            m_ft = self.master_graph.ro_query(ft_q).result_set
            r_ft = self.replica_graph.ro_query(ft_q).result_set
            self.env.assertEquals(m_ft, r_ft)
            self.env.assertTrue(len(m_ft) > 0)

            vec_q = """CALL db.idx.vector.queryNodes('Doc', 'emb', 3, vecf32($q))
                       YIELD node, score
                       RETURN node.score AS docScore, score ORDER BY score"""
            m_vec = self.master_graph.ro_query(vec_q, params={'q': [5, 5, 5, 5]}).result_set
            r_vec = self.replica_graph.ro_query(vec_q, params={'q': [5, 5, 5, 5]}).result_set
            self.env.assertEquals(m_vec, r_vec)
            self.env.assertTrue(len(m_vec) > 0)

        #--------------------------------------------------------------------
        # create data
        #--------------------------------------------------------------------

        q = """UNWIND range(0, 9) AS i
               CREATE (:Doc {score: i, body: 'the quick fox ' + i,
                             emb: vecf32([i, i, i, i])})"""
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.nodes_created, 10)
        self.wait_for_effect() if expect_effect else self.wait_for_query()
        self.master.wait(1, 400)

        assert_indices_agree()

        #--------------------------------------------------------------------
        # update data (property values that back all three index types)
        #--------------------------------------------------------------------

        q = """MATCH (d:Doc) WHERE d.score = 3
               SET d.score = 100, d.body = 'updated lorem ipsum',
                   d.emb = vecf32([100, 100, 100, 100])"""
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.properties_set, 3)
        self.wait_for_effect() if expect_effect else self.wait_for_query()
        self.master.wait(1, 400)

        assert_indices_agree()

        #--------------------------------------------------------------------
        # delete data
        #--------------------------------------------------------------------

        q = "MATCH (d:Doc) WHERE d.score = 100 DELETE d"
        res = self.query_master_and_wait(q)
        self.env.assertEquals(res.nodes_deleted, 1)
        self.wait_for_effect() if expect_effect else self.wait_for_query()
        self.master.wait(1, 400)

        assert_indices_agree()

    def test31_stress_index_and_constraint_create_drop(self, expect_effect=True):
        """
        Stress-test rapid creation and deletion of range/fulltext/vector
        indices and unique/mandatory constraints across many labels and
        relationship-types, with only the minimal replication-ack wait
        between steps (no generous settling delays) - this is the exact
        timing regime that exposed the GraphHub_AddConstraint use-after-free
        (see ApplyCreateConstraint / VerifyAttribute divergence in project
        memory project_constraint_effect_uaf_bug). An unchanged sync_full
        count proves no forced full resync (i.e. no divergence) happened
        along the way, rather than just relying on final-state equality
        after a self-healing resync could have masked the same class of bug.
        """

        self.clear_monitor()

        sync_full_before = self.master.info()["sync_full"]

        def drain(n=1):
            for _ in range(n):
                self.wait_for_effect() if expect_effect else self.wait_for_query()

        node_labels = ["Stress_N0", "Stress_N1", "Stress_N2"]
        edge_types  = ["Stress_E0", "Stress_E1", "Stress_E2"]

        for lbl in node_labels:
            create_node_range_index(self.master_graph, lbl, "a", "b")
            drain()
            # reuse attribute 'a' as a stopword too - the exact condition
            # that originally triggered the corruption this test guards
            # against
            create_node_fulltext_index(self.master_graph, lbl, "c",
                                        stopwords=["the", "a"])
            drain()
            create_node_vector_index(self.master_graph, lbl, "d", dim=4)
            drain()

            create_unique_node_constraint(self.master_graph, lbl, "a")
            drain()
            create_mandatory_node_constraint(self.master_graph, lbl, "b")
            drain()

        for rel in edge_types:
            create_edge_range_index(self.master_graph, rel, "a", "b")
            drain()
            create_edge_fulltext_index(self.master_graph, rel, "c",
                                        stopwords=["the", "a"])
            drain()
            create_edge_vector_index(self.master_graph, rel, "d", dim=4)
            drain()

            create_unique_edge_constraint(self.master_graph, rel, "a")
            drain()
            create_mandatory_edge_constraint(self.master_graph, rel, "b")
            drain()

        # tear everything back down - constraints first, since an index
        # backing a unique constraint can't be dropped while it's alive
        for lbl in node_labels:
            drop_unique_node_constraint(self.master_graph, lbl, "a")
            drain()
            drop_mandatory_node_constraint(self.master_graph, lbl, "b")
            drain()
            drop_node_vector_index(self.master_graph, lbl, "d")
            drain()
            drop_node_fulltext_index(self.master_graph, lbl, "c")
            drain()
            drop_node_range_index(self.master_graph, lbl, "a")
            drain()
            drop_node_range_index(self.master_graph, lbl, "b")
            drain()

        for rel in edge_types:
            drop_unique_edge_constraint(self.master_graph, rel, "a")
            drain()
            drop_mandatory_edge_constraint(self.master_graph, rel, "b")
            drain()
            drop_edge_vector_index(self.master_graph, rel, "d")
            drain()
            drop_edge_fulltext_index(self.master_graph, rel, "c")
            drain()
            drop_edge_range_index(self.master_graph, rel, "a")
            drain()
            drop_edge_range_index(self.master_graph, rel, "b")
            drain()

        # let the replica fully apply and settle
        self.master.wait(1, 400)
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)

        # no divergence-triggered forced full resync happened along the way
        sync_full_after = self.master.info()["sync_full"]
        self.env.assertEquals(sync_full_after, sync_full_before)

        # final state: everything dropped, master and replica agree
        for lbl in node_labels:
            m = list_indicies(self.master_graph, lbl).result_set
            self.env.assertEquals(m, list_indicies(self.replica_graph, lbl).result_set)
            self.env.assertEquals(len(m), 0)

        for rel in edge_types:
            m = list_indicies(self.master_graph, rel).result_set
            self.env.assertEquals(m, list_indicies(self.replica_graph, rel).result_set)
            self.env.assertEquals(len(m), 0)

        self.env.assertEquals(list_constraints(self.master_graph),
                               list_constraints(self.replica_graph))
        self.env.assertEquals(len(list_constraints(self.master_graph)), 0)

    def test32_invalid_duplicate_index_field(self, expect_effect=True):
        """
        A CREATE INDEX statement with a duplicated field must error, leave no
        partial index/attribute/schema behind, and replicate nothing to the
        replica - even though per-field effect emission means the first
        occurrence of the field succeeds internally (schema/attribute/index
        effects already queued) before the duplicate is detected and the
        whole statement is rolled back via the undo log.
        """

        self.clear_monitor()

        try:
            self.master_graph.query("CREATE INDEX FOR (n:BadIdx) ON (n.a, n.a)")
            self.env.assertTrue(False, "expected query to fail")
        except ResponseError as e:
            self.env.assertContains("already indexed", str(e))

        # a failed statement must replicate nothing at all, effect or
        # otherwise
        self.env.assertEquals(len(self.monitor), 0)

        self.assert_graph_eq()
        self.env.assertEquals(
            len(list_indicies(self.master_graph, "BadIdx").result_set), 0)

        # the label/attribute must not have been left half-created - a
        # perfectly valid index on the same field should still work
        create_node_range_index(self.master_graph, "BadIdx", "a")
        self.wait_for_effect() if expect_effect else self.wait_for_query()
        self.master.wait(1, 400)

        self.env.assertEquals(list_indicies(self.master_graph, "BadIdx").result_set,
                               list_indicies(self.replica_graph, "BadIdx").result_set)

    def test33_invalid_duplicate_constraint_property(self, expect_effect=True):
        """
        A CREATE CONSTRAINT statement with a duplicated property must error,
        leave no partial constraint behind, and replicate nothing.
        """

        self.clear_monitor()

        create_node_range_index(self.master_graph, "BadCt", "x")
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        self.clear_monitor()

        try:
            create_unique_node_constraint(self.master_graph, "BadCt", "x", "x")
            self.env.assertTrue(False, "expected constraint creation to fail")
        except ResponseError as e:
            self.env.assertContains("duplicate", str(e).lower())

        self.env.assertEquals(len(self.monitor), 0)

        self.master.wait(1, 400)
        self.env.assertEquals(list_constraints(self.master_graph),
                               list_constraints(self.replica_graph))
        self.env.assertEquals(len(list_constraints(self.master_graph)), 0)

        # the graph must not be left in a broken state - a valid constraint
        # on the same label/property should still work afterward
        create_unique_node_constraint(self.master_graph, "BadCt", "x", sync=True)
        self.wait_for_effect() if expect_effect else self.wait_for_query()

        # give the replica time to converge, including any async
        # re-announcement fired once enforcement completed on the master
        # (see test26's comment on why)
        self.master.wait(1, 400)
        time.sleep(1)
        self.clear_monitor()

        self.env.assertEquals(list_constraints(self.master_graph),
                               list_constraints(self.replica_graph))
        self.env.assertEquals(len(list_constraints(self.master_graph)), 1)

