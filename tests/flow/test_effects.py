import time
import threading
from common import *

GRAPH_ID = "effects"
MONITOR_ATTACHED = False

class testEffects():
    # enable effects replication
    def effects_enable(self):
        self.master.execute_command("GRAPH.CONFIG", "SET", "EFFECTS_THRESHOLD", '0')

    # disable effects replication
    def effects_disable(self):
        res = self.master.execute_command("GRAPH.CONFIG", "SET", "EFFECTS_THRESHOLD", '999999')

    # checks if effects replication is enabled
    def effects_enabled(self):
        conf = self.master.execute_command("GRAPH.CONFIG", "GET", "EFFECTS_THRESHOLD")
        return (conf[1] == 0)

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
        q = "MATCH (n) RETURN n ORDER BY(n)"
        master_resultset = self.master_graph.query(q).result_set
        replica_resultset = self.replica_graph.query(q, read_only=True).result_set
        self.env.assertEquals(master_resultset, replica_resultset)

        q = "MATCH ()-[e]->() RETURN e ORDER BY(e)"
        master_resultset = self.master_graph.query(q).result_set
        replica_resultset = self.replica_graph.query(q, read_only=True).result_set
        self.env.assertEquals(master_resultset, replica_resultset)

    def __init__(self):
        self.env = Env(decodeResponses=True, env='oss', useSlaves=True)
        self.monitor = []
        self.master = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        self.master_graph = Graph(self.master, GRAPH_ID)
        self.replica_graph = Graph(self.replica, GRAPH_ID)

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
        self.env.assertEquals(res.labels_added, 1)
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
        self.env.assertEquals(res.relationships_created, 1)

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
                n.v = vector32f([1.0, 2.0, 3.0])
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
                e.v = vector32f([1.0, 2.0, 3.0])
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
                            v: vector32f([1.0, 2.0, 3.0])
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
                            v: vector32f([1.0, 2.0, 3.0])
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
        q2 = """CREATE ()-[:connect {
                                      ei:1,
                                      s:'str',
                                      eb:True,
                                      a:[1, [2], '3'],
                                      ep:point({latitude: 51, longitude: 0}),
                                      f:3.14,
                                      empty_string: '',
                                      v: vector32f([1.0, 2.0, 3.0])}
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
                    n.v = vector32f([-1.0, -2.0, -3.0])"""

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_set, 0)

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
                v: vector32f([-1.1, 2.2, -3.3])}"""

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
                v: vector32f([-1.2, 2.4, -3.6])}"""

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
                    e.v = vector32f([-1.0, -2.0, -3.0])"""

        res = self.query_master_and_wait(q)
        self.env.assertGreater(res.properties_set, 0)

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
                v: vector32f([-1.1, 2.2, -3.3])}"""

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
                v: vector32f([-1.2, 2.4, -3.6])}"""

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

    def test14_rerun_disable_effects(self):
        # test replication works when effects are disabled

        # no leftovers from previous test
        self.env.assertFalse(self.monitor_containt_effect())

        # update graph key
        global GRAPH_ID
        GRAPH_ID = "effects_disabled"

        # update graph objects to use new graph key
        self.master_graph = Graph(self.master, GRAPH_ID)
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

        # make sure no effects had been recieved
        self.env.assertFalse(self.monitor_containt_effect())

