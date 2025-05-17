import random
import string
from common import *

GRAPH_ID = "intern_string"

# test string interning
# string values are unique across the entire DB
# this reduces our overall memory consumption
#
# the tests validates that intern strings are managed as expected
# and memory savings are visible

def random_string(length=10):
    chars = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    return ''.join(random.choices(chars, k=length))

def assertStringPoolStats(conn, count, avg):
    stats = conn.execute_command("GRAPH.INFO", "ObjectPool")

    # GRAPH.INFO ObjectPool
    # 1) "Object Pool"
    # 2) 1) 1) "Unique Objects in Pool"
    # 2) (integer) 1
    # 2) 1) "Average References per Object"
    # 2) "2"

    objs_in_pool   = int(stats[1][0][1])
    avg_ref_count = float(stats[1][1][1])

    assert(avg_ref_count, avg)
    assert(objs_in_pool, count)

class testInternString():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

        # Synchronous deletion
        self.db.config_set('ASYNC_DELETE', 'no')

    def tearDown(self):
        # clear DB
        self.conn.flushall()
        self.graph = self.db.select_graph(GRAPH_ID)

    def used_memory(self):
        # Purge memory
        self.conn.execute_command('MEMORY PURGE')

        # Get memory information
        memory_info = self.conn.info("memory")
        _used_memory = memory_info['used_memory']

        return _used_memory

    def test_single_graph_string_share(self):
        # create a graph with multiple identical string values
        base_line = self.used_memory()
        assertStringPoolStats(self.conn, 0, 0)

        s = 'A' * 16384 # large string

        # create first node
        q = "CREATE ({value: intern($s)})"
        res = self.graph.query(q, {'s': s})

        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.properties_set, 1)

        # validate string pool stats, expecting a single object
        assertStringPoolStats(self.conn, 1, 1)

        # create multiple nodes all sharing the same string value
        q = "UNWIND range(0, 50) AS x CREATE ({value: intern($s)})"
        res = self.graph.query(q, {'s': s})

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 52)
        
    def test_multi_graph_string_share(self):
        # share string across multiple graphs
        self.graph = self.db.select_graph('A')

        s = 'A' * 16384 # large string

        # create first node
        q = "CREATE ({value: intern($s)})"
        res = self.graph.query(q, {'s': s})

        # create multiple EMPTY graphs
        for _ in range(0, 50):
            g = self.db.select_graph(random_string())
            g.query(q, {'s': s})

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 52)

    def test_delete_shared_string(self):
        # make sure shared string isn't released prematurely
        # create a graph with multiple identical string values

        self.graph = self.db.select_graph('A')
        s = 'A' * 16384 # large string

        # create multiple nodes all sharing the same string value
        q = "UNWIND range(0, 20) AS x CREATE ({value: intern($s)})"
        p = {'s': s}
        res = self.graph.query(q, p)

        # delete nodes one by one, make sure shared string isn't freed prematurely
        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        for i in range(node_count-1):
            res = self.graph.query("MATCH (n) WITH n LIMIT 1 DELETE n")
            self.env.assertEquals(res.nodes_deleted, 1)

            # validate string pool stats
            assertStringPoolStats(self.conn, 1, node_count - i - 1)

            # make sure string wan't freed
            rows = self.graph.query("MATCH (n) RETURN n.value").result_set
            for row in rows:
                self.env.assertEquals(row[0], s)

        # expecting graph to be empty
        res = self.graph.query("MATCH (n) DELETE n")
        self.env.assertEquals(res.nodes_deleted, 1)

        # validate string pool stats
        assertStringPoolStats(self.conn, 0, 0)

        #-----------------------------------------------------------------------

        # Share string across multiple graphs
        # create first node
        q = "CREATE ({value: intern($s)})"
        res = self.graph.query(q, p)

        # create multiple graphs
        node_count = 50
        graphs = []
        for _ in range(0, 50):
            g = self.db.select_graph(random_string())
            graphs.append(g)
            res = g.query(q, p)

        for i, g in enumerate(graphs):
            g.query("MATCH (n) delete n")

            # validate string pool stats
            assertStringPoolStats(self.conn, 1, node_count - i)

        # make sure string is valid
        value = self.graph.query("MATCH (n) RETURN n.value").result_set[0][0]
        self.env.assertEquals(value, s)

        # delete the last graph in the DB
        self.graph.delete()

        # validate string pool stats
        assertStringPoolStats(self.conn, 0, 0)

    def test_nested_shared_string(self):
        # make sure shared string via containers isn't released prematurely

        self.graph = self.db.select_graph('A')
        s = 'A' * 16384 # large string

        # create a node with a string attribute
        p = {'s': s}
        self.graph.query("CREATE (:A {v:intern($s)}), (:B {v:[intern($s)]})", p)

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 2)

        # delete first reference to shared string
        res = self.graph.query("MATCH (a:A) DELETE a")
        self.env.assertEquals(res.nodes_deleted, 1)

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 1)

        # verify that string is still valid
        res = self.graph.query("MATCH (b:B) RETURN b.v[0]").result_set
        self.env.assertEquals(res[0][0], s)

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 1)

        self.graph.delete()

        #-----------------------------------------------------------------------

        self.graph = self.db.select_graph('A')

        # same node containing the same string multiple times
        self.graph.query("CREATE (:A {a: intern($s), b:[intern($s)], c:intern($s)})", p)

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 3)

        # delete first reference
        self.graph.query("MATCH (a:A) SET a.a = NULL")
        res = self.graph.query("MATCH (a:A) RETURN a.b[0], a.c").result_set
        self.env.assertEquals(res[0][0], s)
        self.env.assertEquals(res[0][1], s)

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 2)

        # delete second reference
        self.graph.query("MATCH (a:A) SET a.b = NULL")
        res = self.graph.query("MATCH (a:A) RETURN a.c").result_set
        self.env.assertEquals(res[0][0], s)

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 1)

        # delete last reference & delete node
        self.graph.query("MATCH (a:A) SET a.c = NULL")
        self.graph.query("MATCH (a:A) DELETE a")

        # validate string pool stats
        assertStringPoolStats(self.conn, 0, 0)

    def test_intermidate_intern_string(self):
        # make sure intermidate intern string are removed

        # create a node with a string attribute
        self.graph.query("WITH intern('ABCDEF') AS intermidate RETURN intermidate")

        # validate string pool stats
        assertStringPoolStats(self.conn, 0, 0)

    # TODO: test intern strings and UNDO-LOG

class testInternStringPersistency():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

        # skip test if we're running under Sanitizer
        if VALGRIND or SANITIZER:
            self.env.skip() # sanitizer is not working correctly with bulk

    def tearDown(self):
        # clear DB
        self.conn.flushall()

    def testInternStringPersistent(self):
        # populate DB
        s = 'A' * 16384 # large string

        # create first node
        q = "CREATE ({value: intern($s)})"

        # create multiple EMPTY graphs
        for _ in range(0, 50):
            g = self.db.select_graph(random_string())
            g.query(q, {'s': s})

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 51)

        # Save RDB & Load from RDB
        self.env.dumpAndReload()

        # string-pool stats expected to match former stats before reload
        assertStringPoolStats(self.conn, 1, 51)

class testInternStringReplication():
    def __init__(self):
        # skip test if we're running under Valgrind
        if VALGRIND or SANITIZER:
            Environment.skip(None) # valgrind is not working correctly with replication

        self.env, self.db = Env(env='oss', useSlaves=True)
        self.graph = self.db.select_graph(GRAPH_ID)

        self.source_con = self.env.getConnection()
        self.replica_con = self.env.getSlaveConnection()

        # force effects replication
        self.db.config_set('EFFECTS_THRESHOLD', 0)

    def tearDown(self):
        # clear DB
        self.source_con.flushall()
        self.source_con.execute_command("WAIT", "1", "0")

    def query_and_wait(self, q, p={}):
        res = self.graph.query(q, p)

        # the WAIT command forces master slave sync to complete
        self.source_con.execute_command("WAIT", "1", "0")

        return res

    def test_intern_string_replication(self):
        # both master and replica should be empty
        assertStringPoolStats(self.source_con, 0, 0)
        assertStringPoolStats(self.replica_con, 0, 0)

        # replicate a node creation containing an intern string

        s = 'A' * 16384 # large string

        # create first node
        p = {'s': s}
        q = "CREATE ({value: intern($s)})"
        res = self.query_and_wait(q, p)

        assertStringPoolStats(self.source_con, 1, 1)
        assertStringPoolStats(self.replica_con, 1, 1)

        # replicate an addition of an intern string
        q = "MATCH (n) SET n.s = intern($s)"
        res = self.query_and_wait(q, p)

        assertStringPoolStats(self.source_con, 1, 2)
        assertStringPoolStats(self.replica_con, 1, 2)

        # replicate deletion of an intern string
        q = "MATCH (n) SET n.s = null"
        res = self.query_and_wait(q)

        assertStringPoolStats(self.source_con, 1, 1)
        assertStringPoolStats(self.replica_con, 1, 1)

        # replicate update of an intern string
        q = "MATCH (n) SET n.value = intern('intern-string')"
        res = self.query_and_wait(q)

        assertStringPoolStats(self.source_con, 1, 1)
        assertStringPoolStats(self.replica_con, 1, 1)

        # replicae deletion if a node
        q = "MATCH (n) DELETE n"
        res = self.query_and_wait(q)

        assertStringPoolStats(self.source_con, 0, 0)
        assertStringPoolStats(self.replica_con, 0, 0)

