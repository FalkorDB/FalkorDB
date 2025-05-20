import time
import random
import string
from common import *

GRAPH_ID = "intern_string"
LARGE_STRING = 'A' * 16384 # large string
SMALL_STRING = 'A'

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
    # sleep for a short period of time to allow the DB main thread
    # to catch up with the string-pool stats recent changes
    time.sleep(0.01)
    stats = conn.execute_command("GRAPH.INFO", "ObjectPool")

    # GRAPH.INFO ObjectPool
    # 1) "Object Pool"
    # 2) 1) 1) "Unique Objects in Pool"
    # 2) (integer) 1
    # 2) 1) "Average References per Object"
    # 2) "2"

    objs_in_pool   = int(stats[1][0][1])
    avg_ref_count = float(stats[1][1][1])

    assert avg_ref_count == avg, f"expected avg={avg}, got {avg_ref_count}"
    assert objs_in_pool == count, f"expected count={count}, got {objs_in_pool}"

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

        assertStringPoolStats(self.conn, 0, 0)

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

        # create first node
        q = "CREATE ({value: intern($s)})"
        res = self.graph.query(q, {'s': LARGE_STRING})

        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.properties_set, 1)

        # validate string pool stats, expecting a single object
        assertStringPoolStats(self.conn, 1, 1)

        # create multiple nodes all sharing the same string value
        q = "UNWIND range(0, 10) AS x CREATE ({value: intern($s)})"
        res = self.graph.query(q, {'s': LARGE_STRING})

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 12)
        
    def test_multi_graph_string_share(self):
        # share string across multiple graphs

        # create first node
        q = "CREATE ({value: intern($s)})"
        res = self.graph.query(q, {'s': LARGE_STRING})

        # create multiple EMPTY graphs
        for _ in range(0, 10):
            g = self.db.select_graph(random_string())
            g.query(q, {'s': LARGE_STRING})

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 11)

    def test_delete_shared_string(self):
        # make sure shared string isn't released prematurely
        # create a graph with multiple identical string values

        # create multiple nodes all sharing the same string value
        q = "UNWIND range(0, 9) AS x CREATE ({value: intern($s)})"
        p = {'s': SMALL_STRING}
        self.graph.query(q, p)

        assertStringPoolStats(self.conn, 1, 10)

        # delete nodes one by one, make sure shared string isn't freed prematurely
        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        for i in range(node_count):
            # make sure string wan't freed
            rows = self.graph.query("MATCH (n) RETURN n.value").result_set
            for row in rows:
                self.env.assertEquals(row[0], SMALL_STRING)

            res = self.graph.query("MATCH (n) WITH n LIMIT 1 DELETE n")
            self.env.assertEquals(res.nodes_deleted, 1)

            # validate string pool stats
            avg = node_count - i - 1
            str_count = 1 if avg > 0 else 0
            assertStringPoolStats(self.conn, str_count, avg)

        # expecting graph to be empty
        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        self.env.assertEquals(node_count, 0)

        # validate string pool stats
        assertStringPoolStats(self.conn, 0, 0)

        #-----------------------------------------------------------------------

        # Share string across multiple graphs
        # create first node
        q = "CREATE ({value: intern($s)})"
        res = self.graph.query(q, p)
        assertStringPoolStats(self.conn, 1, 1)

        # create multiple graphs
        graphs = []
        node_count = 10
        for i in range(0, node_count):
            r = random_string()
            g = self.db.select_graph(r)
            graphs.append(g)
            res = g.query(q, p)
            assertStringPoolStats(self.conn, 1, i + 2)

        # delete string from each dedicated graph
        for i, g in enumerate(graphs):
            res = g.query("MATCH (n) delete n")
            self.env.assertEquals(res.nodes_deleted, 1)

            # validate string pool stats
            assertStringPoolStats(self.conn, 1, node_count - i)

        assertStringPoolStats(self.conn, 1, 1)

        # make sure string is valid
        value = self.graph.query("MATCH (n) RETURN n.value").result_set[0][0]
        self.env.assertEquals(value, SMALL_STRING)

        # delete the last graph in the DB
        self.graph.delete()

        # validate string pool stats
        assertStringPoolStats(self.conn, 0, 0)

    def test_nested_shared_string(self):
        # make sure shared string via containers isn't released prematurely

        # create a node with a string attribute
        p = {'s': SMALL_STRING}
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
        self.env.assertEquals(res[0][0], SMALL_STRING)

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 1)

        self.graph.delete()
        assertStringPoolStats(self.conn, 0, 0)

        #-----------------------------------------------------------------------

        # same node containing the same string multiple times
        self.graph.query("CREATE (:A {a: intern($s), b:[intern($s)], c:intern($s)})", p)

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 3)

        # delete first reference
        self.graph.query("MATCH (a:A) SET a.a = NULL")
        res = self.graph.query("MATCH (a:A) RETURN a.b[0], a.c").result_set
        self.env.assertEquals(res[0][0], SMALL_STRING)
        self.env.assertEquals(res[0][1], SMALL_STRING)

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 2)

        # delete second reference
        self.graph.query("MATCH (a:A) SET a.b = NULL")
        res = self.graph.query("MATCH (a:A) RETURN a.c").result_set
        self.env.assertEquals(res[0][0], SMALL_STRING)

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

    def test_undolog(self):
        # create an initial node with an intern string
        q = "CREATE (n:N {v:intern($s)}) RETURN n.v, typeof(n.v)"
        res = self.graph.query(q, {'s': LARGE_STRING})

        # assert query results
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.properties_set, 1)
        self.env.assertEquals(res.result_set[0][0], LARGE_STRING)
        self.env.assertEquals(res.result_set[0][1], "Intern String")
        assertStringPoolStats(self.conn, 1, 1)

        # try to overwrite an intern string with a regular string
        # query should fail
        q = """MATCH (n)
               SET n.v = 'new-string'
               WITH n
               SET n.x = n
               RETURN n.v, typeof(n.v)"""

        try:
            # expecting query to fail
            res = self.graph.query(q)
            self.env.assertFalse("query should fail")
        except:
            pass

        # validate node's intern string was restored
        q = "MATCH (n) RETURN n.v, typeof(n.v)"
        res = self.graph.query(q).result_set
        self.env.assertEquals(res[0][0], LARGE_STRING)
        self.env.assertEquals(res[0][1], "Intern String")
        assertStringPoolStats(self.conn, 1, 1)

    def test_implicit_copy(self):
        # implicit copy of an intern string should produce an intern string
        q = """CREATE (a {v:intern($s)}), (b)
               WITH a, b
               SET b.v = a.v
               RETURN a.v, typeof(a.v), b.v, typeof(b.v)"""

        res = self.graph.query(q, {'s': LARGE_STRING})
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(res.properties_set, 2)
        self.env.assertEquals(LARGE_STRING, res.result_set[0][0])
        self.env.assertEquals("Intern String", res.result_set[0][1])
        self.env.assertEquals(LARGE_STRING, res.result_set[0][2])
        self.env.assertEquals("Intern String", res.result_set[0][3])
        assertStringPoolStats(self.conn, 1, 2)


class testInternStringPersistency():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

        # skip test if we're running under Sanitizer
        if VALGRIND or SANITIZER:
            self.env.skip() # sanitizer is not working correctly with bulk

        # Synchronous deletion
        self.db.config_set('ASYNC_DELETE', 'no')

        # clear DB
        self.conn.flushall()

    def tearDown(self):
        # clear DB
        self.conn.flushall()
        self.graph = self.db.select_graph(GRAPH_ID)

        assertStringPoolStats(self.conn, 0, 0)

    def testInternStringPersistent(self):
        # populate DB

        # create first node
        q = "CREATE ({value: intern($s)})"

        # create multiple EMPTY graphs
        graphs = []
        for _ in range(0, 10):
            g = self.db.select_graph(random_string())
            g.query(q, {'s': SMALL_STRING})
            graphs.append(g)

        # validate string pool stats
        assertStringPoolStats(self.conn, 1, 10)

        # Save RDB & Load from RDB
        self.env.dumpAndReload()

        # string-pool stats expected to match former stats before reload
        assertStringPoolStats(self.conn, 1, 10)

        for g in graphs:
            res = g.query("MATCH (n) RETURN n.value").result_set[0][0]
            self.env.assertEquals(res, SMALL_STRING)

class testInternStringReplication():
    def __init__(self):
        # skip test if we're running under Valgrind
        if VALGRIND or SANITIZER:
            Environment.skip(None) # valgrind is not working correctly with replication

        self.env, self.db = Env(env='oss', useSlaves=True)
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

        self.source_con = self.env.getConnection()
        self.replica_con = self.env.getSlaveConnection()

        # force effects replication
        self.db.config_set('EFFECTS_THRESHOLD', 0)

        # Synchronous deletion
        self.source_con.execute_command("GRAPH.CONFIG", "SET", 'ASYNC_DELETE', 'no')
        self.replica_con.execute_command("GRAPH.CONFIG", "SET", 'ASYNC_DELETE', 'no')

        # clear DB
        self.conn.flushall()
        self.source_con.execute_command("WAIT", "1", "0")

    def tearDown(self):
        # clear DB
        self.source_con.flushall()
        self.source_con.execute_command("WAIT", "1", "0")

        assertStringPoolStats(self.source_con, 0, 0)
        assertStringPoolStats(self.replica_con, 0, 0)

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

        s = LARGE_STRING

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

