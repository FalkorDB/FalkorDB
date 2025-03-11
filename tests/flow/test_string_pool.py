import random
import string
from common import *

GRAPH_ID = "STRING_POOL"

# test string deduplication
# incase DEDUPLICATE_STRINGS is set to true
# string values are unique across the entire DB
# this reduces our overall memory consumption
#
# the tests validates that shared strings are managed as expected
# and memory savings are visible

def random_string(length=10):
    chars = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    return ''.join(random.choices(chars, k=length))

class testStringPool():
    def __init__(self):
        self.env, self.db = Env(moduleArgs="DEDUPLICATE_STRINGS yes")
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def used_memory(self):
        # Get memory information
        memory_info = self.conn.info("memory")
        _used_memory = memory_info['used_memory']

        return _used_memory

    def test_01_single_graph_string_share(self):
        # Create a graph with multiple identical string values
        base_line = self.used_memory()

        s = 'A' * 16384 # large string

        # create first node
        q = "CREATE ({value: $s})"
        res = self.graph.query(q, {'s': s})

        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.properties_set, 1)

        # validate memory consumption increased
        memory_consumption = self.used_memory()
        self.env.assertGreater(memory_consumption, base_line)
        base_line = memory_consumption

        # create multiple nodes all sharing the same string value
        q = "UNWIND range(0, 50) AS x CREATE ({value: $s})"
        res = self.graph.query(q, {'s': s})
        
        # make sure memory consumption didn't increased by a meaningful amount
        memory_consumption = self.used_memory()
        # we expect very little increase since strings are deduplicated
        self.env.assertLess(memory_consumption, base_line * 1.2)

        self.graph.delete()

    def test_02_multi_graph_string_share(self):
        # Share string across multiple graphs
        self.graph = self.db.select_graph('A')

        s = 'A' * 16384 # large string

        # create first node
        q = "CREATE ({value: $s})"
        res = self.graph.query(q, {'s': s})

        # create multiple EMPTY graphs
        graphs = []
        for _ in range(0, 50):
            g = self.db.select_graph(random_string())
            graphs.append(g)
            res = g.query("RETURN 1")

        base_line = self.used_memory()

        # create multiple nodes across multiple graphs
        # all sharing the same string value
        for g in graphs:
            g.query(q, {'s': s})
        
        # make sure memory consumption didn't increased by a meaningful amount
        memory_consumption = self.used_memory()
        self.env.assertLess(memory_consumption, base_line * 4)

        # clear DB
        self.conn.flushall()

    def test_03_delete_shared_string(self):
        # Make sure shared string isn't released prematurely
        # Create a graph with multiple identical string values

        self.graph = self.db.select_graph('A')
        s = 'A' * 16384 # large string

        # create multiple nodes all sharing the same string value
        q = "UNWIND range(0, 20) AS x CREATE ({value: $s})"
        p = {'s': s}
        res = self.graph.query(q, p)

        # delete nodes one by one, make sure shared string isn't freed prematurely
        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        for _ in range(node_count):
            res = self.graph.query("MATCH (n) WITH n LIMIT 1 DELETE n")
            self.env.assertEquals(res.nodes_deleted, 1)

            # make sure string wan't freed
            rows = self.graph.query("MATCH (n) RETURN n.value").result_set
            for row in rows:
                self.env.assertEquals(row[0], s)

        # expecting graph to be empty
        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        self.env.assertEquals(node_count, 0)

        #-----------------------------------------------------------------------

        # Share string across multiple graphs
        # create first node
        q = "CREATE ({value: $s})"
        res = self.graph.query(q, p)

        # create multiple EMPTY graphs
        graphs = [self.graph]
        for _ in range(0, 50):
            g = self.db.select_graph(random_string())
            graphs.append(g)
            res = g.query(q, p)

        for g in graphs:
            v = g.query("MATCH (n) RETURN n.value").result_set[0][0]
            self.env.assertEquals(v, s)
            g.delete()
        
        # clear DB
        self.conn.flushall()

    def test_04_nested_shared_string(self):
        # Make sure shared string via containers isn't released prematurely

        self.graph = self.db.select_graph('A')
        s = 'A' * 16384 # large string

        # create a node with a string attribute
        p = {'s': s}
        self.graph.query("CREATE (:A {v:$s})", p)

        # create a node with an array containing the duplicated string
        self.graph.query("CREATE (:B {v:[$s]})", p)

        # delete first reference to shared string
        res = self.graph.query("MATCH (a:A) DELETE a")
        self.env.assertEquals(res.nodes_deleted, 1)

        # verify that string is still valid
        res = self.graph.query("MATCH (b:B) RETURN b.v[0]").result_set
        self.env.assertEquals(res[0][0], s)

        self.graph.delete()

        #-----------------------------------------------------------------------

        self.graph = self.db.select_graph('A')

        # same node containing the same string multiple times
        self.graph.query("CREATE (:A {a: $s, b:[$s], c:$s})", p)

        # delete first reference
        self.graph.query("MATCH (a:A) SET a.a = NULL")
        res = self.graph.query("MATCH (a:A) RETURN a.b[0], a.c").result_set
        self.env.assertEquals(res[0][0], s)
        self.env.assertEquals(res[0][1], s)

        # delete second reference
        self.graph.query("MATCH (a:A) SET a.b = NULL")
        res = self.graph.query("MATCH (a:A) RETURN a.c").result_set
        self.env.assertEquals(res[0][0], s)

        # delete last reference & delete node
        self.graph.query("MATCH (a:A) SET a.c = NULL")
        self.graph.query("MATCH (a:A) DELETE a")

