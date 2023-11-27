import asyncio
from common import *
from pathos.pools import ProcessPool as Pool
from pathos.helpers import mp as pathos_multiprocess

GRAPH_ID = "G"                      # Graph identifier.
CLIENT_COUNT = 16                   # Number of concurrent connections.
people = ["Roi", "Alon", "Ailon", "Boaz", "Tal", "Omri", "Ori"]


def thread_run_query(query, barrier):
    env, db = Env()
    graph = db.select_graph(GRAPH_ID)

    if barrier is not None:
        barrier.wait()
    
    try:
        result = graph.query(query)
        return { "result_set": result.result_set, 
            "nodes_created": result.nodes_created, 
            "properties_set": result.properties_set }
    except ResponseError as e:
        return str(e)

def delete_graph(graph_id):
    env, db = Env()
    graph = db.select_graph(graph_id)

    # Try to delete graph.
    try:
        graph.delete()
        return True
    except:
        # Graph deletion failed.
        return False

def run_concurrent(queries, f):
    pool = Pool(nodes=CLIENT_COUNT)
    manager = pathos_multiprocess.Manager()
    
    barrier = manager.Barrier(CLIENT_COUNT)
    barriers = [barrier] * CLIENT_COUNT

    # invoke queries
    results = pool.map(f, queries, barriers)

    pool.clear()

    return results

class testConcurrentQueryFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        # skip test if we're running under Valgrind
        if VALGRIND:
            self.env.skip() # valgrind is not working correctly with multi processing

        self.conn  = redis.Redis("localhost", self.env.port)
        self.graph = self.db.select_graph(GRAPH_ID)

        self.populate_graph()

    def populate_graph(self):
        nodes = {}

        # Create entities
        for idx, p in enumerate(people):
            alias = f"n_{idx}"
            node = Node(alias=alias, labels="person", properties={"name": p})
            nodes[alias] = node
        nodes_str = [str(node) for node in nodes.values()]

        # Fully connected graph
        edges = []
        for src in nodes:
            for dest in nodes:
                if src != dest:
                    edges.append(Edge(nodes[src], "know", nodes[dest]))
        edges_str = [str(edge) for edge in edges]

        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")

    # Count number of nodes in the graph
    def test01_concurrent_aggregation(self):
        q = """MATCH (p:person) RETURN count(p)"""
        queries = [q] * CLIENT_COUNT
        results = run_concurrent(queries, thread_run_query)
        for result in results:
            person_count = result["result_set"][0][0]
            self.env.assertEqual(person_count, len(people))
    
    # Concurrently get neighbors of every node.
    def test02_retrieve_neighbors(self):
        q = """MATCH (p:person)-[know]->(n:person) RETURN n.name"""
        queries = [q] * CLIENT_COUNT
        results = run_concurrent(queries, thread_run_query)
        # Fully connected graph + header row.
        expected_resultset_size = len(people) * (len(people)-1)
        for result in results:
            self.env.assertEqual(len(result["result_set"]), expected_resultset_size)

    # Concurrent writes
    def test_03_concurrent_write(self):        
        queries = ["""CREATE (c:country {id:"%d"})""" % i for i in range(CLIENT_COUNT)]
        results = run_concurrent(queries, thread_run_query)
        for result in results:
            self.env.assertEqual(result["nodes_created"], 1)
            self.env.assertEqual(result["properties_set"], 1)
    
    # Try to delete graph multiple times.
    def test_04_concurrent_delete(self):
        pool = Pool(nodes=CLIENT_COUNT)

        # invoke queries
        assertions = pool.map(delete_graph, [GRAPH_ID] * CLIENT_COUNT)

        # Exactly one thread should have successfully deleted the graph.
        self.env.assertEquals(assertions.count(True), 1)

        pool.clear()

    # Try to delete a graph while multiple queries are executing.
    def test_05_concurrent_read_delete(self):
        ##############################################################################################
        # Delete graph via Redis DEL key.
        ##############################################################################################
        self.populate_graph()
        pool = Pool(nodes=CLIENT_COUNT)
        manager = pathos_multiprocess.Manager()
        barrier = manager.Barrier(CLIENT_COUNT)
        barriers = [barrier] * CLIENT_COUNT

        q = """UNWIND (range(0, 10000)) AS x WITH x AS x WHERE (x / 900) = 1 RETURN x"""
        queries = [q] * CLIENT_COUNT
        # invoke queries
        m = pool.amap(thread_run_query, queries, barriers)

        self.conn.delete(GRAPH_ID)

        # wait for processes to return
        m.wait()

        # get the results
        results = m.get()

        # validate result.
        self.env.assertTrue(all([r["result_set"][0][0] == 900 for r in results]))

        # Make sure Graph is empty, e.g. graph was deleted.
        resultset = self.graph.query("MATCH (n) RETURN count(n)").result_set
        self.env.assertEquals(resultset[0][0], 0)
        ##############################################################################################        
        # Delete graph via Redis FLUSHALL.
        ##############################################################################################
        self.populate_graph()
        q = """UNWIND (range(0, 10000)) AS x WITH x AS x WHERE (x / 900) = 1 RETURN x"""
        queries = [q] * CLIENT_COUNT
        barrier = manager.Barrier(CLIENT_COUNT)
        barriers = [barrier] * CLIENT_COUNT
        # invoke queries
        m = pool.amap(thread_run_query, queries, barriers)

        self.conn.flushall()

        # wait for processes to return
        m.wait()

        # get the results
        results = m.get()

        # validate result.
        self.env.assertTrue(all([r["result_set"][0][0] == 900 for r in results]))

        # Make sure Graph is empty, e.g. graph was deleted.
        resultset = self.graph.query("MATCH (n) RETURN count(n)").result_set
        self.env.assertEquals(resultset[0][0], 0)
        ##############################################################################################        
        # Delete graph via GRAPH.DELETE.
        ##############################################################################################
        self.populate_graph()
        q = """UNWIND (range(0, 10000)) AS x WITH x AS x WHERE (x / 900) = 1 RETURN x"""
        queries = [q] * CLIENT_COUNT
        barrier = manager.Barrier(CLIENT_COUNT)
        barriers = [barrier] * CLIENT_COUNT
        # invoke queries
        m = pool.amap(thread_run_query, queries, barriers)

        self.graph.delete()

        # wait for processes to return
        m.wait()

        # get the results
        results = m.get()

        # validate result.
        self.env.assertTrue(all([r["result_set"][0][0] == 900 for r in results]))

        # Make sure Graph is empty, e.g. graph was deleted.
        resultset = self.graph.query("MATCH (n) RETURN count(n)").result_set
        self.env.assertEquals(resultset[0][0], 0)

        pool.clear()

    def test_06_concurrent_write_delete(self):
        # Test setup - validate that graph exists and possible results are None
        self.graph.query("MATCH (n) RETURN n")

        pool = Pool(nodes=1)
        heavy_write_query = """UNWIND(range(0,999999)) as x CREATE(n) RETURN count(n)"""
        writer = pool.apipe(thread_run_query, heavy_write_query, None)
        self.conn.delete(GRAPH_ID)
        writer.wait()
        possible_exceptions = ["Encountered different graph value when opened key " + GRAPH_ID,
                               "Encountered an empty key when opened key " + GRAPH_ID]
        result = writer.get()
        if isinstance(result, str):
            self.env.assertContains(result, possible_exceptions)
        else:
            self.env.assertEquals(1000000, result["result_set"][0][0])

        pool.clear()
    
    def test_07_concurrent_write_rename(self):
        # Test setup - validate that graph exists and possible results are None
        self.graph.query("MATCH (n) RETURN n")

        pool = Pool(nodes=1)
        new_graph = GRAPH_ID + "2"
        # Create new empty graph with id GRAPH_ID + "2"
        self.conn.execute_command("GRAPH.QUERY",new_graph, """MATCH (n) return n""", "--compact")
        heavy_write_query = """UNWIND(range(0,999999)) as x CREATE(n) RETURN count(n)"""
        writer = pool.apipe(thread_run_query, heavy_write_query, None)
        self.conn.rename(GRAPH_ID, new_graph)
        writer.wait()
        # Possible scenarios:
        # 1. Rename is done before query is sent. The name in the graph context is new_graph, so when upon commit, when trying to open new_graph key, it will encounter an empty key since new_graph is not a valid key. 
        #    Note: As from https://github.com/RedisGraph/RedisGraph/pull/820 this may not be valid since the rename event handler might actually rename the graph key, before the query execution.    
        # 2. Rename is done during query executing, so when commiting and comparing stored graph context name (GRAPH_ID) to the retrived value graph context name (new_graph), the identifiers are not the same, since new_graph value is now stored at GRAPH_ID value.

        possible_exceptions = ["Encountered different graph value when opened key " + GRAPH_ID, 
        "Encountered an empty key when opened key " + new_graph]

        result = writer.get()
        if isinstance(result, str):
            self.env.assertContains(result, possible_exceptions)
        else:
            self.env.assertEquals(1000000, result["result_set"][0][0])

        pool.clear()
        
    def test_08_concurrent_write_replace(self):
        # Test setup - validate that graph exists and possible results are None
        self.graph.query("MATCH (n) RETURN n")

        pool = Pool(nodes=1)
        heavy_write_query = """UNWIND(range(0,999999)) as x CREATE(n) RETURN count(n)"""
        writer = pool.apipe(thread_run_query, heavy_write_query, None)
        set_result = self.conn.set(GRAPH_ID, "1")
        writer.wait()
        possible_exceptions = ["Encountered a non-graph value type when opened key " + GRAPH_ID,
                               "WRONGTYPE Operation against a key holding the wrong kind of value"]

        result = writer.get()
        if isinstance(result, str):
            # If the SET command attempted to execute while the CREATE query was running,
            # an exception should have been issued.
            self.env.assertContains(result, possible_exceptions)
        else:
            # Otherwise, both the CREATE query and the SET command should have succeeded.
            self.env.assertEquals(1000000, result.result_set[0][0])
            self.env.assertEquals(set_result, True)

        # Delete the key
        self.conn.delete(GRAPH_ID)

        pool.clear()

    def test_09_concurrent_multiple_readers_after_big_write(self):
        # Test issue #890
        self.graph = Graph(self.conn, GRAPH_ID)
        self.graph.query("""UNWIND(range(0,999)) as x CREATE()-[:R]->()""")
        read_query = """MATCH (n)-[r:R]->(m) RETURN count(r) AS res UNION RETURN 0 AS res"""

        queries = [read_query] * CLIENT_COUNT
        results = run_concurrent(queries, thread_run_query)

        for result in results:
            if isinstance(result, str):
                self.env.assertEquals(0, result)
            else:
                self.env.assertEquals(1000, result["result_set"][0][0])

    def test_10_write_starvation(self):
        # make sure write query do not starve
        # when issuing a large number of read queries
        # alongside a single write query
        # we dont want the write query to have to wait for
        # too long, consider the following sequence:
        # R, W, R, R, R, R, R, R, R...
        # if write is starved our write query might have to wait
        # for all queued read queries to complete while holding
        # Redis global lock, this will hurt performance
        #
        # this test issues a similar sequence of queries and
        # validates that the write query wasn't delayed too much

        self.graph = Graph(self.conn, GRAPH_ID)
        pool = Pool(nodes=CLIENT_COUNT)

        Rq = "UNWIND range(0, 10000) AS x WITH x WHERE x = 9999 RETURN 'R', timestamp()"
        Wq = "UNWIND range(0, 1000) AS x WITH x WHERE x = 27 CREATE ({v:1}) RETURN 'W', timestamp()"
        Slowq = "UNWIND range(0, 100000) AS x WITH x WHERE (x % 73) = 0 RETURN count(1)"

        # issue a number of slow queries, this will give us time to fill up
        # RedisGraph internal threadpool queue
        queries = [Slowq] * CLIENT_COUNT * 5
        nulls = [None] * CLIENT_COUNT * 5

        # issue queries asynchronously
        pool.imap(thread_run_query, queries, nulls)

        # create a long sequence of read queries
        queries = [Rq] * CLIENT_COUNT * 10
        nulls = [None] * CLIENT_COUNT * 10

        # inject a single write query close to the begining on the sequence
        queries[CLIENT_COUNT] = Wq

        # invoke queries
        # execute queries in parallel
        results = pool.map(thread_run_query, queries, nulls)

        # count how many queries completed before the write query
        count = 0
        write_ts = results[CLIENT_COUNT]["result_set"][0][1]
        for result in results:
            row = result["result_set"][0]
            ts = row[1]
            if ts < write_ts:
                count += 1

        # make sure write query wasn't starved
        self.env.assertLessEqual(count, len(queries) * 0.3)

        # delete the key
        self.conn.delete(GRAPH_ID)

        pool.clear()

    def test_11_concurrent_resize_zero_matrix(self):
        if "to_thread" not in dir(asyncio):
            # no need to check
            return

        self.graph = self.db.select_graph(GRAPH_ID)

        self.graph.query("CREATE (:N)")

        def resize_and_query():
            g = Graph(self.env.getConnection(), GRAPH_ID)

            for j in range(1, 10):
                g.query("UNWIND range(1, 10000) AS x CREATE (:M)")
                for i in range(1, 10):
                    g.query("MATCH (n:N)-[r:R]->() RETURN r")
        
        loop = asyncio.get_event_loop()
        tasks = []
        for i in range(1, 10):
            tasks.append(loop.create_task(asyncio.to_thread(resize_and_query)))

        loop.run_until_complete(asyncio.wait(tasks))

