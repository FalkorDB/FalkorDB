import time
from common import *
from time import sleep
from index_utils import *
from pathos.pools import ProcessPool as Pool

graphs   = None # one graph object per client
GRAPH_ID = "stress"  # graph identifier


def query_crud(graph, query_id):
    query_id = int(query_id)
    create_query = "CREATE (n:Node {v:%d}), (n)-[:HAVE]->(:Node {v:%d}), (n)-[:HAVE]->(:Node {v:%d})" % (query_id, query_id, query_id)
    read_query   = "MATCH  (n:Node {v:%d})-[:HAVE]->(n1:Node) RETURN n1.v" % query_id
    update_query = "MATCH  (n:Node {v:%d}) SET n.x = %d" % (query_id, query_id)
    delete_query = "MATCH  (n:Node {v:%d})-[:HAVE*]->(n1:Node) DELETE n, n1" % query_id

    for i in range(10):
        try:
            graph.query(create_query)
            graph.query(read_query)
            graph.query(update_query)
            graph.query(delete_query)
            return True
        except:
            return False

# run n_iterations and create n node in each iteration
def create_nodes(graph, n_iterations):
    for i in range(n_iterations):
        graph.query("CREATE (:Node {v: %d})-[:R]->()" % i)

# run n_iterations and delete n in each iteration
def delete_nodes(graph, n_iterations):
    for i in range(n_iterations):
        graph.query("MATCH (n:Node) WITH n LIMIT 1 DELETE n")

# run n_iterations and update all nodes in each iteration
def update_nodes(graph, n_iterations):
    for i in range(n_iterations):
        graph.query("MATCH (n:Node) WITH n LIMIT 1 SET n.v = 1")

# run n_iterations and execute a read query in each iteration
def read_nodes(graph, n_iterations):
    for i in range(n_iterations):
        graph.query("MATCH (n:Node)-[:R]->() RETURN n LIMIT 1")

# run n_iterations and merge 2 nodes and 1 edge in each iteration
def merge_nodes_and_edges(graph, n_iterations):
    for i in range(n_iterations):
        graph.query("MERGE (a:Node {v: %d}) MERGE (b:Node {v: %d}) MERGE (a)-[:R]->(b)" % (i, i * 10))

# run n_iterations and delete 1 edge in each iteration
def delete_edges(graph, n_iterations):
    for i in range(n_iterations):
        graph.query("MATCH (:Node)-[r]->() WITH r LIMIT 1 DELETE r")

# calls BGSAVE every 0.2 second
def BGSAVE_loop(env, conn, n_iterations):
    results = conn.execute_command("INFO", "persistence")
    cur_bgsave_time = prev_bgsave_time = results['rdb_last_save_time']

    for i in range(n_iterations):
        conn.execute_command("BGSAVE")
        start = time.time()

        while(cur_bgsave_time == prev_bgsave_time):
            # assert and return if the timeout of 5 seconds took place
            if(time.time() - start > 5):
                env.assertTrue(False)
                return

            results = conn.execute_command("INFO", "persistence")
            cur_bgsave_time = results['rdb_last_save_time']
            if cur_bgsave_time == prev_bgsave_time:
                sleep(1) # sleep for 1 second

        prev_bgsave_time = cur_bgsave_time
        env.assertEqual(results['rdb_last_bgsave_status'], "ok")

class testStressFlow():
    def __init__(self):
        # skip test if we're running under Valgrind
        if VALGRIND or SANITIZER != "" or CODE_COVERAGE:
            Env.skip(None) # valgrind is not working correctly with multi process

        self.env, self.db = Env()

        global graphs
        graphs = []

        self.client_count = self.env.getConnection().execute_command("GRAPH.CONFIG", "GET", "THREAD_COUNT")[1] * 5

        for i in range(0, self.client_count):
            graphs.append(Graph(self.env.getConnection(), GRAPH_ID))

    # called before each test function
    def setUp(self):
        # flush DB after each test
        self.env.flush()

    # Count number of nodes in the graph
    def test00_stress(self):
        ids = range(self.client_count)
        pool = Pool(nodes=self.client_count)

        # invoke queries
        pool.map(query_crud, graphs, ids)

        # make sure we did not crashed
        conn = self.env.getConnection()
        conn.ping()
        conn.close()

        pool.clear()

    def test01_bgsave_stress(self):
        n_reads      =  50000
        n_creations  =  50000
        n_updates    =  n_creations/10
        n_deletions  =  n_creations/2

        conn = self.env.getConnection()
        create_node_range_index(graphs[0], 'Node', 'v', sync=True)

        pool = Pool(nodes=5)

        t1 = pool.apipe(create_nodes, graphs[0], n_creations)

        t2 = pool.apipe(delete_nodes, graphs[1], n_deletions)

        t3 = pool.apipe(read_nodes, graphs[2], n_reads)

        t4 = pool.apipe(update_nodes, graphs[3], n_updates)

        t5 = pool.apipe(BGSAVE_loop, self.env, conn, 10000)

        # wait for processes to join
        t1.wait()
        t2.wait()
        t3.wait()
        t4.wait()
        t5.wait()

        # make sure we did not crashed
        conn.ping()
        conn.close()

        pool.clear()

    def test02_write_only_workload(self):
        pool              =  Pool(nodes=3)
        n_creations       =  20000
        n_node_deletions  =  10000
        n_edge_deletions  =  10000

        self.env.start()

        # invoke queries
        t1 = pool.apipe(merge_nodes_and_edges, graphs[0], n_creations)

        t2 = pool.apipe(delete_nodes, graphs[1], n_node_deletions)

        t3 = pool.apipe(delete_edges, graphs[2], n_edge_deletions)

        # wait for processes to join
        t1.wait()
        t2.wait()
        t3.wait()

        # make sure we did not crash
        conn = self.env.getConnection()
        conn.ping()
        conn.close()

        pool.clear()

    def test03_clean_shutdown(self):
        # issue SHUTDOWN while traffic is generated
        indexes = range(self.client_count)
        pool = Pool(nodes=self.client_count)

        # invoke queries
        m = pool.amap(query_crud, graphs, indexes)

        # sleep for half a second, allowing threads to kick in
        sleep(0.2)

        self.env.stop()

        # wait for processes to return
        m.wait()

        self.env.assertTrue(self.env.checkExitCode())

        pool.clear()

