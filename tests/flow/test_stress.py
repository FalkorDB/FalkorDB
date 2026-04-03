from common import Env, Graph
import time
import random
import threading
from queue import Queue, Empty

graph    = None
GRAPH_ID = "stress"  # graph identifier

def query_create(g, i):
    param = {'v': i}
    create_query = "CREATE (:Node {v:$v})<-[:HAVE]-(:Node {v:$v})-[:HAVE]->(:Node {v:$v})"
    g.query(create_query, param)

def query_read(g):
    read_query = "MATCH (n:Node)-[:HAVE]->(m:Node) RETURN n.v, m.v LIMIT 1"
    g.ro_query(read_query)

def query_update(g, i):
    param = {'v': i}
    update_query = "MATCH (n:Node) WITH n LIMIT 1 SET n.x = $v"
    g.query(update_query, param)

def query_delete(g):
    delete_query = "MATCH (n:Node)-[:HAVE*]->(m:Node) WITH n, m LIMIT 1 DELETE n, m"
    g.query(delete_query)

def create_nodes(g, i):
    params = {'v': i}
    g.query("CREATE (:Node {v: $v})-[:R]->()", params)

def delete_nodes(g):
    g.query("MATCH (n:Node) WITH n LIMIT 1 DELETE n")

def delete_edges(g):
    g.query("MATCH (:Node)-[r]->() WITH r LIMIT 1 DELETE r")

def update_nodes(g):
    g.query("MATCH (n:Node) WITH n LIMIT 1 SET n.v = 1")

def read_nodes(g):
    g.ro_query("MATCH (n:Node)-[:R]->() RETURN n LIMIT 1")

def merge_nodes_and_edges(g, i):
    params = {'a': i, 'b': i * 10}
    g.query("MERGE (a:Node {v: $a}) MERGE (b:Node {v: $b}) MERGE (a)-[:R]->(b)", params)

# measure how much time does it takes to perform BGSAVE
# asserts if BGSAVE took too long
# this function is run on a separate thread
def BGSAVE_loop(env, conn, stop_event):
    while not stop_event.is_set():
        conn.bgsave()
        results = conn.execute_command("INFO", "persistence")
        in_progress = results['rdb_bgsave_in_progress']
        max_iterations = 50

        # wait for BGSAVE to finish
        # for 6 seconds max
        for _ in range(max_iterations):
            results = conn.execute_command("INFO", "persistence")
            in_progress = results['rdb_bgsave_in_progress']
            if not in_progress:
                break
            time.sleep(0.1) # sleep 100ms

        env.assertFalse(in_progress)
        env.assertEqual(results['rdb_last_bgsave_status'], "ok")

    conn.close()

def worker(conn, task_queue):
    graph = Graph(conn, GRAPH_ID)

    while True:
        try:
            task = task_queue.get(timeout=1)
        except Empty:
            break

        task_func, args = task
        if args:
            task_func(graph, *args)
        else:
            task_func(graph)

        task_queue.task_done()

    conn.close()

class testStressFlow():
    def __init__(self):
        self.env, _ = Env()
        self.graph = Graph(self.env.getConnection(), GRAPH_ID)

    def setUp(self):
        self.graph.create_node_range_index("Node", "v")

    def tearDown(self):
        self.graph.delete()

    def start_workers(self, worker_count, task_queue):
        threads = []
        for _ in range(worker_count):
            thread = threading.Thread(target=worker, args=(self.env.getConnection(), task_queue))
            thread.start()
            threads.append(thread)
        return threads

    def join_workers(self, threads):
        for thread in threads:
            thread.join()

    def test00_stress(self):
        n_tasks     = 10000 # number of tasks to run
        n_creations = 0.3   # create ratio
        n_deletions = 0.7   # delete ratio
        n_reads     = 0.735 # read ratio
        task_queue  = Queue()

        # Queue up all tasks with correct format
        for i in range(n_tasks):
            r = random.random()
            if r < n_creations:
                task_queue.put((query_create, (i,)))
            elif r < n_deletions:
                task_queue.put((query_delete, ()))
            elif r < n_reads:
                task_queue.put((query_read, ()))
            else:
                task_queue.put((query_update, (i,)))
        
        # start and wait for all tasks to complete
        self.join_workers(self.start_workers(16, task_queue))
        task_queue.join()

    def test01_bgsave_stress(self):
        n_tasks     = 10000 # number of tasks to run
        n_creations = 0.35  # create ratio
        n_deletions = 0.7   # delete ratio
        n_reads     = 0.735 # read ratio
        task_queue  = Queue()

        # Create stop event for BGSAVE thread
        stop_event = threading.Event()

        # Start BGSAVE thread
        bgsave_thread = threading.Thread(
            target=BGSAVE_loop, 
            args=(self.env, self.env.getConnection(), stop_event)
        )

        bgsave_thread.start()

        for i in range(0, n_tasks):
            r = random.random()
            if r < n_creations:
                task_queue.put((create_nodes, (i,)))
            elif r < n_deletions:
                task_queue.put((delete_nodes, ()))
            elif r < n_reads:
                task_queue.put((read_nodes, ()))
            else:
                task_queue.put((update_nodes, ()))

        # start and wait for all tasks to complete
        self.join_workers(self.start_workers(16, task_queue))

        # Stop BGSAVE thread
        stop_event.set()
        bgsave_thread.join(timeout=10)
        self.env.assertFalse(bgsave_thread.is_alive())
        task_queue.join()

    def test02_write_only_workload(self):
        n_tasks           = 10000 # number of tasks to run
        n_creations       = 0.5
        n_node_deletions  = 0.75
        n_edge_deletions  = 1
        task_queue        = Queue()

        for i in range(0, n_tasks):
            r = random.random()
            if r < n_creations:
                task_queue.put((merge_nodes_and_edges, (i,)))
            elif r < n_node_deletions:
                task_queue.put((delete_nodes, ()))
            else:
                task_queue.put((delete_edges, ()))

        # start and wait for all tasks to complete
        self.join_workers(self.start_workers(16, task_queue))
        task_queue.join()

        # make sure we did not crash
        conn = self.env.getConnection()
        conn.ping()
        conn.close()
