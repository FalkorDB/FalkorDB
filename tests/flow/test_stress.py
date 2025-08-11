from common import Env, Graph
import time
import random
import threading
from queue import Queue, Empty

graph    = None
GRAPH_ID = "stress"  # graph identifier
task_queue = Queue()

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
def BGSAVE_loop(env, conn, stop_event):
    while not stop_event.is_set():
        results = conn.execute_command("INFO", "persistence")

        conn.bgsave()
        results = conn.execute_command("INFO", "persistence")
        in_progress = results['rdb_bgsave_in_progress']

        # wait for BGSAVE to finish
        # for 5 seconds max
        for _ in range(50):
            results = conn.execute_command("INFO", "persistence")
            in_progress = results['rdb_bgsave_in_progress']
            if not in_progress:
                break
            time.sleep(0.1) # sleep 100ms

        env.assertFalse(in_progress)
        env.assertEqual(results['rdb_last_bgsave_status'], "ok")

    conn.close()

def worker(graph):
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

class testStressFlow():
    def __init__(self):
        self.env, _ = Env()
        self.graph = Graph(self.env.getConnection(), GRAPH_ID)
        self.thread_local = threading.local()

    def setUp(self):
        self.graph.create_node_range_index("Node", "v")

    def tearDown(self):
        self.graph.delete()

    def start_workers(self, worker_count):
        threads = []
        for _ in range(worker_count):
            thread = threading.Thread(target=worker, args=(self.graph,))
            thread.start()
            threads.append(thread)
        return threads

    def join_workers(self, threads):
        for thread in threads:
            thread.join()
        task_queue.join()

    def test00_stress(self):
        n_tasks     = 10000 # number of tasks to run
        n_creations = 0.3   # create ratio
        n_deletions = 0.7   # delete ratio
        n_reads     = 0.735 # read ratio
        
        workers = self.start_workers(16)

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
        
        # Wait for all tasks to complete
        self.join_workers(workers)

    def test01_bgsave_stress(self):
        n_tasks     = 10000 # number of tasks to run
        n_creations = 0.35  # create ratio
        n_deletions = 0.7   # delete ratio
        n_reads     = 0.735 # read ratio

        # Create stop event for BGSAVE thread
        stop_event = threading.Event()
        workers = self.start_workers(16)

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

        self.join_workers(workers)

        # Stop BGSAVE thread
        stop_event.set()
        bgsave_thread.join(timeout=10)
        self.env.assertFalse(bgsave_thread.is_alive())

    def test02_write_only_workload(self):
        n_tasks           = 10000 # number of tasks to run
        n_creations       = 0.5
        n_node_deletions  = 0.75
        n_edge_deletions  = 1

        for i in range(0, n_tasks):
            r = random.random()
            if r < n_creations:
                task_queue.put((merge_nodes_and_edges, (i)))
            elif r < n_node_deletions:
                task_queue.put((delete_nodes, ()))
            else:
                task_queue.put((delete_edges, ()))

        # make sure we did not crash
        conn = self.env.getConnection()
        conn.ping()
        conn.close()
