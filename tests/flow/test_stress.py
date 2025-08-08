from common import Env, Graph
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
def BGSAVE_loop(env, conn, stop_event):
    while not stop_event.is_set():
        try:
            results = conn.execute_command("INFO", "persistence")
            cur_bgsave_time = prev_bgsave_time = results['rdb_last_save_time']

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
                    time.sleep(1) # sleep for 1 second

            prev_bgsave_time = cur_bgsave_time
            env.assertEqual(results['rdb_last_bgsave_status'], "ok")
            
            # Wait a bit before next BGSAVE
            time.sleep(2)
            
        except Exception as e:
            if not stop_event.is_set():
                print(f"BGSAVE error: {e}")
            break

class testStressFlow():
    def __init__(self):
        self.env, _ = Env()
        self.graph = Graph(self.env.getConnection(), GRAPH_ID)

    def setUp(self):
        self.graph.create_node_range_index("Node", "v")

    def tearDown(self):
        self.graph.delete()

    def worker_task(self, task_func, *args):
        thread_conn = self.env.getConnection()
        thread_graph = Graph(thread_conn, GRAPH_ID)
        try:
            return task_func(thread_graph, *args)
        finally:
            thread_conn.close()

    def test00_stress(self):
        n_tasks     = 10000 # number of tasks to run
        n_creations = 0.3   # create ratio
        n_deletions = 0.7   # delete ratio
        n_reads     = 0.735 # read ratio

        # Use ThreadPoolExecutor to manage threads
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            for i in range(0, n_tasks):
                r = random.random()
                if r < n_creations:
                    future = executor.submit(self.worker_task, query_create, i)
                elif r < n_deletions:
                    future = executor.submit(self.worker_task, query_delete)
                elif r < n_reads:
                    future = executor.submit(self.worker_task, query_read)
                else:
                    future = executor.submit(self.worker_task, query_update, i)
                
                futures.append(future)

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    print(f"Task failed: {e}")

    def test01_bgsave_stress(self):
        n_tasks     = 10000 # number of tasks to run
        n_creations = 0.35  # create ratio
        n_deletions = 0.7   # delete ratio
        n_reads     = 0.735 # read ratio

        # Create stop event for BGSAVE thread
        stop_event = threading.Event()
        
        # Start BGSAVE thread
        conn = self.env.getConnection()
        bgsave_thread = threading.Thread(
            target=BGSAVE_loop, 
            args=(self.env, conn, stop_event),
            daemon=True
        )
        bgsave_thread.start()

        # Use ThreadPoolExecutor to manage threads
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            for i in range(0, n_tasks):
                r = random.random()
                if r < n_creations:
                    future = executor.submit(self.worker_task, create_nodes, i)
                elif r < n_deletions:
                    future = executor.submit(self.worker_task, delete_nodes)
                elif r < n_reads:
                    future = executor.submit(self.worker_task, read_nodes)
                else:
                    future = executor.submit(self.worker_task, update_nodes)
                
                futures.append(future)

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task failed: {e}")

        # Stop BGSAVE thread
        stop_event.set()
        bgsave_thread.join(timeout=5)
        conn.close()

    def test02_write_only_workload(self):
        n_tasks           = 10000 # number of tasks to run
        n_creations       = 0.5
        n_node_deletions  = 0.75
        n_edge_deletions  = 1

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            for i in range(0, n_tasks):
                r = random.random()
                if r < n_creations:
                    future = executor.submit(self.worker_task, merge_nodes_and_edges, i)
                elif r < n_node_deletions:
                    future = executor.submit(self.worker_task, delete_nodes)
                else:
                    future = executor.submit(self.worker_task, delete_edges)
                
                futures.append(future)

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task failed: {e}")

        # make sure we did not crash
        conn = self.env.getConnection()
        conn.ping()
        conn.close()
