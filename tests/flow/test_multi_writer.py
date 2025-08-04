from common import *
import time
import threading
from queue import Queue, Empty

WORKER_COUNT = 16

task_queue = Queue()

def worker(thread_id, db):
    while True:
        try:
            graph_name, query = task_queue.get(timeout=5)
        except Empty:
            break

        try:
            g = db.select_graph(graph_name)
            g.query(query)
        except Exception as e:
            print(f"[Worker-{thread_id}] Error on graph {graph_name}: {e}")

        task_queue.task_done()

class testMultiWriter():

    def __init__(self):
        # Make sure DB utilizes multiple threads
        self.env, self.db = Env(moduleArgs="THREAD_COUNT 4")
        self.conn = self.env.getConnection()

    def tearDown(self):
        self.conn.flushall()

    def test_orphan_writes(self):
        # make sure no writes are left unattended
        GRAPHS = ['A', 'B', 'C', 'D', 'E', 'F']

        # generate queries
        query = "CREATE ()"

        # tasks queue
        #
        # 200 queries for each graph
        #
        # interleave queries:
        #
        # GRAPH.QUERY A "CREATE ()"
        # GRAPH.QUERY B "CREATE ()"
        # ...
        # GRAPH.QUERY F "CREATE ()"
        # GRAPH.QUERY A "CREATE ()"
        # ...

        for i in range(0, 200):
            for graph_name in GRAPHS:
                task_queue.put((graph_name, query))

        # start workers
        workers = []
        for i in range(WORKER_COUNT):
            t = threading.Thread(target=worker, args=(i, self.db))
            t.start()
            workers.append(t)

        # wait for workers to join
        for t in workers:
            t.join()

        task_queue.join()

        # validate each graph has the expected number of nodes
        for graph_name in GRAPHS:
            task_queue.put((graph_name, query))
            g = self.db.select_graph(graph_name)
            node_count = g.query("MATCH (n) RETURN count(n)").result_set[0][0]
            self.env.assertEquals(node_count, 200)

    def test_non_sequential(self):
        # Validate writes to different graph aren't held back
        # issue multiple slow writes against graph A
        # issue a single write against graph B
        # validate that B's query was executed before the last query to A

        slow_query = """UNWIND range(0, 20000) AS x
                        WITH x
                        WHERE x+1 = 1+x
                        CREATE ({v: timestamp()})"""

        fast_query = "CREATE ({v: timestamp()})"

        # enqueued slow queries first
        for i in range(0, 8):
            task_queue.put(("A", slow_query))

        # last query is a fast one against B
        task_queue.put(("B", fast_query))

        workers = []
        for i in range(9):
            t = threading.Thread(target=worker, args=(i, self.db))
            t.start()
            workers.append(t)

        # wait for workers to join
        for t in workers:
            t.join()

        task_queue.join()

        # validate B's timestamp is smaller than A's
        q = "MATCH (n) RETURN max(n.v)"

        g = self.db.select_graph('A')
        A_max_ts = g.query(q).result_set[0][0]

        g = self.db.select_graph('B')
        B_max_ts = g.query(q).result_set[0][0]

        self.env.assertGreater(A_max_ts, B_max_ts)

