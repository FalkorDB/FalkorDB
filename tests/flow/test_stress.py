from common import Env, Graph
from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool

import time
import random
import asyncio

graph    = None
GRAPH_ID = "stress"  # graph identifier


async def query_create(g, i):
    param = {'v': i}
    create_query = "CREATE (:Node {v:$v})<-[:HAVE]-(:Node {v:$v})-[:HAVE]->(:Node {v:$v})"
    await g.query(create_query, param)

async def query_read(g):
    read_query = "MATCH (n:Node)-[:HAVE]->(m:Node) RETURN n.v, m.v LIMIT 1"
    await g.ro_query(read_query)

async def query_update(g, i):
    param = {'v': i}
    update_query = "MATCH (n:Node) WITH n LIMIT 1 SET n.x = $v"
    await g.query(update_query, param)

async def query_delete(g):
    delete_query = "MATCH (n:Node)-[:HAVE*]->(m:Node) WITH n, m LIMIT 1 DELETE n, m"
    await g.query(delete_query)

async def create_nodes(g, i):
    params = {'v': i}
    await g.query("CREATE (:Node {v: $v})-[:R]->()", params)

async def delete_nodes(g):
    await g.query("MATCH (n:Node) WITH n LIMIT 1 DELETE n")

async def delete_edges(g):
    await g.query("MATCH (:Node)-[r]->() WITH r LIMIT 1 DELETE r")

async def update_nodes(g):
    await g.query("MATCH (n:Node) WITH n LIMIT 1 SET n.v = 1")

async def read_nodes(g):
    await g.ro_query("MATCH (n:Node)-[:R]->() RETURN n LIMIT 1")

async def merge_nodes_and_edges(g, i):
    params = {'a': i, 'b': i * 10}
    await g.query("MERGE (a:Node {v: $a}) MERGE (b:Node {v: $b}) MERGE (a)-[:R]->(b)", params)

# measure how much time does it takes to perform BGSAVE
# asserts if BGSAVE took too long
async def BGSAVE_loop(env, conn):
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
            await asyncio.sleep(1) # sleep for 1 second

    prev_bgsave_time = cur_bgsave_time
    env.assertEqual(results['rdb_last_bgsave_status'], "ok")

class testStressFlow():
    def __init__(self):
        self.env, _ = Env()
        self.graph = Graph(self.env.getConnection(), GRAPH_ID)

    # called before each test function
    def setUp(self):
        # create index
        self.graph.create_node_range_index("Node", "v")

    def tearDown(self):
        self.graph.delete()

    # Count number of nodes in the graph
    def test00_stress(self):
        async def run(self):
            # connection pool with 16 connections
            # blocking when there's no connections available
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            n_tasks     = 10000 # number of tasks to run
            n_creations = 0.3   # create ratio
            n_deletions = 0.7   # delete ratio
            n_reads     = 0.735 # read ratio

            tasks = []
            for i in range(0, n_tasks):
                r = random.random()
                if r < n_creations:
                    tasks.append(asyncio.create_task(query_create(g, i)))
                elif r < n_deletions:
                    tasks.append(asyncio.create_task(query_delete(g)))
                elif r < n_reads:
                    tasks.append(asyncio.create_task(query_read(g)))
                else:
                    tasks.append(asyncio.create_task(query_update(g, i)))

            # wait for all tasks to complete
            await asyncio.gather(*tasks)

            # close the connection pool
            await pool.aclose()

        asyncio.run(run(self))

    def test01_bgsave_stress(self):
        async def run(self):
            # connection pool with 16 connections
            # blocking when there's no connections available
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            n_tasks     = 10000 # number of tasks to run
            n_creations = 0.35  # create ratio
            n_deletions = 0.7   # delete ratio
            n_reads     = 0.735 # read ratio

            # async tasks
            conn = self.env.getConnection()
            bgsave_task = asyncio.create_task(BGSAVE_loop(self.env, conn))

            tasks = []
            for i in range(0, n_tasks):
                r = random.random()
                if r < n_creations:
                    tasks.append(asyncio.create_task(create_nodes(g, i)))
                elif r < n_deletions:
                    tasks.append(asyncio.create_task(delete_nodes(g)))
                elif r < n_reads:
                    tasks.append(asyncio.create_task(read_nodes(g)))
                else:
                    tasks.append(asyncio.create_task(update_nodes(g)))

            # wait for all tasks to complete
            await asyncio.gather(*tasks)

            # cancel BGSAVE task
            bgsave_task.cancel()

            # close the connection pool
            await pool.aclose()

        asyncio.run(run(self))

    def test02_write_only_workload(self):
        async def run(self):
            # connection pool with 16 connections
            # blocking when there's no connections available
            pool = BlockingConnectionPool(max_connections=16, timeout=None, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            n_tasks           = 10000 # number of tasks to run
            n_creations       = 0.5
            n_node_deletions  = 0.75
            n_edge_deletions  = 1

            tasks = []
            for i in range(0, n_tasks):
                r = random.random()
                if r < n_creations:
                    tasks.append(asyncio.create_task(merge_nodes_and_edges(g, i)))
                elif r < n_node_deletions:
                    tasks.append(asyncio.create_task(delete_nodes(g)))
                else:
                    tasks.append(asyncio.create_task(delete_edges(g)))

            # wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # make sure we did not crashed
            conn = self.env.getConnection()
            conn.ping()
            conn.close()

            # close the connection pool
            await pool.aclose()

        asyncio.run(run(self))
