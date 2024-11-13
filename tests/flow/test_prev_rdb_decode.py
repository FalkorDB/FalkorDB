import os
import time
import random
from common import *

# decoders versions to tests
VERSIONS = [
        {'decoder_version': 10, 'tag': 'redislabs/redisgraph:2.8.7'},
        {'decoder_version': 11, 'tag': 'redislabs/redisgraph:2.8.12'},
        {'decoder_version': 12, 'tag': 'redislabs/redisgraph:2.8.14'},
        {'decoder_version': 13, 'tag': 'redislabs/redisgraph:2.12.8'},
        {'decoder_version': 14, 'tag': 'falkordb/falkordb:v4.0.7'},
        {'decoder_version': 15, 'tag': 'falkordb/falkordb:v4.2.2'}]

QUERIES = [
        "CREATE (:L1 {val:1, strval: 'str', numval: 5.5, nullval: NULL, boolval: true, array: [1,2,3], point: POINT({latitude: 32, longitude: 34})})-[:E{val:2}]->(:L2{val:3})",
        "CREATE INDEX ON :L1(val)",
        "CREATE INDEX ON :L1(none_existsing)",
        "CREATE (:L3)-[:E2]->(:L4)",
        "MATCH (n1:L3)-[r:E2]->(n2:L4) DELETE n1, r, n2"]

def graph_id(v):
    return f"v{v}_rdb_restore"

# starts db using docker
def run_db(image):
    import docker

    # Initialize the Docker client
    client = docker.from_env()

    random_port = random.randint(49152, 65535)

    # Run the RedisGraph container
    container = client.containers.run(
        image,                            # Image
        detach=True,                      # Run container in the background
        ports={'6379/tcp': random_port},  # Map port 6379
    )

    return container, random_port

# stop and remove docker container
def stop_db(container):
    container.stop()
    container.remove()

# generate a graph dump
def generate_dump(key, port):
    from falkordb import FalkorDB

    # Connect to FalkorDB
    db = FalkorDB(port=port)

    # Select the social graph
    g = db.select_graph(key)

    # Populate graph
    for q in QUERIES:
        g.query(q)

    # Dump key
    return db.connection.dump(key)

# get graph dump from a specified FalkorDB version
# check if dump already exists locally, if not generates and saves dump
# to "./dumps/{v}.dump"
def get_dump(v):
    path = f"./dumps/{v}.dump"

    # get dump
    if not os.path.exists(path):
        # get decoder docker image tag
        tag = [item['tag'] for item in VERSIONS if item ['decoder_version'] == v][0]

        # start Docker container
        container, port = run_db(tag)

        # wait for DB to accept connections
        time.sleep(2)

        # generate dump
        dump = generate_dump(graph_id(v), port)
        print(f"dump: {dump}")

        # ensure the directory exists, create if missing
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # save dump to file
        with open(path, 'wb') as f:
            f.write(dump)
            f.flush()

        # stop db
        stop_db(container)

        return dump
    else:
        with open(path, 'rb') as f:
            return f.read()

class test_prev_rdb_decode(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()

    def test_v10_decode(self):
        decoder_id = 10
        key = graph_id(decoder_id)
        rdb = get_dump(decoder_id)

        # restore dump
        self.redis_con.restore(key, 0, rdb, True)

        # select graph
        graph = self.db.select_graph(key)

        # expected entities
        node0  = Node(node_id=0, labels='L1', properties={'val': 1, 'strval': 'str', 'numval': 5.5, 'boolval': True, 'array': [1,2,3], 'point': {'latitude': 32, 'longitude': 34}})
        node1  = Node(node_id=1, labels='L2', properties={'val': 3})
        edge01 = Edge(src_node=0, relation='E', dest_node=1, edge_id=0, properties={'val':2})

        # validations
        results = graph.query("MATCH (n)-[e]->(m) RETURN n, e, m")
        self.env.assertEqual(results.result_set, [[node0, edge01, node1]])

        plan = str(graph.explain("MATCH (n:L1 {val:1}) RETURN n"))
        self.env.assertIn("Index Scan", plan)

        results = graph.query("MATCH (n:L1 {val:1}) RETURN n")
        self.env.assertEqual(results.result_set, [[node0]])

    def test_v11_decode(self):
        decoder_id = 11
        key = graph_id(decoder_id)
        rdb = get_dump(decoder_id)

        # restore dump
        self.redis_con.restore(key, 0, rdb, True)

        # select graph
        graph = self.db.select_graph(key)

        # expected entities
        node0  = Node(node_id=0, labels='L1', properties={'val': 1, 'strval': 'str', 'numval': 5.5, 'boolval': True, 'array': [1,2,3], 'point': {'latitude': 32, 'longitude': 34}})
        node1  = Node(node_id=1, labels='L2', properties={'val': 3})
        edge01 = Edge(src_node=0, relation='E', dest_node=1, edge_id=0, properties={'val':2})

        # validations
        results = graph.query("MATCH (n)-[e]->(m) RETURN n, e, m")
        self.env.assertEqual(results.result_set, [[node0, edge01, node1]])

        plan = str(graph.explain("MATCH (n:L1 {val:1}) RETURN n"))
        self.env.assertIn("Index Scan", plan)

        results = graph.query("MATCH (n:L1 {val:1}) RETURN n")
        self.env.assertEqual(results.result_set, [[node0]])

    def test_v12_decode(self):
        decoder_id = 12
        key = graph_id(decoder_id)
        rdb = get_dump(decoder_id)

        # restore dump
        self.redis_con.restore(key, 0, rdb, True)

        # select graph
        graph = self.db.select_graph(key)

        # expected entities
        node0  = Node(node_id=0, labels='L1', properties={'val': 1, 'strval': 'str', 'numval': 5.5, 'boolval': True, 'array': [1,2,3], 'point': {'latitude': 32, 'longitude': 34}})
        node1  = Node(node_id=1, labels='L2', properties={'val': 3})
        edge01 = Edge(src_node=0, relation='E', dest_node=1, edge_id=0, properties={'val':2})

        # validations
        results = graph.query("MATCH (n)-[e]->(m) RETURN n, e, m")
        self.env.assertEqual(results.result_set, [[node0, edge01, node1]])

        plan = str(graph.explain("MATCH (n:L1 {val:1}) RETURN n"))
        self.env.assertIn("Index Scan", plan)

        results = graph.query("MATCH (n:L1 {val:1}) RETURN n")
        self.env.assertEqual(results.result_set, [[node0]])

    def test_v13_decode(self):
        decoder_id = 13
        key = graph_id(decoder_id)
        rdb = get_dump(decoder_id)

        # restore dump
        self.redis_con.restore(key, 0, rdb, True)

        # select graph
        graph = self.db.select_graph(key)

        # expected entities
        node0  = Node(node_id=0, labels='L1', properties={'val': 1, 'strval': 'str', 'numval': 5.5, 'boolval': True, 'array': [1,2,3], 'point': {'latitude': 32, 'longitude': 34}})
        node1  = Node(node_id=1, labels='L2', properties={'val': 3})
        edge01 = Edge(src_node=0, relation='E', dest_node=1, edge_id=0, properties={'val':2})

        # validations
        results = graph.query("MATCH (n)-[e]->(m) RETURN n, e, m")
        self.env.assertEqual(results.result_set, [[node0, edge01, node1]])

        plan = str(graph.explain("MATCH (n:L1 {val:1}) RETURN n"))
        self.env.assertIn("Index Scan", plan)

        results = graph.query("MATCH (n:L1 {val:1}) RETURN n")
        self.env.assertEqual(results.result_set, [[node0]])

    def test_v14_decode(self):
        decoder_id = 14
        key = graph_id(decoder_id)
        rdb = get_dump(decoder_id)

        # restore dump
        self.redis_con.restore(key, 0, rdb, True)

        # select graph
        graph = self.db.select_graph(key)

        # expected entities
        node0  = Node(node_id=0, labels='L1', properties={'val': 1, 'strval': 'str', 'numval': 5.5, 'boolval': True, 'array': [1,2,3], 'point': {'latitude': 32, 'longitude': 34}})
        node1  = Node(node_id=1, labels='L2', properties={'val': 3})
        edge01 = Edge(src_node=0, relation='E', dest_node=1, edge_id=0, properties={'val':2})

        # validations
        results = graph.query("MATCH (n)-[e]->(m) RETURN n, e, m")
        self.env.assertEqual(results.result_set, [[node0, edge01, node1]])

        plan = str(graph.explain("MATCH (n:L1 {val:1}) RETURN n"))
        self.env.assertIn("Index Scan", plan)

        results = graph.query("MATCH (n:L1 {val:1}) RETURN n")
        self.env.assertEqual(results.result_set, [[node0]])

    def test_v15_decode(self):
        decoder_id = 15
        key = graph_id(decoder_id)
        rdb = get_dump(decoder_id)

        # restore dump
        self.redis_con.restore(key, 0, rdb, True)

        # select graph
        graph = self.db.select_graph(key)

        # expected entities
        node0  = Node(node_id=0, labels='L1', properties={'val': 1, 'strval': 'str', 'numval': 5.5, 'boolval': True, 'array': [1,2,3], 'point': {'latitude': 32, 'longitude': 34}})
        node1  = Node(node_id=1, labels='L2', properties={'val': 3})
        edge01 = Edge(src_node=0, relation='E', dest_node=1, edge_id=0, properties={'val':2})

        # validations
        results = graph.query("MATCH (n)-[e]->(m) RETURN n, e, m")
        self.env.assertEqual(results.result_set, [[node0, edge01, node1]])

        plan = str(graph.explain("MATCH (n:L1 {val:1}) RETURN n"))
        self.env.assertIn("Index Scan", plan)

        results = graph.query("MATCH (n:L1 {val:1}) RETURN n")
        self.env.assertEqual(results.result_set, [[node0]])

