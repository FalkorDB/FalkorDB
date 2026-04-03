import os
import time
from common import *
from falkordb import FalkorDB

# decoders versions to tests
VERSIONS = [
        {'decoder_version': 10, 'tag': 'redislabs/redisgraph:2.8.7'},
        {'decoder_version': 11, 'tag': 'redislabs/redisgraph:2.8.12'},
        {'decoder_version': 12, 'tag': 'redislabs/redisgraph:2.8.14'},
        {'decoder_version': 13, 'tag': 'redislabs/redisgraph:2.12.8'},
        {'decoder_version': 14, 'tag': 'falkordb/falkordb:v4.0.7'},
        {'decoder_version': 15, 'tag': 'falkordb/falkordb:v4.2.2'},
        {'decoder_version': 16, 'tag': 'falkordb/falkordb:v4.8.5'},
        {'decoder_version': 17, 'tag': 'falkordb/falkordb:v4.10.3'}
        ]

QUERIES = [
        "CREATE (:L1 {val:1, strval: 'str', numval: 5.5, nullval: NULL, boolval: true, array: [1,2,3], point: POINT({latitude: 32, longitude: 34})})-[:E{val:2}]->(:L2{val:3})",
        "CREATE INDEX ON :L1(val)",
        "CREATE INDEX ON :L1(none_existsing)",
        "CREATE (:L3)-[:E2]->(:L4)",
        "MATCH (n1:L3)-[r:E2]->(n2:L4) DELETE n1, r, n2"]

def graph_id(v):
    return f"v{v}_rdb_restore"

def get_image_tag(v):
    return [item['tag'] for item in VERSIONS if item ['decoder_version'] == v][0]

# starts db using docker
def run_db(image):
    import docker
    from random import randint

    # Initialize the Docker client
    client = docker.from_env()

    random_port = randint(49152, 65535)

    # Run the FalkorDB container
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
    # Connect to FalkorDB
    db = FalkorDB(port=port)

    # Select the social graph
    g = db.select_graph(key)
    try:
        g.delete()
    except:
        pass

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
        tag = get_image_tag(v)

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

    with open(path, 'rb') as f:
        return f.read()

class test_prev_rdb_decode():
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()

    def _test_decode(self, decoder_id):
        key = graph_id(decoder_id)
        dump = get_dump(decoder_id)

        # restore dump
        self.redis_con.restore(key, 0, dump, True)

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

    def test_v10_decode(self):
        decoder_id = 10
        self._test_decode(decoder_id)

    def test_v11_decode(self):
        decoder_id = 11
        self._test_decode(decoder_id)

    def test_v12_decode(self):
        decoder_id = 12
        self._test_decode(decoder_id)

    def test_v13_decode(self):
        decoder_id = 13
        self._test_decode(decoder_id)

    def test_v14_decode(self):
        decoder_id = 14
        self._test_decode(decoder_id)

    def test_v15_decode(self):
        decoder_id = 15
        self._test_decode(decoder_id)

    def test_v16_decode(self):
        # under sanitizer we're seeing:
        # Unhandled exception: DUMP payload version or checksum are wrong
        if SANITIZER:
            self.env.skip()
            return

        decoder_id = 16
        self._test_decode(decoder_id)

    def test_v17_decode(self):
        # under sanitizer we're seeing:
        # Unhandled exception: DUMP payload version or checksum are wrong
        if SANITIZER:
            self.env.skip()
            return

        decoder_id = 17
        self._test_decode(decoder_id)

