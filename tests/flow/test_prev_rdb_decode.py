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

    GRAPH_ID = "v9_rdb_restore"
    def _test_v9_decode(self):
        # docker run -p 6379:6379 -it redislabs/redisgraph:2.4.10
        # dump created with the following query (v9 supported property value: integer, double, boolean, string, null, array, point)
        #  graph.query g "CREATE (:L1 {val:1, strval: 'str', numval: 5.5, nullval: NULL, boolval: true, array: [1,2,3], point: POINT({latitude: 32, longitude: 34})})-[:E{val:2}]->(:L2{val:3})"
        #  graph.query g "CREATE INDEX ON :L1(val)"
        #  graph.query g "CREATE INDEX ON :L1(none_existsing)"
        #  graph.query g "CREATE (:L3)-[:E2]->(:L4)"
        #  graph.query g "MATCH (n1:L3)-[r:E2]->(n2:L4) DELETE n1, r, n2"
        #  dump g
        v9_rdb = b"\a\x81\x82\xb6\xa9\x85\xd6\xadh\t\x05\x02g\x00\x02\x02\x02\x01\x02\x04\x02\x02\x02\x00\x02\x00\x02\x01\x02\x05\x02\x01\x02\x02\x02\x02\x02\x02\x02\x03\x02\x01\x02\x04\x02\x01\x02\x05\x02\x01\x02\x00\x02\x01\x02\x00\x02\x06\x02\x00\x02`\x00\x02\x01\x02\x01\x02H\x00\x05\x04str\x00\x02\x02\x02\x80\x00\x00@\x00\x04\x00\x00\x00\x00\x00\x00\x16@\x02\x04\x02P\x00\x02\x01\x02\x05\x02\b\x02\x03\x02`\x00\x02\x01\x02`\x00\x02\x02\x02`\x00\x02\x03\x02\x06\x02\x80\x00\x02\x00\x00\x04\x00\x00\x00\x00\x00\x00@@\x04\x00\x00\x00\x00\x00\x00A@\x02\x01\x02\x01\x02\x01\x02\x01\x02\x00\x02`\x00\x02\x03\x02\x02\x02\x03\x02\x00\x02\x00\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x02\x02\x01\x02\b\x05\x04val\x00\x05\astrval\x00\x05\anumval\x00\x05\bnullval\x00\x05\bboolval\x00\x05\x06array\x00\x05\x06point\x00\x05\x0fnone_existsing\x00\x02\x04\x02\x00\x05\x03L1\x00\x02\x02\x02\x01\x05\x04val\x00\x02\x01\x05\x0fnone_existsing\x00\x02\x01\x05\x03L2\x00\x02\x00\x02\x02\x05\x03L3\x00\x02\x00\x02\x03\x05\x03L4\x00\x02\x00\x02\x02\x02\x00\x05\x02E\x00\x02\x00\x02\x01\x05\x03E2\x00\x02\x00\x00\t\x00\xd7\xd0\x1cB;\xce\x1d>"
        self.redis_con.restore(GRAPH_ID, 0, v9_rdb, True)

        graph  = self.db.select_graph(GRAPH_ID)
        node0  = Node(node_id=0, labels='L1', properties={'val': 1, 'strval': 'str', 'numval': 5.5, 'boolval': True, 'array': [1,2,3], 'point': {'latitude': 32, 'longitude': 34}})
        node1  = Node(node_id=1, labels='L2', properties={'val': 3})
        edge01 = Edge(src_node=0, relation='E', dest_node=1, edge_id=0, properties={'val':2})

        results = graph.query("MATCH (n)-[e]->(m) RETURN n, e, m")
        self.env.assertEqual(results.result_set, [[node0, edge01, node1]])

        plan = str(graph.explain("MATCH (n:L1 {val:1}) RETURN n"))
        self.env.assertIn("Index Scan", plan)

        results = graph.query("MATCH (n:L1 {val:1}) RETURN n")
        self.env.assertEqual(results.result_set, [[node0]])

