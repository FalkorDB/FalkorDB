from collections import OrderedDict
from common import *
import time
import psutil
import random
import threading
from index_utils import *
from click.testing import CliRunner
from falkordb_bulk_loader.bulk_insert import bulk_insert

GRAPH_ID = "persistency"

def get_total_rss_memory(pid):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    all_processes = [parent] + children

    total_rss = sum(p.memory_info().rss for p in all_processes)
    return total_rss  # in bytes

class testGraphPersistency():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

        # skip test if we're running under Sanitizer
        if SANITIZER:
            self.env.skip() # sanitizer is not working correctly with bulk

    def tearDown(self):
        self.conn.flushall()

    def populate_graph(self, graph_name):
        graph = self.db.select_graph(graph_name)
        # quick return if graph already exists
        if graph_name in self.db.list_graphs():
            return graph

        people       = ["Roi", "Alon", "Ailon", "Boaz", "Tal", "Omri", "Ori"]
        visits       = [("Roi", "USA"), ("Alon", "Israel"), ("Ailon", "Japan"), ("Boaz", "UK")]
        countries    = ["Israel", "USA", "Japan", "UK"]
        personNodes  = {}
        countryNodes = {}

        # create nodes
        for p in people:
            person = Node(alias=p, labels="person", properties={"name": p, "height": random.randint(160, 200)})
            personNodes[p] = person

        for c in countries:
            country = Node(alias=c, labels="country", properties={"name": c, "population": random.randint(100, 400)})
            countryNodes[c] = country

        # create edges
        edges = []
        for v in visits:
            person  = v[0]
            country = v[1]
            edges.append(Edge(personNodes[person], 'visit', countryNodes[country], properties={
                        'purpose': 'pleasure'}))

        edges_str = [str(e) for e in edges]
        nodes_str = [str(n) for n in personNodes.values()] + [str(n) for n in countryNodes.values()]
        graph.query(f"CREATE {','.join(nodes_str + edges_str)}")

        # delete nodes, to introduce deleted entries within our datablock
        query = """MATCH (n:person) WHERE n.name = 'Roi' or n.name = 'Ailon' DELETE n"""
        graph.query(query)

        query = """MATCH (n:country) WHERE n.name = 'USA' DELETE n"""
        graph.query(query)

        # create indices
        graph.create_node_range_index("person", "name", "height")
        graph.create_node_range_index("country", "name", "population")
        graph.create_edge_range_index("visit", "purpose")
        graph.query("CALL db.idx.fulltext.createNodeIndex({label: 'person', stopwords: ['A', 'B'], language: 'english'}, { field: 'text', nostem: true, weight: 2, phonetic: 'dm:en' })")
        create_node_vector_index(graph, "person", 'embedding1', dim=128, m=64, efConstruction=10, efRuntime=10)
        create_node_vector_index(graph, "person", 'embedding2', dim=256, similarity_function='cosine', m=32, efConstruction=20, efRuntime=20)
        wait_for_indices_to_sync(graph)

        return graph

    def populate_dense_graph(self, graph_name):
        dense_graph = self.db.select_graph(graph_name)

        # return early if graph exists
        if graph_name in self.db.list_graphs():
            return dense_graph

        nodes = []
        for i in range(10):
            node = Node(alias=f"n_{i}", labels="n", properties={"val": i})
            nodes.append(node)

        edges = []
        for n_idx, n in enumerate(nodes):
            for m_idx, m in enumerate(nodes[:n_idx]):
                edges.append(Edge(n, "connected", m))

        nodes_str = [str(n) for n in nodes]
        edges_str = [str(e) for e in edges]
        dense_graph.query(f"CREATE {','.join(nodes_str + edges_str)}")

        return dense_graph

    def test_save_load(self):
        graph_names = ["G", "{tag}_G"]
        for graph_name in graph_names:
            graph = self.populate_graph(graph_name)
            for i in range(2):
                if i == 1:
                    # Save RDB & Load from RDB
                    self.env.dumpAndReload()

                # Verify
                # Expecting 5 person entities.
                query = """MATCH (p:person) RETURN COUNT(p)"""
                actual_result = graph.query(query)
                nodeCount = actual_result.result_set[0][0]
                self.env.assertEquals(nodeCount, 5)

                query = """MATCH (p:person) WHERE p.name='Alon' RETURN COUNT(p)"""
                actual_result = graph.query(query)
                nodeCount = actual_result.result_set[0][0]
                self.env.assertEquals(nodeCount, 1)

                # Expecting 3 country entities.
                query = """MATCH (c:country) RETURN COUNT(c)"""
                actual_result = graph.query(query)
                nodeCount = actual_result.result_set[0][0]
                self.env.assertEquals(nodeCount, 3)

                query = """MATCH (c:country) WHERE c.name = 'Israel' RETURN COUNT(c)"""
                actual_result = graph.query(query)
                nodeCount = actual_result.result_set[0][0]
                self.env.assertEquals(nodeCount, 1)

                # Expecting 2 visit edges.
                query = """MATCH (n:person)-[e:visit]->(c:country) WHERE e.purpose='pleasure' RETURN COUNT(e)"""
                actual_result = graph.query(query)
                edgeCount = actual_result.result_set[0][0]
                self.env.assertEquals(edgeCount, 2)

                # Verify indices exists
                indices = graph.query("""CALL db.indexes()""").result_set
                expected_indices = {
                        'country': [['name', 'population'], 'english', [], 'NODE'],
                        'person': [['name', 'height', 'text', 'embedding1', 'embedding2'], OrderedDict({'name': ['RANGE'], 'height': ['RANGE'], 'text': ['FULLTEXT'], 'embedding1': ['VECTOR'], 'embedding2': ['VECTOR']}), OrderedDict({'name': OrderedDict({}), 'height': OrderedDict({}), 'text': OrderedDict({}), 'embedding1': OrderedDict({'dimension': 128, 'similarityFunction': 'euclidean', 'M': 64, 'efConstruction': 10, 'efRuntime': 10}), 'embedding2': OrderedDict({'dimension': 256, 'similarityFunction': 'cosine', 'M': 32, 'efConstruction': 20, 'efRuntime': 20})}), 'english', ['a', 'b'], 'NODE'],
                        'visit': [['purpose'], 'english', [], 'RELATIONSHIP']
                }

                self.env.assertEquals(len(indices), len(expected_indices))
                for index in indices:
                    for expected_index in expected_indices[index[0]]:
                        self.env.assertIn(expected_index, index)

    # Verify that edges are not modified after entity deletion
    def test_deleted_entity_migration(self):
        graph_names = ("H", "{tag}_H")
        for graph_name in graph_names:
            graph = self.populate_dense_graph(graph_name)

            query = """MATCH (p) WHERE ID(p) = 0 OR ID(p) = 3 OR ID(p) = 7 OR ID(p) = 9 DELETE p"""
            actual_result = graph.query(query)
            self.env.assertEquals(actual_result.nodes_deleted, 4)

            query = """MATCH (p)-[]->(q) RETURN p.val, q.val ORDER BY p.val, q.val"""
            first_result = graph.query(query)

            # Save RDB & Load from RDB
            self.env.dumpAndReload()

            second_result = graph.query(query)
            self.env.assertEquals(first_result.result_set,
                                  second_result.result_set)

    # Strings, numerics, booleans, array, and point properties should be properly serialized and reloaded
    def test_restore_properties(self):
        graph_names = ("simple_props", "{tag}_simple_props")
        for graph_name in graph_names:
            graph = self.db.select_graph(graph_name)

            query = """CREATE (:p {strval: 'str', numval: 5.5, boolval: true, array: [1,2,3], pointval: point({latitude: 5.5, longitude: 6})})"""
            result = graph.query(query)

            # Verify that node was created correctly
            self.env.assertEquals(result.nodes_created, 1)
            self.env.assertEquals(result.properties_set, 5)

            # Save RDB & Load from RDB
            self.env.dumpAndReload()

            query = """MATCH (p) RETURN p.boolval, p.numval, p.strval, p.array, p.pointval"""
            actual_result = graph.query(query)

            # Verify that the properties are loaded correctly.
            expected_result = [[True, 5.5, 'str', [1, 2, 3], {"latitude": 5.5, "longitude": 6.0}]]
            self.env.assertEquals(actual_result.result_set, expected_result)

    # Verify multiple edges of the same relation between nodes A and B
    # are saved and restored correctly.
    def test_repeated_edges(self):
        graph_names = ["repeated_edges", "{tag}_repeated_edges"]
        for graph_name in graph_names:
            graph = self.db.select_graph(graph_name)
            graph.query("""CREATE (src:P {name: 'src'}), (dest:P {name: 'dest'}),
                        (src)-[:R {val: 1}]->(dest), (src)-[:R {val: 2}]->(dest)""")

            # Verify the new edge
            q = """MATCH (a)-[e]->(b) RETURN e.val, a.name, b.name ORDER BY e.val"""
            actual_result = graph.query(q)

            expected_result = [[1, 'src', 'dest'], [2, 'src', 'dest']]

            self.env.assertEquals(actual_result.result_set, expected_result)

            # Save RDB & Load from RDB
            self.env.dumpAndReload()

            # Verify that the latest edge was properly saved and loaded
            actual_result = graph.query(q)
            self.env.assertEquals(actual_result.result_set, expected_result)

    # Verify that graphs larger than the
    # default capacity are persisted correctly.
    def test_load_large_graph(self):
        graph_name = "LARGE_GRAPH"
        graph = self.db.select_graph(graph_name)
        q = """UNWIND range(1, 50000) AS v CREATE (:L)-[:R {v: v}]->(:L)"""
        actual_result = graph.query(q)
        self.env.assertEquals(actual_result.nodes_created, 100_000)
        self.env.assertEquals(actual_result.relationships_created, 50_000)

        # Save RDB & Load from RDB
        self.env.dumpAndReload()

        expected_result = [[50000]]

        queries = [
            """MATCH (:L)-[r {v: 50000}]->(:L) RETURN r.v""",
            """MATCH (:L)-[r:R {v: 50000}]->(:L) RETURN r.v""",
            """MATCH ()-[r:R {v: 50000}]->() RETURN r.v"""
        ]

        for q in queries:
            actual_result = graph.query(q)
            self.env.assertEquals(actual_result.result_set, expected_result)

    # Verify that graphs created using the GRAPH.BULK endpoint are persisted correctly
    def test_bulk_insert(self):
        port      = self.env.envRunner.port
        runner    = CliRunner()
        graphname = "bulk_inserted_graph"


        csv_path = os.path.dirname(os.path.abspath(__file__)) + '/../../demo/social/resources/bulk_formatted/'
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{port}",
                                          '--nodes', csv_path + 'Person.csv',
                                          '--nodes', csv_path + 'Country.csv',
                                          '--relations', csv_path + 'KNOWS.csv',
                                          '--relations', csv_path + 'VISITED.csv',
                                          graphname])

        # The script should report 27 node creations and 56 edge creations
        self.env.assertEquals(res.exit_code, 0)
        self.env.assertIn('27 nodes created', res.output)
        self.env.assertIn('56 relations created', res.output)

        # Restart the server
        self.env.dumpAndReload()

        graph = self.db.select_graph(graphname)

        query_result = graph.query("""MATCH (p:Person)
                                      RETURN p.name, p.age, p.gender, p.status, ID(p)
                                      ORDER BY p.name""")

        # Verify that the Person label exists, has the correct attributes
        # and is properly populated
        expected_result = [
                ['Ailon Velger',         32,     'male',    'married',  2],
                ['Alon Fital',           32,     'male',    'married',  1],
                ['Boaz Arad',            31,     'male',    'married',  4],
                ['Gal Derriere',         26,     'male',    'single',   11],
                ['Jane Chernomorin',     31,     'female',  'married',  8],
                ['Lucy Yanfital',        30,     'female',  'married',  7],
                ['Mor Yesharim',         31,     'female',  'married',  12],
                ['Noam Nativ',           34,     'male',    'single',   13],
                ['Omri Traub',           33,     'male',    'single',   5],
                ['Ori Laslo',            32,     'male',    'married',  3],
                ['Roi Lipman',           32,     'male',    'married',  0],
                ['Shelly Laslo Rooz',    31,     'female',  'married',  9],
                ['Tal Doron',            32,     'male',    'single',   6],
                ['Valerie Abigail Arad', 31,     'female',  'married',  10]
                ]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Verify that the Country label exists, has the correct attributes, and is properly populated
        query_result = graph.query('MATCH (c:Country) RETURN c.name, ID(c) ORDER BY c.name')
        expected_result = [
                ['Andora',       21],
                ['Canada',       18],
                ['China',        19],
                ['Germany',      24],
                ['Greece',       17],
                ['Italy',        25],
                ['Japan',        16],
                ['Kazakhstan',   22],
                ['Netherlands',  20],
                ['Prague',       15],
                ['Russia',       23],
                ['Thailand',     26],
                ['USA',          14]
        ]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Validate that the expected relations and properties have been constructed
        query_result = graph.query('MATCH (a)-[e:KNOWS]->(b) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name')

        expected_result = [
                ['Ailon Velger', 'friend',   'Noam Nativ'],
                ['Alon Fital',   'friend',   'Gal Derriere'],
                ['Alon Fital',   'friend',   'Mor Yesharim'],
                ['Boaz Arad',    'friend',   'Valerie Abigail Arad'],
                ['Roi Lipman',   'friend',   'Ailon Velger'],
                ['Roi Lipman',   'friend',   'Alon Fital'],
                ['Roi Lipman',   'friend',   'Boaz Arad'],
                ['Roi Lipman',   'friend',   'Omri Traub'],
                ['Roi Lipman',   'friend',   'Ori Laslo'],
                ['Roi Lipman',   'friend',   'Tal Doron'],
                ['Ailon Velger', 'married',  'Jane Chernomorin'],
                ['Alon Fital',   'married',  'Lucy Yanfital'],
                ['Ori Laslo',    'married',  'Shelly Laslo Rooz']
        ]
        self.env.assertEquals(query_result.result_set, expected_result)

        query_result = graph.query('MATCH (a)-[e:VISITED]->(b) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name')

        expected_result = [
                ['Alon Fital',           'business',  'Prague'],
                ['Alon Fital',           'business',  'USA'],
                ['Boaz Arad',            'business',  'Netherlands'],
                ['Boaz Arad',            'business',  'USA'],
                ['Gal Derriere',         'business',  'Netherlands'],
                ['Jane Chernomorin',     'business',  'USA'],
                ['Lucy Yanfital',        'business',  'USA'],
                ['Mor Yesharim',         'business',  'Germany'],
                ['Ori Laslo',            'business',  'China'],
                ['Ori Laslo',            'business',  'USA'],
                ['Roi Lipman',           'business',  'Prague'],
                ['Roi Lipman',           'business',  'USA'],
                ['Tal Doron',            'business',  'Japan'],
                ['Tal Doron',            'business',  'USA'],
                ['Alon Fital',           'pleasure',  'Greece'],
                ['Alon Fital',           'pleasure',  'Prague'],
                ['Alon Fital',           'pleasure',  'USA'],
                ['Boaz Arad',            'pleasure',  'Netherlands'],
                ['Boaz Arad',            'pleasure',  'USA'],
                ['Jane Chernomorin',     'pleasure',  'Greece'],
                ['Jane Chernomorin',     'pleasure',  'Netherlands'],
                ['Jane Chernomorin',     'pleasure',  'USA'],
                ['Lucy Yanfital',        'pleasure',  'Kazakhstan'],
                ['Lucy Yanfital',        'pleasure',  'Prague'],
                ['Lucy Yanfital',        'pleasure',  'USA'],
                ['Mor Yesharim',         'pleasure',  'Greece'],
                ['Mor Yesharim',         'pleasure',  'Italy'],
                ['Noam Nativ',           'pleasure',  'Germany'],
                ['Noam Nativ',           'pleasure',  'Netherlands'],
                ['Noam Nativ',           'pleasure',  'Thailand'],
                ['Omri Traub',           'pleasure',  'Andora'],
                ['Omri Traub',           'pleasure',  'Greece'],
                ['Omri Traub',           'pleasure',  'USA'],
                ['Ori Laslo',            'pleasure',  'Canada'],
                ['Roi Lipman',           'pleasure',  'Japan'],
                ['Roi Lipman',           'pleasure',  'Prague'],
                ['Shelly Laslo Rooz',    'pleasure',  'Canada'],
                ['Shelly Laslo Rooz',    'pleasure',  'China'],
                ['Shelly Laslo Rooz',    'pleasure',  'USA'],
                ['Tal Doron',            'pleasure',  'Andora'],
                ['Tal Doron',            'pleasure',  'USA'],
                ['Valerie Abigail Arad', 'pleasure',  'Netherlands'],
                ['Valerie Abigail Arad', 'pleasure',  'Russia']
                ]
        self.env.assertEquals(query_result.result_set, expected_result)

    # Verify that nodes with multiple labels are saved and restored correctly.
    def test_persist_multiple_labels(self):
        graph_id = "multiple_labels"
        g = self.db.select_graph(graph_id)
        q = "CREATE (a:L0:L1:L2)"
        actual_result = g.query(q)
        self.env.assertEquals(actual_result.nodes_created, 1)
        self.env.assertEquals(actual_result.labels_added, 3)

        # Verify the new node
        q = "MATCH (a) RETURN LABELS(a)"
        actual_result = g.query(q)
        expected_result = [[["L0", "L1", "L2"]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Save RDB & Load from RDB
        self.env.dumpAndReload()

        # Verify that the graph was properly saved and loaded
        actual_result = g.query(q)
        self.env.assertEquals(actual_result.result_set, expected_result)

        queries = [
                "MATCH (a:L0) RETURN count(a)",
                "MATCH (a:L1) RETURN count(a)",
                "MATCH (a:L2) RETURN count(a)",
                "MATCH (a:L0:L0) RETURN count(a)",
                "MATCH (a:L0:L1) RETURN count(a)",
                "MATCH (a:L0:L2) RETURN count(a)",
                "MATCH (a:L1:L0) RETURN count(a)",
                "MATCH (a:L1:L1) RETURN count(a)",
                "MATCH (a:L1:L2) RETURN count(a)",
                "MATCH (a:L2:L0) RETURN count(a)",
                "MATCH (a:L2:L1) RETURN count(a)",
                "MATCH (a:L2:L2) RETURN count(a)",
                "MATCH (a:L0:L1:L2) RETURN count(a)"]

        for q in queries:
            actual_result = g.query(q)
            self.env.assertEquals(actual_result.result_set[0], [1])

    # test encoding and decoding of multiple graphs
    def test_multi_graph(self):
        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        # issue BGSAVE just so we would have something in info persistence rdb_last_save_time
        self.conn.bgsave()

        # make sure at least one second pass between the first save to the next
        time.sleep(1)

        # create 1000 graphs
        # each containing 1000 nodes and 500 edges
        graph_count = 1000
        q = "UNWIND range(0, 499) AS x CREATE (:A)-[:R]->(:B)"

        for i in range(0, graph_count):
            g = self.db.select_graph(GRAPH_ID + str(i))
            g.query(q)

        # Measure the time it takes to encode 1000 graphs
        # Get the last completed save time before issuing BGSAVE
        before = self.conn.info("persistence").get("rdb_last_save_time")

        # Issue BGSAVE
        self.conn.bgsave()
        save_finished = False

        # Wait for BGSAVE to complete for a maximum of 10 seconds
        for _ in range(100):
            time.sleep(0.1)  # poll every 100ms
            now = self.conn.info("persistence").get("rdb_last_save_time")
            if now != before:
                save_finished = True
                break

        self.env.assertTrue(save_finished)

        # Save & Load from RDB
        self.env.dumpAndReload()

        # Make sure reloaded DB contains all graphs
        graphs = self.db.list_graphs()
        graphs = [int(x.replace(GRAPH_ID, "")) for x in graphs if x.startswith(GRAPH_ID)]
        graphs.sort()

        self.env.assertEquals(graphs, list(range(0, graph_count)))

        qs = [
            ("MATCH (n) RETURN count(n)"         , 1000),
            ("MATCH (a:A) RETURN count(a)"       , 500) ,
            ("MATCH (b:B) RETURN count(b)"       , 500) ,
            ("MATCH ()-[e]->() RETURN count(e)"  , 500) ,
            ("MATCH ()-[e:R]->() RETURN count(e)", 500)
        ]

        # Validate all graphs
        for i in range(graph_count):
            g = self.db.select_graph(GRAPH_ID + str(i))
            for q, expected_count in qs:
                result = g.query(q).result_set[0][0]
                if(result != expected_count):
                    self.env.log(f"Graph {i} expected {expected_count}, got {result}")
                    self.env.assertFalse(True)

    # Verify that the DB will respond to PING which taking a snapshot
    def test_ping_while_saving(self):
        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        # issue BGSAVE just so we would have something in info persistence rdb_last_save_time
        self.conn.bgsave()

        # make sure at least one second pass between the first save to the next
        time.sleep(1)

        # create a large graph
        g = self.db.select_graph(GRAPH_ID)
        g.query("UNWIND range(0, 200000) AS x CREATE (:A)-[:R]->(:B)")

        # Get the last completed save time before issuing BGSAVE
        before = self.conn.info("persistence").get("rdb_last_save_time")

        # Start pinging
        def ping_worker(conn, pings):
            while not stop_event.is_set():
                conn.ping()
                pings.append(time.time())
                time.sleep(0.005) # sleep for 5ms

        stop_event = threading.Event()
        pings = []
        thread = threading.Thread(target=ping_worker, args=(self.conn, pings))
        thread.start()

        # Issue BGSAVE
        self.conn.bgsave()

        save_finished = False
        start = time.time()
        # Wait for BGSAVE to complete for a maximum of 10 seconds
        for _ in range(100):
            time.sleep(0.1)  # poll every 100ms
            now = self.conn.info("persistence").get("rdb_last_save_time")
            if now != before:
                save_finished = True
                break
        
        self.env.assertTrue(save_finished)
        end = time.time()

        stop_event.set()  # Signal the thread to stop
        thread.join()

        # Make sure PINGs were answered during the save period
        self.env.assertGreater(len(pings), 5)

        # Make sure PINGs where served while the database was saving
        # TODO: improve PING capture, we only care for the time period
        # where the DB is preparing to fork
        pings_during_save = [s for s in pings if start < s < end]
        self.env.assertGreater(len(pings_during_save), 2)

    # make sure peak memory consumption doesn't goes beyond
    # 50% when taking a snapshot
    def test_bgsave_memory_consumption(self):
        # TODO: unreliable, skipping for now
        self.env.skip()
        return

        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        # issue BGSAVE just so we would have something in info persistence rdb_last_save_time
        # make sure at least one second pass between the first save to the next
        self.conn.bgsave()
        time.sleep(1)

        pid = self.env.envRunner.masterProcess.pid
        before = self.conn.info("persistence").get("rdb_last_save_time")

        # create a large graph
        g = self.db.select_graph(GRAPH_ID)
        g.query("UNWIND range(0, 200000) AS x CREATE (:A)-[:R]->(:B)")

        base_memory_consumption = get_total_rss_memory(pid)
        peak_memory_consumption = base_memory_consumption

        # Issue BGSAVE
        self.conn.bgsave()

        # Wait for BGSAVE to complete
        while True:
            # update peak memory consumption
            peak_memory_consumption = max(peak_memory_consumption, get_total_rss_memory(pid))
            time.sleep(0.005)  # poll every 5ms
            now = self.conn.info("persistence").get("rdb_last_save_time")
            if now != before:
                break

        # assert that peak memory did not cross 1.50 * base_memory_consumption
        self.env.assertLess(peak_memory_consumption, 1.50 * base_memory_consumption)

