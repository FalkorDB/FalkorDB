from common import *

redis_con = None

class test_prev_rdb_decode(FlowTestsBase):
    def __init__(self):
        self.env = Env(decodeResponses=True)
        global redis_con
        redis_con = self.env.getConnection()

    def tearDown(self):
        self.env.flush()

    def test_v9_decode(self):
        graph_name = "v9_rdb_restore"
        # docker run -p 6379:6379 -it redislabs/redisgraph:2.4.10
        # dump created with the following query (v9 supported property value: integer, double, boolean, string, null, array, point)
        #  graph.query g "CREATE (:L1 {val:1, strval: 'str', numval: 5.5, nullval: NULL, boolval: true, array: [1,2,3], point: POINT({latitude: 32, longitude: 34})})-[:E{val:2}]->(:L2{val:3})"
        #  graph.query g "CREATE INDEX ON :L1(val)"
        #  graph.query g "CREATE INDEX ON :L1(none_existsing)"
        #  graph.query g "CREATE (:L3)-[:E2]->(:L4)"
        #  graph.query g "MATCH (n1:L3)-[r:E2]->(n2:L4) DELETE n1, r, n2"
        #  dump g
        v9_rdb = b"\a\x81\x82\xb6\xa9\x85\xd6\xadh\t\x05\x02g\x00\x02\x02\x02\x01\x02\x04\x02\x02\x02\x00\x02\x00\x02\x01\x02\x05\x02\x01\x02\x02\x02\x02\x02\x02\x02\x03\x02\x01\x02\x04\x02\x01\x02\x05\x02\x01\x02\x00\x02\x01\x02\x00\x02\x06\x02\x00\x02`\x00\x02\x01\x02\x01\x02H\x00\x05\x04str\x00\x02\x02\x02\x80\x00\x00@\x00\x04\x00\x00\x00\x00\x00\x00\x16@\x02\x04\x02P\x00\x02\x01\x02\x05\x02\b\x02\x03\x02`\x00\x02\x01\x02`\x00\x02\x02\x02`\x00\x02\x03\x02\x06\x02\x80\x00\x02\x00\x00\x04\x00\x00\x00\x00\x00\x00@@\x04\x00\x00\x00\x00\x00\x00A@\x02\x01\x02\x01\x02\x01\x02\x01\x02\x00\x02`\x00\x02\x03\x02\x02\x02\x03\x02\x00\x02\x00\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x02\x02\x01\x02\b\x05\x04val\x00\x05\astrval\x00\x05\anumval\x00\x05\bnullval\x00\x05\bboolval\x00\x05\x06array\x00\x05\x06point\x00\x05\x0fnone_existsing\x00\x02\x04\x02\x00\x05\x03L1\x00\x02\x02\x02\x01\x05\x04val\x00\x02\x01\x05\x0fnone_existsing\x00\x02\x01\x05\x03L2\x00\x02\x00\x02\x02\x05\x03L3\x00\x02\x00\x02\x03\x05\x03L4\x00\x02\x00\x02\x02\x02\x00\x05\x02E\x00\x02\x00\x02\x01\x05\x03E2\x00\x02\x00\x00\t\x00\xd7\xd0\x1cB;\xce\x1d>"
        redis_con.restore(graph_name, 0, v9_rdb, True)
        redis_graph = Graph(redis_con, graph_name)
        node0 = Node(node_id=0, label='L1', properties={'val': 1, 'strval': 'str', 'numval': 5.5, 'boolval': True, 'array': [1,2,3], 'point': {'latitude': 32, 'longitude': 34}})
        node1 = Node(node_id=1, label='L2', properties={'val': 3})
        edge01 = Edge(src_node=0, relation='E', dest_node=1, edge_id=0, properties={'val':2})
        results = redis_graph.query("MATCH (n)-[e]->(m) RETURN n, e, m")
        self.env.assertEqual(results.result_set, [[node0, edge01, node1]])
        plan = redis_graph.execution_plan("MATCH (n:L1 {val:1}) RETURN n")
        self.env.assertIn("Index Scan", plan)
        results = redis_graph.query("MATCH (n:L1 {val:1}) RETURN n")
        self.env.assertEqual(results.result_set, [[node0]])

