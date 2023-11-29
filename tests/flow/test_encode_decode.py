from common import *
from index_utils import *
from random_graph import create_random_schema, create_random_graph, run_random_graph_ops, ALL_OPS
import re

def compare_nodes_result_set(env, result_set_a, result_set_b):
    env.assertEquals(len(result_set_a), len(result_set_b))
    for i in range(0, len(result_set_a)):
        env.assertEquals(result_set_a[i][0].id, result_set_b[i][0].id)
        env.assertEquals(set(result_set_a[i][0].labels), set(
            result_set_b[i][0].labels))
        env.assertEquals(result_set_a[i][0].properties,
                         result_set_b[i][0].properties)


class test_encode_decode(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env(moduleArgs='VKEY_MAX_ENTITY_COUNT 10 NODE_CREATION_BUFFER 100',
                       enableDebugCommand=True)
        self.redis_con = self.env.getConnection()

    def test01_nodes_over_multiple_keys(self):
        graph_name = "nodes_over_multiple_keys"
        redis_graph = Graph(self.redis_con, graph_name)
        # Create 3 nodes meta keys
        redis_graph.query("UNWIND range(0,20) as i CREATE (:Node {val:i})")
        # Return all the nodes, before and after saving & loading the RDB, and check equality
        query = "MATCH (n:Node) return n"
        expected = redis_graph.query(query)
        # Save RDB & Load from RDB
        self.redis_con.execute_command("DEBUG", "RELOAD")
        actual = redis_graph.query(query)
        self.env.assertEquals(expected.result_set, actual.result_set)

    def test02_no_compaction_on_nodes_delete(self):
        graph_name = "no_compaction_on_nodes_delete"
        redis_graph = Graph(self.redis_con, graph_name)
        # Create 20 nodes meta keys
        redis_graph.query("UNWIND range(0, 20) as i CREATE (:Node)")
        # Return all the nodes, before and after saving & loading the RDB, and check equality
        query = "MATCH (n:Node) WITH n ORDER by id(n) return COLLECT(id(n))"
        expected_full_graph_nodes_id = redis_graph.query(query)

        # Delete 3 nodes.
        redis_graph.query("MATCH (n:Node) WHERE id(n) IN [7, 14, 20] DELETE n")
        expected_nodes_id_after_delete = redis_graph.query(query)

        # Save RDB & Load from RDB
        self.redis_con.execute_command("DEBUG", "RELOAD")

        actual = redis_graph.query(query)
        # Validate no compaction, all IDs are the same
        self.env.assertEquals(
            expected_nodes_id_after_delete.result_set, actual.result_set)

        # Validate reuse of node ids - create 3 nodes.
        redis_graph.query("UNWIND range (0, 2) as i CREATE (:Node)")
        actual = redis_graph.query(query)
        self.env.assertEquals(
            expected_full_graph_nodes_id.result_set, actual.result_set)

    def test03_edges_over_multiple_keys(self):
        graph_name = "edges_over_multiple_keys"
        redis_graph = Graph(self.redis_con, graph_name)
        # Create 3 edges meta keys
        redis_graph.query(
            "UNWIND range(0,20) as i CREATE (:Src)-[:R {val:i}]->(:Dest)")
        # Return all the edges, before and after saving & loading the RDB, and check equality
        query = "MATCH (:Src)-[e:R]->(:Dest) return e"
        expected = redis_graph.query(query)
        # Save RDB & Load from RDB
        self.redis_con.execute_command("DEBUG", "RELOAD")
        actual = redis_graph.query(query)
        self.env.assertEquals(expected.result_set, actual.result_set)

    def test04_no_compaction_on_edges_delete(self):
        graph_name = "no_compaction_on_edges_delete"
        redis_graph = Graph(self.redis_con, graph_name)
        # Create 3 nodes meta keys
        redis_graph.query(
            "UNWIND range(0,20) as i CREATE (:Src)-[:R]->(:Dest)")
        # Return all the edges, before and after saving & loading the RDB, and check equality
        query = "MATCH (:Src)-[e:R]->(:Dest) WITH e ORDER by id(e) return COLLECT(id(e))"
        expected_full_graph_nodes_id = redis_graph.query(query)
        # Delete 3 edges.
        redis_graph.query(
            "MATCH (:Src)-[e:R]->(:Dest) WHERE id(e) IN [7,14,20] DELETE e")
        expected_nodes_id_after_delete = redis_graph.query(query)
        # Save RDB & Load from RDB
        self.redis_con.execute_command("DEBUG", "RELOAD")
        actual = redis_graph.query(query)
        # Validate no compaction, all IDs are the same
        self.env.assertEquals(
            expected_nodes_id_after_delete.result_set, actual.result_set)
        # Validate reuse of edges ids - create 3 edges.
        redis_graph.query(
            "UNWIND range (0,2) as i CREATE (:Src)-[:R]->(:Dest)")
        actual = redis_graph.query(query)
        self.env.assertEquals(
            expected_full_graph_nodes_id.result_set, actual.result_set)

    def test05_multiple_edges_over_multiple_keys(self):
        graph_name = "multiple_edges_over_multiple_keys"
        redis_graph = Graph(self.redis_con, graph_name)
        # Create 3 edges meta keys
        redis_graph.query(
            "CREATE (n1:Src {val:1}), (n2:Dest {val:2}) WITH n1, n2 UNWIND range(0,20) as i CREATE (n1)-[:R {val:i}]->(n2)")
        # Return all the edges, before and after saving & loading the RDB, and check equality
        query = "MATCH (:Src)-[e:R]->(:Dest) return e"
        expected = redis_graph.query(query)
        # Save RDB & Load from RDB
        self.redis_con.execute_command("DEBUG", "RELOAD")
        actual = redis_graph.query(query)
        self.env.assertEquals(expected.result_set, actual.result_set)

    def test06_no_compaction_on_multiple_edges_delete(self):
        graph_name = "no_compaction_on_multiple_edges_delete"
        redis_graph = Graph(self.redis_con, graph_name)
        # Create 3 nodes meta keys
        redis_graph.query(
            "CREATE (n1:Src {val:1}), (n2:Dest {val:2}) WITH n1, n2 UNWIND range(0,20) as i CREATE (n1)-[:R]->(n2)")
        # Return all the edges, before and after saving & loading the RDB, and check equality
        query = "MATCH (:Src)-[e:R]->(:Dest) WITH e ORDER by id(e) return COLLECT(id(e))"
        expected_full_graph_nodes_id = redis_graph.query(query)
        # Delete 3 edges.
        redis_graph.query(
            "MATCH (:Src)-[e:R]->(:Dest) WHERE id(e) IN [7,14,20] DELETE e")
        expected_nodes_id_after_delete = redis_graph.query(query)
        # Save RDB & Load from RDB
        self.redis_con.execute_command("DEBUG", "RELOAD")
        actual = redis_graph.query(query)
        # Validate no compaction, all IDs are the same
        self.env.assertEquals(
            expected_nodes_id_after_delete.result_set, actual.result_set)
        # Validate reuse of edges ids - create 3 edges.
        redis_graph.query(
            "MATCH (n1:Src {val:1}), (n2:Dest {val:2}) WITH n1, n2 UNWIND range (0,2) as i CREATE (n1)-[:R]->(n2)")
        actual = redis_graph.query(query)
        self.env.assertEquals(
            expected_full_graph_nodes_id.result_set, actual.result_set)

    def test07_index_after_encode_decode_in_v7(self):
        graph_name = "index_after_encode_decode_in_v7"
        redis_graph = Graph(self.redis_con, graph_name)
        create_node_range_index(redis_graph, 'N', 'val', sync=True)
        # Verify indices exists.
        plan = str(redis_graph.explain("MATCH (n:N {val:1}) RETURN n"))
        self.env.assertIn("Index Scan", plan)
        # Save RDB & Load from RDB
        self.redis_con.execute_command("DEBUG", "RELOAD")
        # Verify indices exists after loading RDB.
        plan = str(redis_graph.explain("MATCH (n:N {val:1}) RETURN n"))
        self.env.assertIn("Index Scan", plan)

    def test08_multiple_graphs_with_index(self):
        # Create a multi-key graph.
        graph1_name = "v7_graph_1"
        graph1 = Graph(self.redis_con, graph1_name)
        graph1.query(
            "UNWIND range(0,21) AS i CREATE (a:L {v: i})-[:E]->(b:L2 {v: i})")

        # Create a single-key graph.
        graph2_name = "v7_graph_2"
        graph2 = Graph(self.redis_con, graph2_name)
        graph2.query("CREATE (a:L {v: 1})-[:E]->(b:L2 {v: 2})")

        # Add an index to the multi-key graph.
        create_node_range_index(graph1, 'L', 'v', sync=True)

        # Save RDB and reload from RDB
        self.redis_con.execute_command("DEBUG", "RELOAD")

        # The load should be successful and the index should still be built.
        query = "MATCH (n:L {v:1}) RETURN n.v"
        plan = str(graph1.explain(query))
        self.env.assertIn("Index Scan", plan)
        expected = [[1]]
        actual = graph1.query(query)
        self.env.assertEquals(actual.result_set, expected)

    def test08_multiple_reltypes(self):
        graph_name = "multiple_reltypes"
        redis_graph = Graph(self.redis_con, graph_name)
        # Create 10 nodes
        redis_graph.query("UNWIND range(0,10) as v CREATE (:L {v: v})")
        # Create 3 edges of different relation types connecting 6 different nodes
        redis_graph.query(
            "MATCH (a:L {v: 1}), (b:L {v: 2}) CREATE (a)-[:R1]->(b)")
        redis_graph.query(
            "MATCH (a:L {v: 3}), (b:L {v: 4}) CREATE (a)-[:R2]->(b)")
        redis_graph.query(
            "MATCH (a:L {v: 5}), (b:L {v: 6}) CREATE (a)-[:R3]->(b)")

        # Retrieve all the edges before and after saving & loading the RDB to check equality
        query = "MATCH (:L)-[e]->(:L) RETURN ID(e), type(e) ORDER BY ID(e)"
        expected = redis_graph.query(query)

        # Save RDB & Load from RDB
        self.redis_con.execute_command("DEBUG", "RELOAD")

        actual = redis_graph.query(query)
        self.env.assertEquals(expected.result_set, actual.result_set)

    # test changes to the VKEY_MAX_ENTITY_COUNT configuration are reflected in
    # the number of virtual keys created
    def test09_vkey_max_entity_count(self):
        self.redis_con.flushall()

        logfilename = self.env.envRunner._getFileName("master", ".log")
        logfile = open(f"{self.env.logDir}/{logfilename}")
        log = logfile.read()

        # Set configuration
        response = self.redis_con.execute_command(
            "GRAPH.CONFIG SET VKEY_MAX_ENTITY_COUNT 10")
        self.env.assertEqual(response, "OK")

        graph_name = "vkey_max_entity_count"
        redis_graph = Graph(self.redis_con, graph_name)

        # Create 30 nodes
        redis_graph.query("UNWIND range(0, 30) as v CREATE (:L {v: v})")

        # Save RDB & Load from RDB
        self.redis_con.save()

        # Set configuration
        response = self.redis_con.execute_command(
            "GRAPH.CONFIG SET VKEY_MAX_ENTITY_COUNT 5")
        self.env.assertEqual(response, "OK")

        # Save RDB & Load from RDB
        self.redis_con.save()

        log = logfile.read()

        matches = re.findall(
            "Created (.) virtual keys for graph vkey_max_entity_count", log)

        self.env.assertEqual(matches, ['3', '6'])

        matches = re.findall(
            "Deleted (.) virtual keys for graph vkey_max_entity_count", log)

        self.env.assertEqual(matches, ['3', '6'])

    def test10_decode_single_edge_relation_with_deleted_nodes(self):
        self.redis_con.flushall()

        # Set configuration
        response = self.redis_con.execute_command(
            "GRAPH.CONFIG SET VKEY_MAX_ENTITY_COUNT 20000")
        self.env.assertEqual(response, "OK")

        graph_name = "large_deleted_nodes"
        redis_graph = Graph(self.redis_con, graph_name)

        # Create 60000 nodes and 30000 edges
        redis_graph.query(
            "UNWIND range(0, 30000) as v CREATE (:L {v: v})-[:R]->(:M {v: v})")

        # Delete 20000 nodes and 10000 edges
        redis_graph.query(
            "MATCH (n:L)-[:R]->(m:M) WHERE id(n) <= 20000 DELETE n, m")

        res_before = redis_graph.query(
            "MATCH (n:L)-[r:R]->(m:M) RETURN id(n), id(r), id(m)")

        # Save RDB & Load from RDB
        self.redis_con.execute_command("DEBUG", "RELOAD")

        # Validate all data lodaed correctly
        res_after = redis_graph.query(
            "MATCH (n:L)-[r:R]->(m:M) RETURN id(n), id(r), id(m)")
        self.env.assertEquals(res_before.result_set, res_after.result_set)

    def test11_decode_multi_edge_relation_with_deleted_nodes(self):
        self.redis_con.flushall()

        # Set configuration
        response = self.redis_con.execute_command(
            "GRAPH.CONFIG SET VKEY_MAX_ENTITY_COUNT 20000")
        self.env.assertEqual(response, "OK")

        graph_name = "large_deleted_nodes"
        redis_graph = Graph(self.redis_con, graph_name)

        # Create 60000 nodes and 60000 edges
        redis_graph.query(
            "UNWIND range(0, 30000) as v CREATE (n:L {v: v}), (m:M {v: v}) WITH n, m CREATE (n)-[:R]->(m), (n)-[:R]->(m)")

        # Delete 20000 nodes and 40000 edges
        redis_graph.query(
            "MATCH (n:L)-[:R]->(m:M) WHERE id(n) <= 20000 DELETE n, m")

        res_before = redis_graph.query(
            "MATCH (n:L)-[r:R]->(m:M) RETURN id(n), id(r), id(m)")

        # Save RDB & Load from RDB
        self.redis_con.execute_command("DEBUG", "RELOAD")

        # Validate all data lodaed correctly
        res_after = redis_graph.query(
            "MATCH (n:L)-[r:R]->(m:M) RETURN id(n), id(r), id(m)")
        self.env.assertEquals(res_before.result_set, res_after.result_set)

    def test12_random_graph(self):
        self.redis_con.flushall()

        graph_name = "random_graph"
        redis_graph = Graph(self.redis_con, graph_name)

        nodes, edges = create_random_schema()
        res = create_random_graph(redis_graph, nodes, edges)

        nodes_before = redis_graph.query("MATCH (n) RETURN n")
        edges_before = redis_graph.query("MATCH ()-[e]->() RETURN e")

        self.redis_con.execute_command("DEBUG", "RELOAD")

        nodes_after = redis_graph.query("MATCH (n) RETURN n")
        edges_after = redis_graph.query("MATCH ()-[e]->() RETURN e")

        compare_nodes_result_set(self.env, nodes_before.result_set, nodes_after.result_set)
        self.env.assertEquals(edges_before.result_set, edges_after.result_set)

        res = run_random_graph_ops(redis_graph, nodes, edges, ALL_OPS)

        nodes_before = redis_graph.query("MATCH (n) RETURN n")
        edges_before = redis_graph.query("MATCH ()-[e]->() RETURN e")

        self.redis_con.execute_command("DEBUG", "RELOAD")

        nodes_after = redis_graph.query("MATCH (n) RETURN n")
        edges_after = redis_graph.query("MATCH ()-[e]->() RETURN e")

        compare_nodes_result_set(self.env, nodes_before.result_set, nodes_after.result_set)
        self.env.assertEquals(edges_before.result_set, edges_after.result_set)
