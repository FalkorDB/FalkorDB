from common import *
from index_utils import *

people = ["Roi", "Alon", "Ailon", "Boaz", "Tal", "Omri", "Ori"]
redis_graph = None


class testEdgeByIndexScanFlow(FlowTestsBase):
    def __init__(self):
        self.env = Env(decodeResponses=True)

    def setUp(self):
        global redis_graph
        redis_con = self.env.getConnection()
        redis_graph = Graph(redis_con, "social")
        self.populate_graph(redis_graph)
        self.build_indices()

    def tearDown(self):
        self.env.cmd('flushall')
    
    def populate_graph(self, redis_graph):
        nodes = {}

        # Create entities
        node_id = 0
        for p in people:
            node = Node(label="person", properties={"name": p, "created_at": node_id})
            redis_graph.add_node(node)
            nodes[p] = node
            node_id = node_id + 1

        # Fully connected graph
        edge_id = 0
        for src in nodes:
            for dest in nodes:
                if src != dest:
                    edge = Edge(nodes[src], "knows", nodes[dest], properties={"created_at": edge_id * 2})
                    redis_graph.add_edge(edge)
                    edge = Edge(nodes[src], "friend", nodes[dest], properties={"created_at": edge_id * 2 + 1, "updated_at": edge_id * 3})
                    redis_graph.add_edge(edge)
                    edge_id = edge_id + 1

        redis_graph.commit()

    def build_indices(self):
        global redis_graph
        create_node_exact_match_index(redis_graph, "person", "age")
        create_edge_exact_match_index(redis_graph, "friend", "created_at")
        create_edge_exact_match_index(redis_graph, "knows", "created_at", sync=True)

    # Validate that Cartesian products using index and label scans succeed
    def test01_cartesian_product_mixed_scans(self):
        query = "MATCH ()-[f:friend]->(), ()-[k:knows]->() WHERE f.created_at >= 0 RETURN f.created_at, k.created_at ORDER BY f.created_at, k.created_at"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Conditional Traverse', plan)
        indexed_result = redis_graph.query(query)

        query = "MATCH ()-[f:friend]->(), ()-[k:knows]->() RETURN f.created_at, k.created_at ORDER BY f.created_at, k.created_at"
        plan = redis_graph.execution_plan(query)
        self.env.assertNotIn('Edge By Index Scan', plan)
        self.env.assertIn('Conditional Traverse', plan)
        unindexed_result = redis_graph.query(query)

        self.env.assertEquals(indexed_result.result_set, unindexed_result.result_set)

    # Validate that Cartesian products using just index scans succeed
    def test02_cartesian_product_index_scans_only(self):
        query = "MATCH ()-[f:friend]->(), ()-[k:knows]->() WHERE f.created_at >= 0 AND k.created_at >= 0 RETURN f.created_at, k.created_at ORDER BY f.created_at, k.created_at"
        plan = redis_graph.execution_plan(query)
        # The two streams should both use index scans
        self.env.assertEquals(plan.count('Edge By Index Scan'), 2)
        self.env.assertNotIn('Conditional Traverse', plan)
        indexed_result = redis_graph.query(query)

        query = "MATCH ()-[f:friend]->(), ()-[k:knows]->() RETURN f.created_at, k.created_at ORDER BY f.created_at, k.created_at"
        plan = redis_graph.execution_plan(query)
        self.env.assertNotIn('Edge By Index Scan', plan)
        self.env.assertIn('Conditional Traverse', plan)
        unindexed_result = redis_graph.query(query)

        self.env.assertEquals(indexed_result.result_set, unindexed_result.result_set)

    # Validate that the appropriate bounds are respected when a Cartesian product uses the same index in two streams
    def test03_cartesian_product_reused_index(self):
        create_edge_exact_match_index(redis_graph, 'friend', 'updated_at', sync=True)
        query = """MATCH ()-[a:friend]->(), ()-[b:friend]->()
                   WHERE a.created_at >= 80 AND b.updated_at >= 120
                   RETURN a.created_at, b.updated_at
                   ORDER BY a.created_at, b.updated_at"""
        plan = redis_graph.execution_plan(query)
        # The two streams should both use index scans
        self.env.assertEquals(plan.count('Edge By Index Scan'), 2)
        self.env.assertNotIn('Conditional Traverse', plan)


        expected_result = [[81, 120], [81, 123], [83, 120], [83, 123]]
        result = redis_graph.query(query)

        self.env.assertEquals(result.result_set, expected_result)

    # Validate index utilization when filtering on a numeric field with the `IN` keyword.
    def test04_test_in_operator_numerics(self):
        # Validate the transformation of IN to multiple OR expressions.
        query = "MATCH ()-[f:friend]-() WHERE f.created_at IN [1,2,3] RETURN f"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)

        # Validate that nested arrays are not scanned in index.
        query = "MATCH ()-[f:friend]-() WHERE f.created_at IN [[1,2],3] RETURN f"
        plan = redis_graph.execution_plan(query)
        self.env.assertNotIn('Edge By Index Scan', plan)
        self.env.assertIn('Conditional Traverse', plan)

        # Validate the transformation of IN to multiple OR, over a range.
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at IN range(0,30) RETURN DISTINCT n.name ORDER BY n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)

        expected_result = [['Ailon'], ['Alon'], ['Roi']]
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

         # Validate the transformation of IN to empty index iterator.
        query = "MATCH ()-[f:friend]-() WHERE f.created_at IN [] RETURN f.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)

        expected_result = []
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Validate the transformation of IN OR IN to empty index iterators.
        query = "MATCH ()-[f:friend]->() WHERE f.created_at IN [] OR f.created_at IN [] RETURN f.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)

        expected_result = []
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Validate the transformation of multiple IN filters.
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at IN [0, 1, 2] OR f.created_at IN [14, 15, 16] RETURN n.name ORDER BY n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)

        expected_result = [['Alon'], ['Roi']]
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Validate the transformation of multiple IN filters.
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at IN [0, 1, 2] OR f.created_at IN [14, 15, 16] OR f.created_at IN [] RETURN n.name ORDER BY n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)

        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

    def test05_index_scan_and_id(self):
        query = """MATCH (n)-[f:friend]->()
                   WHERE id(f)>=10 AND f.created_at<15
                   RETURN n.name
                   ORDER BY n.name"""
        plan = redis_graph.execution_plan(query)
        query_result = redis_graph.query(query)
        self.env.assertIn('Filter', plan)
        self.env.assertIn('Edge By Index Scan', plan)

        self.env.assertEqual(2, len(query_result.result_set))
        expected_result = [['Alon'], ['Roi']]
        self.env.assertEquals(expected_result, query_result.result_set)

    # Validate placement of index scans and filter ops when not all filters can be replaced.
    def test06_index_scan_multiple_filters(self):
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at = 31 AND NOT EXISTS(f.fakeprop) RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertNotIn('Conditional Traverse', plan)
        self.env.assertIn('Filter', plan)

        query_result = redis_graph.query(query)
        expected_result = ["Ailon"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

    def test07_index_scan_with_params(self):
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at = $time RETURN n.name"
        params = {'time': 31}
        plan = redis_graph.execution_plan(query, params=params)
        self.env.assertIn('Edge By Index Scan', plan)
        query_result = redis_graph.query(query, params=params)
        expected_result = ["Ailon"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

    def test08_index_scan_with_param_array(self):
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at in $times RETURN n.name"
        params = {'times': [31]}
        plan = redis_graph.execution_plan(query, params=params)
        self.env.assertIn('Edge By Index Scan', plan)
        query_result = redis_graph.query(query, params=params)
        expected_result = ["Ailon"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

    def test09_single_index_multiple_scans(self):
        query = """MATCH (p1:person {name: 'Roi'}), (p2:person {name: 'Alon'})
                   MERGE (p1)-[:friend {created_at: 100}]->(p2)
                   MERGE (p1)-[:friend {created_at: 101}]->(p2)"""
        plan = redis_graph.execution_plan(query)
        # Two index scans should be performed.
        self.env.assertEqual(plan.count("Edge By Index Scan"), 2)

        query_result = redis_graph.query(query)
        # Two new nodes should be created.
        self.env.assertEquals(query_result.relationships_created, 2)

    def test10_runtime_index_utilization(self):
        # find all person nodes with age in the range 33-37
        # current age (x) should be resolved at runtime
        # index query should be constructed for each age value
        q = """UNWIND range(33, 37) AS x
        MATCH (n)-[f:friend {created_at: x}]->()
        RETURN n.name
        ORDER BY n.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Edge By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [['Ailon'], ['Ailon'], ['Boaz']]
        self.env.assertEquals(query_result.result_set, expected_result)

        # similar to the query above, only this time the filter is specified
        # by an OR condition
        q = """WITH 33 AS min, 37 AS max 
        MATCH (n)-[f:friend]->()
        WHERE f.created_at = min OR f.created_at = max
        RETURN n.name
        ORDER BY n.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Edge By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [['Ailon'], ['Boaz']]
        self.env.assertEquals(query_result.result_set, expected_result)

        # find all person nodes with age equals 33 'x'
        # 'x' value is known only at runtime
        q = """WITH 33 AS x
        MATCH (n)-[f:friend {created_at: x}]->()
        RETURN n.name
        ORDER BY n.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Edge By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [["Ailon"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # find all person nodes with age equals x + 1
        # the expression x+1 is evaluated to the constant 33 only at runtime
        # expecting index query to be constructed at runtime
        q = """WITH 32 AS x
        MATCH (n)-[f:friend]->()
        WHERE f.created_at = (x + 1)
        RETURN n.name
        ORDER BY n.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Edge By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [["Ailon"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # same idea as previous query only we've switched the position of the
        # operands, queried entity (p.age) is now on the right hand side of the
        # filter, expecting the same behavior
        q = """WITH 32 AS x
        MATCH (n)-[f:friend]->()
        WHERE (x + 1) = f.created_at
        RETURN n.name
        ORDER BY n.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Edge By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [["Ailon"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # make sure all node scan not removed because we need to filter
        q = """MATCH (a)-[e:friend]->()
        WHERE a.created_at > 5 AND e.created_at > a.created_at
        RETURN DISTINCT a.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Filter', plan)
        self.env.assertIn('All Node Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [["Ori"]]
        self.env.assertEquals(query_result.result_set, expected_result)

    def test11_index_scan_and_label_filter(self):
        query = "MATCH (n)-[f:friend]->(m) WHERE f.created_at = 1 RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertNotIn('All Node Scan', plan)
        self.env.assertNotIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)-[f:friend]->(m) WHERE f.created_at = 1 RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertNotIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)-[f:friend]->(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Roi'})-[f:friend]->(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Alon'})-[f:friend]->(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        self.env.assertEquals(query_result.result_set, [])

        query = "MATCH (n)<-[f:friend]-(m) WHERE f.created_at = 1 RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertNotIn('All Node Scan', plan)
        self.env.assertNotIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)<-[f:friend]-(m) WHERE f.created_at = 1 RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertNotIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)<-[f:friend]-(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Roi'})<-[f:friend]-(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        self.env.assertEquals(query_result.result_set, [])

        query = "MATCH (n:person {name: 'Alon'})<-[f:friend]-(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

    def test12_index_scan_and_with(self):
        query = "MATCH (n)-[f:friend]->(m) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertNotIn('All Node Scan', plan)
        self.env.assertNotIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)-[f:friend]->(m) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertNotIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)-[f:friend]->(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Roi'})-[f:friend]->(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Alon'})-[f:friend]->(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        self.env.assertEquals(query_result.result_set, [])

        query = "MATCH (n)<-[f:friend]-(m) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertNotIn('All Node Scan', plan)
        self.env.assertNotIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)<-[f:friend]-(m) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertNotIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)<-[f:friend]-(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Roi'})<-[f:friend]-(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        self.env.assertEquals(query_result.result_set, [])

        query = "MATCH (n:person {name: 'Alon'})<-[f:friend]-(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Edge By Index Scan', plan)
        self.env.assertIn('Node By Label Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

    def test13_index_scan_numeric_accuracy(self):
        redis_graph = Graph(self.env.getConnection(), 'large_index_values')

        create_edge_exact_match_index(redis_graph, 'R1', 'id', sync=True)
        create_edge_exact_match_index(redis_graph, 'R2', 'id1', 'id2', sync=True)
        redis_graph.query("UNWIND range(1, 5) AS v CREATE ()-[:R1 {id: 990000000262240068 + v}]->()")
        redis_graph.query("UNWIND range(1, 5) AS v CREATE ()-[:R2 {id1: 990000000262240068 + v, id2: 990000000262240068 - v}]->()")

        # test index search
        result = redis_graph.query("MATCH ()-[u:R1{id: 990000000262240069}]->() RETURN u.id")
        expected_result = [[990000000262240069]]
        self.env.assertEquals(result.result_set, expected_result)

        # test index search from child
        result = redis_graph.query("MATCH ()-[u:R1]->() WITH min(u.id) as id MATCH ()-[u:R1{id: id}]->() RETURN u.id")
        expected_result = [[990000000262240069]]
        self.env.assertEquals(result.result_set, expected_result)

        # test index search with or
        result = redis_graph.query("MATCH ()-[u:R1]->() WHERE u.id = 990000000262240069 OR u.id = 990000000262240070 RETURN u.id ORDER BY u.id")
        expected_result = [[990000000262240069], [990000000262240070]]
        self.env.assertEquals(result.result_set, expected_result)

        # test resetting index scan operation
        result = redis_graph.query("MATCH ()-[u1:R1]->(), ()-[u2:R1]->() WHERE u1.id = 990000000262240069 AND (u2.id = 990000000262240070 OR u2.id = 990000000262240071) RETURN u1.id, u2.id ORDER BY u1.id, u2.id")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEquals(result.result_set, expected_result)

        # test resetting index scan operation when using the consume from child function
        result = redis_graph.query("MATCH ()-[u:R1]->() WITH min(u.id) as id MATCH ()-[u1:R1]->(), ()-[u2:R1]->() WHERE u1.id = 990000000262240069 AND (u2.id = 990000000262240070 OR u2.id = 990000000262240071) RETURN u1.id, u2.id ORDER BY u1.id, u2.id")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEquals(result.result_set, expected_result)

        # test resetting index scan operation when rebuild index is required
        result = redis_graph.query("MATCH ()-[u:R1]->() WITH min(u.id) as id MATCH ()-[u1:R1]->(), ()-[u2:R1]->() WHERE u1.id = id AND (u2.id = 990000000262240070 OR u2.id = 990000000262240071) RETURN u1.id, u2.id ORDER BY u1.id, u2.id")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEquals(result.result_set, expected_result)

        # test index scan with 2 different attributes
        result = redis_graph.query("MATCH ()-[u:R2]->() WHERE u.id1 = 990000000262240069 AND u.id2 = 990000000262240067 RETURN u.id1, u.id2")
        expected_result = [[990000000262240069, 990000000262240067]]
        self.env.assertEquals(result.result_set, expected_result)

    def test14_create_index_multi_edge(self):
        redis_graph = Graph(self.env.getConnection(), 'index_multi_edge')

        result = redis_graph.query("CREATE (a:A), (b:B)")
        self.env.assertEquals(result.nodes_created, 2)

        result = redis_graph.query("MATCH (a:A), (b:B) UNWIND range(1, 500) AS x CREATE (a)-[:R{v:x}]->(b)")
        self.env.assertEquals(result.relationships_created, 500)

        result = create_edge_exact_match_index(redis_graph, 'R', 'v', sync=True)
        self.env.assertEquals(result.indices_created, 1)

        result = redis_graph.query("MATCH (a:A)-[r:R]->(b:B) WHERE r.v > 0 RETURN count(r)")
        self.env.assertEquals(result.result_set[0][0], 500)

    def test15_self_referencing_edge(self):
        # make sure edge connecting node 0 to itself is indexed
        # (0)->(0)
        g = Graph(self.env.getConnection(), 'self_ref_edge')

        res = g.query("CREATE (a)-[e:R{v:1}]->(a) RETURN a, e")
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.relationships_created, 1)

        # validate IDs
        self.env.assertEquals(res.result_set[0][0].id, 0)
        self.env.assertEquals(res.result_set[0][1].id, 0)

        # create index over R.v
        create_edge_exact_match_index(g, "R", "v", sync=True)

        # make sure edge can be located via index scan
        q = "MATCH ()-[e:R{v:1}]->() RETURN e"

        # validate index is utilized
        plan = g.execution_plan(q)
        self.env.assertIn("Edge By Index Scan", plan)

        # get result using index scan
        res = g.query(q)
        self.env.assertEquals(len(res.result_set), 1)
        actual = res.result_set

        # get results without index
        res = g.query("MATCH ()-[e]->() RETURN e")
        expected = res.result_set

        # make sure the same edge is returned
        self.env.assertEquals(expected, actual)

