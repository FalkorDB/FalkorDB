from common import *
from index_utils import *

GRAPH_ID = "edge_index_scan"

people = ["Roi", "Alon", "Ailon", "Boaz", "Tal", "Omri", "Ori"]

class testEdgeByIndexScanFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    def setUp(self):
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()
        self.build_indices()

    def tearDown(self):
        self.graph.delete()
    
    def populate_graph(self):
        nodes = {}

        # Create entities
        node_id = 0
        for idx, p in enumerate(people):
            node = Node(alias=f"n_{idx}", labels="person", properties={"name": p, "created_at": node_id})
            nodes[node.alias] = node
            node_id = node_id + 1
        nodes_str = [str(node) for node in nodes.values()]

        # Fully connected graph
        edges = []
        edge_id = 0
        for src in nodes:
            for dest in nodes:
                if src != dest:
                    edge = Edge(nodes[src], "knows", nodes[dest], properties={"created_at": edge_id * 2})
                    edges.append(edge)
                    edge = Edge(nodes[src], "friend", nodes[dest], properties={"created_at": edge_id * 2 + 1, "updated_at": edge_id * 3})
                    edges.append(edge)
                    edge_id = edge_id + 1
        edges_str = [str(edge) for edge in edges]

        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")

    def build_indices(self):
        create_node_range_index(self.graph, "person", "age")
        create_edge_range_index(self.graph, "friend", "created_at")
        create_edge_range_index(self.graph, "knows", "created_at", sync=True)

    # Validate that Cartesian products using index and label scans succeed
    def test01_cartesian_product_mixed_scans(self):
        query = "MATCH ()-[f:friend]->(), ()-[k:knows]->() WHERE f.created_at >= 0 RETURN f.created_at, k.created_at ORDER BY f.created_at, k.created_at"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Conditional Traverse', plan)
        indexed_result = self.graph.query(query)

        query = "MATCH ()-[f:friend]->(), ()-[k:knows]->() RETURN f.created_at, k.created_at ORDER BY f.created_at, k.created_at"
        plan = str(self.graph.explain(query))
        self.env.assertNotContains('Edge By Index Scan', plan)
        self.env.assertContains('Conditional Traverse', plan)
        unindexed_result = self.graph.query(query)

        self.env.assertEqual(indexed_result.result_set, unindexed_result.result_set)

    # Validate that Cartesian products using just index scans succeed
    def test02_cartesian_product_index_scans_only(self):
        query = "MATCH ()-[f:friend]->(), ()-[k:knows]->() WHERE f.created_at >= 0 AND k.created_at >= 0 RETURN f.created_at, k.created_at ORDER BY f.created_at, k.created_at"
        plan = str(self.graph.explain(query))
        # The two streams should both use index scans
        self.env.assertEqual(plan.count('Edge By Index Scan'), 2)
        self.env.assertNotContains('Conditional Traverse', plan)
        indexed_result = self.graph.query(query)

        query = "MATCH ()-[f:friend]->(), ()-[k:knows]->() RETURN f.created_at, k.created_at ORDER BY f.created_at, k.created_at"
        plan = str(self.graph.explain(query))
        self.env.assertNotContains('Edge By Index Scan', plan)
        self.env.assertContains('Conditional Traverse', plan)
        unindexed_result = self.graph.query(query)

        self.env.assertEqual(indexed_result.result_set, unindexed_result.result_set)

    # Validate that the appropriate bounds are respected when a Cartesian product uses the same index in two streams
    def test03_cartesian_product_reused_index(self):
        create_edge_range_index(self.graph, 'friend', 'updated_at', sync=True)
        query = """MATCH ()-[a:friend]->(), ()-[b:friend]->()
                   WHERE a.created_at >= 80 AND b.updated_at >= 120
                   RETURN a.created_at, b.updated_at
                   ORDER BY a.created_at, b.updated_at"""
        plan = str(self.graph.explain(query))
        # The two streams should both use index scans
        self.env.assertEqual(plan.count('Edge By Index Scan'), 2)
        self.env.assertNotContains('Conditional Traverse', plan)


        expected_result = [[81, 120], [81, 123], [83, 120], [83, 123]]
        result = self.graph.query(query)

        self.env.assertEqual(result.result_set, expected_result)

    # Validate index utilization when filtering on a numeric field with the `IN` keyword.
    def test04_test_in_operator_numerics(self):
        # Validate the transformation of IN to multiple OR expressions.
        query = "MATCH ()-[f:friend]-() WHERE f.created_at IN [1,2,3] RETURN f"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)

        # Validate that nested arrays are not scanned in index.
        query = "MATCH ()-[f:friend]-() WHERE f.created_at IN [[1,2],3] RETURN f"
        plan = str(self.graph.explain(query))
        self.env.assertNotContains('Edge By Index Scan', plan)
        self.env.assertContains('Conditional Traverse', plan)

        # Validate the transformation of IN to multiple OR, over a range.
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at IN range(0,30) RETURN DISTINCT n.name ORDER BY n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)

        expected_result = [['Ailon'], ['Alon'], ['Roi']]
        result = self.graph.query(query)
        self.env.assertEqual(result.result_set, expected_result)

         # Validate the transformation of IN to empty index iterator.
        query = "MATCH ()-[f:friend]-() WHERE f.created_at IN [] RETURN f.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)

        expected_result = []
        result = self.graph.query(query)
        self.env.assertEqual(result.result_set, expected_result)

        # Validate the transformation of IN OR IN to empty index iterators.
        query = "MATCH ()-[f:friend]->() WHERE f.created_at IN [] OR f.created_at IN [] RETURN f.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)

        expected_result = []
        result = self.graph.query(query)
        self.env.assertEqual(result.result_set, expected_result)

        # Validate the transformation of multiple IN filters.
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at IN [0, 1, 2] OR f.created_at IN [14, 15, 16] RETURN n.name ORDER BY n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)

        expected_result = [['Alon'], ['Roi']]
        result = self.graph.query(query)
        self.env.assertEqual(result.result_set, expected_result)

        # Validate the transformation of multiple IN filters.
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at IN [0, 1, 2] OR f.created_at IN [14, 15, 16] OR f.created_at IN [] RETURN n.name ORDER BY n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)

        result = self.graph.query(query)
        self.env.assertEqual(result.result_set, expected_result)

    def test05_index_scan_and_id(self):
        query = """MATCH (n)-[f:friend]->()
                   WHERE id(f)>=10 AND f.created_at<15
                   RETURN n.name
                   ORDER BY n.name"""
        plan = str(self.graph.explain(query))
        query_result = self.graph.query(query)
        self.env.assertContains('Filter', plan)
        self.env.assertContains('Edge By Index Scan', plan)

        self.env.assertEqual(2, len(query_result.result_set))
        expected_result = [['Alon'], ['Roi']]
        self.env.assertEqual(expected_result, query_result.result_set)

    # Validate placement of index scans and filter ops when not all filters can be replaced.
    def test06_index_scan_multiple_filters(self):
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at = 31 AND NOT EXISTS(f.fakeprop) RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertNotContains('Conditional Traverse', plan)
        self.env.assertContains('Filter', plan)

        query_result = self.graph.query(query)
        expected_result = ["Ailon"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH ({created_at:1})-[:friend {created_at:31}]->({created_at:2}) RETURN 1"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertNotContains('Conditional Traverse', plan)
        self.env.assertContains('Filter', plan)

    def test07_index_scan_with_params(self):
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at = $time RETURN n.name"
        params = {'time': 31}
        plan = str(self.graph.explain(query, params=params))
        self.env.assertContains('Edge By Index Scan', plan)
        query_result = self.graph.query(query, params=params)
        expected_result = ["Ailon"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

    def test08_index_scan_with_param_array(self):
        query = "MATCH (n)-[f:friend]->() WHERE f.created_at in $times RETURN n.name"
        params = {'times': [31]}
        plan = str(self.graph.explain(query, params=params))
        self.env.assertContains('Edge By Index Scan', plan)
        query_result = self.graph.query(query, params=params)
        expected_result = ["Ailon"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

    def test09_runtime_index_utilization(self):
        # find all person nodes with age in the range 33-37
        # current age (x) should be resolved at runtime
        # index query should be constructed for each age value
        q = """UNWIND range(33, 37) AS x
        MATCH (n)-[f:friend {created_at: x}]->()
        RETURN n.name
        ORDER BY n.name"""
        plan = str(self.graph.explain(q))
        self.env.assertContains('Edge By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [['Ailon'], ['Ailon'], ['Boaz']]
        self.env.assertEqual(query_result.result_set, expected_result)

        # similar to the query above, only this time the filter is specified
        # by an OR condition
        q = """WITH 33 AS min, 37 AS max 
        MATCH (n)-[f:friend]->()
        WHERE f.created_at = min OR f.created_at = max
        RETURN n.name
        ORDER BY n.name"""
        plan = str(self.graph.explain(q))
        self.env.assertContains('Edge By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [['Ailon'], ['Boaz']]
        self.env.assertEqual(query_result.result_set, expected_result)

        # find all person nodes with age equals 33 'x'
        # 'x' value is known only at runtime
        q = """WITH 33 AS x
        MATCH (n)-[f:friend {created_at: x}]->()
        RETURN n.name
        ORDER BY n.name"""
        plan = str(self.graph.explain(q))
        self.env.assertContains('Edge By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["Ailon"]]
        self.env.assertEqual(query_result.result_set, expected_result)

        # find all person nodes with age equals x + 1
        # the expression x+1 is evaluated to the constant 33 only at runtime
        # expecting index query to be constructed at runtime
        q = """WITH 32 AS x
        MATCH (n)-[f:friend]->()
        WHERE f.created_at = (x + 1)
        RETURN n.name
        ORDER BY n.name"""
        plan = str(self.graph.explain(q))
        self.env.assertContains('Edge By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["Ailon"]]
        self.env.assertEqual(query_result.result_set, expected_result)

        # same idea as previous query only we've switched the position of the
        # operands, queried entity (p.age) is now on the right hand side of the
        # filter, expecting the same behavior
        q = """WITH 32 AS x
        MATCH (n)-[f:friend]->()
        WHERE (x + 1) = f.created_at
        RETURN n.name
        ORDER BY n.name"""
        plan = str(self.graph.explain(q))
        self.env.assertContains('Edge By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["Ailon"]]
        self.env.assertEqual(query_result.result_set, expected_result)

        # make sure all node scan not removed because we need to filter
        q = """MATCH (a)-[e:friend]->()
        WHERE a.created_at > 5 AND e.created_at > a.created_at
        RETURN DISTINCT a.name"""
        plan = str(self.graph.explain(q))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Filter', plan)
        self.env.assertContains('All Node Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["Ori"]]
        self.env.assertEqual(query_result.result_set, expected_result)

    def test10_index_scan_and_label_filter(self):
        query = "MATCH (n)-[f:friend]->(m) WHERE f.created_at = 1 RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertNotContains('All Node Scan', plan)
        self.env.assertNotContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)-[f:friend]->(m) WHERE f.created_at = 1 RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertNotContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)-[f:friend]->(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Roi'})-[f:friend]->(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Alon'})-[f:friend]->(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        self.env.assertEqual(query_result.result_set, [])

        query = "MATCH (n)<-[f:friend]-(m) WHERE f.created_at = 1 RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertNotContains('All Node Scan', plan)
        self.env.assertNotContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)<-[f:friend]-(m) WHERE f.created_at = 1 RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertNotContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)<-[f:friend]-(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Roi'})<-[f:friend]-(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        self.env.assertEqual(query_result.result_set, [])

        query = "MATCH (n:person {name: 'Alon'})<-[f:friend]-(m:person) WHERE f.created_at = 1 RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

    def test11_index_scan_and_with(self):
        query = "MATCH (n)-[f:friend]->(m) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertNotContains('All Node Scan', plan)
        self.env.assertNotContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)-[f:friend]->(m) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertNotContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)-[f:friend]->(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Roi'})-[f:friend]->(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Roi"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Alon'})-[f:friend]->(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        self.env.assertEqual(query_result.result_set, [])

        query = "MATCH (n)<-[f:friend]-(m) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertNotContains('All Node Scan', plan)
        self.env.assertNotContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)<-[f:friend]-(m) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertNotContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person)<-[f:friend]-(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

        query = "MATCH (n:person {name: 'Roi'})<-[f:friend]-(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        self.env.assertEqual(query_result.result_set, [])

        query = "MATCH (n:person {name: 'Alon'})<-[f:friend]-(m:person) WHERE f.created_at = 1 WITH n RETURN n.name"
        plan = str(self.graph.explain(query))
        self.env.assertContains('Edge By Index Scan', plan)
        self.env.assertContains('Node By Label Scan', plan)
        self.env.assertContains('Filter', plan)
        query_result = self.graph.query(query)
        expected_result = ["Alon"]
        self.env.assertEqual(query_result.result_set[0], expected_result)

    def test12_index_scan_numeric_accuracy(self):
        create_edge_range_index(self.graph, 'R1', 'id', sync=True)
        create_edge_range_index(self.graph, 'R2', 'id1', 'id2', sync=True)
        self.graph.query("UNWIND range(1, 5) AS v CREATE ()-[:R1 {id: 990000000262240068 + v}]->()")
        self.graph.query("UNWIND range(1, 5) AS v CREATE ()-[:R2 {id1: 990000000262240068 + v, id2: 990000000262240068 - v}]->()")

        # test index search
        result = self.graph.query("MATCH ()-[u:R1{id: 990000000262240069}]->() RETURN u.id")
        expected_result = [[990000000262240069]]
        self.env.assertEqual(result.result_set, expected_result)

        # test index search from child
        result = self.graph.query("MATCH ()-[u:R1]->() WITH min(u.id) as id MATCH ()-[u:R1{id: id}]->() RETURN u.id")
        expected_result = [[990000000262240069]]
        self.env.assertEqual(result.result_set, expected_result)

        # test index search with or
        result = self.graph.query("MATCH ()-[u:R1]->() WHERE u.id = 990000000262240069 OR u.id = 990000000262240070 RETURN u.id ORDER BY u.id")
        expected_result = [[990000000262240069], [990000000262240070]]
        self.env.assertEqual(result.result_set, expected_result)

        # test resetting index scan operation
        result = self.graph.query("MATCH ()-[u1:R1]->(), ()-[u2:R1]->() WHERE u1.id = 990000000262240069 AND (u2.id = 990000000262240070 OR u2.id = 990000000262240071) RETURN u1.id, u2.id ORDER BY u1.id, u2.id")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEqual(result.result_set, expected_result)

        # test resetting index scan operation when using the consume from child function
        result = self.graph.query("MATCH ()-[u:R1]->() WITH min(u.id) as id MATCH ()-[u1:R1]->(), ()-[u2:R1]->() WHERE u1.id = 990000000262240069 AND (u2.id = 990000000262240070 OR u2.id = 990000000262240071) RETURN u1.id, u2.id ORDER BY u1.id, u2.id")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEqual(result.result_set, expected_result)

        # test resetting index scan operation when rebuild index is required
        result = self.graph.query("MATCH ()-[u:R1]->() WITH min(u.id) as id MATCH ()-[u1:R1]->(), ()-[u2:R1]->() WHERE u1.id = id AND (u2.id = 990000000262240070 OR u2.id = 990000000262240071) RETURN u1.id, u2.id ORDER BY u1.id, u2.id")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEqual(result.result_set, expected_result)

        # test index scan with 2 different attributes
        result = self.graph.query("MATCH ()-[u:R2]->() WHERE u.id1 = 990000000262240069 AND u.id2 = 990000000262240067 RETURN u.id1, u.id2")
        expected_result = [[990000000262240069, 990000000262240067]]
        self.env.assertEqual(result.result_set, expected_result)

    def test13_create_index_multi_edge(self):
        result = self.graph.query("CREATE (a:A), (b:B)")
        self.env.assertEqual(result.nodes_created, 2)

        result = self.graph.query("MATCH (a:A), (b:B) UNWIND range(1, 500) AS x CREATE (a)-[:R{v:x}]->(b)")
        self.env.assertEqual(result.relationships_created, 500)

        result = create_edge_range_index(self.graph, 'R', 'v', sync=True)
        self.env.assertEqual(result.indices_created, 1)

        result = self.graph.query("MATCH (a:A)-[r:R]->(b:B) WHERE r.v > 0 RETURN count(r)")
        self.env.assertEqual(result.result_set[0][0], 500)

    def test14_self_referencing_edge(self):
        self.graph.delete()
        # make sure edge connecting node 0 to itself is indexed
        # (0)->(0)

        res = self.graph.query("CREATE (a)-[e:R{v:1}]->(a) RETURN a, e")
        self.env.assertEqual(res.nodes_created, 1)
        self.env.assertEqual(res.relationships_created, 1)

        # validate IDs
        self.env.assertEqual(res.result_set[0][0].id, 0)
        self.env.assertEqual(res.result_set[0][1].id, 0)

        # create index over R.v
        create_edge_range_index(self.graph, "R", "v", sync=True)

        # make sure edge can be located via index scan
        q = "MATCH ()-[e:R{v:1}]->() RETURN e"

        # validate index is utilized
        plan = str(self.graph.explain(q))
        self.env.assertContains("Edge By Index Scan", plan)

        # get result using index scan
        res = self.graph.query(q)
        self.env.assertEqual(len(res.result_set), 1)
        actual = res.result_set

        # get results without index
        res = self.graph.query("MATCH ()-[e]->() RETURN e")
        expected = res.result_set

        # make sure the same edge is returned
        self.env.assertEqual(expected, actual)


# Regression tests for PR #393 review feedback. Placed in their own
# class so they can own a dedicated graph without polluting the main
# suite's setup.
class testEdgeByIndexScanRegressionsFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    def test15_selectivity_large_type(self):
        """
        Creating >BATCH_SIZE (10 000) edges of a single type and then
        running a highly selective `WHERE r.v = …` query must:
          - complete the background population cursor without losing
            any edges (regresses the edge-id-as-row-cursor bug);
          - materialize endpoints directly from the index without a
            full tensor scan (regresses the O(|E_type|) path — the
            query should be comfortably sub-second on 10k edges).
        """
        g = self.db.select_graph("edge_index_selectivity")
        try:
            n = 10_500  # BATCH_SIZE + margin so population spans > 1 batch
            g.query(f"UNWIND range(0, {n - 1}) AS i CREATE ()-[:T {{v: i}}]->()")
            create_edge_range_index(g, "T", "v", sync=True)

            q = "MATCH ()-[r:T]->() WHERE r.v = 7777 RETURN r.v"
            plan = str(g.explain(q))
            self.env.assertContains("Edge By Index Scan", plan)

            res = g.query(q)
            self.env.assertEqual(res.result_set, [[7777]])
        finally:
            g.delete()

    def test16_rewrite_gating(self):
        """
        The rewriter must NOT substitute `EdgeByIndexScan` for
        multi-type OR patterns. These cases the operator can't serve
        faithfully, so the plan should keep `CondTraverse`.
        (Bidirectional `-[]-` with a single type IS supported and
        exercised elsewhere in this file — see test04.)
        """
        g = self.db.select_graph("edge_index_gating")
        try:
            g.query("CREATE ()-[:A {v: 1}]->()")
            g.query("CREATE ()-[:B {v: 1}]->()")
            create_edge_range_index(g, "A", "v")
            create_edge_range_index(g, "B", "v", sync=True)

            # Multi-type OR: two indexes, our op only serves one.
            q = "MATCH ()-[r:A|B]->() WHERE r.v = 1 RETURN r"
            plan = str(g.explain(q))
            self.env.assertNotContains("Edge By Index Scan", plan)
            self.env.assertContains("Conditional Traverse", plan)
        finally:
            g.delete()

    def test17_child_var_reference_safety(self):
        """
        `prune_all_node_scan_child` must not drop the source scan when
        the edge-index query still references the scan's output.
        `WHERE r.p = n.q` references `n` through `r.p`'s peer — we
        don't currently rewrite patterns whose filter depends on the
        child, but this test guards the invariant that rewrites which
        DO fire never strand child-bound variables.
        """
        g = self.db.select_graph("edge_index_child_ref")
        try:
            g.query("CREATE (a {q: 7})-[:T {p: 7}]->(b)")
            g.query("CREATE (a {q: 8})-[:T {p: 8}]->(b)")
            create_edge_range_index(g, "T", "p", sync=True)

            q = "MATCH (n)-[r:T]->() WHERE r.p = n.q RETURN r.p ORDER BY r.p"
            # Must execute correctly regardless of whether the
            # rewriter fires. We don't assert plan shape here — only
            # correctness — because the optimizer is free to keep
            # CondTraverse when the filter references `n`.
            res = g.query(q)
            self.env.assertEqual(res.result_set, [[7], [8]])
        finally:
            g.delete()

    def test18_non_indexable_literal_retains_filter(self):
        """
        `WHERE r.v = [1,2,3]` produces an `IndexQuery::Equal` whose
        value is a list — the runtime `can_utilize_index` correctly
        rejects this, but the optimizer must retain the original
        Filter above the index scan so the fallback path doesn't
        return every edge of the type.
        """
        g = self.db.select_graph("edge_index_non_indexable_literal")
        try:
            g.query("CREATE ()-[:T {v: 1}]->()")
            g.query("CREATE ()-[:T {v: 2}]->()")
            create_edge_range_index(g, "T", "v", sync=True)

            # No row should match since `r.v` is a scalar int, never a list.
            q = "MATCH ()-[r:T]->() WHERE r.v = [1, 2, 3] RETURN r.v"
            res = g.query(q)
            self.env.assertEqual(res.result_set, [])
        finally:
            g.delete()

    def test19_cascade_delete_cleans_edge_index(self):
        """
        Deleting a node must remove its incident indexed edges from
        the edge index (via `delete_implicit_edges`), not just the
        relationship tensor. Otherwise a subsequent index query sees
        stale (src, dst) triples for edges that no longer exist.
        """
        g = self.db.select_graph("edge_index_cascade_delete")
        try:
            g.query("CREATE (:N {id: 1})-[:T {v: 100}]->(:N {id: 2})")
            g.query("CREATE (:N {id: 3})-[:T {v: 200}]->(:N {id: 4})")
            create_edge_range_index(g, "T", "v", sync=True)

            # Pre-sanity: both edges are visible via the index.
            q = "MATCH ()-[r:T]->() WHERE r.v > 0 RETURN r.v ORDER BY r.v"
            res = g.query(q)
            self.env.assertEqual(res.result_set, [[100], [200]])

            # Cascade-delete one endpoint; the incident T-edge is
            # implicitly deleted.
            g.query("MATCH (n:N {id: 1}) DELETE n")

            # The remaining edge is still indexed; the deleted edge
            # must not appear.
            res = g.query(q)
            self.env.assertEqual(res.result_set, [[200]])
        finally:
            g.delete()

    def test20_pre_bound_destination_respected(self):
        """
        When the child has already bound `rp.to.alias` (e.g. via a
        prior MATCH component), the edge scan must filter by both
        endpoints — not just `from` — so edges pointing at the wrong
        destination don't leak through and clobber the binding.
        """
        g = self.db.select_graph("edge_index_pre_bound_to")
        try:
            g.query("""CREATE
                      (a {tag: 'a'}),
                      (b {tag: 'b'}),
                      (c {tag: 'c'}),
                      (a)-[:T {v: 1}]->(b),
                      (a)-[:T {v: 2}]->(c)""")
            create_edge_range_index(g, "T", "v", sync=True)

            q = """MATCH (src {tag: 'a'}), (dst {tag: 'b'})
                   MATCH (src)-[r:T]->(dst)
                   WHERE r.v > 0
                   RETURN r.v"""
            res = g.query(q)
            # Only the edge to `b` (v=1) should be returned —
            # the edge to `c` (v=2) has the wrong destination.
            self.env.assertEqual(res.result_set, [[1]])
        finally:
            g.delete()

    def test21_sparse_attribute_population(self):
        """
        Background edge-index population must not declare a type
        "exhausted" just because few docs were emitted in the current
        batch. Here only every 100th edge carries the indexed attr,
        so if `exhausted` were based on `batch.len()` (docs produced)
        vs. `BATCH_SIZE`, the population would stop prematurely and
        later hits would be missing.
        """
        g = self.db.select_graph("edge_index_sparse_attr")
        try:
            # 15_000 edges > BATCH_SIZE (10k); only the ~150 multiples
            # of 100 carry the indexed `v` attr, the rest have none.
            g.query(
                "UNWIND range(0, 14999) AS i "
                "CREATE ()-[:T]->()"
            )
            g.query(
                "MATCH ()-[r:T]->() "
                "WITH r, ID(r) AS eid "
                "WHERE eid % 100 = 0 "
                "SET r.v = eid"
            )
            create_edge_range_index(g, "T", "v", sync=True)

            # Grab an indexed value that lives deep in the second
            # batch (by source-row ordering). It must be findable.
            q = "MATCH ()-[r:T]->() WHERE r.v = 14900 RETURN r.v"
            res = g.query(q)
            self.env.assertEqual(res.result_set, [[14900]])
        finally:
            g.delete()

    def test22_in_list_with_parameter(self):
        """
        Regression for the narrowed `needs_post_filter` whitelist:
        `WHERE r.v IN $lst` must retain the filter because the list
        is a runtime `Parameter`, not a scalar-literal `ExprIR::List`.
        At runtime the parameter could evaluate to non-indexable
        elements (e.g. a list of dates); the retained filter keeps
        correctness even when the index path devolves to `Or([])`.
        """
        g = self.db.select_graph("edge_index_in_param")
        try:
            g.query("CREATE ()-[:T {v: 1}]->()")
            g.query("CREATE ()-[:T {v: 2}]->()")
            create_edge_range_index(g, "T", "v", sync=True)

            # A parameter list that would produce an empty index
            # query after non-scalar filtering. No row should match.
            q = "MATCH ()-[r:T]->() WHERE r.v IN $lst RETURN r.v"
            res = g.query(q, {"lst": []})
            self.env.assertEqual(res.result_set, [])

            # Same parameter shape but legal values — correctness
            # must hold on the index path too.
            res = g.query(q, {"lst": [1]})
            self.env.assertEqual(res.result_set, [[1]])
        finally:
            g.delete()

    def test23_self_loop_alias(self):
        """
        Regression for self-loop MATCH patterns:
        `MATCH (n)-[r:T]->(n)` shares an alias on both endpoints.
        Without the `from_id == to_id` filter, every edge of the
        type would match (both endpoints leak through unbound) and
        `drain_pending` would overwrite the from-alias with the
        destination value.
        """
        g = self.db.select_graph("edge_index_self_loop")
        try:
            # Three edges: one self-loop + two non-loops.
            g.query("CREATE (a {tag: 'a'})-[:T {v: 1}]->(a)")
            g.query("CREATE (b {tag: 'b'})-[:T {v: 2}]->(c {tag: 'c'})")
            g.query("CREATE (d {tag: 'd'})-[:T {v: 3}]->(e {tag: 'e'})")
            create_edge_range_index(g, "T", "v", sync=True)

            # Only the self-loop should match.
            q = "MATCH (n)-[r:T]->(n) WHERE r.v > 0 RETURN n.tag, r.v"
            res = g.query(q)
            self.env.assertEqual(res.result_set, [["a", 1]])
        finally:
            g.delete()

    def test25_in_list_precision_losing_int(self):
        """
        Regression for the scalar-literal whitelist: an int64 that
        can't round-trip through f64 is non-indexable at runtime.
        `WHERE r.v IN [<big int>]` must keep the filter — otherwise
        the runtime rejects the query and the fallback returns every
        edge of the type.
        """
        g = self.db.select_graph("edge_index_precision_int_in_list")
        try:
            g.query("CREATE ()-[:T {v: 1}]->()")
            g.query("CREATE ()-[:T {v: 2}]->()")
            create_edge_range_index(g, "T", "v", sync=True)

            # 2^53 + 1 is the smallest positive int64 that can't
            # round-trip through f64. No edge has `v` equal to it,
            # so the query must return zero rows.
            big = 9007199254740993
            q = f"MATCH ()-[r:T]->() WHERE r.v IN [{big}] RETURN r.v"
            res = g.query(q)
            self.env.assertEqual(res.result_set, [])
        finally:
            g.delete()

    def test24_multi_edge_populate(self):
        """
        Regression for the BatchCursor edge_id tracking:
        `MULTI EDGES` between the same `(src, dst)` pair must all
        be indexed even when a batch boundary falls mid-group.
        Uses a BATCH_SIZE-sized multi-edge group to force the
        cursor to resume inside it.
        """
        g = self.db.select_graph("edge_index_multi_edge_populate")
        try:
            # 11_000 edges between the same (a)-(b) pair — crosses
            # the BATCH_SIZE=10_000 boundary inside one multi-edge
            # group. Each edge carries a unique `v` we can probe.
            g.query("CREATE (a {tag: 'a'}), (b {tag: 'b'})")
            g.query(
                "MATCH (a {tag: 'a'}), (b {tag: 'b'}) "
                "UNWIND range(0, 10999) AS i "
                "CREATE (a)-[:T {v: i}]->(b)"
            )
            create_edge_range_index(g, "T", "v", sync=True)

            # Values on both sides of the 10 000 boundary must be
            # findable through the index.
            for target in [0, 5000, 9999, 10000, 10999]:
                q = f"MATCH ()-[r:T]->() WHERE r.v = {target} RETURN r.v"
                res = g.query(q)
                self.env.assertEqual(res.result_set, [[target]])
        finally:
            g.delete()


# RDB round-trip regression for edge indexes. Needs DEBUG RELOAD so it
# lives in its own class with `enableDebugCommand=True`. Matches
# FalkorDB C's behavior: edge indexes are encoded and restored under
# the relationship-schema block, symmetric with node indexes.
class testEdgeIndexRdbRoundtripFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)

    def test26_rdb_roundtrip_edge_index(self):
        g = self.db.select_graph("edge_index_rdb_roundtrip")
        try:
            g.query("CREATE ()-[:T {v: 1}]->()")
            g.query("CREATE ()-[:T {v: 2}]->()")
            g.query("CREATE ()-[:T {v: 3}]->()")
            create_edge_range_index(g, "T", "v", sync=True)

            probe = "MATCH ()-[r:T]->() WHERE r.v = 2 RETURN r.v"
            plan_before = str(g.explain(probe))
            self.env.assertContains("Edge By Index Scan", plan_before)
            self.env.assertEqual(g.query(probe).result_set, [[2]])

            self.env.dumpAndReload()

            # After reload the edge index must still exist and the
            # planner must still route the probe through it.
            plan_after = str(g.explain(probe))
            self.env.assertContains("Edge By Index Scan", plan_after)
            self.env.assertEqual(g.query(probe).result_set, [[2]])

            # The probe must also see edges inserted after reload —
            # the index is live, not a frozen snapshot.
            g.query("CREATE ()-[:T {v: 42}]->()")
            q = "MATCH ()-[r:T]->() WHERE r.v = 42 RETURN r.v"
            self.env.assertContains("Edge By Index Scan", str(g.explain(q)))
            self.env.assertEqual(g.query(q).result_set, [[42]])
        finally:
            g.delete()

