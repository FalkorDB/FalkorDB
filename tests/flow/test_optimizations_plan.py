from common import *

people = ["Roi", "Alon", "Ailon", "Boaz"]
GRAPH_ID = "optimizations_plan"

class testOptimizationsPlan(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        nodes = {}
        # Create entities
        for idx, p in enumerate(people):
            node = Node(alias=p, labels="person", properties={"name": p, "val": idx})
            nodes[p] = node

        # Fully connected graph
        edges = []
        for src in nodes:
            for dest in nodes:
                if src != dest:
                    edges.append(Edge(nodes[src], "know", nodes[dest]))
                    edges.append(Edge(nodes[src], "works_with", nodes[dest]))

        nodes_str = [str(n) for n in nodes.values()]
        edges_str = [str(e) for e in edges]
        self.graph.query(f"CREATE {','.join(nodes_str+edges_str)}")

        query = """MATCH (a)-[:know]->(b) CREATE (a)-[:know]->(b)"""
        self.graph.query(query)

    def test01_typeless_edge_count(self):
        query = """MATCH ()-[r]->() RETURN COUNT(r)"""
        resultset = self.graph.query(query).result_set
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Project", executionPlan)
        self.env.assertIn("Results", executionPlan)
        self.env.assertNotIn("All Node Scan", executionPlan)
        self.env.assertNotIn("Conditional Traverse", executionPlan)
        self.env.assertNotIn("Aggregate", executionPlan)
        expected = [[36]]
        self.env.assertEqual(resultset, expected)

    def test02_typed_edge_count(self):
        query = """MATCH ()-[r:know]->() RETURN COUNT(r)"""
        resultset = self.graph.query(query).result_set
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Project", executionPlan)
        self.env.assertIn("Results", executionPlan)
        self.env.assertNotIn("All Node Scan", executionPlan)
        self.env.assertNotIn("Conditional Traverse", executionPlan)
        self.env.assertNotIn("Aggregate", executionPlan)
        expected = [[24]]
        self.env.assertEqual(resultset, expected)

    def test03_unknown_typed_edge_count(self):
        query = """MATCH ()-[r:unknown]->() RETURN COUNT(r)"""
        resultset = self.graph.query(query).result_set
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Project", executionPlan)
        self.env.assertIn("Results", executionPlan)
        self.env.assertNotIn("All Node Scan", executionPlan)
        self.env.assertNotIn("Conditional Traverse", executionPlan)
        self.env.assertNotIn("Aggregate", executionPlan)
        expected = [[0]]
        self.env.assertEqual(resultset, expected)

    def test04_typeless_edge_count_with_alias(self):
        query = """MATCH ()-[r]->() RETURN COUNT(r) as c"""
        resultset = self.graph.query(query).result_set
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Project", executionPlan)
        self.env.assertIn("Results", executionPlan)
        self.env.assertNotIn("All Node Scan", executionPlan)
        self.env.assertNotIn("Conditional Traverse", executionPlan)
        self.env.assertNotIn("Aggregate", executionPlan)
        expected = [[36]]
        self.env.assertEqual(resultset, expected)

    def test05_typed_edge_count_with_alias(self):
        query = """MATCH ()-[r:know]->() RETURN COUNT(r) as c"""
        resultset = self.graph.query(query).result_set
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Project", executionPlan)
        self.env.assertIn("Results", executionPlan)
        self.env.assertNotIn("All Node Scan", executionPlan)
        self.env.assertNotIn("Conditional Traverse", executionPlan)
        self.env.assertNotIn("Aggregate", executionPlan)
        expected = [[24]]
        self.env.assertEqual(resultset, expected)

    def test06_multiple_typed_edge_count_with_alias(self):
        query = """MATCH ()-[r:know | :works_with]->() RETURN COUNT(r) as c"""
        resultset = self.graph.query(query).result_set
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Project", executionPlan)
        self.env.assertIn("Results", executionPlan)
        self.env.assertNotIn("All Node Scan", executionPlan)
        self.env.assertNotIn("Conditional Traverse", executionPlan)
        self.env.assertNotIn("Aggregate", executionPlan)
        expected = [[36]]
        self.env.assertEqual(resultset, expected)

    def test07_count_unreferenced_edge(self):
        query = """MATCH ()-[:know]->(b) RETURN COUNT(b)"""
        # This count in this query cannot be reduced, as the traversal op doesn't store
        # data about non-referenced edges.
        resultset = self.graph.query(query).result_set
        executionPlan = str(self.graph.explain(query))
        # Verify that the optimization was not applied.
        self.env.assertNotIn("Project", executionPlan)
        self.env.assertIn("Aggregate", executionPlan)
        self.env.assertIn("All Node Scan", executionPlan)
        self.env.assertIn("Conditional Traverse", executionPlan)
        expected = [[12]]
        self.env.assertEqual(resultset, expected)

    def test08_non_labeled_node_count(self):
        query = """MATCH (n) RETURN COUNT(n)"""
        resultset = self.graph.query(query).result_set
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Project", executionPlan)
        self.env.assertIn("Results", executionPlan)
        self.env.assertNotIn("All Node Scan", executionPlan)
        self.env.assertNotIn("Node By Label Scan", executionPlan)
        self.env.assertNotIn("Aggregate", executionPlan)
        expected = [[4]]
        self.env.assertEqual(resultset, expected)

    def test09_non_labeled_node_count_with_alias(self):
        query = """MATCH (n) RETURN COUNT(n) as c"""
        resultset = self.graph.query(query).result_set
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Project", executionPlan)
        self.env.assertIn("Results", executionPlan)
        self.env.assertNotIn("All Node Scan", executionPlan)
        self.env.assertNotIn("Node By Label Scan", executionPlan)
        self.env.assertNotIn("Aggregate", executionPlan)
        expected = [[4]]
        self.env.assertEqual(resultset, expected)

    def test10_labled_node_count(self):
        query = """MATCH (n:person) RETURN COUNT(n)"""
        resultset = self.graph.query(query).result_set
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Project", executionPlan)
        self.env.assertIn("Results", executionPlan)
        self.env.assertNotIn("All Node Scan", executionPlan)
        self.env.assertNotIn("Node By Label Scan", executionPlan)
        self.env.assertNotIn("Aggregate", executionPlan)
        expected = [[4]]
        self.env.assertEqual(resultset, expected)

    def test11_value_hash_join(self):
        # Issue a query that joins two streams on a node property.
        query = """MATCH (p1:person)-[:know]->({name: 'Roi'}), (p2)-[]->(:person {name: 'Alon'}) WHERE p1.name = p2.name RETURN p2.name ORDER BY p2.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Value Hash Join", executionPlan)
        self.env.assertNotIn("Cartesian Product", executionPlan)

        resultset = self.graph.query(query).result_set
        expected = [['Ailon'], ['Boaz']]
        self.env.assertEqual(resultset, expected)

        # Issue a query that joins two streams on a function call.
        query = """MATCH (p1:person)-[:know]->({name: 'Roi'}) MATCH (p2)-[]->(:person {name: 'Alon'}) WHERE ID(p1) = ID(p2) RETURN p2.name ORDER BY p2.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Value Hash Join", executionPlan)
        self.env.assertNotIn("Cartesian Product", executionPlan)

        resultset = self.graph.query(query).result_set
        self.env.assertEqual(resultset, expected) # same results expected

        query = """MATCH (p1:person)-[:know]->({name: 'Roi'}) MATCH (p2)-[]->(:person {name: 'Alon'}) WHERE p1 = p2 RETURN p2.name ORDER BY p2.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Value Hash Join", executionPlan)
        self.env.assertNotIn("Cartesian Product", executionPlan)

        resultset = self.graph.query(query).result_set
        self.env.assertEqual(resultset, expected) # same results expected

    def test12_multiple_stream_value_hash_join(self):
        # Issue a query that joins three streams.
        query = """MATCH (p1:person)-[:know]->({name: 'Roi'}), (p2)-[]->(:person {name: 'Alon'}), (p3)
                   WHERE p1.name = p2.name AND ID(p2) = ID(p3)
                   RETURN p2.name
                   ORDER BY p2.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Value Hash Join", executionPlan)
        self.env.assertNotIn("Cartesian Product", executionPlan)

        resultset = self.graph.query(query).result_set
        expected = [['Ailon'], ['Boaz']]
        self.env.assertEqual(resultset, expected)

        # Issue a query that joins four streams that all resolve the same entity.
        query = """MATCH (p1 {name: 'Ailon'}), (p2), (p3), (p4)
                   WHERE ID(p1) = ID(p2) AND ID(p2) = ID(p3) AND p3.name = p4.name
                   RETURN p4.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Value Hash Join", executionPlan)
        self.env.assertNotIn("Cartesian Product", executionPlan)

        expected = [['Ailon']]
        resultset = self.graph.query(query).result_set
        self.env.assertEqual(resultset, expected)

        # Issue a query that joins four streams that all resolve the same entity, with multiple reapeating filter (issue #869).
        query = """MATCH (p1 {name: 'Ailon'}), (p2), (p3), (p4)
                   WHERE ID(p1) = ID(p2)   AND
                         ID(p2) = ID(p3)   AND
                         ID(p3) = ID(p2)   AND
                         ID(p2) = ID(p1)   AND
                         p3.name = p4.name AND
                         p4.name = p3.name
                   RETURN p4.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Value Hash Join", executionPlan)
        self.env.assertNotIn("Cartesian Product", executionPlan)

        expected = [['Ailon']]
        resultset = self.graph.query(query).result_set
        self.env.assertEqual(resultset, expected)

    def test13_duplicate_filter_placement(self):
        # Issue a query that joins three streams and contains a redundant filter.
        query = """MATCH (p0), (p1), (p2)
                   WHERE id(p2) = id(p0) AND id(p1) = id(p2) AND id(p1) = id(p2)
                   RETURN p2.name ORDER BY p2.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Value Hash Join", executionPlan)
        self.env.assertNotIn("Cartesian Product", executionPlan)

        resultset = self.graph.query(query).result_set
        expected = [['Ailon'], ['Alon'], ['Boaz'], ['Roi']]
        self.env.assertEqual(resultset, expected)

    def test14_distinct_aggregations(self):
        # Verify that the Distinct operation is removed from the aggregating query.
        query = """MATCH (src:person)-[:know]->(dest) RETURN DISTINCT src.name, COUNT(dest) ORDER BY src.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Aggregate", executionPlan)
        self.env.assertNotIn("Distinct", executionPlan)

        resultset = self.graph.query(query).result_set
        expected = [['Ailon', 3],
                    ['Alon', 3],
                    ['Boaz', 3],
                    ['Roi', 3]]
        self.env.assertEqual(resultset, expected)


        # Verify that the Distinct operation is not removed from a valid projection.
        query = """MATCH (src:person) WITH DISTINCT src MATCH (src)-[:know]->(dest) RETURN src.name, COUNT(dest) ORDER BY src.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Aggregate", executionPlan)
        self.env.assertIn("Distinct", executionPlan)

        resultset = self.graph.query(query).result_set
        # This query should emit the same result.
        self.env.assertEqual(resultset, expected)

    def test15_test_splitting_cartesian_product(self):
        query = """MATCH (p1), (p2), (p3) WHERE p1.name <> p2.name AND p2.name <> p3.name RETURN DISTINCT p2.name ORDER BY p2.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertEqual(2, executionPlan.count("Cartesian Product"))
        expected = [['Ailon'],
                    ['Alon'],
                    ['Boaz'],
                    ['Roi']]
        resultset = self.graph.query(query).result_set
        self.env.assertEqual(resultset, expected)
    
    def test16_test_splitting_cartesian_product_with_multiple_filters(self):
        query = """MATCH (p1), (p2), (p3) WHERE p1.name <> p2.name AND ID(p1) <> ID(p2) RETURN DISTINCT p2.name ORDER BY p2.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertEqual(2, executionPlan.count("Cartesian Product"))
        expected = [['Ailon'],
                    ['Alon'],
                    ['Boaz'],
                    ['Roi']]
        resultset = self.graph.query(query).result_set
        self.env.assertEqual(resultset, expected)

    def test17_test_multiple_branch_filter_cp_optimization(self):
        query = """MATCH (p1), (p2), (p3), (p4) WHERE p1.val + p2.val = p3.val AND p3.val > 0 RETURN DISTINCT p3.name ORDER BY p3.name"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertEqual(2, executionPlan.count("Cartesian Product"))
        expected = [['Ailon'],
                    ['Alon'],
                    ['Boaz']]
        resultset = self.graph.query(query).result_set
        self.env.assertEqual(resultset, expected)

    def test18_test_semi_apply_and_cp_optimize(self):
        self.graph.query ("CREATE ({val:0}), ({val:1})-[:R]->({val:2})-[:R]->({val:3})")
        # The next query generates the execution plan:
        # 1) "Results"
        # 2) "    Sort"
        # 3) "        Distinct"
        # 4) "            Project"
        # 5) "                Semi Apply"
        # 6) "                    Cartesian Product"
        # 7) "                        All Node Scan | (n4)"
        # 8) "                        Filter"
        # 9) "                            Cartesian Product"
        # 10) "                                All Node Scan | (n1)"
        # 11) "                                Filter"
        # 12) "                                    All Node Scan | (n3)"
        # 13) "                                All Node Scan | (n2)"
        # 14) "                    Expand Into | (n3)->(n4)"
        # 15) "                        Filter"
        # 16) "                            Argument"
        # We want to make sure the optimization is not misplacing the semi apply bounded branch.
        resultset = self.graph.query("MATCH (n1), (n2), (n3), (n4) WHERE (n3)-[:R]->(n4 {val:n3.val+1}) AND n1.val + n2.val = n3.val AND n3.val > 1  RETURN DISTINCT n3.val ORDER BY n3.val").result_set
        expected = [[2]]
        self.env.assertEqual(resultset, expected)
    
    def test19_test_filter_compaction_remove_true_filter(self):
        query = "MATCH (n) WHERE 1 = 1 RETURN n"
        executionPlan = str(self.graph.explain(query))
        self.env.assertNotIn("Filter", executionPlan)

    def test20_test_filter_compaction_not_removing_false_filter(self):
        query = "MATCH (n) WHERE 1 > 1 RETURN n"
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Filter", executionPlan)
        resultset = self.graph.query(query).result_set
        expected = []
        self.env.assertEqual(resultset, expected)

    # ExpandInto should be applied where possible on projected graph entities.
    def test21_expand_into_projected_endpoints(self):
        query = """MATCH (a)-[]->(b) WITH a, b MATCH (a)-[e]->(b) RETURN a.val, b.val ORDER BY a.val, b.val LIMIT 3"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertIn("Expand Into", executionPlan)
        resultset = self.graph.query(query).result_set
        expected = [[0, 1],
                    [0, 2],
                    [0, 3]]
        self.env.assertEqual(resultset, expected)

    # Variables bound in one scope should not be used to introduce ExpandInto ops in later scopes.
    def test22_no_expand_into_across_scopes(self):
        query = """MATCH (reused_1)-[]->(reused_2) WITH COUNT(reused_2) as edge_count MATCH (reused_1)-[]->(reused_2) RETURN edge_count, reused_1.val, reused_2.val ORDER BY reused_1.val, reused_2.val LIMIT 3"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertNotIn("Expand Into", executionPlan)
        resultset = self.graph.query(query).result_set
        expected = [[14, 0, 1],
                    [14, 0, 2],
                    [14, 0, 3]]
        self.env.assertEqual(resultset, expected)

    # Test limit propagation, execution-plan operations such as
    # conditional traverse accumulate a batch of records before processing
    # knowladge about limit can benifit such operation as they can reduce
    # their batch size to match the current limit.
    def test23_limit_propagation(self):
        graph_id = "limit-propagation"
        graph = self.db.select_graph(graph_id)

        # create graph
        query = """UNWIND range(0, 64) AS x CREATE ()-[:R]->()-[:R]->()"""
        graph.query(query)

        # query with LIMIT 1
        query = """CYPHER l=1
                   MATCH (a)-[]->(b)
                   WITH b AS b
                   MATCH (b)-[]->(c)
                   RETURN c
                   LIMIT $l"""

        # profile query
        profile = graph.profile(query)

        # make sure 'a' to 'b' traversal operation is aware of limit
        traverse_op = profile.collect_operations("Conditional Traverse")[1]
        self.env.assertEqual(traverse_op.records_produced, 1)
        #self.env.assertIn("Conditional Traverse | (a)->(b) | Records produced: 1", profile)

        # query with LIMIT 1
        query = """CYPHER l=1
                   MATCH (a), (b)
                   WITH a AS a, b AS b
                   MATCH (a)-[]->(b)
                   WITH b AS b
                   MATCH (b)-[]->(c)
                   RETURN c
                   LIMIT $l"""

        # profile query
        profile = graph.profile(query)
        expand_into_op = profile.collect_operations("Expand Into")[0]
        self.env.assertEqual(expand_into_op.records_produced, 1)

        # make sure 'a' to 'b' expand into traversal operation is aware of limit
        #self.env.assertIn("Expand Into | (a)->(b) | Records produced: 1", profile)

        # aggregation should reset limit, otherwise we'll take a performance hit
        # recall aggregation operations are eager
        query = """CYPHER l=1 MATCH (a)-[]->(b) WITH count(a) AS src, b AS b MATCH (b)-[]->(c) RETURN c LIMIT $l"""

        # profile query
        profile = graph.profile(query)

        # traversal from a to b shouldn't be effected by the limit.
        traverse_op = profile.collect_operations("Conditional Traverse")[0]
        self.env.assertEqual(traverse_op.records_produced, 130)
        #self.env.assertNotIn("Conditional Traverse | (a)->(b) | Records produced: 130", profile)

    # "WHERE true" predicates should not build filter ops.
    def test24_compact_true_predicates(self):
        query = """MATCH (a) WHERE true RETURN a"""
        executionPlan = str(self.graph.explain(query))
        self.env.assertNotIn("Filter", executionPlan)

    # Cartesian product filter placement should not recurse into earlier scopes.
    def test25_optimize_cartesian_product_scoping(self):
        query = """MATCH (a {name: 'Ailon'})-[]->(b {name: 'Roi'}) WITH 'const' AS c MATCH (a), (b) WHERE a.val = 3 OR b.val = 3 RETURN a.val, b.val ORDER BY a.val, b.val LIMIT 3"""
        resultset = self.graph.query(query).result_set
        expected = [[0, 3],
                    [0, 3],
                    [0, 3]]
        self.env.assertEqual(resultset, expected)

    # Constant filters should not break Cartesian Product placement.
    def test26_optimize_cartesian_product_constant_filters(self):
        query = """MATCH (a) WHERE 2 > rand() MATCH (a), (b) RETURN a.val, b.val ORDER BY a.val, b.val DESC LIMIT 3"""
        resultset = self.graph.query(query).result_set
        expected = [[0, 3],
                    [0, 3],
                    [0, 3]]
        self.env.assertEqual(resultset, expected)

    # Filters on single Cartesian Product branches should be placed properly.
    def test27_optimize_cartesian_product_complex_filter_trees(self):
        query = """MATCH (a), (b), (c) WHERE a.val = 0 OR 'lit' > 3 AND b.val <> b.fake RETURN a.val, b.val ORDER BY a.val, b.val DESC LIMIT 3"""
        resultset = self.graph.query(query).result_set
        expected = [[0, 3],
                    [0, 3],
                    [0, 3]]
        self.env.assertEqual(resultset, expected)

    # Labels' order should be replaced properly.
    def test28_optimize_label_scan_switch_labels(self):
        # Create three nodes with label N, two with label M, one of them in common.
        self.graph.query("CREATE (:N), (:N), (:N:M), (:M)")

        # Make sure that the M is traversed first.
        query = "MATCH (n:N:M) RETURN n"
        plan = str(self.graph.explain(query))
        self.env.assertIn("Node By Label Scan | (n:M)", plan)

        # Make sure multi-label is enforced, we're expecting only the node with
        # both :N and :M to be returned.
        res = self.graph.query(query)
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0], Node(alias='n', labels=['N', 'M']))

    # in cases where a referred label doesn't exist, the UNKNOW_LABEL_ID 
    # is being cached. Once the label is created we want to make sure that 
    # the UNKNOW_LABEL_ID is replaced with the actual label ID. this test 
    # illustrates this scenario by traversing from a non-existing label 
    # (populating our execution-plan cache) which afterwards is being 
    # created. once created we want to make sure the correct label ID is used.
    def test29_optimize_label_scan_cached_label_id(self):
        self.graph.delete()

        # Create node with label Q
        self.graph.query("CREATE (n:Q)")

        # Make sure N is traversed first, as it has no nodes. (none existing)
        plan = str(self.graph.explain("MATCH (n:N:Q) RETURN n"))
        self.env.assertIn("Node By Label Scan | (n:N)", plan)

        # Add label `N` to only node in the graph
        query = """MATCH (n:Q) SET n:N"""
        self.graph.query(query)

        # Make sure #nodes labeled as Q > #nodes labeled as N
        self.graph.query("CREATE (n:Q)")

        # Make sure N is traversed first, N is now associated with an ID
        # |N| < |Q|
        query = """MATCH (n:N:Q) RETURN count(n)"""
        res = self.graph.query(query)
        self.env.assertEquals(res.result_set, [[1]])

        plan = str(self.graph.explain(query))
        self.env.assertIn("Node By Label Scan | (n:N)", plan)

    # mandatory match labels should not be replaced with optional ones in
    # optimize-label-scan
    def test30_optimize_mandatory_labels_order_only(self):
        # clean db
        self.graph.delete()

        # create a node with label N
        query = """CREATE (n:N {v: 1})"""
        self.graph.query(query)

        query = """MATCH (n:N) OPTIONAL MATCH (n:Q) RETURN n.v"""
        plan = str(self.graph.explain(query))

        # make sure N is traversed first, even though there are no nodes with
        # label Q
        self.env.assertIn("Node By Label Scan | (n:N)", plan)
        res = self.graph.query(query)
        self.env.assertEquals(res.result_set, [[1]])

        # create nodes so there are two nodes with label N, and one with label Q.
        self.graph.query("CREATE (:N:Q {v: 2})")

        # The most tempting label to start traversing from is Z, as there are
        # no nodes of label Z, but it is optional, so the second most tempting
        # label (Q) must be traversed first (order swapped with N)
        queries = ["MATCH (n:N) MATCH (n:Q) OPTIONAL MATCH (n:Z) RETURN n",
                   "MATCH (n:Q) MATCH (n:N) OPTIONAL MATCH (n:Z) RETURN n"]

        for q in queries:
            plan = str(self.graph.explain(q))
            self.env.assertIn("Node By Label Scan | (n:Q)", plan)
            self.env.assertIn("Conditional Traverse | (n:N)->(n:N)", plan)

            # assert correctness of the results
            res = self.graph.query(q)
            self.env.assertEquals(len(res.result_set), 1)
            self.env.assertEquals(res.result_set[0][0], Node(labels=['N', 'Q'], properties={'v': 2}))

    def test31_optimize_optional_labels(self):
        """Tests that the optimization of the Label-Scan op works on optional
        labels properly"""

        # create a node with label `N`
        self.graph.query("CREATE (:N)")

        plan = str(self.graph.explain("OPTIONAL MATCH (n:N:M) RETURN n"))

        # make sure `M` is traversed first, as it has less labels
        self.env.assertIn("Node By Label Scan | (n:M)", plan)
        self.env.assertIn("Conditional Traverse | (n:N)->(n:N)", plan)

        # make sure that labels from different `OPTIONAL MATCH` clauses are not
        # "mixed" in Label-Scan optimization
        query = "OPTIONAL MATCH (n:N) OPTIONAL MATCH (n:M) RETURN n"
        plan = str(self.graph.explain(query))

        # make sure `N` is the first label traversed, even though there are less
        # labels with label `M`
        self.env.assertIn("Node By Label Scan | (n:N)", plan)
        self.env.assertIn("Conditional Traverse | (n:M)->(n:M)", plan)

    def test32_remove_redundant_filters(self):
        # test that filter reduction is a run-time optimization
        # we can't remove redundant filters e.g. WHERE 1 = 1
        # at compile time and cache the resulting execution-plan
        # for the following reason:
        # CYPHER param=1 WITH 4 AS X WHERE $param = 1 RETURN X

        q = "WITH 4 AS X WHERE $param = 1 RETURN X"

        # param = 1, WHERE $param = 1 evaluates to True
        # expecting 'Filter' operation to be removed
        params = {'param': 1}
        plan = str(self.graph.explain(q, params))
        self.env.assertNotIn('Filter', plan)

        # validate result-set
        res = self.graph.query(q, params).result_set
        self.env.assertEquals(len(res), 1)
        self.env.assertEquals(res[0][0], 4)

        # param = 2, WHERE $param = 1 evaluates to False
        # expecting 'Filter' operation to show up in execution-plan
        params = {'param': 2}
        plan = str(self.graph.explain(q, params))
        self.env.assertIn('Filter', plan)

        # validate result-set
        res = self.graph.query(q, params).result_set
        self.env.assertEquals(len(res), 0)

    def test33_cartesian_product_count_optimization(self):
        # Create test data
        test_graph_id = "cartesian_count_test"
        test_graph = self.db.select_graph(test_graph_id)
        
        # Create 100 nodes of each type N1 and N2
        test_graph.query("UNWIND range(1, 100) AS i CREATE (n:N1 {id: i})")
        test_graph.query("UNWIND range(1, 100) AS i CREATE (n:N2 {id: i})")
        
        # Test the problematic query that should be optimized
        query = """MATCH (n1:N1), (n2:N2)
                   WITH COUNT(*) AS c
                   MATCH (n1:N1), (n2:N2)
                   RETURN COUNT(*)"""
        
        # First verify the result is correct (100 * 100 = 10000)
        resultset = test_graph.query(query).result_set
        expected = [[10000]]
        self.env.assertEqual(resultset, expected)
        
        # Verify both aggregations are calculated properly
        # Test 1: Test the intermediate aggregation separately
        intermediate_query = """MATCH (n1:N1), (n2:N2)
                               WITH COUNT(*) AS c
                               RETURN c"""
        intermediate_result = test_graph.query(intermediate_query).result_set
        expected_intermediate = [[10000]]
        self.env.assertEqual(intermediate_result, expected_intermediate)
        
        # Test 2: Test that intermediate values work correctly with operations
        intermediate_usage_query = """MATCH (n1:N1), (n2:N2)
                                     WITH COUNT(*) AS c
                                     RETURN c, c * 2 AS doubled"""
        intermediate_usage_result = test_graph.query(intermediate_usage_query).result_set
        expected_usage = [[10000, 20000]]
        self.env.assertEqual(intermediate_usage_result, expected_usage)
        
        # Test 3: Test both aggregations work together in a chained query
        chained_query = """MATCH (n1:N1), (n2:N2)
                          WITH COUNT(*) AS intermediate_count
                          MATCH (n3:N1), (n4:N2) 
                          WITH intermediate_count, COUNT(*) AS final_count
                          RETURN intermediate_count, final_count"""
        chained_result = test_graph.query(chained_query).result_set
        expected_chained = [[10000, 10000]]
        self.env.assertEqual(chained_result, expected_chained)
        
        # Test 4: Test the original problematic pattern with validation of intermediate result
        # This ensures the WITH COUNT(*) AS c step produces correct intermediate aggregation
        validation_query = """MATCH (n1:N1), (n2:N2)
                             WITH COUNT(*) AS c
                             RETURN 'intermediate_result' AS step, c AS count
                             UNION ALL
                             MATCH (n1:N1), (n2:N2)
                             WITH COUNT(*) AS c
                             MATCH (n1:N1), (n2:N2)
                             RETURN 'final_result' AS step, COUNT(*) AS count"""
        validation_result = test_graph.query(validation_query).result_set
        # Should return both intermediate and final counts as 10000
        expected_validation = [['intermediate_result', 10000], ['final_result', 10000]]
        self.env.assertEqual(validation_result, expected_validation)
        
        # Check that the execution plan is optimized (should not contain nested Aggregate operations)
        executionPlan = str(test_graph.explain(query))
        self.env.assertIn("Project", executionPlan)
        self.env.assertIn("Results", executionPlan)
        # Should not have nested Cartesian Products or multiple Aggregate operations
        self.env.assertTrue(executionPlan.count("Aggregate") <= 1)
        
        # Test simpler case: single cartesian product count
        simple_query = """MATCH (n1:N1), (n2:N2) RETURN COUNT(*)"""
        simple_resultset = test_graph.query(simple_query).result_set
        self.env.assertEqual(simple_resultset, expected)
        
        # Check that the simple query is also optimized
        simple_plan = str(test_graph.explain(simple_query))
        self.env.assertIn("Project", simple_plan)
        self.env.assertNotIn("Cartesian Product", simple_plan)
        self.env.assertNotIn("Node By Label Scan", simple_plan)
        self.env.assertNotIn("Aggregate", simple_plan)
        
        # Test with mixed node types
        mixed_query = """MATCH (n1:N1), (n2) RETURN COUNT(*)"""
        # Should be 100 * 204 = 20400 (100 N1 nodes + 100 N2 nodes + 4 person nodes)
        mixed_resultset = test_graph.query(mixed_query).result_set
        expected_mixed = [[20400]]
        self.env.assertEqual(mixed_resultset, expected_mixed)
        
        # Additional test: Verify COUNT(*) works correctly in multiple scenarios
        # Test 1: Single label count
        single_label_query = """MATCH (n:N1) RETURN COUNT(*)"""
        single_result = test_graph.query(single_label_query).result_set
        expected_single = [[100]]
        self.env.assertEqual(single_result, expected_single)
        
        # Test 2: Verify aggregation with WHERE clause (should not be optimized)
        where_query = """MATCH (n1:N1), (n2:N2) WHERE n1.id = n2.id RETURN COUNT(*)"""
        where_result = test_graph.query(where_query).result_set
        expected_where = [[100]]  # Only matching IDs (1-100 match)
        self.env.assertEqual(where_result, expected_where)
        
        # Test 3: Verify that optimization doesn't break with additional aggregations
        sum_query = """MATCH (n1:N1) RETURN COUNT(*), SUM(n1.id)"""
        sum_result = test_graph.query(sum_query).result_set
        # COUNT(*) = 100, SUM(1..100) = 5050
        expected_sum = [[100, 5050]]
        self.env.assertEqual(sum_result, expected_sum)
        
        # Clean up test data
        test_graph.delete()
