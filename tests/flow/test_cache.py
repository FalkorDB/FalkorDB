import asyncio
from common import *
from index_utils import *

CACHE_SIZE = 16

class testCache():
    def __init__(self):
        # Have only one thread handling queries
        self.env, self.db = Env(moduleArgs=f"THREAD_COUNT 8 CACHE_SIZE {CACHE_SIZE}")

    def compare_uncached_to_cached_query_plans(self, query, params=None):
        plan_graph = self.db.select_graph('Cache_Test_plans')
        # Create graph
        plan_graph.query("RETURN 1")
        uncached_plan = str(plan_graph.explain(query, params))
        cached_plan = str(plan_graph.explain(query, params))
        self.env.assertEqual(uncached_plan, cached_plan)
        plan_graph.delete()

    def test_01_sanity_check(self):
        graph = self.db.select_graph('Cache_Sanity_Check')
        for i in range(CACHE_SIZE + 1):
            result = graph.query("MATCH (n) WHERE n.value = {val} RETURN n".format(val=i))
            self.env.assertFalse(result.cached_execution)
        
        for i in range(1, CACHE_SIZE + 1):
            result = graph.query("MATCH (n) WHERE n.value = {val} RETURN n".format(val=i))
            self.env.assertTrue(result.cached_execution)
        
        result = graph.query("MATCH (n) WHERE n.value = 0 RETURN n")
        self.env.assertFalse(result.cached_execution)

        graph.delete()

    def test_02_test_create(self):
        # Both queries do exactly the same operations
        graph = self.db.select_graph('Cache_Test_Create')
        query = "CREATE ()"
        self.compare_uncached_to_cached_query_plans(query)
        uncached_result = graph.query(query)
        cached_result = graph.query(query)
        self.env.assertFalse(uncached_result.cached_execution)
        self.env.assertTrue(cached_result.cached_execution)
        self.env.assertEqual(uncached_result.nodes_created, cached_result.nodes_created)
        graph.delete()
        
    def test_03_test_create_with_params(self):
        # Both queries do exactly the same operations
        graph = self.db.select_graph('Cache_Test_Create_With_Params')
        params = {'val' : 1}
        query = "CREATE ({val:$val})"
        self.compare_uncached_to_cached_query_plans(query, params)
        uncached_result = graph.query(query, params)
        params = {'val' : 2}
        cached_result = graph.query(query, params)
        self.env.assertFalse(uncached_result.cached_execution)
        self.env.assertTrue(cached_result.cached_execution)
        self.env.assertEqual(uncached_result.nodes_created, cached_result.nodes_created)
        graph.delete()

    def test_04_test_delete(self):
        # Both queries do exactly the same operations
        graph = self.db.select_graph('Cache_Test_Delete')
        for i in range(2):
            params = {'val' : i}
            query = "CREATE ({val:$val})-[:R]->()"
            graph.query(query, params)
        
        params = {'val': 0}
        query = "MATCH (n {val:$val}) DELETE n"
        self.compare_uncached_to_cached_query_plans(query, params)
        uncached_result = graph.query(query, params)
        params = {'val': 1}
        cached_result = graph.query(query, params)
        self.env.assertFalse(uncached_result.cached_execution)
        self.env.assertTrue(cached_result.cached_execution)
        self.env.assertEqual(uncached_result.relationships_deleted, cached_result.relationships_deleted)
        self.env.assertEqual(uncached_result.nodes_deleted, cached_result.nodes_deleted)
        graph.delete()

    def test_05_test_merge(self):
        # Different outcome, same execution plan.
        graph = self.db.select_graph('Cache_Test_Merge')    
        params = {'create_val': 0, 'match_val':1}
        query = "MERGE (n) ON CREATE SET n.val = $create_val ON MATCH SET n.val = $match_val RETURN n.val"
        self.compare_uncached_to_cached_query_plans(query, params)
        uncached_result = graph.query(query, params)
        cached_result = graph.query(query, params)
        self.env.assertFalse(uncached_result.cached_execution)
        self.env.assertTrue(cached_result.cached_execution)
        self.env.assertEqual(uncached_result.properties_set, cached_result.properties_set)
        self.env.assertEqual([[0]], uncached_result.result_set)
        self.env.assertEqual(1, uncached_result.nodes_created)
        self.env.assertEqual([[1]], cached_result.result_set)
        self.env.assertEqual(0, cached_result.nodes_created)

        graph.delete()

    def test_06_test_branching_with_path_filter(self):
        # Different outcome, same execution plan.
        graph = self.db.select_graph('Cache_Test_Path_Filter') 
        query = "CREATE ({val:1})-[:R]->({val:2})-[:R2]->({val:3})"
        graph.query(query)
        query = "MATCH (n) WHERE (n)-[:R]->({val:$val}) OR (n)-[:R2]->({val:$val}) RETURN n.val"
        params = {'val':2}
        self.compare_uncached_to_cached_query_plans(query, params)
        uncached_result = graph.query(query, params)
        params = {'val':3}
        cached_result = graph.query(query, params)
        self.env.assertFalse(uncached_result.cached_execution)
        self.env.assertTrue(cached_result.cached_execution)
        self.env.assertEqual([[1]], uncached_result.result_set)
        self.env.assertEqual([[2]], cached_result.result_set)
        graph.delete()


    def test_07_test_optimizations_index(self):
        graph = self.db.select_graph('Cache_Test_Index')
        create_node_range_index(graph, 'N', 'val', sync=True)
        query = "CREATE (:N{val:1}), (:N{val:2})"
        graph.query(query)
        query = "MATCH (n:N{val:$val}) RETURN n.val"
        params = {'val':1}
        self.compare_uncached_to_cached_query_plans(query, params)
        uncached_result = graph.query(query, params)
        params = {'val':2}
        cached_result = graph.query(query, params)
        self.env.assertFalse(uncached_result.cached_execution)
        self.env.assertTrue(cached_result.cached_execution)
        self.env.assertEqual([[1]], uncached_result.result_set)
        self.env.assertEqual([[2]], cached_result.result_set)
        graph.delete()


    def test_08_test_optimizations_id_scan(self):
        graph = self.db.select_graph('Cache_Test_ID_Scan')
        query = "CREATE (), ()"
        graph.query(query)
        query = "MATCH (n) WHERE ID(n)=$id RETURN id(n)"
        params = {'id':0}
        self.compare_uncached_to_cached_query_plans(query, params)
        uncached_result = graph.query(query, params)
        params = {'id':1}
        cached_result = graph.query(query, params)
        self.env.assertFalse(uncached_result.cached_execution)
        self.env.assertTrue(cached_result.cached_execution)
        self.env.assertEqual([[0]], uncached_result.result_set)
        self.env.assertEqual([[1]], cached_result.result_set)
        graph.delete()


    def test_09_test_join(self):
        graph = self.db.select_graph('Cache_Test_Join')
        query = "CREATE ({val:1}), ({val:2}), ({val:3}),({val:4})"
        graph.query(query)
        query = "MATCH (a {val:$val}), (b) WHERE a.val = b.val-1 RETURN a.val, b.val "
        params = {'val':1}
        self.compare_uncached_to_cached_query_plans(query, params)
        uncached_result = graph.query(query, params)
        params = {'val':3}
        cached_result = graph.query(query, params)
        self.env.assertFalse(uncached_result.cached_execution)
        self.env.assertTrue(cached_result.cached_execution)
        self.env.assertEqual([[1, 2]], uncached_result.result_set)
        self.env.assertEqual([[3, 4]], cached_result.result_set)
        graph.delete()

    def test_10_test_edge_merge(self):
        # In this scenario, the same query is executed twice.
        # In the first time, the relationship `leads` is unknown to the graph so it is created.
        # In the second time the relationship should be known to the graph, so it will be returned by the match.
        # The test validates that a valid edge is returned.
        graph = self.db.select_graph('Cache_Test_Edge_Merge')
        query = "CREATE ({val:1}), ({val:2})"
        graph.query(query)
        query = "MATCH (a {val:1}), (b {val:2}) MERGE (a)-[e:leads]->(b) RETURN e"
        self.compare_uncached_to_cached_query_plans(query)
        uncached_result = graph.query(query)
        self.env.assertEqual(1, uncached_result.relationships_created)
        cached_result = graph.query(query)
        self.env.assertEqual(0, cached_result.relationships_created)
        self.env.assertEqual(uncached_result.result_set, cached_result.result_set)

    def test_11_test_labelscan_update(self):
        # In this scenario a label scan is made for non existing label
        # than the label is created and the label scan query is re-used.
        graph = self.db.select_graph('Cache_test_labelscan_update')
        query = "MATCH (n:Label) return n"
        result = graph.query(query)
        self.env.assertEqual(0, len(result.result_set))
        query = "MERGE (n:Label)"
        result = graph.query(query)
        self.env.assertEqual(1, result.nodes_created)
        query = "MATCH (n:Label) return n"
        result = graph.query(query)
        self.env.assertEqual(1, len(result.result_set))
        self.env.assertEqual("Label", result.result_set[0][0].labels[0])

    def test_12_test_index_scan_update(self):
        # In this scenario a label scan and Update op are made for non-existent label,
        # then the label is created and an index are subsequently created.
        # When the cached query is reused, it should rely on valid label data.
        graph = self.db.select_graph('Cache_test_index_scan_update')
        params = {'v': 1}
        query = "MERGE (n:Label {v: 1}) SET n.v = $v"
        result = graph.query(query, params)
        self.env.assertEqual(0, len(result.result_set))
        self.env.assertEqual(1, result.nodes_created)
        self.env.assertEqual(1, result.labels_added)

        result = create_node_range_index(graph, 'Label', 'v', sync=True)
        self.env.assertEqual(1, result.indices_created)

        params = {'v': 5}
        query = "MERGE (n:Label {v: 1}) SET n.v = $v"
        result = graph.query(query, params)
        self.env.assertEqual(0, result.nodes_created)
        self.env.assertEqual(1, result.properties_set)

    def test_13_test_skip_limit(self):
        # Test using parameters for skip and limit values,
        # ensuring cached executions always use the parameterized values.
        graph = self.db.select_graph('Cache_Empty_Key')
        query = "UNWIND [1,2,3,4] AS arr RETURN arr SKIP $s LIMIT $l"
        params = {'s': 1, 'l': 1}
        uncached_result = graph.query(query, params)
        expected_result = [[2]]
        cached_result = graph.query(query, params)
        self.env.assertEqual(expected_result, cached_result.result_set)
        self.env.assertEqual(uncached_result.result_set, cached_result.result_set)
        self.env.assertFalse(uncached_result.cached_execution)
        self.env.assertTrue(cached_result.cached_execution)

        # Update the params
        params = {'s': 2, 'l': 2}
        # The new result should respect the new skip and limit.
        expected_result = [[3], [4]]
        cached_result = graph.query(query, params)
        self.env.assertEqual(expected_result, cached_result.result_set)
        self.env.assertTrue(cached_result.cached_execution)

    def test_14_cache_eviction(self):
        # this tests spawns a new graph env` with a query-cache with just
        # a single slot, then multiple clients are issuing a similar query
        # only with a small variation to cause a cache miss which implies
        # cache eviction of the only solt
        # we want to make sure the execution of a recently evicted query
        # runs to completion successfuly

        # skip if 'to_thread' is missing or if test under valgrind
        if VALGRIND or "to_thread" not in dir(asyncio):
            self.env.skip()

        # stop previous env
        self.env.flush()
        self.env.stop()

        self.env, self.db = Env(moduleArgs='THREAD_COUNT 8 CACHE_SIZE 1')

        # eviction
        graph = self.db.select_graph('cache_eviction')

        # populate graph
        graph.query("UNWIND range(0, 10000) as x CREATE ({v:'/'})")

        # _run_query is expected to be issued by multiple threads
        def _run_query(i):
            #random param name
            param_name = 'p_' + str(i)
            q = f"MATCH (n) WHERE n.v = ${param_name} RETURN count(n)"
            params = {param_name : '/'}
            g = self.db.select_graph('cache_eviction')
            count = g.query(q, params).result_set[0][0]
            self.env.assertEqual(count, 10001)

        tasks = []
        loop = asyncio.get_event_loop()
        for i in range(1, 50):
            tasks.append(loop.create_task(asyncio.to_thread(_run_query, i)))

        loop.run_until_complete(asyncio.wait(tasks))

