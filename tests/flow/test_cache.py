import asyncio
from common import *
from index_utils import *
from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool

GRAPH_IDS = ["Cache_Test_plans", "Cache_Sanity_Check", 'Cache_Test_Create',
        'Cache_Test_Create_With_Params', 'Cache_Test_Delete', 'Cache_Test_Merge',
        'Cache_Test_Path_Filter', 'Cache_Test_Index', 'Cache_Test_ID_Scan',
        'Cache_Test_Join', 'Cache_Test_Edge_Merge', 'Cache_test_labelscan_update',
        'Cache_test_index_scan_update', 'Cache_Empty_Key', 'cache_eviction']
CACHE_SIZE = 16

class testCache():
    @classmethod
    def setup_class(cls):
        # Have only one thread handling queries
        cls.env, cls.db = Env(moduleArgs=f"THREAD_COUNT 8 CACHE_SIZE {CACHE_SIZE}")
        cls.conn = cls.env.getConnection()

    def setup_method(self):
        self.conn.delete(*GRAPH_IDS)

    def compare_uncached_to_cached_query_plans(self, query, params=None):
        plan_graph = self.db.select_graph('Cache_Test_plans')
        # Create graph
        plan_graph.query("RETURN 1")
        uncached_plan = str(plan_graph.explain(query, params))
        cached_plan = str(plan_graph.explain(query, params))
        assert uncached_plan == cached_plan

    def test_01_sanity_check(self):
        graph = self.db.select_graph('Cache_Sanity_Check')
        for i in range(CACHE_SIZE + 1):
            result = graph.query("MATCH (n) WHERE n.value = {val} RETURN n".format(val=i))
            assert not result.cached_execution
        
        for i in range(1, CACHE_SIZE + 1):
            result = graph.query("MATCH (n) WHERE n.value = {val} RETURN n".format(val=i))
            assert result.cached_execution
        
        result = graph.query("MATCH (n) WHERE n.value = 0 RETURN n")
        assert not result.cached_execution


    def test_02_test_create(self):
        # Both queries do exactly the same operations
        graph = self.db.select_graph('Cache_Test_Create')
        query = "CREATE ()"
        self.compare_uncached_to_cached_query_plans(query)
        uncached_result = graph.query(query)
        cached_result = graph.query(query)
        assert not uncached_result.cached_execution
        assert cached_result.cached_execution
        assert uncached_result.nodes_created == cached_result.nodes_created
        
    def test_03_test_create_with_params(self):
        # Both queries do exactly the same operations
        graph = self.db.select_graph('Cache_Test_Create_With_Params')
        params = {'val' : 1}
        query = "CREATE ({val:$val})"
        self.compare_uncached_to_cached_query_plans(query, params)
        uncached_result = graph.query(query, params)
        params = {'val' : 2}
        cached_result = graph.query(query, params)
        assert not uncached_result.cached_execution
        assert cached_result.cached_execution
        assert uncached_result.nodes_created == cached_result.nodes_created

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
        assert not uncached_result.cached_execution
        assert cached_result.cached_execution
        assert uncached_result.relationships_deleted == cached_result.relationships_deleted
        assert uncached_result.nodes_deleted == cached_result.nodes_deleted

    def test_05_test_merge(self):
        # Different outcome, same execution plan.
        graph = self.db.select_graph('Cache_Test_Merge')    
        params = {'create_val': 0, 'match_val':1}
        query = "MERGE (n) ON CREATE SET n.val = $create_val ON MATCH SET n.val = $match_val RETURN n.val"
        self.compare_uncached_to_cached_query_plans(query, params)
        uncached_result = graph.query(query, params)
        cached_result = graph.query(query, params)
        assert not uncached_result.cached_execution
        assert cached_result.cached_execution
        assert uncached_result.properties_set == cached_result.properties_set
        assert [[0]] == uncached_result.result_set
        assert 1 == uncached_result.nodes_created
        assert [[1]] == cached_result.result_set
        assert 0 == cached_result.nodes_created


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
        assert not uncached_result.cached_execution
        assert cached_result.cached_execution
        assert [[1]] == uncached_result.result_set
        assert [[2]] == cached_result.result_set


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
        assert not uncached_result.cached_execution
        assert cached_result.cached_execution
        assert [[1]] == uncached_result.result_set
        assert [[2]] == cached_result.result_set


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
        assert not uncached_result.cached_execution
        assert cached_result.cached_execution
        assert [[0]] == uncached_result.result_set
        assert [[1]] == cached_result.result_set


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
        assert not uncached_result.cached_execution
        assert cached_result.cached_execution
        assert [[1, 2]] == uncached_result.result_set
        assert [[3, 4]] == cached_result.result_set

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
        assert 1 == uncached_result.relationships_created
        cached_result = graph.query(query)
        assert 0 == cached_result.relationships_created
        assert uncached_result.result_set == cached_result.result_set

    def test_11_test_labelscan_update(self):
        # In this scenario a label scan is made for non existing label
        # than the label is created and the label scan query is re-used.
        graph = self.db.select_graph('Cache_test_labelscan_update')
        query = "MATCH (n:Label) return n"
        result = graph.query(query)
        assert 0 == len(result.result_set)
        query = "MERGE (n:Label)"
        result = graph.query(query)
        assert 1 == result.nodes_created
        query = "MATCH (n:Label) return n"
        result = graph.query(query)
        assert 1 == len(result.result_set)
        assert "Label" == result.result_set[0][0].labels[0]

    def test_12_test_index_scan_update(self):
        # In this scenario a label scan and Update op are made for non-existent label,
        # then the label is created and an index are subsequently created.
        # When the cached query is reused, it should rely on valid label data.
        graph = self.db.select_graph('Cache_test_index_scan_update')
        params = {'v': 1}
        query = "MERGE (n:Label {v: 1}) SET n.v = $v"
        result = graph.query(query, params)
        assert 0 == len(result.result_set)
        assert 1 == result.nodes_created
        assert 1 == result.labels_added

        result = create_node_range_index(graph, 'Label', 'v', sync=True)
        assert 1 == result.indices_created

        params = {'v': 5}
        query = "MERGE (n:Label {v: 1}) SET n.v = $v"
        result = graph.query(query, params)
        assert 0 == result.nodes_created
        assert 1 == result.properties_set

    def test_13_test_skip_limit(self):
        # Test using parameters for skip and limit values,
        # ensuring cached executions always use the parameterized values.
        graph = self.db.select_graph('Cache_Empty_Key')
        query = "UNWIND [1,2,3,4] AS arr RETURN arr SKIP $s LIMIT $l"
        params = {'s': 1, 'l': 1}
        uncached_result = graph.query(query, params)
        expected_result = [[2]]
        cached_result = graph.query(query, params)
        assert expected_result == cached_result.result_set
        assert uncached_result.result_set == cached_result.result_set
        assert not uncached_result.cached_execution
        assert cached_result.cached_execution

        # Update the params
        params = {'s': 2, 'l': 2}
        # The new result should respect the new skip and limit.
        expected_result = [[3], [4]]
        cached_result = graph.query(query, params)
        assert expected_result == cached_result.result_set
        assert cached_result.cached_execution

    def test_14_cache_eviction(self):
        # this tests spawns a new graph env` with a query-cache with just
        # a single slot, then multiple clients are issuing a similar query
        # only with a small variation to cause a cache miss which implies
        # cache eviction of the only solt
        # we want to make sure the execution of a recently evicted query
        # runs to completion successfuly

        # stop previous env
        self.env.stop()

        type(self).env, type(self).db = Env(moduleArgs='THREAD_COUNT 8 CACHE_SIZE 1')

        # eviction

        async def run(self):
            # connection pool with 16 connections
            # blocking when there's no connections available
            pool = BlockingConnectionPool(max_connections=16, timeout=None, host=self.env.host, port=self.env.port, decode_responses=True)
            db = FalkorDB(connection_pool=pool)
            g = db.select_graph('cache_eviction')

            tasks = []
            for i in range(1, 50):
                # random param name
                param_name = 'p_' + str(i)
                q = f"UNWIND range(0, 50000) as x WITH x WHERE x >= ${param_name} RETURN count(x)"
                params = {param_name : 0}
                tasks.append(asyncio.create_task(g.query(q,params)))

            results = await asyncio.gather(*tasks)
            for r in results:
                assert r.result_set[0][0] == 50001

            await pool.aclose()

        asyncio.run(run(self))

