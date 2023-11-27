from common import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from demo import QueryInfo

GRAPH_ID = "params"


class testParams(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def setUp(self):
        self.env.flush()
    
    def test_simple_params(self):
        params = [1, 2.3, -1, -2.3, "str", True, False, None, [0, 1, 2]]
        query = "RETURN $param"
        for param in params:
            expected_results = [[param]]
            query_info = QueryInfo(query = query, description="Tests simple params", expected_result = expected_results)
            self._assert_resultset_equals_expected(self.graph.query(query, {'param': param}), query_info)

    def test_invalid_param(self):
        invalid_queries = [
                "CYPHER param=a RETURN $param",                            # 'a' is undefined
                "CYPHER param=a MATCH (a) RETURN $param",                  # 'a' is undefined
                "CYPHER param=f(1) RETURN $param",                         # 'f' doesn't exists
                "CYPHER param=2+f(1) RETURN $param",                       # 'f' doesn't exists
                "CYPHER param=[1, f(1)] UNWIND $param AS x RETURN x",      # 'f' doesn't exists
                "CYPHER param=[1, [2, f(1)]] UNWIND $param AS x RETURN x", # 'f' doesn't exists
                "CYPHER param={'key':f(1)} RETURN $param",                 # 'f' doesn't exists
                "CYPHER param=1*'a' RETURN $param",                        # 1*'a' isn't defined
                "CYPHER param=abs(1)+f(1) RETURN $param",                  # 'f' doesn't exists
                "CYPHER param= RETURN 1",                                  # undefined parameter
                "CYPHER param=count(1) RETURN $param"                      # aggregation function can't be used as a parameter
                "CYPHER param=2+count(1) RETURN $param",                   # aggregation function can't be used as a parameter
                "CYPHER param=[1, count(1)] UNWIND $param AS x RETURN x",  # aggregation function can't be used as a parameter
                "CYPHER param={'key':count(1)} RETURN $param",             # aggregation function can't be used as a parameter
                "CYPHER param={'key':1*'a'} RETURN $param",                # 1*'a' isn't defined
                "CYPHER param=[1, 1*'a'] UNWIND $param AS x RETURN x",     # 1*'a' isn't defined
                "CYPHER param={'key':a} RETURN $param",                    # 'a' isn't defined
                "CYPHER param=[1, a] UNWIND $param AS x RETURN x",         # 'a' isn't defined
                "CYPHER param0=1 param1=$param0 RETURN $param1",           # paramers shouldn't refer to one another
                "RETURN ({1})--({})"                                       # old params syntax
                ]
        for q in invalid_queries:
            try:
                result = self.graph.query(q)
                assert(False)
            except redis.exceptions.ResponseError as e:
                pass

    def test_expression_on_param(self):
        params = {'param': 1}
        query = "RETURN $param + 1"
        expected_results = [[2]]
            
        query_info = QueryInfo(query = query, description="Tests expression on param", expected_result = expected_results)
        self._assert_resultset_equals_expected(self.graph.query(query, params), query_info)

    def test_node_retrival(self):
        p0 = Node(node_id=0, labels="Person", properties={'name': 'a'})
        p1 = Node(node_id=1, labels="Person", properties={'name': 'b'})
        p2 = Node(node_id=2, labels="NoPerson", properties={'name': 'a'})
        self.graph.query(f"CREATE {p0}, {p1}, {p2}")

        params = {'name': 'a'}
        query = "MATCH (n :Person {name:$name}) RETURN n"
        expected_results = [[p0]]
            
        query_info = QueryInfo(query = query, description="Tests expression on param", expected_result = expected_results)
        self._assert_resultset_equals_expected(self.graph.query(query, params), query_info)

    def test_parameterized_skip_limit(self):
        params = {'skip': 1, 'limit': 1}
        query = "UNWIND [1,2,3] AS X RETURN X SKIP $skip LIMIT $limit"
        expected_results = [[2]]
            
        query_info = QueryInfo(query = query, description="Tests skip limit as params", expected_result = expected_results)
        self._assert_resultset_equals_expected(self.graph.query(query, params), query_info)

        # Set one parameter to non-integer value
        params = {'skip': '1', 'limit': 1}
        try:
            self.graph.query(query, params)
            assert(False)
        except redis.exceptions.ResponseError as e:
            pass

    def test_missing_parameter(self):
        # Make sure missing parameters are reported back as an error.
        query = "RETURN $missing"
        try:
            self.graph.query(query)
            assert(False)
        except:
            # Expecting an error.
            pass

        try:
            self.graph.profile(query)
            assert(False)
        except:
            # Expecting an error.
            pass

        try:
            self.graph.explain(query)
            assert(False)
        except:
            # Expecting an error.
            pass

        query = "MATCH (a) WHERE a.v = $missing RETURN a"
        try:
            self.graph.query(query)
            assert(False)
        except:
            # Expecting an error.
            pass

        query = "MATCH (a) SET a.v = $missing RETURN a"
        try:
            self.graph.query(query)
            assert(False)
        except:
            # Expecting an error.
            pass

    def test_id_scan(self):
        self.graph.query("CREATE ({val:1})")
        expected_results = [[1]]
        params = {'id': 0}
        query = "MATCH (n) WHERE id(n)=$id return n.val"
        query_info = QueryInfo(query=query, description="Test id scan with params", expected_result=expected_results)
        self._assert_resultset_equals_expected(self.graph.query(query, params), query_info)
        plan = str(self.graph.explain(query, params=params))
        self.env.assertIn('NodeByIdSeek', plan)
