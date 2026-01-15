import random
import string
from common import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from demo import QueryInfo

GRAPH_ID = "params"


class testParams(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()
    
    def test_simple_params(self):
        params = [1, 2.3, -1, -2.3, "str", True, False, None, [0, 1, 2]]
        query = "RETURN $param"
        for param in params:
            expected_results = [[param]]
            query_info = QueryInfo(query = query, description="Tests simple params", expected_result = expected_results)
            self._assert_resultset_equals_expected(self.graph.query(query, {'param': param}), query_info)

    def test_valid_param(self):
        queries = [
            # --- Numbers ---

            ("CYPHER x = 1", 1),
            ("CYPHER x = -1", -1),
            ("CYPHER x = .1", 0.1),
            ("CYPHER x = .123", 0.123),
            ("CYPHER x = 0.123", 0.123),
            ("CYPHER x = -0.123", -0.123),
            ("CYPHER x = -.123", -0.123),

            # Integer base, positive exponent
            ("CYPHER x = 2e2",  2e2),
            ("CYPHER x = 2e+2", 2e+2),
            ("CYPHER x = 2e-2", 2e-2),

            # Integer base, negative number
            ("CYPHER x = -2e2",  -2e2),
            ("CYPHER x = -2e+2", -2e+2),
            ("CYPHER x = -2e-2", -2e-2),

            # Decimal base, positive exponent
            ("CYPHER x = 1.2e2", 1.2e2),
            ("CYPHER x = 1.2e+2", 1.2e+2),
            ("CYPHER x = 1.2e-2", 1.2e-2),

            # Decimal base, negative number
            ("CYPHER x = -1.2e2", -1.2e2),
            ("CYPHER x = -1.2e+2", -1.2e+2),
            ("CYPHER x = -1.2e-2", -1.2e-2),
    
            # --- Strings ---

            ("CYPHER x = ''", ''),           # Empty single-quoted
            ("CYPHER x = \"\"", ""),         # Empty double-quoted
            ("CYPHER x = 'a'", 'a'),         # 'a'
            ("CYPHER x = \"a\"", "a"),       # "a"
            ("CYPHER x = '\"'", '"'),        # Double quote inside string
            ("CYPHER x = '\\''", "'"),       # Escaped single quote
            ("CYPHER x = '\\\"'", '"'),      # Escaped double quote
            ("CYPHER x = 'a\\nb'", "a\nb"),  # Escaped newline
            ("CYPHER x = 'a\\\\b'", "a\\b"), # Escaped backslash
            ("CYPHER x = 'aBc'", "aBc"),     # "aBc"
    
            # --- Booleans ---

            ("CYPHER x = true",  True),
            ("CYPHER x = True",  True),
            ("CYPHER x = TRUE",  True),
            ("CYPHER x = false", False),
            ("CYPHER x = False", False),
            ("CYPHER x = FALSE", False),
    
            # --- Null ---
            ("CYPHER x = null", None),
            ("CYPHER x = Null", None),
            ("CYPHER x = NULL", None),
    
            # --- Arrays ---
            ("CYPHER x = []", []),
            ("CYPHER x = [1, 2, 3]", [1, 2, 3]),
            ("CYPHER x = ['a', 'b']", ["a", "b"]),
            ("CYPHER x = [1, 'a', true, null]", [1, "a", True, None]),
            ("CYPHER x = [[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
            ("CYPHER x = [[1, [2]], [3]]", [[1, [2]], [3]]),
            ("CYPHER x = [[], []]", [[], []]),
    
            # --- Maps ---
            ("CYPHER x = {}", {}),
            ("CYPHER x = {a: 1}", {'a': 1}),
            ("CYPHER x = {a: 'b'}", {'a': 'b'}),
            ("CYPHER x = {a: true, b: null, c: [1, 2]}", {'a': True, 'b': None, 'c': [1, 2]}),
            ("CYPHER x = {nested: {inner: 'val'}}", {'nested': {'inner': 'val'}}),
            ("CYPHER x = {a: {}, b: []}", {'a': {}, 'b': []}),

            # --- Arrays with Maps ---
            ("CYPHER x = [{}]", [{}]),
            ("CYPHER x = [{a: 1}]", [{'a': 1}]),
            ("CYPHER x = [{a: true, b: null, c: [1, 2]}]", [{'a': True, 'b': None, 'c': [1, 2]}]),
            ("CYPHER x = [[{a: 1}], [{b: 2}]]", [[{'a': 1}], [{'b': 2}]]),
            ("CYPHER x = [1, {a: 'str'}, [2, {b: [3, {c: 'deep'}]}]]", [1, {'a': 'str'}, [2, {'b': [3, {'c': 'deep'}]}]]),
            ("CYPHER x = [[[], [{}, {a: {b: [1, 2, {c: 'x'}]}}]]]", [[[], [{}, {'a': {'b': [1, 2, {'c': 'x'}]}}]]]),
            ("CYPHER x = [[{nested: {inner: {x: [1, 2, {y: 'z'}]}}}]]", [[{'nested': {'inner': {'x': [1, 2, {'y': 'z'}]}}}]]),

            # --- Maps with Arrays ---
            ("CYPHER x = {a: []}", {'a': []}),
            ("CYPHER x = {a: [1, 2, 3]}", {'a': [1, 2, 3]}),
            ("CYPHER x = {a: ['x', 'y'], b: [true, false]}", {'a': ['x', 'y'], 'b': [True, False]}),
            ("CYPHER x = {nested: [[1], [2, 3]]}", {'nested': [[1], [2, 3]]}),
            ("CYPHER x = {mixed: [1, 'a', true, null]}", {'mixed': [1, 'a', True, None]}),
            ("CYPHER x = {m: [{x: 1}, {y: 2}]}", {'m': [{'x': 1}, {'y': 2}]}),
            ("CYPHER x = {deep: [1, {a: [2, {b: 'x'}]}]}", {'deep': [1, {'a': [2, {'b': 'x'}]}]}),
            ("CYPHER x = {emptyMap: {}, emptyArray: [], combo: [{}, []]}", {'emptyMap': {}, 'emptyArray': [], 'combo': [{}, []]}),
            ("CYPHER x = {a: {b: {c: [1, 2, {d: []}]}}}", {'a': {'b': {'c': [1, 2, {'d': []}]}}})
        ]

        for q, expected in queries:
            actual = self.graph.query(q + " RETURN $x").result_set[0][0]
            self.env.assertEqual(expected, actual)

        #-----------------------------------------------------------------------
        # random value utilities
        #-----------------------------------------------------------------------

        def random_scalar():
            options = [
                lambda: random.randint(-1000, 1000),
                lambda: round(random.uniform(-1000.0, 1000.0), random.randint(0, 6)),
                lambda: ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 10))),
                lambda: random.choice([True, False]),
                lambda: None
            ]
            return random.choice(options)()

        def random_array(max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return [random_scalar() for _ in range(random.randint(0, 5))]
            arr = []
            for _ in range(random.randint(0, 5)):
                choice = random.choice(['scalar', 'array', 'dict'])
                if choice == 'scalar':
                    arr.append(random_scalar())
                elif choice == 'array':
                    arr.append(random_array(max_depth, current_depth + 1))
                else:
                    arr.append(random_dict(max_depth, current_depth + 1))
            return arr

        def random_dict(max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return {
                    random_key(): random_scalar()
                    for _ in range(random.randint(0, 5))
                }
            d = {}
            for _ in range(random.randint(0, 5)):
                key = random_key()
                choice = random.choice(['scalar', 'array', 'dict'])
                if choice == 'scalar':
                    d[key] = random_scalar()
                elif choice == 'array':
                    d[key] = random_array(max_depth, current_depth + 1)
                else:
                    d[key] = random_dict(max_depth, current_depth + 1)
            return d

        def random_key(length=None):
            if length is None:
                length = random.randint(1, 10)

            key = random.choices(string.ascii_letters)[0]
            key += ''.join(random.choices(string.ascii_letters + string.digits, k=length-1))
            return key

        def random_value():
            return random.choice([
                random_scalar,
                random_array,
                random_dict
            ])()

        for i in range(0, 50):
            q = "RETURN $x"
            expected = random_value()
            actual = self.graph.query(q, {'x': expected}).result_set[0][0]
            self.env.assertEqual(expected, actual)

    def test_escaping_param(self):
        queries = [
        # --- Valid recognized escapes ---
        ("CYPHER x = '\\a' RETURN $x as param, '\\a' as raw", "\a"),
        ("CYPHER x = '\\b' RETURN $x as param, '\\b' as raw", "\b"),
        ("CYPHER x = '\\f' RETURN $x as param, '\\f' as raw", "\f"),
        ("CYPHER x = '\\n' RETURN $x as param, '\\n' as raw", "\n"),
        ("CYPHER x = '\\r' RETURN $x as param, '\\r' as raw", "\r"),
        ("CYPHER x = '\\t' RETURN $x as param, '\\t' as raw", "\t"),
        ("CYPHER x = '\\v' RETURN $x as param, '\\v' as raw", "\v"),
        ("CYPHER x = '\\\\' RETURN $x as param, '\\\\' as raw", "\\"),
        ("CYPHER x = '\\'' RETURN $x as param, '\\'' as raw", "'"),
        ("CYPHER x = '\\\"' RETURN $x as param, '\\\"' as raw", "\""),
        ("CYPHER x = '\\?' RETURN $x as param, '\\?' as raw", "?"),

        # --- Unrecognized escape sequences ---
        ("CYPHER x = '\\x' RETURN $x as param, '\\x' as raw", "\\x"),
        ("CYPHER x = '\\!' RETURN $x as param, '\\!' as raw", "\\!"),
        ("CYPHER x = '\\1' RETURN $x as param, '\\1' as raw", "\\1"),

        # --- Mixed content with recognized and unrecognized escapes ---
        ("CYPHER x = 'Line1\\nLine2\\xEnd' RETURN $x as param, 'Line1\\nLine2\\xEnd' as raw", "Line1\nLine2\\xEnd"),
        ("CYPHER x = 'Tab\\tThen\\!Now\\n' RETURN $x as param, 'Tab\\tThen\\!Now\\n' as raw", "Tab\tThen\\!Now\n"),
        ("CYPHER x = 'Mix\\\\path\\nof\\xfiles' RETURN $x as param, 'Mix\\\\path\\nof\\xfiles' as raw", "Mix\\path\nof\\xfiles"),
        ("CYPHER x = 'Quotes: \\\"hello\\\" and \\\'bye\\\'' RETURN $x as param, 'Quotes: \\\"hello\\\" and \\\'bye\\\'' as raw", "Quotes: \"hello\" and 'bye'"),
        ("CYPHER x = '\\nStart and end\\t' RETURN $x as param, '\\nStart and end\\t' as raw", "\nStart and end\t"),
    ]

        for q, expected in queries:
            res = self.graph.query(q).result_set[0]
            actual = res[0]
            raw = res[1]
            self.env.assertEqual(expected, actual)
            self.env.assertEqual(expected, raw)

    def test_backtick_param_name(self):
        queries = [("CYPHER `param`    = 1 RETURN $`param`",    1),
                   ("CYPHER `.pa.ram.` = 1 RETURN $`.pa.ram.`", 1),
                   ("CYPHER `pa\ram`   = 1 RETURN $`pa\ram`",   1),
                   ("CYPHER `p~aram`   = 1 RETURN $`p~aram`",   1),

                   ("CYPHER `p~aram` = 1 `x` = 2 RETURN $`p~aram` + $`x`", 3),
                   ("CYPHER `p~aram` = 1 x = 2   RETURN $`p~aram` + $x",   3)
                   ]

        for q, expected in queries:
            res = self.graph.query(q).result_set[0]
            actual = res[0]
            self.env.assertEqual(expected, actual)

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
                "CYPHER param=count(1) RETURN $param",                     # aggregation function can't be used as a parameter
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

    def test_multi_cypher_directives(self):
        # cypher allows for multiple CYPHER directives to be specified
        # make sure we're able to parse such cases correctly
        queries = [
                ("CYPHER CYPHER RETURN 1", [[1]]),
                ("CYPHER a=1 CYPHER b=2 RETURN $a, $b", [[1,2]]),
                ("CYPHER CYPHER a=1 b=2 RETURN $a, $b", [[1,2]]),
                ("CYPHER a=1 b=2 CYPHER CYPHER RETURN $a, $b", [[1,2]]),
                ("CYPHER CYPHER a=1 CYPHER CYPHER b=2 CYPHER CYPHER RETURN $a, $b", [[1,2]])
            ]

        for q, expected in queries:
            actual = self.graph.query(q).result_set
            self.env.assertEqual(actual, expected)

    def test_id_scan(self):
        self.graph.query("CREATE ({val:1})")
        expected_results = [[1]]
        params = {'id': 0}
        query = "MATCH (n) WHERE id(n)=$id return n.val"
        query_info = QueryInfo(query=query, description="Test id scan with params", expected_result=expected_results)
        self._assert_resultset_equals_expected(self.graph.query(query, params), query_info)
        plan = str(self.graph.explain(query, params=params))
        self.env.assertIn('NodeByIdSeek', plan)

