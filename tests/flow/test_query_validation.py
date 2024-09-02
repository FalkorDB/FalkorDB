from common import *

GRAPH_ID = "query_validation"

class testQueryValidationFlow(FlowTestsBase):

    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()
    
    def populate_graph(self):
        # Create a single graph.
        self.graph.query("CREATE ({age:34})")

    # Expect an error when trying to use a function which does not exists.
    def test01_none_existing_function(self):
        query = """MATCH (n) RETURN noneExistingFunc(n.age) AS cast"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    # Make sure function validation is type case insensitive.
    def test02_case_insensitive_function_name(self):
        try:
            query = """MATCH (n) RETURN mAx(n.age)"""
            self.graph.query(query)
        except redis.exceptions.ResponseError:
            # function validation should be case insensitive.
            self.env.assertTrue(False)
    
    def test03_edge_missing_relation_type(self):
        try:
            query = """CREATE (n:Person {age:32})-[]->(:person {age:30})"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test04_escaped_quotes(self):
       query = r"CREATE (:escaped{prop1:'single \' char', prop2: 'double \" char', prop3: 'mixed \' and \" chars'})"
       actual_result = self.graph.query(query)
       self.env.assertEquals(actual_result.nodes_created, 1)
       self.env.assertEquals(actual_result.properties_set, 3)

       query = r"MATCH (a:escaped) RETURN a.prop1, a.prop2, a.prop3"
       actual_result = self.graph.query(query)
       expected_result = [["single ' char", 'double " char', 'mixed \' and " chars']]
       self.env.assertEquals(actual_result.result_set, expected_result)

    def test05_invalid_entity_references(self):
        try:
            query = """MATCH (a) RETURN e"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

        try:
            query = """MATCH (a) RETURN a ORDER BY e"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

        try:
            query = """MATCH (@anon_0) RETURN @anon_0"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test06_where_references(self):
        try:
            query = """MATCH (a) WHERE fake = true RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test07_with_references(self):
        try:
            query = """MATCH (a) WITH e RETURN e"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test08_count_distinct_star(self):
        try:
            query = """MATCH (a) RETURN COUNT(DISTINCT *)"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test09_invalid_apply_all(self):
        try:
            query = """MATCH (a) RETURN SUM(*)"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test10_missing_params(self):
        try:
            query = """MATCH (a {name:$name}) RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass
    
    def test11_param_error(self):
        try:
            query = """CYPHER name=({name:'a'}) MATCH (a {name:$name}) RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test12_invalid_query_order(self):
        try:
            query = """MERGE (a) MATCH (a)-[]->(b) RETURN b"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test13_create_bound_variables(self):
        try:
            query = """MATCH (a)-[e]->(b) CREATE (a)-[e]->(b)"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test14_treat_path_as_entity(self):
        self.graph.query("CREATE ()-[:R]->()")
        try:
            query= """MATCH x=()-[]->() RETURN x.name"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test15_dont_crash_on_multiple_errors(self):
        try:
            query = """MATCH (a) where id(a) IN range(0) OR id(a) in range(1)"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    # Run a query in which a parsed parameter introduces a type in an unsupported context.
    def test16_param_introduces_unhandled_type(self):
        try:
            query = """CYPHER props={a:1,b:2} CREATE (a:A $props)"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("Encountered unhandled type" in str(e))
            pass

    # Validate that the module fails properly with incorrect argument counts.
    def test17_query_arity(self):
        # Call GRAPH.QUERY with a missing query argument.
        try:
            res = self.redis_con.execute_command("GRAPH.QUERY", "G")
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("wrong number of arguments" in str(e))
            pass

    # Run queries in which compile-time variables are accessed but not defined.
    def test18_undefined_variable_access(self):
        try:
            query = """CREATE (:person{name:bar[1]})"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("not defined" in str(e))
            pass

        try:
            query = """MATCH (a {val: undeclared}) RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("not defined" in str(e))
            pass

        try:
            query = """UNWIND [fake] AS ref RETURN ref"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("not defined" in str(e))
            pass

    def test19_invalid_cypher_options(self):
        query = "EXPLAIN MATCH (p:president)-[:born]->(:state {name:'Hawaii'}) RETURN p"
        try:
            self.graph.query(query)
            assert(False)
        except:
            # Expecting an error.
            pass

        query = "PROFILE MATCH (p:president)-[:born]->(:state {name:'Hawaii'}) RETURN p"
        try:
            self.graph.query(query)
            assert(False)
        except:
            # Expecting an error.
            pass

        query = "CYPHER val=1 EXPLAIN MATCH (p:president)-[:born]->(:state {name:'Hawaii'}) RETURN p"
        try:
            self.graph.query(query)
            assert(False)
        except:
            # Expecting an error.
            pass

        query = "CYPHER val=1 PROFILE MATCH (p:president)-[:born]->(:state {name:'Hawaii'}) RETURN p"
        try:
            self.graph.query(query)
            assert(False)
        except:
            # Expecting an error.
            pass

    # Undirected edges are not allowed in CREATE clauses.
    def test20_undirected_edge_creation(self):
        try:
            query = """CREATE (:Endpoint)-[:R]-(:Endpoint)"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("Only directed relationships" in str(e))
            pass

    # Applying a filter for non existing entity.
    def test20_non_existing_graph_entity(self):
        try:
            query = """MATCH p=() WHERE p.name='value' RETURN p"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("Type mismatch: expected Map, Node, Edge, Null, or Point but was Path" in str(e))
            pass

    # Comments should not affect query functionality.
    def test21_ignore_query_comments(self):
        query = """MATCH (n)  // This is a comment
                   /* This is a block comment */
                   WHERE EXISTS(n.age)
                   RETURN n.age /* Also a block comment*/"""
        actual_result = self.graph.query(query)
        expected_result = [[34]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """/* A block comment*/ MATCH (n)  // This is a comment
                /* This is a block comment */
                WHERE EXISTS(n.age)
                RETURN n.age /* Also a block comment*/"""
        actual_result = self.graph.query(query)
        expected_result = [[34]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """// This is a comment
                MATCH (n)  // This is a comment
                /* This is a block comment */
                WHERE EXISTS(n.age)
                RETURN n.age /* Also a block comment*/"""
        actual_result = self.graph.query(query)
        expected_result = [[34]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (n)  /* This is a block comment */ WHERE EXISTS(n.age)
                RETURN n.age /* Also a block comment*/"""
        actual_result = self.graph.query(query)
        expected_result = [[34]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Validate procedure call refrences and definitions
    def test22_procedure_validations(self):
        try:
            # procedure call refering to a none existing alias 'n'
            query = """CALL db.idx.fulltext.queryNodes(n, 'B') YIELD node RETURN node"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("not defined" in str(e))
            pass

        # refer to procedure call original output when output is aliased.
        try:
            query = """CALL db.idx.fulltext.queryNodes('A', 'B') YIELD node AS n RETURN node"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("not defined" in str(e))
            pass

        # valid procedure call, no output aliasing
        query = """CALL db.idx.fulltext.queryNodes('A', 'B') YIELD node RETURN node"""
        self.graph.query(query)

        # valid procedure call, output aliasing
        query = """CALL db.idx.fulltext.queryNodes('A', 'B') YIELD node AS n RETURN n"""
        self.graph.query(query)

    # Referencing a variable before defining it should raise a compile-time error.
    def test24_reference_before_definition(self):
        try:
            query = """MATCH ({prop: reference}) MATCH (reference) RETURN *"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("not defined" in str(e))
            pass

    # Invalid filters in cartesian products should raise errors.
    def test25_cartesian_product_invalid_filter(self):
        try:
            query = """MATCH p1=(), (n), ({prop: p1.path_val}) RETURN *"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("Type mismatch: expected Map, Node, Edge, Null, or Point but was Path" in str(e))
            pass

    # invalid predicates should raise errors.
    def test26_invalid_filter_predicate(self):
        queries = [
            """WITH 1 AS a WHERE '' RETURN a""",
            """MATCH (a) WHERE 1 RETURN a""",
            """MATCH (a) WHERE -1 RETURN a""",
            """MATCH (a) WHERE -1 OR true RETURN a""",
            """MATCH (a) WHERE true OR -1 RETURN a""",
            """MATCH (a) WHERE true AND -1 RETURN a""",
            """MATCH (a:Author) WHERE a.name CONTAINS 'Ernest' OR 'Amor' RETURN a""",
            """MATCH () RETURN [()<-[]-() WHERE 1 | TRUE]"""]

        for query in queries:
            try:
                self.graph.query(query)
                assert(False)
            except redis.exceptions.ResponseError as e:
                # Expecting an error.
                assert("Expected boolean predicate" in str(e))
                pass

    # The NOT operator does not compare left and right side expressions.
    def test28_invalid_filter_binary_not(self):
        try:
            # Query should have been:
            # MATCH (u) where u.v IS NOT NULL RETURN u
            query = """MATCH (u) where u.v NOT NULL RETURN u"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("Invalid usage of 'NOT' filter" in str(e))
            pass

    def test29_invalid_filter_non_boolean_constant(self):
        try:
            query = """MATCH (a) WHERE a RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            assert("expected Boolean but was Node" in str(e))
            pass

        try:
            query = """MATCH (a) WHERE 1+rand() RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            assert("expected Boolean but was Float" in str(e))
            pass

        try:
            query = """CYPHER p=3 WITH 1 AS a WHERE $p RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            assert("expected Boolean but was Integer" in str(e))
            pass

        # 'val' is a boolean, so this query is valid.
        query = """WITH true AS val WHERE val return val"""
        self.graph.query(query)

        # Non-existent properties are treated as NULLs, which are boolean in Cypher's 3-valued logic.
        query = """MATCH (a) WHERE a.fakeprop RETURN a"""
        self.graph.query(query)

    # Encountering traversals as property values should raise compile-time errors.
    def test30_unexpected_traversals(self):
        query = """MATCH (a {prop: ()-[]->()}) RETURN a"""
        try:
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("Encountered unhandled type" in str(e))

    def test31_set_invalid_property_type(self):
        queries = ["""MATCH (a) CREATE (:L {v: a})""",
                   """MATCH (a), (b) WHERE b.age IS NOT NULL SET b.age = a""",
                   """MERGE (a) ON MATCH SET a.age = a"""]
        for q in queries:
            try:
                self.graph.query(q)
                assert(False)
            except redis.exceptions.ResponseError as e:
                # Expecting an error.
                assert("Property values can only be of primitive types" in str(e))
                pass

    def test32_return_following_clauses(self):
        # After a RETURN clause we're expecting only the following clauses:
        # SKIP, LIMIT, ORDER-BY and UNION, given that SKIP and LIMIT are
        # actually attributes of the RETURN clause this leaves us with
        # ORDER-BY and UNION.

        invalid_queries = ["""RETURN 1 CREATE ()""",
                """RETURN 1 RETURN 2""",
                """MATCH(n) RETURN n DELETE n""",
                """MATCH(n) RETURN n SET n.v = 1""",
                """RETURN 1 MERGE ()""",
                """RETURN 1 MATCH (n) RETURN n""",
                """RETURN 1 WITH 1 as one RETURN one""" ]

        # Invalid queries, expecting errors.
        for q in invalid_queries:
            try:
                self.graph.query(q)
                assert(False)
            except redis.exceptions.ResponseError as e:
                # Expecting an error.
                assert("Unexpected clause following RETURN" in str(e))
                pass

    # Parameters cannot reference aliases.
    def test33_alias_reference_in_param(self):
        try:
            query = """CYPHER A=[a] RETURN 5"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("'a' not defined" in str(e))
            pass

    def test34_self_referential_properties(self):
        try:
            # The server should emit an error on trying to create a node with a self-referential property.
            query = """CREATE (a:L {v: a.v})"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            self.env.assertIn("undefined attribute", str(e))

        # MATCH clauses should be able to use self-referential properties as existential filters.
        query = """MATCH (a {age: a.age}) RETURN a.age"""
        actual_result = self.graph.query(query)
        expected_result = [[34]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Test a query that allocates a large buffer.
    def test35_large_query(self):
        retval = "abcdef" * 1_000
        query = "RETURN " + "\"" + retval + "\""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], retval)

    def test36_multiple_proc_calls(self):
        query = """MATCH (a)
                   CALL algo.BFS(a, 3, NULL) YIELD nodes as ns1
                   MATCH (b)
                   CALL algo.BFS(b, 3, NULL) YIELD nodes as ns2
                   RETURN ns1"""
        plan = str(self.graph.explain(query))
        self.env.assertTrue(plan.count("ProcedureCall") == 2)

    def test37_list_comprehension_missuse(self):
        # all expect list comprehension,
        # unfortunately this isn't enforced by the parser
        # as such it is possible for a user miss-use this function
        # and our current arithmetic expression construction logic will
        # construct a malformed function call

        # make sure we're reciving an exception for each miss-use query
        queries = ["WITH 1 AS x RETURN all(x > 2)",
                "WITH 1 AS x RETURN all([1],2,3)"]

        for q in queries:
            try:
                self.graph.query(q)
                assert(False)
            except redis.exceptions.ResponseError as e:
                pass

    def test38_return_star_union(self):
        # queries of the form [...] RETURN * UNION [...] should have
        # all relevant validations on their column names enforced
        queries = ["WITH 5 AS x RETURN * UNION WITH 10 AS y RETURN *",
                   "WITH 5 AS x RETURN * UNION WITH 10 AS y RETURN y",
                   "WITH 5 AS x, 8 AS y RETURN * UNION WITH 10 AS y RETURN y"]
        for q in queries:
            try:
                self.graph.query(q)
                assert(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("All sub queries in a UNION must have the same column names", str(e))

    def test39_non_single_statement_query(self):
        queries = [";",
                   " ;",
                   " ",
                   "cypher"]
        for q in queries:
            try:
                self.graph.query(q)
                assert(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("empty query", str(e))
        
        queries = ["MATCH (n) RETURN n; MATCH"]
        for q in queries:
            try:
                self.graph.query(q)
                assert(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("query with more than one statement is not supported", str(e))

        queries = ["RETURN 1;",
                   "RETURN 1;;"]
        for q in queries:
            res = self.graph.query(q)
            self.env.assertEquals(res.result_set, [[1]])

    def test40_compile_time_errors_in_star_projections(self):
        # validate that parser errors are handled correctly
        # in queries containing star projections
        queries = ["MATCH (a)-[r:]->(b) RETURN *",
                   "MATCH (a)-[r:]->(b) WITH b RETURN *"]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError:
                pass

        # check that AST validation errors are handled correctly
        # in queries containing star projections
        queries = ["WITH 1 RETURN *",
                   "RETURN *",
                   "CREATE () RETURN DISTINCT *",
                   "MATCH () WITH * RETURN z",
                   "MATCH () WITH * RETURN *",
                   "MATCH () WITH * WHERE n.v > 1 RETURN *"]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError:
                pass

    # Test returning multiple occurrence of an expression.
    def test41_return_duplicate_expression(self):
        queries = ["""MATCH (a) RETURN max(a.val), max(a.val)""",
                """MATCH (a) return max(a.val) as x, max(a.val) as x""",
                """MATCH (a) RETURN a.val, a.val LIMIT 1""",
                """MATCH (a) return a.val as x, a.val as x LIMIT 1""",
                """WITH 1 AS a, 1 AS a RETURN a""",
                """MATCH (n) WITH n, n RETURN n"""]

        for q in queries:
            try:
                self.graph.query(q)
                assert(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("Multiple result columns with the same name are not supported", str(e))

    # Test fail with unknown function.
    def test42_unknown_function(self):
        queries = ["""MATCH (a { v: x()}) RETURN a""",
                """MERGE (a { v: x()}) RETURN a""",
                """MERGE (a) ON CREATE SET a.v = x() RETURN a""",
                """CREATE (a { v: x()}) RETURN a""",
                """MATCH (n) RETURN shortestPath(n, n)""",
                """MATCH p=()-[*1..5]->() RETURN shortestPath(p)""",
                """RETURN ge(1, 2)"""]

        for q in queries:
            try:
                self.graph.query(q)
                assert(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("Unknown function", str(e))
    
    # Variable length edges are not allowed in CREATE or MERGE clauses.
    def test43_invalid_variable_length_edge_use(self):
        queries = [
            """CREATE (a:A)-[e:E1*]->(b:B)""",
            """CREATE (a:A)-[e1:E1]->(b:B)-[e2:E2*]->(c:C)""",
            """MERGE (a:A)-[e:E1*]->(b:B)""",
            """MERGE (a:A)-[e1:E1]->(b:B)-[e2:E2*]->(c:C)""",
        ]
        for q in queries:
            try:
                self.graph.query(q)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("Variable length relationships cannot be used in", str(e))

    def test44_undefined_variables(self):
        # invalid usage of undefined variables in a `WITH` clause
        invalid_queries = [
            "WITH a RETURN a",
            "WITH a AS a RETURN a",
            "WITH [a] AS a RETURN a",
            "WITH [a[a[a]]] AS a RETURN a",
            "WITH a AS b, b AS c, c AS a RETURN a",
            "WITH {a:a} AS a RETURN a",
            "WITH a RETURN 0",
            "WITH 3 AS a, 4 AS b, a + b AS c RETURN c",
            "WITH [x in a | x.prop1] AS a RETURN 1",
            "WITH [(n)-[x:R]->(m) | a.prop1] AS a RETURN 1"
        ]
        for query in invalid_queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                # Expecting an error.
                self.env.assertIn("'a' not defined", str(e))

        # invalid usage of undefined variables in a `RETURN` clause
        invalid_queries = [
            "RETURN a AS a",
            "RETURN [a] AS a",
            "RETURN [a[a[a]]] AS a",
            "RETURN a AS b, b AS c, c AS a",
            "RETURN {a:a} AS a",
            "RETURN [x in a | x.prop1] AS a",
            "RETURN [(n)-[x:R]->(m) | a.prop1] AS a"
        ]
        for query in invalid_queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                # Expecting an error.
                self.env.assertIn("'a' not defined", str(e))

    def test45_union_scope(self):
        # make sure OPTIONAL MATCH followed by a MATCH clause in a different
        # UNION scope do not effect one another
        # in case the scopes had been mixed we would encounted an error

        q = "OPTIONAL MATCH (a) RETURN a UNION MATCH (a) RETURN a"
        self.graph.query(q)
