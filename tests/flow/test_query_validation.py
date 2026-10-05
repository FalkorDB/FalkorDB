from common import *
from index_utils import wait_for_indices_to_sync

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

    def fresh_graph(self, name):
        # a run interrupted before its cleanup (e.g. by a server crash) can
        # leave the graph behind; start from an empty one
        if name in self.db.list_graphs():
            self.db.select_graph(name).delete()
        return self.db.select_graph(name)

    # Expect an error when trying to use a function which does not exists.
    def test01_none_existing_function(self):
        query = """MATCH (n) RETURN noneExistingFunc(n.age) AS cast"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    # Make sure function validation is type case insensitive.
    def test02_case_insensitive_function_name(self):
        try:
            query = """MATCH (n) RETURN mAx(n.age)"""
            self.graph.query(query)
        except redis.ResponseError:
            # function validation should be case insensitive.
            self.env.assertTrue(False)
    
    def test03_edge_missing_relation_type(self):
        try:
            query = """CREATE (n:Person {age:32})-[]->(:person {age:30})"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    def test04_escaped_quotes(self):
       query = r"CREATE (:escaped{prop1:'single \' char', prop2: 'double \" char', prop3: 'mixed \' and \" chars'})"
       actual_result = self.graph.query(query)
       self.env.assertEqual(actual_result.nodes_created, 1)
       self.env.assertEqual(actual_result.properties_set, 3)

       query = r"MATCH (a:escaped) RETURN a.prop1, a.prop2, a.prop3"
       actual_result = self.graph.query(query)
       expected_result = [["single ' char", 'double " char', 'mixed \' and " chars']]
       self.env.assertEqual(actual_result.result_set, expected_result)

    def test05_invalid_entity_references(self):
        try:
            query = """MATCH (a) RETURN e"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

        try:
            query = """MATCH (a) RETURN a ORDER BY e"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

        try:
            query = """MATCH (@anon_0) RETURN @anon_0"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    def test06_where_references(self):
        try:
            query = """MATCH (a) WHERE fake = true RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    def test07_with_references(self):
        try:
            query = """MATCH (a) WITH e RETURN e"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    def test08_count_distinct_star(self):
        try:
            query = """MATCH (a) RETURN COUNT(DISTINCT *)"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    def test09_invalid_apply_all(self):
        try:
            query = """MATCH (a) RETURN SUM(*)"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    def test10_missing_params(self):
        try:
            query = """MATCH (a {name:$name}) RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass
    
    def test11_param_error(self):
        try:
            query = """CYPHER name=({name:'a'}) MATCH (a {name:$name}) RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    def test12_invalid_query_order(self):
        try:
            query = """MERGE (a) MATCH (a)-[]->(b) RETURN b"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    def test13_create_bound_variables(self):
        try:
            query = """MATCH (a)-[e]->(b) CREATE (a)-[e]->(b)"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    def test14_treat_path_as_entity(self):
        self.graph.query("CREATE ()-[:R]->()")
        try:
            query= """MATCH x=()-[]->() RETURN x.name"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    def test15_dont_crash_on_multiple_errors(self):
        try:
            query = """MATCH (a) where id(a) IN range(0) OR id(a) in range(1)"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError:
            # Expecting an error.
            pass

    # Run a query in which a parsed parameter introduces a type in an unsupported context.
    def test16_param_introduces_unhandled_type(self):
        try:
            query = """CYPHER props={a:1,b:2} CREATE (a:A $props)"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            assert("Encountered unhandled type" in str(e))
            pass

    # Validate that the module fails properly with incorrect argument counts.
    def test17_query_arity(self):
        # Call GRAPH.QUERY with a missing query argument.
        try:
            res = self.redis_con.execute_command("GRAPH.QUERY", "G")
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            assert("wrong number of arguments" in str(e))
            pass

    # Run queries in which compile-time variables are accessed but not defined.
    def test18_undefined_variable_access(self):
        try:
            query = """CREATE (:person{name:bar[1]})"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            assert("not defined" in str(e))
            pass

        try:
            query = """MATCH (a {val: undeclared}) RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            assert("not defined" in str(e))
            pass

        try:
            query = """UNWIND [fake] AS ref RETURN ref"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
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
        except redis.ResponseError as e:
            # Expecting an error.
            assert("Only directed relationships" in str(e))
            pass

    # Applying a filter for non existing entity.
    def test20_non_existing_graph_entity(self):
        try:
            query = """MATCH p=() WHERE p.name='value' RETURN p"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            assert("Type mismatch: expected Map, Node, Edge, Datetime, Date, Time, Duration, Null, or Point but was Path" in str(e))
            pass

    # Comments should not affect query functionality.
    def test21_ignore_query_comments(self):
        query = """MATCH (n)  // This is a comment
                   /* This is a block comment */
                   WHERE EXISTS(n.age)
                   RETURN n.age /* Also a block comment*/"""
        actual_result = self.graph.query(query)
        expected_result = [[34]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        query = """/* A block comment*/ MATCH (n)  // This is a comment
                /* This is a block comment */
                WHERE EXISTS(n.age)
                RETURN n.age /* Also a block comment*/"""
        actual_result = self.graph.query(query)
        expected_result = [[34]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        query = """// This is a comment
                MATCH (n)  // This is a comment
                /* This is a block comment */
                WHERE EXISTS(n.age)
                RETURN n.age /* Also a block comment*/"""
        actual_result = self.graph.query(query)
        expected_result = [[34]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        query = """MATCH (n)  /* This is a block comment */ WHERE EXISTS(n.age)
                RETURN n.age /* Also a block comment*/"""
        actual_result = self.graph.query(query)
        expected_result = [[34]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    # Validate procedure call refrences and definitions
    def test22_procedure_validations(self):
        try:
            # procedure call refering to a none existing alias 'n'
            query = """CALL db.idx.fulltext.queryNodes(n, 'B') YIELD node RETURN node"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            assert("not defined" in str(e))
            pass

        # refer to procedure call original output when output is aliased.
        try:
            query = """CALL db.idx.fulltext.queryNodes('A', 'B') YIELD node AS n RETURN node"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
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
        except redis.ResponseError as e:
            # Expecting an error.
            assert("not defined" in str(e))
            pass

    # Invalid filters in cartesian products should raise errors.
    def test25_cartesian_product_invalid_filter(self):
        try:
            query = """MATCH p1=(), (n), ({prop: p1.path_val}) RETURN *"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            assert("Type mismatch: expected Map, Node, Edge, Datetime, Date, Time, Duration, Null, or Point but was Path" in str(e))
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
            #@todo barak implement list comprehension predicates
            # """MATCH () RETURN [()<-[]-() WHERE 1 | TRUE]"""
            ]

        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                # Expecting an error.
                self.env.assertContains("Expected boolean predicate", str(e))
                pass

    # The NOT operator does not compare left and right side expressions.
    def test28_invalid_filter_binary_not(self):
        try:
            # Query should have been:
            # MATCH (u) where u.v IS NOT NULL RETURN u
            query = """MATCH (u) where u.v NOT NULL RETURN u"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            assert("Invalid usage of 'NOT' filter" in str(e))
            pass

    def test29_invalid_filter_non_boolean_constant(self):
        try:
            query = """MATCH (a) WHERE a RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            assert("expected Boolean but was Node" in str(e))
            pass

        try:
            query = """MATCH (a) WHERE 1+rand() RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            assert("expected Boolean but was Float" in str(e))
            pass

        try:
            query = """CYPHER p=3 WITH 1 AS a WHERE $p RETURN a"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
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
        except redis.ResponseError as e:
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
            except redis.ResponseError as e:
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
            except redis.ResponseError as e:
                # Expecting an error.
                assert("Unexpected clause following RETURN" in str(e))
                pass

    # Parameters cannot reference aliases.
    def test33_alias_reference_in_param(self):
        try:
            query = """CYPHER A=[a] RETURN 5"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            # expecting an error
            pass

    def test34_self_referential_properties(self):
        try:
            # The server should emit an error on trying to create a node with a self-referential property.
            query = """CREATE (a:L {v: a.v})"""
            self.graph.query(query)
            assert(False)
        except redis.ResponseError as e:
            # Expecting an error.
            self.env.assertContains("undefined attribute", str(e))

        # MATCH clauses should be able to use self-referential properties as existential filters.
        query = """MATCH (a {age: a.age}) RETURN a.age"""
        actual_result = self.graph.query(query)
        expected_result = [[34]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        # A self-reference nested inside a comprehension is the same error, but
        # it used to reach evaluation and crash the server in GraphEntity_Keys.
        # The query must be *rejected*: returning an empty key/property list
        # for an entity that does not exist yet would silently accept an
        # invalid query.
        # See https://github.com/FalkorDB/FalkorDB/issues/415
        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]

        queries = [
                # the original report: 'child1' is referenced from within an
                # any() predicate, and the inner comprehension variable 'root'
                # shadows the node created by the same clause
                ("""CREATE (root:Root {name: 'x'}),
                           (child1:TextNode {var: floor(any(v4 IN [2] WHERE child1 = [root IN keys(root)]))}),
                           (child2:IntNode {v0: 0})""", "'child1' not defined"),
                # minimal forms of the same thing
                ("""CREATE (a:L {v: keys(a)})""", "'a' not defined"),
                ("""CREATE (a:L {v: properties(a)})""", "'a' not defined"),
                ("""CREATE (a:L {v: [x IN keys(a) | x]})""", "'a' not defined"),
                ("""CREATE (a:L {v: any(x IN [1] WHERE a.q = 1)})""", "'a' not defined"),
                # the same holds for a relationship reading itself
                ("""CREATE (a:L)-[r:R {v: keys(r)}]->(b:L)""", "'r' not defined"),
                ("""CREATE (a:L)-[r:R {v: properties(r)}]->(b:L)""", "'r' not defined")]

        for query, expected in queries:
            try:
                self.graph.query(query)
                assert(False)
            except redis.ResponseError as e:
                self.env.assertContains(expected, str(e))

        # the rejected CREATEs must not have partially applied
        self.env.assertEqual(
                self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0],
                node_count)

    # Test a query that allocates a large buffer.
    def test35_large_query(self):
        retval = "abcdef" * 1_000
        query = "RETURN " + "\"" + retval + "\""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.result_set[0][0], retval)

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
            except redis.ResponseError as e:
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
            except redis.ResponseError as e:
                self.env.assertContains("All sub queries in a UNION must have the same column names", str(e))

    def test39_non_single_statement_query(self):
        queries = [";",      # Error: could not parse query
                   " ;",     # Error: query with more than one statement is not supported.
                   " ",      # Error: query with more than one statement is not supported.
                   "cypher"] # Error: empty query.
        for q in queries:
            try:
                self.graph.query(q)
                assert(False)
            except redis.ResponseError as e:
                pass
        
        queries = ["MATCH (n) RETURN n; MATCH"]
        for q in queries:
            try:
                self.graph.query(q)
                assert(False)
            except redis.ResponseError as e:
                self.env.assertContains("query with more than one statement is not supported", str(e))

        queries = ["RETURN 1;",
                   "RETURN 1;;",
                   "RETURN 1; ",
                   "RETURN 1;\n",
                   "RETURN 1; \n;"]
        for q in queries:
            res = self.graph.query(q)
            self.env.assertEqual(res.result_set, [[1]])

    def test40_compile_time_errors_in_star_projections(self):
        # validate that parser errors are handled correctly
        # in queries containing star projections
        queries = ["MATCH (a)-[r:]->(b) RETURN *",
                   "MATCH (a)-[r:]->(b) WITH b RETURN *"]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.ResponseError:
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
            except redis.ResponseError:
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
            except redis.ResponseError as e:
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
            except redis.ResponseError as e:
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
            except redis.ResponseError as e:
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
            except redis.ResponseError as e:
                # Expecting an error.
                self.env.assertContains("'a' not defined", str(e))

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
            except redis.ResponseError as e:
                # Expecting an error.
                self.env.assertContains("'a' not defined", str(e))

    def test45_union_scope(self):
        # make sure OPTIONAL MATCH followed by a MATCH clause in a different
        # UNION scope do not effect one another
        # in case the scopes had been mixed we would encounted an error

        q = "OPTIONAL MATCH (a) RETURN a UNION MATCH (a) RETURN a"
        self.graph.query(q)

    def test46_repeated_relationship_variable(self):
        # reusing a relationship variable used to crash the server in several
        # shapes: within a single pattern, inside a pattern predicate reached
        # through WITH, and across a chain of WITH/WHERE clauses. Every shape
        # must now produce either rows or a clean error, and the server must
        # survive it.
        g = self.db.select_graph("repeated_rel_var")

        try:
            g.query("""CREATE (a:P {name: 'A'}), (a)-[:R]->(a),
                              (:P {name: 'B'})-[:R]->(:P {name: 'C'})""")

            # the same variable twice in one pattern can never match and is
            # rejected - it used to trip an assertion in query graph
            # construction and take the server down
            # https://github.com/FalkorDB/FalkorDB/issues/465
            try:
                g.query("WITH NULL AS a0 MATCH ()-[a0]-()-[a0]-() RETURN 1")
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                # Expecting an error.
                self.env.assertContains("same relationship variable", str(e))

            # referencing a bound relationship variable from a pattern
            # predicate is legal - each pattern uses it once - and used to be
            # rejected with that same 'multiple patterns' error
            # https://github.com/FalkorDB/FalkorDB/issues/1335
            actual = g.query("""MATCH (n0)-[r0]->(n0)
                                WHERE ()<-[r0]-(n0)
                                RETURN n0.name""")
            self.env.assertEqual(actual.result_set, [['A']])

            # a relationship variable repeated inside a pattern predicate,
            # reached through WITH, used to segfault. Rejecting it and
            # returning no rows are both acceptable - crashing is not.
            # https://github.com/FalkorDB/FalkorDB/issues/1982
            # https://github.com/FalkorDB/FalkorDB/issues/1333
            queries = ["""MATCH (p:P)-[r:R]-()
                          WITH p, r
                          WHERE (p)-[r:R]->()<-[r:R]-()
                          RETURN p.name""",
                       """MATCH (n0)--(n0)
                          WITH *
                          WHERE (n0)--(n0)
                          MATCH (n1)-[r1]-()
                          WITH *
                          WHERE (n1)-[r1]->()<-[r1]-()
                          RETURN *"""]

            for q in queries:
                try:
                    self.env.assertEqual(g.query(q).result_set, [])
                except redis.ResponseError:
                    # a clean rejection is acceptable, a crash is not
                    pass

                # the server must still answer - the crash used to arrive
                # asynchronously, after the reply
                self.env.assertEqual(
                        g.query("MATCH (n:P) RETURN count(n)").result_set, [[3]])
        finally:
            g.delete()

    def test47_pattern_expression_in_inlined_properties(self):
        # a pattern comprehension or pattern predicate inside a pattern's
        # inline property map is never lowered by the planner, and used to
        # reach the evaluator and take the server down as soon as the outer
        # pattern had a candidate row. It is rejected up front, as FalkorDB C
        # does.
        # https://github.com/FalkorDB/FalkorDB/issues/2308
        g = self.fresh_graph("inlined_pattern_expr")

        try:
            g.query("CREATE (a {k:1})-[:R {k:1}]->(b {k:1})")

            queries = [
                # relationship properties - the reported shape
                "MATCH ()-[{k:size([(a)-[{k:1}]-()|a.k])}]-() RETURN 1",
                # variable-length relationship properties
                "MATCH ()-[*1..2 {k:size([(a)-->()|1])}]->() RETURN 1",
                # node properties
                "MATCH (n {k:[(n)-->(m) | m.k][0]}) RETURN n",
                "MATCH (n {k:exists((n)-->())}) RETURN n",
                # write clauses evaluate inline properties too
                "CREATE ({k:size([(a)-->()|1])})",
                "MERGE ({k:size([(a)-->()|1])})",
            ]

            for q in queries:
                try:
                    g.query(q)
                    self.env.assertTrue(False)
                except redis.ResponseError as e:
                    self.env.assertContains(
                            "Encountered unhandled type in inlined properties",
                            str(e))

            # inside a pattern predicate's properties it is rejected too; the
            # predicate parse falls back, so only require a clean error
            rejected = False
            try:
                g.query("""MATCH (n)
                           WHERE (n)-[{k:size([(a)-->()|1])}]->()
                           RETURN 1""")
            except redis.ResponseError:
                rejected = True
            self.env.assertTrue(rejected)

            # index OPTIONS are evaluated once, with no input row to run a
            # sub-plan on
            for q in ["""CREATE VECTOR INDEX FOR (n:A) ON (n.v) OPTIONS
                         {dimension: size([(a)-->() | 1]) + 1,
                          similarityFunction: 'euclidean'}""",
                      """CREATE FULLTEXT INDEX FOR (n:A) ON (n.name) OPTIONS
                         {language: head([(a)-->() | 'english'])}"""]:
                try:
                    g.query(q)
                    self.env.assertTrue(False)
                except redis.ResponseError as e:
                    self.env.assertContains("not supported in index OPTIONS", str(e))

            # the server survived, and a pattern comprehension outside inline
            # properties, over a pattern with inline properties, still works
            actual = g.query("""MATCH (a)-[{k:1}]->()
                                RETURN [(a)-[{k:1}]->(x) | x.k] AS r""")
            self.env.assertEqual(actual.result_set, [[[1]]])
        finally:
            g.delete()

    def test48_pattern_comprehension_in_clause_expressions(self):
        # the planner lowers a pattern comprehension into its own sub-plan only
        # where it plans for one; in SET, DELETE, procedure arguments and
        # YIELD ... WHERE it used to reach the evaluator un-lowered and crash
        # the server. Every shape must now produce correct rows or a clean
        # error.
        # https://github.com/FalkorDB/FalkorDB/issues/2308
        g = self.fresh_graph("clause_pattern_comprehension")

        def reset():
            g.query("MATCH (n) DETACH DELETE n")
            g.query("""CREATE (:A {name: 'a'})-[:R]->(:A {name: 'b'})""")

        try:
            g.query("RETURN 1")
            reset()

            # SET, in every form, reading the graph through a comprehension
            queries = [
                ("MATCH (n:A) SET n.p = size([(n)-->() | 1])",
                 "MATCH (n:A) RETURN n.name, n.p ORDER BY n.name",
                 [['a', 1], ['b', 0]]),
                ("MATCH (n:A) SET n += {p: [(n)-->(y) | y.name]}",
                 "MATCH (n:A) RETURN n.name, n.p ORDER BY n.name",
                 [['a', ['b']], ['b', []]]),
                ("MATCH (n:A) SET n = {name: n.name, p: [(n)<--(y) | y.name]}",
                 "MATCH (n:A) RETURN n.name, n.p ORDER BY n.name",
                 [['a', []], ['b', ['a']]]),
                ("MATCH (n:A) SET n.p = size([(n)-->() | 1]), n.q = size([()-->(n) | 1])",
                 "MATCH (n:A) RETURN n.name, n.p, n.q ORDER BY n.name",
                 [['a', 1, 0], ['b', 0, 1]]),
                ("MATCH (n:A) FOREACH (x IN [1] | SET n.p = size([(n)-->() | x]))",
                 "MATCH (n:A) RETURN n.name, n.p ORDER BY n.name",
                 [['a', 1], ['b', 0]]),
                # a CALL body that ends the query used to lose its scope
                # table, so any sub-plan in it indexed past the end
                ("MATCH (n:A) CALL { WITH n SET n.p = size([(n)<--() | 1]) }",
                 "MATCH (n:A) RETURN n.name, n.p ORDER BY n.name",
                 [['a', 0], ['b', 1]]),
                ("MATCH (n:A) CALL { WITH n UNWIND [(n)-->(y) | y] AS y SET y.p = 1 }",
                 "MATCH (n:A) RETURN n.name, n.p ORDER BY n.name",
                 [['a', None], ['b', 1]]),
                ("MATCH (n:A) CALL { WITH n WITH n WHERE size([(n)-->() | 1]) > 0 SET n.p = 1 }",
                 "MATCH (n:A) RETURN n.name, n.p ORDER BY n.name",
                 [['a', 1], ['b', None]]),
                # REMOVE from an entity picked by a comprehension
                ("MATCH (n:A {name: 'a'}) REMOVE head([(n)-->(y) | y]).name",
                 "MATCH (n:A) RETURN n.name ORDER BY n.name",
                 [['a'], [None]]),
                # DELETE an entity picked by a comprehension
                ("MATCH (n:A {name: 'a'}) DETACH DELETE head([(n)-->(y) | y])",
                 "MATCH (n:A) RETURN n.name",
                 [['a']]),
            ]
            for write, read, expected in queries:
                g.query(write)
                self.env.assertEqual(g.query(read).result_set, expected)
                reset()

            # SET applies its items in order: a comprehension sees what the
            # items before it set, in the same SET or a consecutive one, and
            # not what the items after it will set
            q = """MATCH (n:A {name: 'a'})
                   SET n.p = 1, n.q = head([(n)-->() | n.p]), n.r = n.q + 1
                   SET n.s = head([(n)-->() | n.r]), n.p = 5
                   RETURN n.p, n.q, n.r, n.s"""
            self.env.assertEqual(g.query(q).result_set, [[5, 1, 2, 2]])
            q = """MATCH (n:A {name: 'a'})
                   SET n.t = head([(n)-->() | n.u]), n.u = 1
                   RETURN n.t, n.u"""
            self.env.assertEqual(g.query(q).result_set, [[None, 1]])
            reset()

            # a SET feeding a RETURN inside a UNION branch; the branch's
            # variables must not be clobbered by the comprehension's result
            q = """MATCH (n:A) SET n.p = size([(n)-->() | 1])
                   RETURN n.name AS name, n.p AS p
                   UNION
                   RETURN 'z' AS name, 9 AS p"""
            actual = sorted(g.query(q).result_set)
            self.env.assertEqual(actual, [['a', 1], ['b', 0], ['z', 9]])
            reset()

            # procedure arguments and YIELD ... WHERE
            q = """MATCH (n:A {name: 'a'})
                   CALL db.labels() YIELD label
                   WHERE size([(n)-->() | 1]) = 1
                   RETURN label"""
            self.env.assertEqual(g.query(q).result_set, [['A']])

            g.query("CREATE FULLTEXT INDEX FOR (n:A) ON (n.name)")
            wait_for_indices_to_sync(g)
            q = """MATCH (n:A {name: 'a'})
                   CALL db.idx.fulltext.queryNodes('A', head([(n)-->(y) | y.name]))
                   YIELD node
                   RETURN node.name"""
            self.env.assertEqual(g.query(q).result_set, [['b']])
            q = """CALL db.idx.fulltext.queryNodes('A', 'a|b') YIELD node
                   WHERE size([(node)-->() | 1]) = 1
                   RETURN node.name"""
            self.env.assertEqual(g.query(q).result_set, [['a']])

            # the index drop is planned with a label known up front, so a
            # computed one is rejected rather than tripping the planner
            try:
                g.query("""MATCH (n:A {name: 'a'})
                           CALL db.idx.fulltext.drop(head([(n)-->() | 'A']))""")
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                self.env.assertContains("must be a string literal", str(e))

            # MERGE applies ON CREATE / ON MATCH itself, after matching, so
            # there is no place to run a sub-plan; rejected as in FalkorDB C
            for q in ["MERGE (n:A {name: 'a'}) ON MATCH SET n.p = size([(n)-->() | 1])",
                      "MERGE (n:B) ON CREATE SET n.p = size([(x)-->() | 1])"]:
                try:
                    g.query(q)
                    self.env.assertTrue(False)
                except redis.ResponseError as e:
                    self.env.assertContains("not supported in MERGE", str(e))

            # the server is still up
            self.env.assertEqual(
                    g.query("MATCH (n:A) RETURN count(n)").result_set, [[2]])
        finally:
            g.delete()

    def test49_pattern_comprehension_in_order_by(self):
        # ORDER BY is resolved against the projected scope, but its pattern
        # comprehensions were planned below the projection, so every such
        # query failed with "Variable ? not found".
        # https://github.com/FalkorDB/FalkorDB/issues/2308
        g = self.fresh_graph("order_by_pattern_comprehension")
        try:
            g.query("""CREATE (:N {name: 'a'})-[:R]->(:N {name: 'b'}),
                              (:N {name: 'c'})""")

            queries = [
                # n is not projected: the comprehension must still see the
                # row's n, not a fresh pattern-local one
                ("""MATCH (n:N) RETURN n.name AS nm
                    ORDER BY size([(n)<--() | 1]) DESC, nm""",
                 [['b'], ['a'], ['c']]),
                ("""MATCH (n:N) RETURN DISTINCT n.name AS nm
                    ORDER BY size([(n)-->() | 1]) DESC, nm""",
                 [['a'], ['b'], ['c']]),
                # a projected alias
                ("""MATCH (n:N) RETURN n.name AS nm, n AS m
                    ORDER BY size([(m)<--() | 1]) DESC, nm""",
                 None),
                # alongside a projection comprehension, with SKIP / LIMIT
                ("""MATCH (n:N) RETURN n.name AS nm, size([(n)-->() | 1]) AS c
                    ORDER BY size([(n)<--() | 1]) DESC, nm SKIP 1 LIMIT 1""",
                 [['a', 1]]),
                ("""MATCH (n:N) WITH n ORDER BY size([(n)<--() | 1]) DESC LIMIT 1
                    RETURN n.name""",
                 [['b']]),
            ]
            for q, expected in queries:
                actual = g.query(q).result_set
                if expected is None:
                    actual = [row[:1] for row in actual]
                    expected = [['b'], ['a'], ['c']]
                self.env.assertEqual(actual, expected)
        finally:
            g.delete()

    def test50_unit_subquery_keeps_outer_cardinality(self):
        # a CALL subquery that returns nothing passes each outer row through
        # exactly once, however many rows its body produced; an uncorrelated
        # body used to multiply the outer rows by its own row count
        g = self.fresh_graph("unit_subquery_cardinality")
        try:
            g.query("CREATE (:A {name: 'a'}), (:A {name: 'b'})")

            q = "CALL { MATCH (n:A) SET n.p = 1 } RETURN 1 AS one"
            self.env.assertEqual(g.query(q).result_set, [[1]])

            q = """CALL { MATCH (n:A) SET n.p = 2 }
                   MATCH (m:A) RETURN m.name, m.p ORDER BY m.name"""
            self.env.assertEqual(g.query(q).result_set, [['a', 2], ['b', 2]])

            # a correlated body matching several rows per outer row
            q = """MATCH (n:A {name: 'a'})
                   CALL { WITH n MATCH (m:A) SET m.p = 3 }
                   RETURN n.name"""
            self.env.assertEqual(g.query(q).result_set, [['a']])

            # a body producing no rows still keeps the outer row
            q = "CALL { MATCH (n:Missing) SET n.p = 1 } RETURN 1 AS one"
            self.env.assertEqual(g.query(q).result_set, [[1]])

            # every row of the body still runs
            q = """UNWIND [1, 2] AS i
                   CALL { WITH i UNWIND range(1, i) AS j CREATE (:T {i: i, j: j}) }
                   RETURN i"""
            self.env.assertEqual(g.query(q).result_set, [[1], [2]])
            self.env.assertEqual(
                    g.query("MATCH (t:T) RETURN count(t)").result_set, [[3]])
        finally:
            g.delete()

    def test51_pattern_comprehension_reading_a_loop_variable(self):
        # a pattern comprehension that reads a list comprehension, quantifier
        # or reduce variable was hoisted out of the loop and evaluated once,
        # before the variable was bound, silently giving wrong results. It is
        # now planned as a nested plan run for every iteration, as Neo4j does.
        # https://github.com/FalkorDB/FalkorDB/issues/2308
        g = self.fresh_graph("loop_variable_pattern_comprehension")
        try:
            g.query("CREATE (:N {name: 'a', k: 1})-[:R]->(:N {name: 'b', k: 2})")
            ns = "MATCH (n:N) WITH n ORDER BY n.name WITH collect(n) AS ns "

            queries = [
                # the loop variable in the pattern
                (ns + "RETURN [x IN ns | size([(x)-->() | 1])]", [[[1, 0]]]),
                (ns + "RETURN [x IN ns | [(x)-->(y) | y.name]]", [[[['b'], []]]]),
                (ns + "RETURN [x IN ns WHERE size([(x)<--() | 1]) > 0 | x.name]",
                 [[['b']]]),
                (ns + "RETURN [x IN ns | [y IN ns | size([(x)-->(y) | 1])]]",
                 [[[[0, 1], [0, 0]]]]),
                # in the comprehension's WHERE / result only
                ("""MATCH (n:N) RETURN n.name,
                          [x IN [1, 2] | [(n)-->(y) WHERE y.k = x | y.name]]
                   ORDER BY n.name""",
                 [['a', [[], ['b']]], ['b', [[], []]]]),
                ("""MATCH (n:N) RETURN n.name, [x IN range(1, 2) | [(n)-->() | x]]
                   ORDER BY n.name""",
                 [['a', [[1], [2]]], ['b', [[], []]]]),
                # quantifiers and reduce, including the accumulator
                (ns + """RETURN any(x IN ns WHERE size([(x)-->() | 1]) > 1),
                                all(x IN ns WHERE size([(x)--() | 1]) = 1)""",
                 [[False, True]]),
                (ns + "RETURN reduce(s = 0, x IN ns | s + size([(x)-->() | 1]))",
                 [[1]]),
                (ns + """RETURN reduce(s = 0, x IN ns |
                                s + size([(x)<--(y) WHERE y.k = s + 1 | 1]))""",
                 [[1]]),
                # a bare pattern in a comprehension's own WHERE is an
                # existence test too, in a projection as in a filter
                ("""MATCH (n:N) RETURN n.name, [(n)-->(m) WHERE NOT (m)-->() | m.name]
                    ORDER BY n.name""",
                 [['a', ['b']], ['b', []]]),
                # a nested plan inside another comprehension reads that
                # comprehension's pattern variables
                (ns + """RETURN [x IN ns | [(x)-->(m) |
                                  [y IN [1] | size([(m)<--(z) WHERE z = x | y])]]]""",
                 [[[[[1]], []]]]),
                # a bare pattern in a loop predicate is an existence test
                (ns + "RETURN [x IN ns WHERE (x)-->() | x.name]", [[['a']]]),
                (ns + "RETURN none(x IN ns WHERE (x)<--()), single(x IN ns WHERE (x)-->())",
                 [[False, True]]),
            ]
            for q, expected in queries:
                self.env.assertEqual(g.query(q).result_set, expected)

            # in a write clause
            g.query(ns + "FOREACH (x IN ns | SET x.c = size([(x)-->() | 1]))")
            actual = g.query("MATCH (n:N) RETURN n.name, n.c ORDER BY n.name").result_set
            self.env.assertEqual(actual, [['a', 1], ['b', 0]])
        finally:
            g.delete()
