from common import *
import itertools
import re

values = ["str1", "str2", False, True, 5, 10.5]
GRAPH_ID = "with_caluse"

class testWithClause(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()
 
    def populate_graph(self):
        # Populate a graph with two labels, each containing the same property values but different keys.
        # Each node pair is connected by an edge from label_a to label_b
        nodes = []
        edges = []
        for idx, v in enumerate(values):
            src  = Node(alias=f"s{idx}", labels="label_a", properties={"a_val": v, "a_idx": idx})
            dest = Node(alias=f"t{idx}", labels="label_b", properties={"b_val": v, "b_idx": idx})
            e    = Edge(src, 'connects', dest, properties={"edgeval": idx})
            nodes.append(src)
            nodes.append(dest)
            edges.append(e)

        nodes_str = [str(n) for n in nodes]
        edges_str = [str(e) for e in edges]
        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")
    
    # Verify that graph entities specified in a WITH clause are returned properly
    def test01_with_scalar_read_queries(self):
        query = """MATCH (a:label_a) WITH a.a_val AS val RETURN val ORDER BY val"""
        actual_result = self.graph.query(query)
        expected = [['str1'],
                    ['str2'],
                    [False],
                    [True],
                    [5],
                    [10.5]]
        self.env.assertEqual(actual_result.result_set, expected)


        query = """MATCH (a:label_a) WITH a.a_val AS val ORDER BY val SKIP 1 LIMIT 1 RETURN val"""
        actual_result = self.graph.query(query)
        expected = [['str2']]
        self.env.assertEqual(actual_result.result_set, expected)


        query = """MERGE (a:label_a {a_val: 5}) WITH a.a_val AS val ORDER BY val RETURN val"""
        actual_result = self.graph.query(query)
        expected = [[5]]
        self.env.assertEqual(actual_result.properties_set, 0)

        self.env.assertEqual(actual_result.result_set, expected)


        # Merge on existing edge
        query = """MERGE ()-[e:connects {edgeval: 5}]->() WITH e.edgeval AS val RETURN val"""
        actual_result = self.graph.query(query)
        expected = [[5]]
        self.env.assertEqual(actual_result.properties_set, 0)

        self.env.assertEqual(actual_result.result_set, expected)

    def test02_with_arithmetic_op_read_queries(self):
        # Iterate over nodes
        query = """MATCH (a) WITH ID(a) AS id RETURN id ORDER BY id"""
        actual_result = self.graph.query(query)
        expected = [[0],
                    [1],
                    [2],
                    [3],
                    [4],
                    [5],
                    [6],
                    [7],
                    [8],
                    [9],
                    [10],
                    [11]]
        self.env.assertEqual(actual_result.result_set, expected)

        # Iterate over edges
        query = """MATCH ()-[e]->() WITH ID(e) AS id RETURN id ORDER BY id"""
        actual_result = self.graph.query(query)

        expected = [[0],
                    [1],
                    [2],
                    [3],
                    [4],
                    [5]]
        self.env.assertEqual(actual_result.result_set, expected)

        # Perform chained arithmetic ops
        query = """MATCH (a)-[]->(b) WHERE a.a_val > 0 AND b.b_val > 0 WITH a.a_val * 2 + b.b_val AS val ORDER BY val RETURN val"""
        actual_result = self.graph.query(query)

        expected = [[15],
                    [31.5]]
        self.env.assertEqual(actual_result.result_set, expected)

    def test03_with_aggregate_op_read_queries(self):
        query = """MATCH (a)-[e]->() WITH COUNT(a.a_val) AS count_res, SUM(ID(e)) AS sum_res RETURN count_res, sum_res"""
        actual_result = self.graph.query(query)
        expected = [[6, 15]]
        self.env.assertEqual(actual_result.result_set, expected)

    # TODO UNWIND support needs to be extended for combinations like UNWIND...MATCH
    def test04_with_unwind_expressions(self):
        query = """UNWIND [1, 2, 3] AS x WITH x AS y RETURN y"""
        actual_result = self.graph.query(query)

        expected = [[1],
                    [2],
                    [3]]
        self.env.assertEqual(actual_result.result_set, expected)

        query = """UNWIND [1, 2, 3] AS x WITH x * 2 AS y RETURN y"""
        actual_result = self.graph.query(query)

        expected = [[2],
                    [4],
                    [6]]
        self.env.assertEqual(actual_result.result_set, expected)

        query = """UNWIND [1, 2, 3] AS x WITH x * 2 AS y WITH y * 2 AS z RETURN z"""
        actual_result = self.graph.query(query)

        expected = [[4],
                    [8],
                    [12]]
        self.env.assertEqual(actual_result.result_set, expected)

        query = """UNWIND [1, 2, 3] AS x WITH x * 2 AS y WITH y * 2 AS z WITH MIN(z) as min RETURN min"""
        actual_result = self.graph.query(query)

        expected = [[4]]
        self.env.assertEqual(actual_result.result_set, expected)

    def test05_with_create_expressions(self):
        query = """CREATE (c:c_label {c_val: 25}) WITH c AS c RETURN c.c_val AS val"""
        actual_result = self.graph.query(query)
        expected = [[25]]
        self.env.assertEqual(actual_result.result_set, expected)
        self.env.assertEqual(actual_result.labels_added, 1)
        self.env.assertEqual(actual_result.nodes_created, 1)
        self.env.assertEqual(actual_result.properties_set, 1)

        # TODO Update CREATE to accept variable arguments from UNWIND, WITH, etc
        query = """UNWIND [5] AS a WITH a AS b CREATE (:unwind_label {prop: 'some_constant'})"""
        actual_result = self.graph.query(query)

        self.env.assertEqual(actual_result.labels_added, 1)
        self.env.assertEqual(actual_result.nodes_created, 1)
        self.env.assertEqual(actual_result.properties_set, 1)
        query = """MATCH (a:unwind_label) RETURN a.prop"""
        actual_result = self.graph.query(query)

        # Verify that the entity was created properly
        expected = [['some_constant']]
        self.env.assertEqual(actual_result.result_set, expected)

    def test06_update_expressions(self):
        query = """MATCH (c:c_label) SET c.c_val = 50 WITH c.c_val AS val RETURN val"""
        actual_result = self.graph.query(query)
        expected = [[50]]
        self.env.assertEqual(actual_result.result_set, expected)
        self.env.assertEqual(actual_result.properties_set, 1)

    # Verify that projected nodes, edges, and scalars can be returned properly
    def test07_projected_graph_entities(self):
        query = """MATCH (a)-[e]->(b) WITH a, e, b.b_val AS b_val ORDER BY a.a_val LIMIT 2 RETURN *"""
        actual_result = self.graph.query(query)

        # Validate the header strings of the 3 columns.
        # NOTE - currently, RETURN * populates values in alphabetical order, but that is subject to later change.
        self.env.assertEqual(actual_result.header[0][1], 'a')
        self.env.assertEqual(actual_result.header[1][1], 'b_val')
        self.env.assertEqual(actual_result.header[2][1], 'e')

        # Verify that 2 rows and 3 columns are returned.
        self.env.assertEqual(len(actual_result.result_set), 2)
        self.env.assertEqual(len(actual_result.result_set[0]), 3)

    def test08_filter_projected(self):
        # If x is odd then ceil(x/2) > floor(x/2)
        # otherwise, x is even and ceil(x/2) == floor(x/2).
        # ceil(5/2) = 3, floor(5/2) = 2
        # ceil(6/2) = 3, floor(6/2) = 3.
        query = """unwind range(0, 10) as x with x as x where ceil(x/2.0) > floor(x/2.0) return count(x)"""
        actual_result = self.graph.query(query)
        # Expecting count of 5: [1,3,5,7,9].
        self.env.assertEqual(actual_result.result_set[0], [5])

    # Verify that filters can properly be placed in the scope up to WITH and the expressions projected by it.
    def test09_filter_placement(self):
        # Place a filter on a projected expression.
        query = """UNWIND [1,2,3] AS a WITH a WHERE a = 2 RETURN a"""
        actual_result = self.graph.query(query)
        expected = [[2]]
        self.env.assertEqual(actual_result.result_set, expected)

        # Place a filter on a projected expression that is aliased by the WITH clause.
        query = """UNWIND [1,2,3] AS a WITH a AS b WHERE a = 2 RETURN b"""
        actual_result = self.graph.query(query)
        expected = [[2]]
        self.env.assertEqual(actual_result.result_set, expected)

    def test10_filter_placement_validate_scopes(self):
        # Verify that filters cannot be placed in earlier scopes.
        query = """UNWIND ['scope1'] AS a
                   WITH a AS b
                   UNWIND ['scope2'] AS a
                   WITH a
                   WHERE a = 'scope1'
                   RETURN a"""
        actual_result = self.graph.query(query)
        expected = [] # No results should be returned
        self.env.assertEqual(actual_result.result_set, expected)

        query = """UNWIND ['scope1'] AS a WITH a AS b UNWIND ['scope2'] AS a WITH a WHERE a = 'scope2' RETURN a"""
        actual_result = self.graph.query(query)
        expected = [['scope2']]
        self.env.assertEqual(actual_result.result_set, expected)

        # Verify that WITH filters are properly placed in scope without violating Apply restrictions.
        query = """MATCH (a) OPTIONAL MATCH (b) WITH a, b WHERE b.fakeprop = true RETURN a, b"""
        actual_result = self.graph.query(query)
        expected = [] # No results should be returned
        self.env.assertEqual(actual_result.result_set, expected)
        # TODO - See what if we can do it
        # # Verify that the Filter op appears directly above the Apply operation in the ExecutionPlan.
        # plan = str(self.graph.explain(query))
        # self.env.assertTrue(re.search('Filter\s+Apply', plan))

        # Verify that filters on projected aliases do not get placed before the projection op.
        query = """UNWIND [1] AS a WITH a AS b, 'projected' AS a WHERE a = 1 RETURN a"""
        plan = str(self.graph.explain(query))
        actual_result = self.graph.query(query)
        expected = [] # No results should be returned
        self.env.assertEqual(actual_result.result_set, expected)
        self.env.assertTrue(re.search('Filter\s+Project', plan))

        query = """UNWIND [1] AS a WITH a AS b, 'projected' AS a WHERE a = 'projected' RETURN a"""
        plan = str(self.graph.explain(query))
        actual_result = self.graph.query(query)
        expected = [['projected']] # The projected string should be returned
        self.env.assertTrue(re.search('Filter\s+Project', plan))

    def test11_valid_order_by_aliases(self):
        # Verify that ORDER BY aliases match previously defined references
        query = """UNWIND [1,2,3] AS a WITH a ORDER BY a RETURN a"""
        actual_result = self.graph.query(query)
        expected_result = [[1],
                           [2],
                           [3]]
        self.env.assertEqual(actual_result.result_set, expected_result)

        invalid_queries = ["UNWIND [1,2,3] AS a WITH a ORDER BY nonexistent RETURN a",
                           "UNWIND [1,2,3] AS a WITH a ORDER BY a, nonexistent RETURN a",
                           "UNWIND [1,2,3] AS a WITH a AS b UNWIND [1,2,3] AS c WITH c ORDER BY a RETURN c"]
        for query in invalid_queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                # Expecting an error.
                self.env.assertContains("not defined", str(e))

    def test12_cartesian_product_reset_single_response(self):
        # Verify that WITH projections that with no children are reset
        # properly by CartesianProduct ops
        query = """WITH 1 AS x MATCH (a:label_a), (b:label_b) RETURN a.v, b.v"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(len(actual_result.result_set), 36)

    def test13_rematch_multiple_carried_variables(self):
        # Re-MATCHing several variables that a WITH already bound used to
        # either crash the server or fail to resolve the alias the WHERE
        # referred to. Re-matching a single variable always worked, so the
        # regression only shows up from the second carried variable onwards.
        # https://github.com/FalkorDB/FalkorDB/issues/1949
        # https://github.com/FalkorDB/FalkorDB/issues/2004
        g = self.db.select_graph("with_rematch_bound")

        try:
            g.query("""CREATE (:Person {name: 'Alice', age: 30}),
                              (:Person {name: 'Bob', age: 25})""")

            # the reported repro - re-matching both carried variables
            res = g.query("""MATCH (a:Person {name: 'Alice'}),
                                   (b:Person {name: 'Bob'})
                             WITH a, b
                             MATCH (a), (b)
                             RETURN a.name AS name, b.name AS name2""")
            self.env.assertEqual(res.result_set, [['Alice', 'Bob']])

            # the same with a WHERE filtering both re-matched aliases
            res = g.query("""MATCH (a:Person), (b:Person)
                             WITH a, b
                             MATCH (a), (b)
                             WHERE a.age > 25 AND b.age < 30
                             RETURN a.name AS a, b.name AS b""")
            self.env.assertEqual(res.result_set, [['Alice', 'Bob']])

            # a variable carried under an alias by WITH * is re-matchable too
            # https://github.com/FalkorDB/FalkorDB/issues/1280
            res = g.query("""OPTIONAL MATCH (n0)
                             WITH *, n0 AS alias4
                             MATCH (n0), (alias4)
                             RETURN n0.name, alias4.name ORDER BY n0.name""")
            self.env.assertEqual(res.result_set,
                                 [['Alice', 'Alice'], ['Bob', 'Bob']])

            # re-matching inside OPTIONAL MATCH, filtering a bound alias
            # https://github.com/FalkorDB/FalkorDB/issues/1300
            res = g.query("""MATCH (n1), (n0)
                             OPTIONAL MATCH (n0), (n1)
                             WHERE n0
                             RETURN n0.name, n1.name
                             ORDER BY n0.name, n1.name""")
            self.env.assertEqual(res.result_set,
                                 [['Alice', 'Alice'], ['Alice', 'Bob'],
                                  ['Bob', 'Alice'], ['Bob', 'Bob']])

            # the same re-match without a WHERE - no alias to resolve, yet it
            # still crashed the server outright
            # https://github.com/FalkorDB/FalkorDB/issues/1281
            res = g.query("""MATCH (n), (n1)
                             OPTIONAL MATCH (n), (n1)
                             RETURN n.name, n1.name
                             ORDER BY n.name, n1.name""")
            self.env.assertEqual(res.result_set,
                                 [['Alice', 'Alice'], ['Alice', 'Bob'],
                                  ['Bob', 'Alice'], ['Bob', 'Bob']])

            # re-matching a single variable kept working throughout
            res = g.query("""MATCH (a:Person {name: 'Alice'})
                             WITH a MATCH (a) RETURN a.name AS name""")
            self.env.assertEqual(res.result_set, [['Alice']])
        finally:
            g.delete()

    def test14_rematch_filtering_a_projected_scalar_alias(self):
        # The alias a WITH carries into a later MATCH need not be a node - a
        # projected *scalar* used as the WHERE predicate hit a separate
        # resolution path and failed with "Unable to resolve filtered alias".
        # https://github.com/FalkorDB/FalkorDB/issues/1301
        g = self.db.select_graph("with_rematch_scalar_alias")

        try:
            g.query("""CREATE (:T {name: 'n1'})<-[:E]-(:T {name: 'n0', k1: true})
                              <-[:E]-(:T {name: 'n2'})""")

            # the reported repro - alias0 is n0.k1, a boolean, and drives the
            # filter of a MATCH that re-binds variables the WITH carried
            res = g.query("""MATCH p0 = (n1)<-[]-(n0)<-[]-(n2)
                             WITH *, n0.k1 AS alias0, n1
                             MATCH (n2), (n1)
                             WHERE alias0
                             RETURN n1.name, n2.name, alias0""")
            self.env.assertEqual(res.result_set, [['n1', 'n2', True]])

            # negating the same alias keeps it resolvable and filters the row
            # out, proving the predicate is really evaluated rather than
            # dropped
            res = g.query("""MATCH p0 = (n1)<-[]-(n0)<-[]-(n2)
                             WITH *, n0.k1 AS alias0, n1
                             MATCH (n2), (n1)
                             WHERE NOT alias0
                             RETURN n1.name""")
            self.env.assertEqual(res.result_set, [])

            # a non-boolean projected alias must still resolve - the failure
            # has to be the type of the predicate, never the alias lookup
            g.query("MATCH (t:T {name: 'n0'}) SET t.k1 = 7")
            try:
                g.query("""MATCH p0 = (n1)<-[]-(n0)<-[]-(n2)
                           WITH *, n0.k1 AS alias0, n1
                           MATCH (n2), (n1)
                           WHERE alias0
                           RETURN n1.name""")
                assert(False)
            except redis.ResponseError as e:
                self.env.assertContains("Type mismatch", str(e))
        finally:
            g.delete()

    def test15_hidden_projection_variables_keep_their_slots(self):
        # Variables a WITH/RETURN copies in only for its WHERE/ORDER BY/SKIP/
        # LIMIT, and the locals of a pattern comprehension, go out of scope
        # after it, but the next variable of that scope must not get their
        # record slot: it would read their value.
        g = self.db.select_graph("with_hidden_slots")
        g.query("CREATE (:A {v:1})-[:R {w:2}]->(:B {v:0}), (:B {v:5})")
        try:
            cases = [
                ("MATCH (n) WITH n.v AS k WHERE n.v = 0 MATCH (x)-->(y) RETURN k, x.v, y.v",
                 [[0, 1, 0]]),
                ("MATCH (n) WITH n.v AS k WHERE n.v >= 0 RETURN k, [(x)-->(y) | y.v] AS l ORDER BY k",
                 [[0, [0]], [1, [0]], [5, [0]]]),
                ("MATCH (n) WITH n, 1 AS o ORDER BY size([(n)-[r]->(m) | m]) MATCH (x)-->(y) "
                 "RETURN n.v, x.v, y.v ORDER BY n.v",
                 [[0, 1, 0], [1, 1, 0], [5, 1, 0]]),
                ("MATCH (n:A) WHERE size([(n)-[r]->(m) | m]) > 0 MATCH (x)-[s]->(y) "
                 "WHERE size([(x)-[r2]->(m2) | m2]) > 0 RETURN n.v, x.v, s.w, y.v",
                 [[1, 1, 2, 0]]),
            ]
            for q, expected in cases:
                self.env.assertEqual(g.query(q).result_set, expected, message=q)
        finally:
            g.delete()

    def test16_order_by_variables_do_not_cross_the_projection(self):
        # A variable ORDER BY reads from before the projection is not
        # projected: it is not visible after WITH/RETURN, and not returned
        # from a CALL {} body.
        g = self.db.select_graph("with_order_by_scope")
        g.query("CREATE (:A {v:1}), (:B {v:0}), (:B {v:5})")
        try:
            try:
                g.query("MATCH (n) WITH n.v AS k ORDER BY n.v RETURN n")
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                self.env.assertContains("'n' not defined", str(e))

            res = g.query("MATCH (n) WITH n.v AS k ORDER BY n.v RETURN *")
            self.env.assertEqual(res.header[0][1], 'k')
            self.env.assertEqual(len(res.header), 1)
            self.env.assertEqual(res.result_set, [[0], [1], [5]])

            res = g.query("MATCH (n) CALL { WITH n RETURN n.v AS k ORDER BY n.v } RETURN k ORDER BY k")
            self.env.assertEqual(res.result_set, [[0], [1], [5]])
        finally:
            g.delete()

    def test17_hidden_slot_key_does_not_clobber_an_alias(self):
        # Hidden variables used to be kept under a made-up string key such as
        # `_slot_1_2` (the copy of `n`, scope 1, id 2), which is also a legal
        # alias. It overwrote the alias, so the later MATCH lost its rows.
        # They are now kept in a nameless list, so no alias can reach them.
        g = self.db.select_graph("with_slot_alias")
        g.query("CREATE (:A {v:1})-[:R {w:2}]->(:B {v:0}), (:B {v:5})")
        try:
            res = g.query("""MATCH (n) WITH n.v AS k, n.v + 0 AS _slot_1_2 WHERE n.v = 0
                             MATCH (x)-->(y) RETURN k, _slot_1_2, x.v, y.v""")
            self.env.assertEqual(res.result_set, [[0, 0, 1, 0]])
        finally:
            g.delete()

    def test18_hidden_slot_is_not_nameable_and_underscore_locals_leave_scope(self):
        g = self.db.select_graph("with_slot_nameable")
        g.query("CREATE (:A {v:1})-[:R {w:2}]->(:B {v:0}), (:B {v:5})")
        try:
            # A later alias spelling an old hidden key gets a fresh variable,
            # not the filter-only copy of `n`: the MATCH sees (_slot_1_2) unbound.
            res = g.query("""MATCH (n) WITH n.v AS k, n.v + 0 AS o WHERE n.v = 0
                             MATCH (_slot_1_2)-->(y) RETURN k, y.v""")
            self.env.assertEqual(res.result_set, [[0, 0]])
            for alias in ("_slot_#1_2", "_slot_1_2_"):
                res = g.query(f"""MATCH (n) WITH n.v AS k, n.v + 0 AS o WHERE n.v = 0
                                  MATCH (`{alias}`)-->(y) RETURN k, y.v""")
                self.env.assertEqual(res.result_set, [[0, 0]], depth=1)

            # A `_`-prefixed local of a pattern predicate is not visible after it,
            # like any other local.
            try:
                g.query("MATCH (n:A) WHERE (n)-->(_l) RETURN _l")
                self.env.assertTrue(False)
            except redis.ResponseError as e:
                self.env.assertContains("'_l' not defined", str(e))
        finally:
            g.delete()

    def test19_names_spelling_internal_reservations_behave_like_any_other(self):
        # Every scope-table reservation (hidden WITH/ORDER BY copies, pattern
        # locals, list/quantifier/reduce variables, FOREACH locals, aggregate
        # order placeholders) used to live under a string key a query could
        # spell. Each query below runs twice: with the colliding names, and
        # with those names replaced by `w0`, `w1`, ...: the results must match.
        g = self.db.select_graph("with_internal_names")
        g.query("CREATE (:A {id:0, v:1})-[:R {k:2}]->(:B {id:1, v:0}), (:B {id:2, v:5})-[:S {k:3}]->(:A {id:3, v:1})")
        names = ["_slot_0_1", "_slot_1_2", "_slot_", "_foreach_0_1", "_quant_0_1", "_lc_0_1",
                 "_reduce_acc_0_1", "_reduce_iter_0_2", "__agg_placeholder_0", "__agg_placeholder_1",
                 "__agg_order_by_placeholder__"]
        queries = [
            # hidden WHERE / ORDER BY copies of a WITH
            "MATCH (n) WITH n.v AS @0 WHERE n.v = 0 MATCH (x)-->(y) RETURN @0, x.v, y.v",
            "MATCH (n) WITH n.v AS @0, n.v + 0 AS @1 WHERE n.v = 0 MATCH (@2)-->(y) RETURN @0, @1, y.v",
            "MATCH (n) WITH n.v AS @0 ORDER BY n.v MATCH (x)-->(y) RETURN @0, x.v, y.v ORDER BY @0",
            "MATCH (n) WITH n.v AS @0 ORDER BY n.v LIMIT 2 WITH @0 MATCH (x)-->(y) RETURN @0, x.v, y.v",
            # aggregation
            "MATCH (n) WITH n.v AS @0, count(*) AS @1 WHERE @1 >= 0 MATCH (x)-->(y) RETURN @0, @1, x.v, y.v",
            "MATCH (n) WITH n.v AS @0, count(*) AS @1 ORDER BY @1 RETURN @0, @1",
            "MATCH (n) WITH n, count(*) AS @0 RETURN n.id, @0, collect(n.id) AS c ORDER BY n.id",
            "MATCH (n) WITH n.v AS @0 RETURN @0, count(*) AS @1 ORDER BY @0",
            # pattern predicates / comprehensions
            "MATCH (n) WITH n.v AS k WHERE n.v >= 0 RETURN k, [(n)-->(@0) | @0.v] AS l ORDER BY k",
            "MATCH (n:A) WHERE (n)-->(@0) MATCH (x)-[s]->(y) RETURN n.id, x.id, s.k, y.id ORDER BY n.id, x.id",
            "MATCH (n:A) WHERE size([(@0)-->(m) | m]) >= 0 MATCH (x)-[s]->(y) RETURN n.id, x.id, y.id ORDER BY n.id, x.id",
            # list comprehension, quantifier, reduce
            "MATCH (n) WITH n, [@0 IN [1, 2] WHERE @0 > 0 | @0 + n.v] AS l UNWIND [0, 1] AS @1 RETURN n.id, l, @1 ORDER BY n.id, @1",
            "MATCH (n) WHERE any(@0 IN [0, 1] WHERE @0 = n.v) MATCH (m) RETURN n.id, m.id ORDER BY n.id, m.id",
            "MATCH (n) WHERE all(@0 IN [0, 1, 2, 5] WHERE @0 >= 0) WITH n.v AS @1 ORDER BY n.v RETURN @1 ORDER BY @1",
            "MATCH (n) RETURN n.id, reduce(@0 = 0, @1 IN [1, 2] | @0 + @1 + n.v) AS r ORDER BY n.id",
            "MATCH (n) WITH n.v AS @2 ORDER BY n.v RETURN @2, reduce(@0 = 0, @1 IN [1, 2] | @0 + @1) AS r ORDER BY @2",
            # CALL {}
            "MATCH (n) CALL { WITH n RETURN n.v AS @0 ORDER BY n.v } RETURN n.id, @0 ORDER BY n.id",
            "MATCH (n:A) CALL { WITH n MATCH (n)-[:R]->(@0) RETURN @0 } MATCH (@1:B) RETURN n.id, @0.id, @1.id ORDER BY @1.id",
            "MATCH (n) CALL { WITH n MATCH (n)-->(z) RETURN count(z) AS @0 } WITH n.id AS i, @0 ORDER BY n.v RETURN i, @0 ORDER BY i",
            "MATCH (n) WITH n.v AS @0 WHERE n.v >= 0 CALL { WITH @0 RETURN @0 + 1 AS @1 } RETURN @0, @1 ORDER BY @0",
            # UNION
            "MATCH (n:A) WITH n.v AS @0 ORDER BY n.v RETURN @0 AS o UNION MATCH (m:B) WITH m.v AS @0 ORDER BY m.v RETURN @0 AS o",
            "MATCH (n:A) CALL { WITH n RETURN n.v AS @0 UNION WITH n RETURN 7 AS @0 } RETURN n.id, @0 ORDER BY n.id, @0",
            # OPTIONAL MATCH, UNWIND, RETURN *
            "MATCH (n:A) OPTIONAL MATCH (n)-[@0:R]->(@1) WITH n, @0, @1 WHERE n.v >= 0 UNWIND [0, 1] AS @2 RETURN n.id, @0.k, @1.id, @2 ORDER BY @2",
            "MATCH (n) WITH n.v AS k ORDER BY n.v MATCH (@0) RETURN k, @0.id ORDER BY k, @0.id",
        ]
        # FOREACH (writes), checked by read-back
        foreach = [
            ("MATCH (n:A) FOREACH (@0 IN [1, 2] | SET n.f = @0) WITH n MATCH (x)-->(y) RETURN n.id, n.f, x.id, y.id ORDER BY n.id, x.id", None),
            ("MATCH (n:A) FOREACH (@0 IN [1, 2] | CREATE (:T {x: @0})) WITH n MATCH (@1:T) RETURN n.id, @1.x ORDER BY n.id, @1.x", None),
        ]
        try:
            def rows(q):
                try:
                    # compared as multisets: most queries have no total order
                    return sorted(g.query(q).result_set, key=repr)
                except redis.ResponseError as e:
                    return "ERR " + str(e)
            def fill(tmpl, vals):
                for i, v in enumerate(vals):
                    tmpl = tmpl.replace(f"@{i}", v)
                return tmpl

            for i, tmpl in enumerate(queries + [f for f, _ in foreach]):
                for combo in itertools.islice(itertools.permutations(names, 3), i % 7, None, 41):
                    q = fill(tmpl, combo)
                    plain = fill(tmpl, ("w0", "w1", "w2"))
                    if tmpl in [f for f, _ in foreach]:
                        g.query("MATCH (n:A) REMOVE n.f WITH n MATCH (t:T) DELETE t")
                    got = rows(q)
                    if tmpl in [f for f, _ in foreach]:
                        g.query("MATCH (n:A) REMOVE n.f WITH n MATCH (t:T) DELETE t")
                    want = rows(plain)
                    self.env.assertEqual(got, want, message=q)
                    self.env.assertFalse(isinstance(want, str) and want.startswith("ERR 'w"), message=plain)
        finally:
            g.delete()

    def test20_foreach_after_hidden_variables_does_not_multiply_reservations(self):
        # FOREACH keeps the ids of its body's variables reserved. It must not
        # re-reserve the ones the scope already holds (a hidden WITH copy, an
        # earlier FOREACH's locals), or every FOREACH doubles the scope table:
        # 24 of them took 5 s and 2 GB. The scope-table size itself is asserted
        # by the unit test `scope_tables_list_each_variable_once`; this query is
        # kept small enough to fail fast rather than stall.
        g = self.db.select_graph("with_foreach_reservations")
        g.query("CREATE (:A {v: 1})")
        try:
            body = " ".join(f"FOREACH (i{j} IN [1] | CREATE (:T {{x: i{j}}}))" for j in range(12))
            res = g.query(f"MATCH (n:A) WITH n.v AS k WHERE n.v >= 0 {body} RETURN k")
            self.env.assertEqual(res.result_set, [[1]])
            self.env.assertEqual(res.nodes_created, 12)
        finally:
            g.delete()
