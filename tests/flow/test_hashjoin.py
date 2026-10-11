from common import *

GRAPH_ID = "hashjoin"


class testHashJoin(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def test_multi_hashjoins(self):
        # See issue https://github.com/RedisGraph/RedisGraph/issues/1124
        # Construct a 4 node graph, (v1),(v2),(v3),(v4)
        self.graph.query("CREATE ({val:1}), ({val:2}), ({val:3}), ({val:4})")

        # Find nodes a,b,c such that a.v = 1, a.v = b.v-1 and b.v = c.v-1
        q = "MATCH (a {val:1}), (b), (c) WHERE a.val = b.val-1 AND b.val = c.val-1 RETURN a.val, b.val, c.val"
        plan = str(self.graph.explain(q))

        # Make sure plan contains 2 Value Hash Join operations
        self.env.assertEquals(plan.count("Value Hash Join"), 2)

        # Validate results
        expected_result = [[1,2,3]]
        actual_result = self.graph.query(q)

        self.env.assertEquals(actual_result.result_set, expected_result)

    def test_argument_injection(self):
        # make sure ValueHashJoin is taken into account
        # when searching for Argument operations for data injection

        q = "CREATE (:A {id: 1}), (:B {id:2, A_id: 1})"
        self.graph.query(q)

        q = """UNWIND [1] AS current_id
        MATCH (a:A {id: current_id}), (b:B)
        WHERE b['A_id'] = a.id
        MERGE (a)-[:R]->(b)
        RETURN *
        """

        res = self.graph.query(q)
        self.env.assertEquals(res.relationships_created, 1)

        queries = [
            """UNWIND [1] AS current_id
            MATCH (a:A {id: current_id}), (b:B)
            WHERE b.A_id = a.id
            CREATE (a)-[:R]->(b)
            RETURN *
            """, 

             """UNWIND [1] AS current_id
            MATCH (a:A {id: current_id}), (b:B)
            WHERE b.A_id = a['id']
            CREATE (a)-[:R]->(b)
            RETURN *
            """, 

            """UNWIND [1] AS current_id
            MATCH (a:A {id: current_id}), (b:B)
            WHERE b['A_id'] = a['id']
            CREATE (a)-[:R]->(b)
            RETURN *
            """
        ]

        for q in queries:
            res = self.graph.query(q)
            self.env.assertEquals(res.relationships_created, 1)


    def test_join_on_maps_with_mixed_value_types(self):
        # joining on maps whose values are of different types
        # used to crash the server in OpBase_CloneRecord
        q = """CREATE (:N)-[r:R]->(:N), (h:H)-[s:S]->(h)
               SET r.x = 1.0/3.0, s.x = 'x'"""
        self.graph.query(q)

        q = """MATCH ()-[r:R]->()
               MATCH (h:H)-[s:S]->(h)
               WHERE properties(r) = properties(s)
               RETURN count(r)"""
        plan = str(self.graph.explain(q))
        self.env.assertIn("Value Hash Join", plan)

        res = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[0]])

        # one value of each type, the property is absent for null
        values = [
            "1.0/3.0",                                  # float
            "1",                                        # int
            "1.0",                                      # float equal to int
            "'x'",                                      # string
            "true",                                     # bool
            None,                                       # null
            "[1, 'x']",                                 # list
            "vecf32([0.5, 0.25])",                      # vecf32
            "date('2024-01-01')",                       # date
            "localdatetime('2024-01-01T00:00:00')",     # datetime
            "duration('P1D')",                          # duration
        ]

        self.graph.delete()
        nodes = []
        for label in ["A", "B"]:
            for v in values:
                props = "" if v is None else f" {{x: {v}}}"
                nodes.append(f"(:{label}{props})")
        self.graph.query(f"CREATE {', '.join(nodes)}")

        q = "MATCH (n) WHERE n.x IS NULL RETURN id(n)"
        null_ids = {r[0] for r in self.graph.query(q).result_set}
        self.env.assertEquals(len(null_ids), 2)

        # every (a, b) pair and the scalar comparison of their values
        # expressions are evaluated per pair, without a join
        exps = {
            "scalar":     "a.x = b.x",
            "props":      "properties(a) = properties(b)",
            "map":        "{x: a.x} = {x: b.x}",
            "nested_map": "{x: {y: a.x}} = {x: {y: b.x}}",
            "map_list":   "[{x: a.x}] = [{x: b.x}]",
            "map_in":     "{x: a.x} IN [{x: b.x}]",
            "scalar_in":  "a.x IN [b.x]",
        }
        q = f"""MATCH (a:A), (b:B)
                RETURN id(a), id(b), {', '.join(exps.values())}"""
        res = self.graph.query(q).result_set
        self.env.assertEquals(len(res), len(values) ** 2)

        expected = {name: set() for name in exps}
        for row in res:
            pair = (row[0], row[1])
            cols = dict(zip(exps.keys(), row[2:]))

            # a map holding mixed value types compares
            # the same way as the values themselves
            for name in ["map", "nested_map", "map_list"]:
                self.env.assertEquals(cols[name], cols["scalar"],
                                      message=f"{name} {pair}")
            self.env.assertEquals(cols["map_in"], cols["scalar_in"],
                                  message=f"map_in {pair}")

            # properties() drops null values, two empty maps are equal
            if cols["scalar"] is None:
                both_null = row[0] in null_ids and row[1] in null_ids
                self.env.assertEquals(cols["props"], both_null,
                                      message=f"props {pair}")
            else:
                self.env.assertEquals(cols["props"], cols["scalar"],
                                      message=f"props {pair}")

            for name, v in cols.items():
                if v is True:
                    expected[name].add(pair)

        # the same comparisons as a WHERE filter, planned as a hash join,
        # and as a projected boolean, must return exactly the 'true' pairs
        for name in ["props", "map", "nested_map", "map_list"]:
            exp = exps[name]
            q = f"MATCH (a:A), (b:B) WHERE {exp} RETURN id(a), id(b)"
            plan = str(self.graph.explain(q))
            self.env.assertIn("Value Hash Join", plan)

            actual = {(r[0], r[1]) for r in self.graph.query(q).result_set}
            self.env.assertEquals(actual, expected[name], message=name)

            q = f"""MATCH (a:A), (b:B)
                    WITH a, b, {exp} AS same
                    WHERE same
                    RETURN id(a), id(b)"""
            actual = {(r[0], r[1]) for r in self.graph.query(q).result_set}
            self.env.assertEquals(actual, expected[name], message=name)

    def test_join_on_nan(self):
        # NaN is not equal to NaN, a join must not match it
        self.graph.query("CREATE (:A {v: 0.0/0.0}), (:B {v: 0.0/0.0})")

        q = "MATCH (a:A), (b:B) WHERE a.v = b.v RETURN count(*)"
        plan = str(self.graph.explain(q))
        self.env.assertIn("Value Hash Join", plan)

        res = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[0]])

        q = "MATCH (a:A), (b:B) WHERE {v: a.v} = {v: b.v} RETURN count(*)"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[0]])
