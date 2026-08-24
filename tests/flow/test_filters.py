from common import *

GRAPH_ID = "filters"

class testFilters():
    def __init__(self):
        self.env, self.db = Env()
        self.g = self.db.select_graph(GRAPH_ID)

    def test01_filter_with_different_predicates(self):
        self.g.query("UNWIND range(1, 5) AS x CREATE (:N { v: x, b: x % 2 = 0 })")

        # test and operation
        expected = [[i, j] for i in range(1, 6) for j in range(1, 6) if i % 2 == 0 and j % 2 == 0]
        result = self.g.query("MATCH (n:N), (m:N) WHERE n.b AND m.b RETURN n.v, m.v ORDER BY n.v, m.v")
        self.env.assertEqual(result.result_set,  expected)

        # test or operation
        expected = [[i, j] for i in range(1, 6) for j in range(1, 6) if i % 2 == 0 or j % 2 == 0]
        result = self.g.query("MATCH (n:N), (m:N) WHERE n.b OR m.b RETURN n.v, m.v ORDER BY n.v, m.v")
        self.env.assertEqual(result.result_set,  expected)

        # test xor operation
        expected = [[i, j] for i in range(1, 6) for j in range(1, 6) if i % 2 != j % 2]
        result = self.g.query("MATCH (n:N), (m:N) WHERE n.b XOR m.b RETURN n.v, m.v ORDER BY n.v, m.v")
        self.env.assertEqual(result.result_set,  expected)

        # test negation of and operation
        expected = [[i, j] for i in range(1, 6) for j in range(1, 6) if not (i % 2 == 0 and j % 2 == 0)]
        result = self.g.query("MATCH (n:N), (m:N) WHERE NOT (n.b AND m.b) RETURN n.v, m.v ORDER BY n.v, m.v")
        self.env.assertEqual(result.result_set,  expected)

        # test negation of or operation
        expected = [[i, j] for i in range(1, 6) for j in range(1, 6) if not (i % 2 == 0 or j % 2 == 0)]
        result = self.g.query("MATCH (n:N), (m:N) WHERE NOT (n.b OR m.b) RETURN n.v, m.v ORDER BY n.v, m.v")
        self.env.assertEqual(result.result_set,  expected)

        # test negation of xor operation
        expected = [[i, j] for i in range(1, 6) for j in range(1, 6) if not (i % 2 != j % 2)]
        result = self.g.query("MATCH (n:N), (m:N) WHERE NOT (n.b XOR m.b) RETURN n.v, m.v ORDER BY n.v, m.v")
        self.env.assertEqual(result.result_set,  expected)

    def test02_filter_with_null(self):
        conditions = [("null", None), ("true", True), ("false", False), ("x", True), ("y", False), ("z", None)]
        for c in conditions:
            q = "WITH true AS x, false AS y, null AS z WHERE %s RETURN x" % c[0]
            result = self.g.query(q)
            expected = [[True]] if c[1] else []
            self.env.assertEqual(result.result_set,  expected)

        def null_and(a, b):
            if a is not None and b is not None:
                return a and b
            elif a is not None and not a:
                return False
            elif b is not None and not b:
                return False
            return None
        def null_or(a, b):
            if a is not None and b is not None:
                return a or b
            elif a is not None and a:
                return True
            elif b is not None and b:
                return True
            return None
        ops = [("AND", lambda a, b : null_and(a, b)), ("OR", lambda a, b : null_or(a, b)), ("XOR", lambda a, b : None if a is None or b is None else a ^ b)]
        for op in ops:
            for c1 in conditions:
                for c2 in conditions:
                    q = "WITH true AS x, false AS y, null AS z WHERE %s %s %s RETURN x" % (c1[0], op[0], c2[0])
                    result = self.g.query(q)
                    expected = [[True]] if op[1](c1[1], c2[1]) else []
                    if result.result_set != expected:
                        print(q)
                    self.env.assertEqual(result.result_set,  expected)

        for op1 in ops:
            for op2 in ops:
                for c1 in conditions:
                    for c2 in conditions:
                        for c3 in conditions:
                            q = "WITH true AS x, false AS y, null AS z WHERE (%s %s %s) %s %s RETURN x" % (c1[0], op1[0], c2[0], op2[0], c3[0])
                            result = self.g.query(q)
                            expected = [[True]] if op2[1](op1[1](c1[1], c2[1]), c3[1]) else []
                            if result.result_set != expected:
                                print(q)
                            self.env.assertEqual(result.result_set,  expected)

    def test03_filter_with_nan(self):
        res = self.g.query("WITH 1 AS x WHERE 0.0 / 0.0 = 0.0 / 0.0 RETURN x")
        self.env.assertEqual(res.result_set, [])

        res = self.g.query("WITH 1 AS x WHERE 0.0 / 0.0 <> 0.0 / 0.0 RETURN x")
        self.env.assertEqual(res.result_set, [[1]])

    def test04_redundant_filter(self):
        q = """MATCH (n), (), ()
               WHERE ('a' <= ('km' + 'X'))
               RETURN *"""
        plan = self.g.explain(q)
        self.env.assertFalse('Filter' in plan)


    def test05_bulk_filter_matches_scalar_eval(self):
        # A filter on `var.prop <op> constant` runs through the columnar
        # comparison kernels, while the same expression in a RETURN is
        # evaluated per row. The two must agree for every property type.
        #
        # Issue #2582: the kernel answered `false` for every non-string cell of
        # `prop <> 'string'`, so `WHERE n.v <> 'abc'` dropped rows that
        # `RETURN n.v <> 'abc'` called true, and `WHERE a AND b` disagreed with
        # `WHERE a WITH n WHERE b`.
        g = self.db.select_graph("bulk_filter_parity")
        g.query("""CREATE (:V {k:'int', v:1}), (:V {k:'float', v:1.5}),
                          (:V {k:'bool', v:true}), (:V {k:'str', v:'abc'}),
                          (:V {k:'other', v:'Dhaka'}), (:V {k:'list', v:[1,2]}),
                          (:V {k:'missing'})""")
        g.query("MATCH (n:V {k:'point'}) DELETE n")
        g.query("CREATE (:V {k:'point'})")
        g.query("MATCH (n:V {k:'point'}) SET n.v = point({latitude:31.18, longitude:34.22})")

        for op in ['=', '<>', '<', '<=', '>', '>=']:
            for const in ["'abc'", '1', '1.5', 'true']:
                # what the per-row evaluator says, keyed by node
                scalar = g.query(f"MATCH (n:V) RETURN n.k, n.v {op} {const} ORDER BY n.k")
                expected = sorted([[k] for k, r in scalar.result_set if r is True])
                # what the columnar kernel says for the same predicate
                bulk = g.query(f"MATCH (n:V) WHERE n.v {op} {const} RETURN n.k ORDER BY n.k")
                self.env.assertEqual(sorted(bulk.result_set), expected)

        # `WHERE a AND b` must equal `WHERE a WITH n WHERE b`: the first is one
        # conjunction kernel, the second two separate filters.
        conj = g.query("MATCH (n:V) WHERE n.k <> 'zz' AND n.v <> 'abc' RETURN n.k ORDER BY n.k")
        split = g.query("MATCH (n:V) WHERE n.k <> 'zz' WITH n WHERE n.v <> 'abc' RETURN n.k ORDER BY n.k")
        self.env.assertEqual(conj.result_set, split.result_set)
        # and parentheses, which keep the predicate off the kernel, must not
        # change the answer either
        paren = g.query("MATCH (n:V) WHERE (n.v <> 'abc') RETURN n.k ORDER BY n.k")
        plain = g.query("MATCH (n:V) WHERE n.v <> 'abc' RETURN n.k ORDER BY n.k")
        self.env.assertEqual(paren.result_set, plain.result_set)

    def test06_bulk_paths_keep_integer_precision(self):
        # A mixed int/float property column must not be promoted to f64 by the
        # bulk paths: 9007199254740993 and 9007199254740992 are the same double,
        # so promotion would make the filter match the wrong row, print the
        # wrong number, and merge two grouping keys into one.
        g = self.db.select_graph("bulk_precision")
        g.query("CREATE (:P {v: 9007199254740993}), (:P {v: 1.5})")

        res = g.query("MATCH (n:P) WHERE n.v = 9007199254740992 RETURN count(n)")
        self.env.assertEqual(res.result_set, [[0]])

        res = g.query("MATCH (n:P) WHERE n.v = 9007199254740993 RETURN n.v")
        self.env.assertEqual(res.result_set, [[9007199254740993]])

        res = g.query("MATCH (n:P) WITH n.v AS k, count(*) AS c RETURN k ORDER BY k")
        self.env.assertEqual(res.result_set, [[1.5], [9007199254740993]])

    def test07_columnar_expressions_match_per_row(self):
        # Filters, projections and aggregate inputs are all evaluated column
        # at a time. The columnar walk must answer exactly what the per-row
        # evaluator answers for every operator, including three-valued logic
        # and the null/type edge cases.
        #
        # `UNWIND` builds the same row set as a literal list, and evaluating
        # the expression inside `WITH ... AS` (per-row) against `WHERE` and
        # `RETURN` (columnar) is the cross-check.
        g = self.db.select_graph("columnar_expr_parity")
        g.query("""CREATE (:E {i:1, f:1.5, s:'abc', b:true, z:0}),
                          (:E {i:2, f:2.5, s:'abd', b:false, z:1}),
                          (:E {i:3, f:-1.0, s:'ABC', b:true, z:2}),
                          (:E {i:-4, s:'', b:false, z:3}),
                          (:E {i:0, f:0.0, z:4})""")

        exprs = [
            "n.i > 1", "n.i >= 1", "n.i < 1", "n.i <= 1", "n.i = 1", "n.i <> 1",
            "n.i % 2 = 0", "n.i * 2 + 1 > 3", "n.i - 1 <= 0", "n.i / 2 = 0",
            "n.f > 1.0", "n.i + n.f > 2.0", "n.i > n.z", "n.i = n.z",
            "n.f IS NULL", "n.f IS NOT NULL", "n.missing = 1", "n.missing <> 1",
            "n.b", "NOT n.b", "n.b AND n.i > 1", "n.b OR n.i > 1",
            "n.b XOR n.i > 1", "NOT (n.b AND n.i > 1)", "NOT n.missing",
            "n.i > 1 AND n.f > 1.0 AND n.s <> 'abc'",
            "n.i > 1 OR n.f > 1.0 OR n.s = 'abc'",
            "n.s STARTS WITH 'ab'", "n.s CONTAINS 'b'", "n.s ENDS WITH 'c'",
            "toUpper(n.s) = 'ABC'", "size(n.s) > 2", "n.s =~ 'ab.'",
            "n.i IN [1, 2]", "n.s IN ['abc', 'ABC']",
            "(CASE WHEN n.i > 1 THEN 'big' ELSE 'small' END) = 'big'",
            "(CASE n.i WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'many' END) = 'one'",
            # CASE must not evaluate a branch for rows that did not select it:
            # `n.z = 0` would divide by zero on the first row otherwise.
            "(CASE WHEN n.z = 0 THEN 0 ELSE 10 / n.z END) > 3",
            "coalesce(n.f, 0.0) > 1.0", "n.i > 1 = true",
        ]

        for expr in exprs:
            # per-row: the expression is computed by a projection the engine
            # cannot turn into a column reference, then compared as a variable
            scalar = g.query(f"MATCH (n:E) RETURN n.z AS z, {expr} AS v ORDER BY z")
            expected = sorted([[z] for z, v in scalar.result_set if v is True])
            columnar = g.query(f"MATCH (n:E) WHERE {expr} RETURN n.z AS z ORDER BY z")
            self.env.assertEqual(sorted(columnar.result_set), expected)

        # Aggregate inputs and grouping keys take the same columnar path.
        for expr in ["n.i * 3 + 1", "n.i % 2", "n.f * 2.0", "toUpper(n.s)",
                     "CASE WHEN n.i > 1 THEN n.i ELSE 0 END"]:
            rows = g.query(f"MATCH (n:E) RETURN n.z AS z, {expr} AS v ORDER BY z").result_set
            values = [v for _, v in rows if v is not None]
            agg = g.query(f"MATCH (n:E) RETURN count({expr}), collect({expr})")
            self.env.assertEqual(agg.result_set[0][0], len(values))
            self.env.assertEqual(sorted(map(str, agg.result_set[0][1])),
                                 sorted(map(str, values)))

    def test08_columnar_and_or_short_circuit(self):
        # `A AND B` must not evaluate B for rows where A is already false: the
        # columnar walk narrows rows per conjunct, so a division by zero in B
        # stays unreached exactly as it does per row.
        g = self.db.select_graph("columnar_short_circuit")
        g.query("CREATE (:D {d: 0}), (:D {d: 1}), (:D {d: 2})")

        # d=1 -> 10, d=2 -> 5, both > 1; d=0 never reaches the division.
        res = g.query("MATCH (n:D) WHERE n.d <> 0 AND 10 / n.d > 1 RETURN count(n)")
        self.env.assertEqual(res.result_set, [[2]])

        res = g.query("MATCH (n:D) WHERE n.d = 0 OR 10 / n.d > 100 RETURN count(n)")
        self.env.assertEqual(res.result_set, [[1]])

        # …and a division that is genuinely reachable still raises.
        try:
            g.query("MATCH (n:D) WHERE 10 / n.d > 1 RETURN count(n)")
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("Division by zero", str(e))
