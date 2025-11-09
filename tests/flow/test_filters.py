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
        self.env.assertEquals(res.result_set, [])

        res = self.g.query("WITH 1 AS x WHERE 0.0 / 0.0 <> 0.0 / 0.0 RETURN x")
        self.env.assertEquals(res.result_set, [[1]])

    def test04_redundant_filter(self):
        q = """MATCH (n), (), ()
               WHERE ('a' <= ('km' + 'X'))
               RETURN *"""
        plan = self.g.explain(q)
        self.env.assertFalse('Filter' in plan)

