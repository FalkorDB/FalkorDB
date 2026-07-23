from common import *

GRAPH_ID = "operator_precedence"


class testOperatorPrecedenceFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def _assert_cases(self, cases):
        for query, expected in cases:
            actual = self.graph.query(query)
            self.env.assertEquals(actual.result_set, expected)

    # Repro for #2181 and #423: IS (NOT) NULL should bind after arithmetic and unary minus.
    def test01_is_null_precedence(self):
        cases = [
            ("RETURN 1.0 + 2.0 IS NULL AS result", [[False]]),
            ("RETURN 3.0 * 4.0 IS NULL AS result", [[False]]),
            ("RETURN 5.0 - 1.0 IS NULL AS result", [[False]]),
            ("RETURN 1 % 2 IS NULL AS result", [[False]]),
            ("RETURN 1 > 2 IS NULL AS result", [[False]]),
            ("RETURN 1.0 + 2.0 IS NOT NULL AS result", [[True]]),
            ("RETURN -1 IS NULL AS result", [[False]]),
            ("RETURN -1 IS NOT NULL AS result", [[True]]),
        ]
        self._assert_cases(cases)

    # Repro for #2231: unary minus literal should be a valid left operand for IN.
    def test02_unary_minus_with_in(self):
        cases = [
            ("WITH 0 AS x WHERE -1 IN [-1] RETURN x", [[0]]),
            ("WITH 0 AS x WHERE (-1 IN [-1]) RETURN x", [[0]]),
        ]
        self._assert_cases(cases)

    # Expanded coverage: CONTAINS should bind after arithmetic/string concatenation.
    def test03_contains_precedence(self):
        cases = [
            ("RETURN 'ab' + 'cd' CONTAINS 'bc' AS result", [[True]]),
            ("RETURN 'ab' + 'cd' CONTAINS 'bd' AS result", [[False]]),
            ("RETURN 'a' + toString(1 + 2) CONTAINS 'a3' AS result", [[True]]),
            ("RETURN toString(5 - 3) + 'x' CONTAINS '2x' AS result", [[True]]),
        ]
        self._assert_cases(cases)

    # Expanded coverage: IN should bind after arithmetic, comparison, and unary minus.
    def test04_in_precedence(self):
        cases = [
            ("RETURN 1 + 1 IN [2] AS result", [[True]]),
            ("RETURN 5 - 1 IN [4] AS result", [[True]]),
            ("RETURN 3 * 4 IN [12] AS result", [[True]]),
            ("RETURN 5.0 / 2.0 IN [2.5] AS result", [[True]]),
            ("RETURN 5 % 2 IN [1] AS result", [[True]]),
            ("RETURN 1 > 2 IN [false] AS result", [[True]]),
            ("RETURN -1 IN [-1] AS result", [[True]]),
            ("RETURN 1 + 2 IN [1, 2, 3] AS result", [[True]]),
        ]
        self._assert_cases(cases)

    # Repro for #751: this query should not crash on a self-looping relation.
    def test05_self_loop_crash_repro(self):
        self.graph.query(
            "CREATE (n0:L28 {k161:-392104257, k162:-60652336, k158:false, id:5, k159:'dGThjm0QF'})"
        )
        self.graph.query(
            "MATCH (n0 {id:5}), (n1 {id:5}) "
            "MERGE (n0)-[r:T38 {k490:true, k486:true, k485:-1782193271, k487:true, id:10, k489:true}]->(n1)"
        )

        query = (
            "MATCH (n0)<-[r0]-(n1), (n2)<-[r1]-(n1), (n0)<-[r2]-(n3) "
            "OPTIONAL MATCH (n0)<-[r3]-(n3), (n2)<-[r4]-(n1) "
            "WHERE (n1.k159) + '1' CONTAINS '1' "
            "AND r3.id <> r4.id "
            "AND r4.id <> r3.id "
            "AND r3.k485 > r4.k485 "
            "AND r4.k485 > r0.k485 "
            "RETURN *"
        )
        self.graph.query(query)

        self.env.assertTrue(True)
