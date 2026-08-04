from common import *

GRAPH_ID = "operator-precedence"

class testOperatorPrecedence():
    def __init__(self):
        self.env, self.db = Env()
        self.g = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.g.delete()

    def test01_string_predicates_after_concatenation(self):
        self.g.query("CREATE (:N {name:'Alice'})")

        queries = {
            "MATCH (n) WHERE (n.name)+'1' CONTAINS '1' RETURN n.name": [['Alice']],
            "MATCH (n) WHERE (n.name)+'1' STARTS WITH 'Alice' RETURN n.name": [['Alice']],
            "MATCH (n) WHERE (n.name)+'1' ENDS WITH '1' RETURN n.name": [['Alice']],
            "MATCH (n) WHERE n.name CONTAINS '1' RETURN n.name": [],
        }

        for query, expected in queries.items():
            result = self.g.query(query)
            self.env.assertEqual(result.result_set, expected)

    def test02_list_append_precedence_before_in(self):
        query = """RETURN [1]+2 IN [3]+4 AS a,
                          ([1]+2) IN ([3]+4) AS b,
                          [1]+(2 IN [3])+4 AS c"""
        result = self.g.query(query)
        self.env.assertEqual(result.result_set, [[False, False, [1, False, 4]]])

    def test03_list_concat_precedence_before_in(self):
        query = """RETURN [1]+[2] IN [3]+[4] AS a,
                          ([1]+[2]) IN ([3]+[4]) AS b,
                          (([1]+[2]) IN [3])+[4] AS c,
                          [1]+([2] IN [3])+[4] AS d"""
        result = self.g.query(query)
        self.env.assertEqual(result.result_set,
            [[False, False, [False, 4], [1, False, 4]]])

    def test04_chained_rhs_value_expression_before_in(self):
        query = """RETURN 2 IN [0]+1+2 AS a,
                          2 IN ([0]+1+2) AS b,
                          [0]+(1 IN [1])+2 AS c"""
        result = self.g.query(query)
        self.env.assertEqual(result.result_set, [[True, True, [0, True, 2]]])

    def test05_issue_751_reproduction(self):
        self.g.query("""CREATE (n0 :L28{k161 : -392104257, k162 : -60652336,
                       k158 : false, id : 5, k159 : 'dGThjm0QF'})""")
        self.g.query("""MATCH (n0 {id : 5}), (n1 {id : 5})
                       MERGE(n0)-[r :T38{k490 : true, k486 : true,
                       k485 : -1782193271, k487 : true, id : 10,
                       k489 : true}]->(n1)""")

        query = """MATCH (n0)<-[r0]-(n1), (n2)<-[r1]-(n1),
                  (n0)<-[r2]-(n3)
                  OPTIONAL MATCH (n0)<-[r3]-(n3), (n2)<-[r4]-(n1)
                  WHERE (n1.k159)+'1' CONTAINS '1'
                    AND r3.id <> r4.id
                    AND r4.id <> r3.id
                    AND r3.k485 > r4.k485
                    AND r4.k485 > r0.k485
                  RETURN count(*)"""
        result = self.g.query(query)
        self.env.assertTrue(result.result_set[0][0] >= 0)
