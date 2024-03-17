from common import *
import re

GRAPH_ID = "load_csv"

class testLoadCSV():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
 
    def test01_invalid_call(self):
        queries = ["LOAD CSV FROM a AS row RETURN row",
                   "LOAD CSV FROM 2 AS row RETURN row",
                   "LOAD CSV FROM $arr AS row RETURN row",
                   "WITH 2 AS x LOAD CSV FROM x AS row RETURN row"]

        for q in queries:
            try:
                self.graph.query(q, {'arr': []})
                self.env.assertFalse(True)
            except Exception as e:
                self.env.assertEquals(str(e), "path to CSV must be a string")

    def test02_project_csv_rows(self):
        # project all rows in a CSV file
        q = """LOAD CSV FROM 'data.csv' AS row
               RETURN row"""

        result = self.graph.query(q).result_set
        self.env.assertEquals(result[0][0], ['AAA', 'BB', 'C'])

    def test03_load_csv_multiple_times(self):
        # project the same CSV multiple times
        q = """UNWIND [1,2,3] AS x
               LOAD CSV FROM 'data.csv' AS row
               RETURN x, row
               ORDER BY x"""

        result = self.graph.query(q).result_set
        self.env.assertEquals(result[0], [1, ['AAA', 'BB', 'C']])
        self.env.assertEquals(result[1], [2, ['AAA', 'BB', 'C']])
        self.env.assertEquals(result[2], [3, ['AAA', 'BB', 'C']])

    def test04_dynamic_csv_path(self):
        # project all rows in a CSV file
        q = """UNWIND ['a', 'b'] AS x
               LOAD CSV FROM x + '.csv' AS row
               RETURN x, row
               ORDER BY x"""

        result = self.graph.query(q).result_set
        self.env.assertEquals(result[0], ['a', ['AAA', 'BB', 'C']])
        self.env.assertEquals(result[1], ['b', ['AAA', 'BB', 'C']])

