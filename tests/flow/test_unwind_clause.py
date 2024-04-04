from common import *
import re

GRAPH_ID = "unwind"

class testUnwindClause():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
 
    def test01_unwind_null(self):
        query = """UNWIND null AS x RETURN x"""
        actual_result = self.graph.query(query)
        expected = []
        self.env.assertEqual(actual_result.result_set, expected)

    def test02_unwind_input_types(self):
        # map list input
        query = """UNWIND ([{x:3, y:5}]) AS q RETURN q"""
        actual_result = self.graph.query(query)
        expected = [[{'x':3, 'y':5}]]
        self.env.assertEqual(actual_result.result_set, expected)

        # map input
        query = """UNWIND ({x:3, y:5}) AS q RETURN q"""
        actual_result = self.graph.query(query)
        expected = [[{'x': 3, 'y': 5}]]
        self.env.assertEqual(actual_result.result_set, expected)

        # map containing a key with the value NULL
        query = """UNWIND ({x:null}) AS q RETURN q"""
        actual_result = self.graph.query(query)
        expected = [[{'x': None}]]
        self.env.assertEqual(actual_result.result_set, expected)

        # integer input
        query = """UNWIND 5 AS q RETURN q"""
        actual_result = self.graph.query(query)
        expected = [[5]]
        self.env.assertEqual(actual_result.result_set, expected)

        # string input
        query = """UNWIND 'abc' AS q RETURN q"""
        actual_result = self.graph.query(query)
        expected = [['abc']]
        self.env.assertEqual(actual_result.result_set, expected)

        # floating-point input
        query = """UNWIND 7.5 AS q RETURN q"""
        actual_result = self.graph.query(query)
        expected = [[7.5]]
        self.env.assertEqual(actual_result.result_set, expected)

        # nested list
        query = """WITH [[1, 2], [3, 4], 5] AS nested UNWIND nested AS x RETURN x"""
        actual_result = self.graph.query(query)
        expected = [[[1, 2]], [[3, 4]], [5]]
        self.env.assertEqual(actual_result.result_set, expected)

        # nested list double unwind
        query = """WITH [[1, 2], [3, 4], 5] AS nested UNWIND nested AS x UNWIND x AS y RETURN y"""
        actual_result = self.graph.query(query)
        expected = [[1], [2], [3], [4], [5]]
        self.env.assertEqual(actual_result.result_set, expected)

        # empty list
        query = """UNWIND [] AS x RETURN x"""
        actual_result = self.graph.query(query)
        expected = []
        self.env.assertEqual(actual_result.result_set, expected)

        # list with null at the last position
        query = """UNWIND [1, 2, null] AS x RETURN x"""
        actual_result = self.graph.query(query)
        expected = [[1], [2], [None]]
        self.env.assertEqual(actual_result.result_set, expected)

        # list with null before the last position
        query = """UNWIND [1, null, 2] AS x RETURN x"""
        actual_result = self.graph.query(query)
        expected = [[1], [None], [2]]
        self.env.assertEqual(actual_result.result_set, expected)

        # list with null at first position
        query = """UNWIND [null, 1, 2] AS x RETURN x"""
        actual_result = self.graph.query(query)
        expected = [[None], [1], [2]]
        self.env.assertEqual(actual_result.result_set, expected)

    def test03_unwind_heap_allocated_value(self):
        # make sure access to unwinded heap allocated values is safe
        # the second UNWIND will free its internal list every time it pulls
        # from the former UNWIND list, once freed we want to make sure access
        # to its former elements is still valid
        query = """UNWIND [1, 2, 3] AS i
                   UNWIND [[tostring(i), tostring(i+1)]] AS j
                   MERGE (n:N {v:j})
                   RETURN n.v
                   ORDER BY n.v"""
        res = self.graph.query(query)
        self.env.assertEqual(res.nodes_created, 3)
        expected_result = [[['1','2']], [['2','3']], [['3','4']]]
        self.env.assertEqual(res.result_set, expected_result)

    def test04_unwind_set(self):
        # delete property
        query = """CREATE (n:N {x:3})"""
        actual_result = self.graph.query(query)
        query = """UNWIND ({x:null}) AS q MATCH (n:N) SET n.x= q.x RETURN n"""
        actual_result = self.graph.query(query)
        self.env.assertEqual(actual_result.properties_removed, 1)

    def test05_overwrite_var(self):
        queries = ["UNWIND [0, 1] AS i UNWIND [2, 3] AS i RETURN i",
                   "MATCH (i) UNWIND [0, 1] as i RETURN i"]

        for q in queries:
            try:
                self.graph.query(q)
                # should not reach this point
                self.env.assertTrue(False)
            except Exception as e:
                self.env.assertTrue("Variable `i` already declared" in str(e))

    def test06_access_undefined_var(self):
        query = "UNWIND [0, i, 1] AS i RETURN i"
        try:
            self.graph.query(query)
            # should not reach this point
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertTrue("'i' not defined" in str(e))

    def test07_nested_unwind(self):
        # n0 is a heap allocated array
        # which gets free on the third call to consume of the nested UNWIND
        query = """WITH [0] AS n0
                   UNWIND [0, 0] AS n1
                   WITH *
                   UNWIND [0, 0] AS n2
                   MERGE ({n3:0})"""
        
        result = self.graph.query(query)
        self.env.assertEqual(result.nodes_created, 1)
        self.env.assertEqual(result.properties_set, 1)

