from common import *

GRAPH_ID = "list_datatype"

# tests the list datatype
class testList(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_collect(self):
        self.graph.query("UNWIND range(0, 10) AS x CREATE ()")

        query = """MATCH (n) RETURN collect(n)"""
        result = self.graph.query(query)
        result_set = result.result_set
        self.env.assertEquals(len(result_set), 1)
        self.env.assertTrue(all(isinstance(n, Node) for n in result_set[0][0]))

    def test02_unwind(self):
        query = """CREATE ()"""
        self.graph.query(query)
        query = """unwind(range(0,10)) as x return x"""
        result_set = self.graph.query(query).result_set
        expected_result = [[0], [1], [2], [3],
                           [4], [5], [6], [7], [8], [9], [10]]
        self.env.assertEquals(result_set, expected_result)

    # List functions should handle null inputs appropriately.
    def test03_null_list_function_inputs(self):
        expected_result = [[None]]

        # NULL list argument to subscript.
        query = """WITH NULL as list RETURN list[0]"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # NULL list argument to slice.
        query = """WITH NULL as list RETURN list[0..5]"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # NULL list argument to HEAD.
        query = """WITH NULL as list RETURN head(list)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # NULL list argument to LAST.
        query = """WITH NULL as list RETURN last(list)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # NULL list argument to TAIL.
        query = """WITH NULL as list RETURN tail(list)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # NULL list argument to IN.
        query = """WITH NULL as list RETURN 'val' in list"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # NULL list argument to SIZE.
        query = """WITH NULL as list RETURN size(list)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # NULL subscript argument.
        query = """WITH ['a'] as list RETURN list[NULL]"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # NULL IN non-empty list should return NULL.
        query = """RETURN NULL in ['val']"""
        actual_result = self.graph.query(query)
        expected_result = [[None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # NULL arguments to slice.
        query = """WITH ['a'] as list RETURN list[0..NULL]"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # NULL range argument should produce an error.
        query = """RETURN range(NULL, 5)"""
        try:
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

        # NULL IN empty list should return false.
        query = """RETURN NULL in []"""
        actual_result = self.graph.query(query)
        expected_result = [[False]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test04_head_function(self):
        # Test empty list input
        query = """RETURN head([])"""
        actual_result = self.graph.query(query)
        expected_result = [[None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test integer list input
        query = """WITH [1, 2, 3] as list RETURN head(list)"""
        actual_result = self.graph.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test string list input
        query = """WITH ['a', 'b', 'c'] as list RETURN head(list)"""
        actual_result = self.graph.query(query)
        expected_result = [['a']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test NULL value in list input
        query = """WITH [NULL, 'b', NULL] as list RETURN head(list)"""
        actual_result = self.graph.query(query)
        expected_result = [[None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test wrong type inputs
        try:
            query = """RETURN head(1)"""
            self.graph.query(query)
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Integer", str(e))

        try:
            query = """RETURN head('a')"""
            self.graph.query(query)
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was String", str(e))

        try:
            query = """RETURN head(true)"""
            self.graph.query(query)
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Boolean", str(e))

    def test05_last_function(self):
        # Test empty list input
        query = """RETURN last([])"""
        actual_result = self.graph.query(query)
        expected_result = [[None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test integer list input
        query = """WITH [1, 2, 3] as list RETURN last(list)"""
        actual_result = self.graph.query(query)
        expected_result = [[3]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test string list input
        query = """WITH ['a', 'b', 'c'] as list RETURN last(list)"""
        actual_result = self.graph.query(query)
        expected_result = [['c']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test NULL value in list input
        query = """WITH [NULL, 'b', NULL] as list RETURN last(list)"""
        actual_result = self.graph.query(query)
        expected_result = [[None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test wrong type inputs
        try:
            query = """RETURN last(1)"""
            self.graph.query(query)
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Integer", str(e))

        try:
            query = """RETURN last('a')"""
            self.graph.query(query)
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was String", str(e))

        try:
            query = """RETURN last(true)"""
            self.graph.query(query)
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Boolean", str(e))

    def test06_toBooleanList(self):
        # NULL input should return NULL
        expected_result = [None]
        query = """RETURN toBooleanList(null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # List of NULL values input should return list of NULL values
        query = """RETURN toBooleanList([null, null])"""
        expected_result = [[None, None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # Test list with mixed type values
        query = """RETURN toBooleanList(['abc', true, 'false', null, ['a','b']])"""
        expected_result = [[None, True, False, None, None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # Tests with empty array result
        query = """MATCH (n:X) RETURN toBooleanList(n)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (n:X) RETURN toBooleanList([n])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH ()-[e]->() RETURN toBooleanList(e)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH ()-[e]->() RETURN toBooleanList([e])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[]->() RETURN toBooleanList(p)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[]->() RETURN toBooleanList([p])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test of type mismatch input
        try:
            self.graph.query("RETURN toBooleanList(true)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Boolean", str(e))

        try:
            self.graph.query("RETURN toBooleanList(7.05)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Float", str(e))

        try:
            self.graph.query("RETURN toBooleanList(7)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Integer", str(e))

        try:
            self.graph.query("RETURN toBooleanList('abc')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was String", str(e))

        try:
            query = """CREATE (a:X) RETURN toBooleanList(a)"""
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Node", str(e))

        try:
            query = """CREATE (a:X)-[r:R]->(b:Y) RETURN toBooleanList(r)"""
            self.graph.query(query)
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Edge", str(e))

        try:
            query = """CREATE (a:X), (b:Y)"""
            self.graph.query(query)
            query = """MATCH (a:X), (b:Y) CREATE (a)-[r:R]->(b)"""
            self.graph.query(query)
            query = """MATCH p=()-[]->() RETURN toBooleanList(p)"""
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Path", str(e))
            query = """MATCH (a:X),(b:Y) DELETE a, b"""
            self.graph.query(query)

        # List of node input should return list of NULL
        query = """CREATE (a:X) RETURN toBooleanList([a])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X) DELETE a"""
        self.graph.query(query)

        # List of edge input should return list of NULL
        query = """CREATE (a:X)-[r:R]->(b:Y) RETURN toBooleanList([r])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X),(b:Y) DELETE a, b"""
        self.graph.query(query)

        # List of path input should return list of NULL
        query = """CREATE (a:X), (b:Y)"""
        self.graph.query(query)
        query = """MATCH (a:X), (b:Y) create (a)-[e:R]->(b)"""
        self.graph.query(query)
        query = """MATCH p=()-[]->() RETURN toBooleanList([p])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X),(b:Y) DELETE a, b"""
        self.graph.query(query)

        # Test without input argument
        try:
            query = """RETURN toBooleanList()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'toBooleanList', expected at least 1", str(e)) 

    def test07_toFloatList(self):
        # NULL input should return NULL
        expected_result = [None]
        query = """RETURN toFloatList(null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # List of NULL values input should return list of NULL values
        query = """RETURN toFloatList([null, null])"""
        expected_result = [[None, None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # Test list with mixed type values
        query = """RETURN toFloatList(['abc', 1.5, 7.0578, null, ['a','b']]) """
        expected_result = [[None, 1.5, 7.0578, None, None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        query = """RETURN toFloatList(['7.0578']) """
        expected_result = 7.0578
        actual_result = self.graph.query(query)
        self.env.assertAlmostEqual(float(actual_result.result_set[0][0][0]), expected_result, 1e-5)

        # Tests with empty array result
        query = """MATCH (n:X) RETURN toFloatList(n)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (n:X) RETURN toFloatList([n])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH ()-[e]->() RETURN toFloatList(e)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH ()-[e]->() RETURN toFloatList([e])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[]->() RETURN toFloatList(p)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[]->() RETURN toFloatList([p])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        try:
            self.graph.query("RETURN toFloatList(true)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Boolean", str(e))

        try:
            self.graph.query("RETURN toFloatList(7.05)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Float", str(e))

        try:
            self.graph.query("RETURN toFloatList(7)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Integer", str(e))

        try:
            self.graph.query("RETURN toFloatList('abc')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was String", str(e))

        try:
            query = """CREATE (a:X) RETURN toFloatList(a)"""
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Node", str(e))
            query = """MATCH (a:X) DELETE a"""
            self.graph.query(query)

        try:
            query = """CREATE (a:X)-[r:R]->(b:Y) RETURN toFloatList(r)"""
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Edge", str(e))
            query = """MATCH (a:X),(b:Y) DELETE a, b"""
            self.graph.query(query)

        try:
            query = """CREATE (a:X), (b:Y)"""
            self.graph.query(query)
            query = """MATCH (a:X), (b:Y) CREATE (a)-[r:R]->(b)"""
            self.graph.query(query)
            query = """MATCH p=()-[]->() RETURN toFloatList(p)"""
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Path", str(e))
            query = """MATCH (a:X),(b:Y) DELETE a, b"""
            self.graph.query(query)

        # List of node input should return list of NULL
        query = """CREATE (a:X) RETURN toFloatList([a])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X) DELETE a"""
        self.graph.query(query)

        # List of edge input should return list of NULL
        query = """CREATE (a:X)-[r:R]->(b:Y) RETURN toFloatList([r])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X),(b:Y) DELETE a, b"""
        self.graph.query(query)

        # List of path input should return list of NULL
        query = """CREATE (a:X), (b:Y)"""
        self.graph.query(query)
        query = """MATCH (a:X), (b:Y) create (a)-[e:R]->(b)"""
        self.graph.query(query)
        query = """MATCH p=()-[]->() RETURN toFloatList([p])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X), (b:Y) DELETE a, b"""
        self.graph.query(query)

        # Test without input argument
        try:
            query = """RETURN toFloatList()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'toFloatList', expected at least 1", str(e))

    def test08_toIntegerList(self):
        # NULL input should return NULL
        expected_result = [None]
        query = """RETURN toIntegerList(null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # List of NULL values input should return list of NULL values
        query = """RETURN toIntegerList([null, null])"""
        expected_result = [[None, None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # Test list with mixed type values
        query = """RETURN toIntegerList(['abc', 7, '5', null, ['a','b']]) """
        expected_result = [[None, 7, 5, None, None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # Tests with empty array result
        query = """MATCH (n:X) RETURN toIntegerList(n)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (n:X) RETURN toIntegerList([n])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH ()-[e]->() RETURN toIntegerList(e)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH ()-[e]->() RETURN toIntegerList([e])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[]->() RETURN toIntegerList(p)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[]->() RETURN toIntegerList([p])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        try:
            self.graph.query("RETURN toIntegerList(true)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Boolean", str(e))

        try:
            self.graph.query("RETURN toIntegerList(7.05)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Float", str(e))

        try:
            self.graph.query("RETURN toIntegerList(7)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Integer", str(e))

        try:
            self.graph.query("RETURN toIntegerList('abc')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was String", str(e))

        try:
            query = """CREATE (a:X) RETURN toIntegerList(a)"""
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Node", str(e))
            query = """MATCH (a:X) DELETE a"""
            self.graph.query(query)

        try:
            query = """CREATE (a:X)-[r:R]->(b:Y) RETURN toIntegerList(r)"""
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Edge", str(e))
            query = """MATCH (a:X), (b:Y) DELETE a, b"""
            self.graph.query(query)

        try:
            query = """CREATE (a:X), (b:Y)"""
            self.graph.query(query)
            query = """MATCH (a:X), (b:Y) CREATE (a)-[r:R]->(b)"""
            self.graph.query(query)
            query = """MATCH p=()-[]->() RETURN toIntegerList(p)"""
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Path", str(e))
            query = """MATCH (a:X),(b:Y) DELETE a, b"""
            self.graph.query(query)

        # List of node input should return list of NULL
        query = """CREATE (a:X) RETURN toIntegerList([a])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X) DELETE a"""
        self.graph.query(query)

        # List of edge input should return list of NULL
        query = """CREATE (a:X)-[r:R]->(b:Y) RETURN toIntegerList([r])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X),(b:Y) DELETE a, b"""
        self.graph.query(query)

        # List of path input should return list of NULL
        query = """CREATE (a:X), (b:Y)"""
        self.graph.query(query)
        query = """MATCH (a:X), (b:Y) create (a)-[e:R]->(b)"""
        self.graph.query(query)
        query = """MATCH p=()-[]->() RETURN toIntegerList([p])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X), (b:Y) DELETE a, b"""
        self.graph.query(query)

        # Test without input argument
        try:
            query = """RETURN toIntegerList()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'toIntegerList', expected at least 1", str(e))

    def test09_toStringList(self):
        # NULL input should return NULL
        expected_result = [None]
        query = """RETURN toStringList(null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # List of NULL values input should return list of NULL values
        query = """RETURN toStringList([null, null])"""
        expected_result = [[None, None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result) 

        # Test list with string values
        query = """RETURN toStringList(['abc', '5.32', 'true', 'this is a test'])"""
        expected_result = [['abc', '5.32', 'true', 'this is a test']]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # Test list with float values
        query = """RETURN toStringList([0.32425, 5.32, 7.1])"""
        expected_result = [['0.324250', '5.320000', '7.100000']]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # Test list with mixed type values
        query = """RETURN toStringList(['abc', 7, '5.32', null, ['a','b']]) """
        expected_result = [['abc', '7', '5.32', None, None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # Tests with empty array result
        query = """MATCH (n:X) RETURN toStringList(n)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH (n:X) RETURN toStringList([n])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH ()-[e]->() RETURN toStringList(e)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH ()-[e]->() RETURN toStringList([e])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[]->() RETURN toStringList(p)"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """MATCH p=()-[]->() RETURN toStringList([p])"""
        expected_result = []
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)

        try:
            self.graph.query("RETURN toStringList(true)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Boolean", str(e))

        try:
            self.graph.query("RETURN toStringList(7.05)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Float", str(e))

        try:
            self.graph.query("RETURN toStringList(7)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Integer", str(e))

        try:
            self.graph.query("RETURN toStringList('abc')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was String", str(e))

        try:
            query = """CREATE (a:X) RETURN toStringList(a)"""
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Node", str(e))
            query = """MATCH (a:X) DELETE a"""
            self.graph.query(query)

        try:
            query = """CREATE (a:X)-[r:R]->(b:Y) RETURN toStringList(r)"""
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Edge", str(e))
            query = """MATCH (a:X),(b:Y) DELETE a, b"""
            self.graph.query(query)

        try:
            query = """CREATE (a:X), (b:Y)"""
            self.graph.query(query)
            query = """MATCH (a:X), (b:Y) CREATE (a)-[r:R]->(b)"""
            self.graph.query(query)
            query = """MATCH p=()-[]->() RETURN toStringList(p)"""
            actual_result = self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was Path", str(e))
            query = """MATCH (a:X),(b:Y) DELETE a, b"""
            self.graph.query(query)

        # List of node input should return list of NULL
        query = """CREATE (a:X) RETURN toStringList([a])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X) DELETE a"""
        self.graph.query(query)

        # List of edge input should return list of NULL
        query = """CREATE (a:X)-[r:R]->(b:Y) RETURN toStringList([r])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X),(b:Y) DELETE a, b"""
        self.graph.query(query)

        # List of path input should return list of NULL
        query = """CREATE (a:X), (b:Y)"""
        self.graph.query(query)
        query = """MATCH (a:X), (b:Y) create (a)-[e:R]->(b)"""
        self.graph.query(query)
        query = """MATCH p=()-[]->() RETURN toStringList([p])"""
        expected_result = [[None]]
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """MATCH (a:X), (b:Y) DELETE a, b"""
        self.graph.query(query)

        # Test without input argument
        try:
            query = """RETURN toStringList()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'toStringList', expected at least 1", str(e))

    def test09_remove(self):
        # NULL input should return NULL
        expected_result = [None]
        query = """WITH NULL as list RETURN list.remove(null, 2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # 2nd arg should be integer
        try:
            self.graph.query("RETURN list.remove([1,2,3], '2')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected Integer but was String", str(e))

        # 3rd arg shoulf be integer
        try:
            self.graph.query("RETURN list.remove([1,2,3], 2, '1')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected Integer but was String", str(e))

        # Test without input argument
        try:
            query = """RETURN list.remove()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'list.remove', expected at least 2", str(e))

        # Test with 1 input argument
        try:
            query = """RETURN list.remove([1,2,3])"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 1 arguments to function 'list.remove', expected at least 2", str(e))

        # Test with 4 input argument
        try:
            query = """RETURN list.remove([1,2,3], 2, 1, 3)"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 4 arguments to function 'list.remove', expected at most 3", str(e))

        ### Test valid inputs ###

        expected_result = [[1]]
        query = """RETURN list.remove([1,2,3], 1, 2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,4]]
        query = """RETURN list.remove([1,2,3,4], 1, 2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2]]
        query = """RETURN list.remove([1,2,3], 2, 1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """RETURN list.remove([1,2,3], 2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test negative index
        expected_result = [[1,2,3]]
        query = """RETURN list.remove([1,2,3,4], -1, 1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test negative index
        expected_result = [[2,3,4]]
        query = """RETURN list.remove([1,2,3,4], -4, 1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test out of range removal
        expected_result = [[1,2,3]]
        query = """RETURN list.remove([1,2,3,4], -1, 2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test out of range removal
        expected_result = [[1]]
        query = """RETURN list.remove([1,2,3,4], -3, 5)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test out of range removal
        expected_result = [[1]]
        query = """RETURN list.remove([1,2,3,4], 1, 5)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test out bound index
        expected_result = [[1,2,3,4]]
        query = """RETURN list.remove([1,2,3,4], -5, 5)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test out bound index
        expected_result = [[1,2,3,4]]
        query = """RETURN list.remove([1,2,3,4], 4, 5)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3]]
        query = """RETURN list.remove([1,2,3], 1, 0)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3]]
        query = """RETURN list.remove([1,2,3], 1, -1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)


    def test10_sort(self):
        # NULL input should return NULL
        expected_result = [None]
        query = """WITH NULL as list RETURN list.sort(null, true)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # 2nd arg should be bool
        try:
            self.graph.query("RETURN list.sort([1,3,2], '2')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected Boolean but was String", str(e))

        # Test without input argument
        try:
            query = """RETURN list.sort()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'list.sort', expected at least 1", str(e))

        # Test with 3 input argument
        try:
            query = """RETURN list.sort([1,2,3], 2, 1)"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 3 arguments to function 'list.sort', expected at most 2", str(e))

        ### Test valid inputs ###
        expected_result = [[1]]
        query = """RETURN list.sort([1])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3]]
        query = """RETURN list.sort([1,3,2])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """RETURN list.sort([1,3,2], true)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """RETURN list.sort([3,1,2])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """RETURN list.sort([1,2,3])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """RETURN list.sort([3,2,1])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[3,2,1]]
        query = """RETURN list.sort([1,3,2], false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """RETURN list.sort([1,3,2], false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """RETURN list.sort([3,1,2], false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """RETURN list.sort([1,2,3], false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)
        query = """RETURN list.sort([3,2,1], false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[[1,2,3],[4,5,6]]]
        query = """RETURN list.sort([[4,5,6], [1,2,3]])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[[1,2,3],[1,2,3,4]]]
        query = """RETURN list.sort([[1,2,3,4], [1,2,3]])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        query = """WITH {a: 1, b: 2, c: 3} as map RETURN list.sort([map, 1, [2,1,3]])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0][0], {'a': 1, 'b': 2, 'c': 3})
        self.env.assertEquals(actual_result.result_set[0][0][1], [2,1,3])
        self.env.assertEquals(actual_result.result_set[0][0][2], 1)

    def test11_insert(self):
        # NULL input should return NULL
        expected_result = [None]
        query = """WITH NULL as list RETURN list.insert(null, 2, 3)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # 2nd arg should be integer
        try:
            self.graph.query("RETURN list.insert([1,2,3], '2', 3)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected Integer but was String", str(e))

        # 4th arg shoulf be bool
        try:
            self.graph.query("RETURN list.insert([1,2,3], 2, '1', 3)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected Boolean but was Integer", str(e))

        # Test without input argument
        try:
            query = """RETURN list.insert()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'list.insert', expected at least 3", str(e))

        # Test with 1 input argument
        try:
            query = """RETURN list.insert([1,2,3])"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 1 arguments to function 'list.insert', expected at least 3", str(e))

        # Test with 5 input argument
        try:
            query = """RETURN list.insert([1,2,3], 2, 1, true, 1)"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 5 arguments to function 'list.insert', expected at most 4", str(e))


        ### Test valid inputs ###

        expected_result = [[4,1,2,3]]
        query = """RETURN list.insert([1,2,3], 0, 4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,4,2,3]]
        query = """RETURN list.insert([1,2,3], 1, 4, true)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,4,3]]
        query = """RETURN list.insert([1,2,3], 2, 4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3,4]]
        query = """RETURN list.insert([1,2,3], 3, 4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3,4]]
        query = """RETURN list.insert([1,2,3], -1, 4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,4,3]]
        query = """RETURN list.insert([1,2,3], -2, 4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,4,2,3]]
        query = """RETURN list.insert([1,2,3], -3, 4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[4,1,2,3]]
        query = """RETURN list.insert([1,2,3], -4, 4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[4]]
        query = """RETURN list.insert([], 0, 4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[4]]
        query = """RETURN list.insert([], -1, 4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3,[1,3]]]
        query = """RETURN list.insert([1,2,3], 3, [1,2+1])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test out of range index
        expected_result = [[1,2,3]]
        query = """RETURN list.insert([1,2,3], 4, 5)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test out of range index
        expected_result = [[1,2,3]]
        query = """RETURN list.insert([1,2,3], -5, 5)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test val of type NULL
        expected_result = [[1,2,3]]
        query = """RETURN list.insert([1,2,3], 1, null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test dup=true
        expected_result = [[1,2,3]]
        query = """RETURN list.insert([1,2,3], 0, 2, false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test dup=true
        expected_result = [[[1,3],1,2,3]]
        query = """RETURN list.insert([[1,3],1,2,3], 4, [1,2+1], false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

    def test12_insertListElements(self):
        # NULL input should return NULL
        expected_result = [None]
        query = """WITH NULL as list RETURN list.insertListElements(null, [1,2,3], 2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # 2nd arg should be list
        try:
            self.graph.query("RETURN list.insertListElements([1,2,3], '2', 3)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was String", str(e))

        # 3th arg should be Integer
        try:
            self.graph.query("RETURN list.insertListElements([1,2,3], [2], '1')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected Integer but was String", str(e))

        # Test without input argument
        try:
            query = """RETURN list.insertListElements()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'list.insertListElements', expected at least 3", str(e))

        # Test with 1 input argument
        try:
            query = """RETURN list.insertListElements([1,2,3])"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 1 arguments to function 'list.insertListElements', expected at least 3", str(e))

        # Test with 5 input argument
        try:
            query = """RETURN list.insertListElements([1,2,3], [2], 1, true, 1)"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 5 arguments to function 'list.insertListElements', expected at most 4", str(e))


        ### Test valid inputs ###

        expected_result = [[4,1,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [4], 0)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,4,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [4], 1, true)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,4,3]]
        query = """RETURN list.insertListElements([1,2,3], [4], 2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3,4]]
        query = """RETURN list.insertListElements([1,2,3], [4], 3)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3,4]]
        query = """RETURN list.insertListElements([1,2,3], [4], -1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,4,3]]
        query = """RETURN list.insertListElements([1,2,3], [4], -2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,4,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [4], -3)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[4,1,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [4], -4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[4]]
        query = """RETURN list.insertListElements([], [4], 0)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[4]]
        query = """RETURN list.insertListElements([], [4], -1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3,[1,3]]]
        query = """RETURN list.insertListElements([1,2,3], [[1,2+1]], 3)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test out of range index
        expected_result = [[1,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [5], 4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test out of range index
        expected_result = [[1,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [5], -5)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test val of type NULL
        expected_result = [[1,2,3]]
        query = """RETURN list.insertListElements([1,2,3], null, 1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test dup=true
        expected_result = [[1,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [2], 0, false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test dup=true
        expected_result = [[[1,3],1,2,3]]
        query = """RETURN list.insertListElements([[1,3],1,2,3], [[1,2+1]], 4, false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        ### test multiple values in list2 ###

        expected_result = [[4,5,6,1,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [4,5,6], 0)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,4,5,6,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [4,5,6], 1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,4,5,6,3]]
        query = """RETURN list.insertListElements([1,2,3], [4,5,6], 2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3,4,5,6]]
        query = """RETURN list.insertListElements([1,2,3], [4,5,6], 3)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3,4,5,6]]
        query = """RETURN list.insertListElements([1,2,3], [4,5,6], -1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,4,5,6,3]]
        query = """RETURN list.insertListElements([1,2,3], [4,5,6], -2)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,4,5,6,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [4,5,6], -3)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[4,5,6,1,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [4,5,6], -4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[4,5,6]]
        query = """RETURN list.insertListElements([], [4,5,6], 0)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[4,5,6]]
        query = """RETURN list.insertListElements([], [4,5,6], -1)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[1,2,3,[1,3],4]]
        query = """RETURN list.insertListElements([1,2,3], [[1,2+1], 4], 3)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test out of range index
        expected_result = [[1,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [5, 6], 4)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test out of range index
        expected_result = [[1,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [5, 9], -5)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test dup=true
        expected_result = [[9,7,1,2,3]]
        query = """RETURN list.insertListElements([1,2,3], [9,3,2,7], 0, false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # test dup=true
        expected_result = [[1,[1,3],2,3,[1,4],8,[5]]]
        query = """RETURN list.insertListElements([1,[1,3],2,3,[5]], [[1,4],[1,2+1],8,[5]], 4, false)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

    def test13_dedup(self):
        # NULL input should return NULL
        expected_result = [None]
        query = """WITH NULL as list RETURN list.dedup(null)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        # 1st arg should be list
        try:
            self.graph.query("RETURN list.dedup('2')")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Type mismatch: expected List or Null but was String", str(e))

        # Test without input argument
        try:
            query = """RETURN list.dedup()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 0 arguments to function 'list.dedup', expected at least 1", str(e))

        # Test with 2 input argument
        try:
            query = """RETURN list.dedup([1,2,3], 2)"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Received 2 arguments to function 'list.dedup', expected at most 1", str(e))


        ### Test valid inputs ###

        expected_result = [[1,2,3,4]]
        query = """RETURN list.dedup([1,2,3,2,2,1,4])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[3]]
        query = """RETURN list.dedup([3,3,3])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

        expected_result = [[3,[1,2],[1]]]
        query = """RETURN list.dedup([3,[1,2],3,[1],[1,2]])"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0], expected_result)

