from common import *

GRAPH_ID = "REDUCE"

# test reduce function
# reduce (sum = 0, n in [1,2,3] | sum + n)
# the reduce function is composed of 4 components:
#    1. accumulator                  `sum`
#    2. accumulator init expression  `0`
#    3. list expression              `[1,2,3]`
#    4. variable                     `n`
#    5. eval expression              `sum + n`

# this test class verifies the correctnes reduce
# in addition to its error handeling

class testReduce():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test_sum_reduction(self):
        # sum = 0, for n in [1,2,3], sum += n
        q = "RETURN reduce(sum = 0, n in [1,2,3] | sum + n)"
        expected = 6
        actual = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(actual, expected)
        #-----------------------------------------------------------------------

        # sum = 0, for n in [1,2,3], sum -= n
        q = "RETURN reduce(sum = 0, n in [1,2,3] | sum - n)"
        expected = -6
        actual = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(actual, expected)
        #-----------------------------------------------------------------------

        # sum = 0, for n in ['1','2','3'], sum += int(n)
        q = "RETURN reduce(sum = 0, n in ['1','2','3'] | sum + tointeger(n))"
        expected = 6
        actual = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(actual, expected)

    def test_overwrite_reduction(self):
        # last = 0, for n in [1,2,3], last = n
        q = "RETURN reduce(last = 0, n in [1,2,3] | n)"
        expected = 3
        actual = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(actual, expected)

    def test_string_reduction(self):
        # msg = 'hello ' + 'world'
        q = "RETURN reduce(msg='hello ', c in ['w', 'o', 'r', 'l', 'd'] | msg+c)"
        expected = "hello world"
        actual = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(actual, expected)

    def test_array_reduction(self):
        # arr = [1,2] + [2,3]
        q = "RETURN reduce(arr=[1,2], n in [2,3] | arr+n)"
        expected = [1,2,2,3]
        actual = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(actual, expected)

    def test_variable_reduction(self):
        # sum = 1 + 1 + 2 + 3 - 3
        q = """WITH 1 AS base, [1,2,3] AS arr, -1 AS bias
               RETURN reduce(sum=base, n in arr | sum + n + bias)"""
        expected = 4
        actual = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(actual, expected)

    def test_multiple_reductions(self):
        q = """UNWIND [[1,2,3], [4,5,6]] AS arr
               RETURN reduce(sum=1, n in arr | sum + n)"""

        expected = [[7], [16]]
        actual = self.graph.query(q).result_set
        self.env.assertEquals(actual, expected)

    def test_missing_sections_reduction(self):
        # missing accumulator expression
        q = "RETURN reduce(sum=0, n in [1,2,3])"
        try:
            self.graph.query(q).result_set
        except ResponseError as e:
            self.env.assertIn("Unknown function 'reduce'", str(e))
        #-----------------------------------------------------------------------

        # missing list expression
        q = "RETURN reduce(sum=0 | x+1)"
        try:
            self.graph.query(q).result_set
        except ResponseError as e:
            self.env.assertIn("Invalid input '|'", str(e))
        #-----------------------------------------------------------------------

        # missing accumulator init
        q = "RETURN reduce(n in [1,2,3] | n)"
        try:
            self.graph.query(q).result_set
        except ResponseError as e:
            self.env.assertIn("Invalid input '|'", str(e))

    def test_missing_variables_reduction(self):
        # `x` isn't defined
        q = "RETURN reduce(sum=x, n in [1,2,3] | sum+n)"
        try:
            actual = self.graph.query(q).result_set
        except ResponseError as e:
            self.env.assertIn("'x' not defined", str(e))
        #-----------------------------------------------------------------------

        q = "RETURN reduce(sum=0, n in x | sum+n)"
        try:
            actual = self.graph.query(q).result_set
        except ResponseError as e:
            self.env.assertIn("'x' not defined", str(e))
        #-----------------------------------------------------------------------

        q = "RETURN reduce(sum=0, n in [1,2,3] | sum+x)"
        try:
            actual = self.graph.query(q).result_set
        except ResponseError as e:
            self.env.assertIn("'x' not defined", str(e))

    def test_nested_reduction(self):
        # sum = 1 + 1
        # n in [1,2]
        # sum += i in [1,2]
        q = """RETURN reduce(
               sum=reduce(x=1, n in [1] | x+n),
               n in reduce(arr=[1], n in [2] | arr+n)
               | sum + n)"""

        expected = 5
        actual = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(actual, expected)

    def test_nested_shadowed_reduce_aliases(self):
        self.graph.query("CREATE (:Node)")

        q = """MATCH (n)
               WHERE [x IN [0] |
                     reduce(s=0, x IN [0] | s +
                            reduce(s=0, x IN [0] | s + 1))][0]
               RETURN 1"""

        expected = [[1]]
        actual = self.graph.query(q).result_set
        self.env.assertEquals(actual, expected)

    def test_empty_reduction(self):
        # 1 + nothing is 1
        q = "RETURN reduce(sum=1, n in [] | sum + n)"
        expected = 1
        actual = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(actual, expected)

    def test_type_missmatch_reduction(self):
        # 'a' * 1 is an invalid operation
        q = "RETURN reduce(sum='a', n in [1,2,3] | sum * n)"
        try:
            actual = self.graph.query(q).result_set
        except ResponseError as e:
            self.env.assertIn("Type mismatch", str(e))
        #-----------------------------------------------------------------------

        # '2' isn't an array
        q = "RETURN reduce(sum=1, n in 2 | sum + n)"
        try:
            actual = self.graph.query(q).result_set
        except ResponseError as e:
            self.env.assertIn("Type mismatch", str(e))

    def test_null_reduction(self):
        q = "RETURN reduce(sum=NULL, n in [1,2,3] | sum+n)"
        actual = self.graph.query(q).result_set[0][0]
        expected = None
        self.env.assertEquals(actual, expected)
        #-----------------------------------------------------------------------

        q = "RETURN reduce(sum=1, n in NULL | sum+n)"
        actual = self.graph.query(q).result_set[0][0]
        expected = None
        self.env.assertEquals(actual, expected)
        #-----------------------------------------------------------------------

        q = "RETURN reduce(sum=1, n in [1,2,3] | sum + n + NULL)"
        actual = self.graph.query(q).result_set[0][0]
        expected = None
        self.env.assertEquals(actual, expected)

    def test_invalid_use(self):
        queries = [
            "return reduce(1,[1],$a)",
            "with 1 as x return reduce(x,x,x)",
            "with {a:1, b:2} as x return reduce(x,x,x)",
            "return reduce()",
            "return reduce(1)"
        ]

        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                # Expecting an error.
                self.env.assertContains(str(e), "Unknown function 'reduce'")
                pass
    
    def test_aggregate_in_reduce(self):
        queries = [
            "RETURN reduce(x = 0, n in [1] | min(n))",
            "WITH [1,2,3] as l RETURN reduce(x = 0, n in l | min(n))",
            "WITH [1,2,3] as l RETURN reduce(x = 0, n in l | min(n))",
            "WITH [1,2,3] as l RETURN reduce(x = min(l), n in l | n)",
            "WITH [1,2,3] as l RETURN reduce(x = 0, n in min(l) | n)"
        ]

        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                # Expecting an error.
                self.env.assertContains(str(e), "Invalid use of aggregating function 'min'")
                pass
