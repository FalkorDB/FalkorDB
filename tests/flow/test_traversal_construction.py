from common import *
from index_utils import *

GRAPH_ID = "TraversalConstruction"

class testTraversalConstruction():
    def __init__(self):
        self.env, self.db = Env()
        redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        # Create the graph
        self.graph.query("RETURN 1")

    # Test differing starting points for the same search pattern
    def test_starting_point(self):
        # Neither the source nor the destination are labeled
        # perform an AllNodeScan from the source node.
        query = """MATCH (a)-[]->(b) RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertIn("All Node Scan | (a)", plan)

        # Destination is labeled, perform a LabelScan from the destination node.
        query = """MATCH (a)-[]->(b:B) RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertIn("Node By Label Scan | (b:B)", plan)

        # Destination is filtered, perform an AllNodeScan from the destination node.
        query = """MATCH (a)-[]->(b) WHERE b.v = 2 RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertIn("All Node Scan | (b)", plan)

        # Destination is labeled but source is filtered, perform an AllNodeScan from the source node.
        query = """MATCH (a)-[]->(b:B) WHERE a.v = 1 OR a.v = 3 RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertIn("All Node Scan | (a)", plan)

        # Both are labeled and source is filtered, perform a LabelScan from the source node.
        query = """MATCH (a:A)-[]->(b:B) WHERE a.v = 3 RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertIn("Node By Label Scan | (a:A)", plan)

        # Both are labeled and dest is filtered, perform a LabelScan from the dest node.
        query = """MATCH (a:A)-[]->(b:B) WHERE b.v = 2 RETURN a, b"""
        plan = str(self.graph.explain(query))
        self.env.assertIn("Node By Label Scan | (b:B)", plan)

    # make sure traversal begins with labeled entity
    def test_start_with_label(self):
        queries = ["MATCH (A:L)-->(B)-->(C) RETURN 1",
                   # "MATCH (A)-->(B:L)-->(C) RETURN 1", # improve on this case
                   "MATCH (A)-->(B)-->(C:L) RETURN 1"]

        for q in queries:
            plan = str(self.graph.explain(q))
            ops = plan.split(os.linesep)
            ops.reverse()
            self.env.assertTrue("Node By Label Scan" in ops[0])

    # make sure traversal begins with filtered entity
    def test_start_with_filter(self):
        # MATCH (A)-->(B)-->(C) WHERE A.val = 1 RETURN *
        # MATCH (A)-->(B)-->(C) WHERE B.val = 1 RETURN *
        # MATCH (A)-->(B)-->(C) WHERE C.val = 1 RETURN *
        entities = ['A', 'B', 'C']
        for e in entities:
            q = """MATCH (A)-->(B)-->(C) WHERE {}.val = 1 RETURN *""".format(e)
            plan = str(self.graph.explain(q))
            ops = plan.split(os.linesep)
            ops.reverse()

            self.env.assertTrue("All Node Scan | ({})".format(e) in ops[0])
            self.env.assertTrue("Filter" in ops[1])

    # make sure traversal begins with bound entity
    def test_start_with_bound(self):
        # MATCH (X) WITH X as A MATCH (A)-->(B)-->(C) RETURN *
        # MATCH (X) WITH X as B MATCH (A)-->(B)-->(C) RETURN *
        # MATCH (X) WITH X as C MATCH (A)-->(B)-->(C) RETURN *
        entities = ['A', 'B', 'C']
        for e in entities:
            q = "MATCH (X) WITH X as {} MATCH (A)-->(B)-->(C) RETURN *".format(e)
            plan = str(self.graph.explain(q))
            ops = plan.split(os.linesep)
            ops.reverse()
            self.env.assertTrue("Conditional Traverse | ({}".format(e) in ops[2])

    # make sure traversal begins with bound entity and follows with filter
    def test_start_with_bound_follows_with_filter(self):
        queries = ["MATCH (X) WITH X AS B MATCH (A {v:1})-->(B)-->(C) RETURN *",
                "MATCH (X) WITH X AS B MATCH (A)-->(B)-->(C {v:1}) RETURN *"]
        for q in queries:
            plan = str(self.graph.explain(q))
            ops = plan.split(os.linesep)
            ops.reverse()
            self.env.assertTrue("Filter" in ops[3])

    def test_filter_as_early_as_possible(self):
        q = """MATCH (A:L {v: 1})-->(B)-->(C), (B)-->(D:L {v: 1}) RETURN 1"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        ops.reverse()
        self.env.assertTrue("Node By Label Scan" in ops[0]) # scan either A or D
        self.env.assertTrue("Filter" in ops[1]) # filter either A or D
        self.env.assertTrue("Conditional Traverse" in ops[2]) # traverse from A to D or from D to A
        self.env.assertTrue("Conditional Traverse" in ops[3]) # traverse from A to D or from D to A
        self.env.assertTrue("Filter" in ops[4]) # filter either A or D

    def test_long_pattern(self):
        q = """match (a)--(b)--(c)--(d)--(e)--(f)--(g)--(h)--(i)--(j)--(k)--(l) return *"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        self.env.assertEqual(len(ops), 14)

    def test_start_with_index_filter(self):
        # TODO: enable this test, once we'll score higher filters that
        # have the potential turn into index scan
        return

        create_node_range_index(graph, 'L', 'v', sync=True)

        q = """MATCH (a:L {v:1})-[]-(b:L {x:1}) RETURN a, b"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        ops.reverse()
        self.env.assertTrue("Index Scan" in ops[0]) # start with index scan

        q = """MATCH (a:L {x:1})-[]-(b:L {v:1}) RETURN a, b"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        ops.reverse()
        self.env.assertTrue("Index Scan" in ops[0]) # start with index scan

    def test_variable_length_traversal_placement(self):
        # cyclic traversal followed by variable-length traversal
        q = """MATCH (b)<-[*]-(a:L {v: 5})<--(a) WHERE b.v = 10 RETURN a"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        ops.reverse()
        self.env.assertTrue("Node By Label Scan | (a:L)" in ops[0]) # scan A
        self.env.assertTrue("Filter" in ops[1]) # filter A
        self.env.assertTrue("Expand Into" in ops[2]) # traverse from A to itself
        self.env.assertTrue("Conditional Variable Length Traverse" in ops[3]) # var-len traverse from A to B

        # bidirectional variable-length traversal
        q = """MATCH (a:L {v: 5})-[*]-(b:L) WHERE a <> b RETURN b"""
        plan = str(self.graph.explain(q))
        ops = plan.split(os.linesep)
        ops.reverse()
        self.env.assertTrue("Node By Label Scan | (a:L)" in ops[0]) # scan A
        self.env.assertTrue("Filter" in ops[1]) # filter A
        self.env.assertTrue("Conditional Variable Length Traverse" in ops[2]) # bidirectional var-len traverse from A to B

    def test_traverse_zero_length_edge(self):
        # populate graph
        self.graph.query("CREATE (:A{v:1})-[:R{x:1}]->(:B{v:2})-[:R{x:2}]->(:C{v:3})")

        # traverse from 'a' to itself
        q1 = """MATCH (a)-[*0]->(b) RETURN a, b"""
        q2 = """MATCH (a:A)-[*0]->(b) RETURN a, b"""
        q3 = """MATCH (a)-[*0]->(b:A) RETURN a, b"""

        queries = [q1, q2, q3]

        for q in queries:
            plan = self.graph.explain(q)
            root = plan.structured_plan
            self.env.assertTrue(root.name == "Results")

            child = root.children[0]
            self.env.assertTrue(child.name == "Project")

            child = child.children[0]
            self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

            child = child.children[0]
            self.env.assertTrue(child.name == "All Node Scan" in child.name or
                                "Node By Label Scan" in child.name)

            # validate that 'a' == 'b'
            result = self.graph.query(q).result_set
            for row in result:
                self.env.assertTrue(row[0] == row[1])

        #-----------------------------------------------------------------------

        # traverse from 'a' back to itself via a 0 length edge
        q1 = """MATCH (a)-[*0]->(a) RETURN a"""
        q2 = """MATCH (a:A)-[*0]->(a) RETURN a"""
        q3 = """MATCH (a)-[*0]->(a:A) RETURN a"""
        q4 = """MATCH (a:A)-[*0]->(a:A) RETURN a"""

        queries = [q1, q2, q3, q4]

        for q in queries:
            plan = self.graph.explain(q)
            root = plan.structured_plan
            self.env.assertTrue(root.name == "Results")

            child = root.children[0]
            self.env.assertTrue(child.name == "Project")

            child = child.children[0]
            self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

            child = child.children[0]
            self.env.assertTrue(child.name == "All Node Scan" in child.name or
                                "Node By Label Scan" in child.name)

            # validate 'a' was found
            result = self.graph.query(q).result_set
            self.env.assertTrue(len(result) > 0)

        #-----------------------------------------------------------------------

        # traverse from 'a' to 'b' using 0 length edge
        q = """MATCH (a:A)-[*0]->(b:B) RETURN a, b"""
        plan = self.graph.explain(q)

        root = plan.structured_plan
        self.env.assertTrue(root.name == "Results")

        child = root.children[0]
        self.env.assertTrue(child.name == "Project")

        child = child.children[0]
        self.env.assertTrue("Conditional Traverse" in child.name)

        child = child.children[0]
        self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

        child = child.children[0]
        self.env.assertTrue("Node By Label Scan" in child.name)

        # make sure 'b' isn't reachable
        result = self.graph.query(q).result_set
        self.env.assertTrue(len(result) == 0)

        #-----------------------------------------------------------------------

        # create a multi label node
        q = "CREATE (:X:Y)"
        result = self.graph.query(q)
        self.env.assertEquals(result.nodes_created, 1)

        # traverse from a multi label node 'a' to itself using a 0 length edge
        q1 = """MATCH (a:X)-[*0]->(b:Y) RETURN a, b"""
        q2 = """MATCH (a:Y)-[*0]->(b:X) RETURN a, b"""
        q3 = """MATCH (a:X:Y)-[*0]->(b:X) RETURN a, b"""
        q4 = """MATCH (a:X:Y)-[*0]->(b:Y) RETURN a, b"""
        q5 = """MATCH (a:X:Y)-[*0]->(b:Y:X) RETURN a, b"""
        queries = [q1, q2, q3, q4, q5]

        for q in queries:
            plan = self.graph.explain(q)

            root = plan.structured_plan
            self.env.assertTrue(root.name == "Results")

            child = root.children[0]
            self.env.assertTrue(child.name == "Project")

            child = child.children[0]
            self.env.assertTrue("Conditional Traverse" in child.name or
                                "Expand Into" in child.name)

            child = child.children[0]
            self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

            child = child.children[0]
            self.env.assertTrue("Node By Label Scan" in child.name or
                                "Conditional Traverse" in child.name)

            # make sure 'a' == 'b'
            result = self.graph.query(q).result_set
            self.env.assertTrue(len(result) == 1)
            self.env.assertTrue(result[0][0] == result[0][1])

        #-----------------------------------------------------------------------

        # traverse from 'a' to itself
        q1 = """MATCH (a)-[*0]->(b{v:1}) RETURN a, b"""
        q2 = """MATCH (a{v:1})-[*0]->(b) RETURN a, b"""

        queries = [q1, q2]

        for q in queries:
            plan = self.graph.explain(q)

            root = plan.structured_plan
            self.env.assertTrue(root.name == "Results")

            child = root.children[0]
            self.env.assertTrue(child.name == "Project")

            child = child.children[0]
            self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

            child = child.children[0]
            self.env.assertTrue(child.name == "Filter")

            child = child.children[0]
            self.env.assertTrue("All Node Scan" in child.name)

            # validate that 'a' == 'b'
            result = self.graph.query(q).result_set
            self.env.assertTrue(len(result) == 1)
            for row in result:
                self.env.assertTrue(row[0] == row[1])

        #-----------------------------------------------------------------------

        # traverse from 'a' to itself
        q = """MATCH (a{v:1})-[*0]->(b{v:1}) RETURN a, b"""
        plan = self.graph.explain(q)

        root = plan.structured_plan
        self.env.assertTrue(root.name == "Results")

        child = root.children[0]
        self.env.assertTrue(child.name == "Project")

        child = child.children[0]
        self.env.assertTrue(child.name == "Filter")

        child = child.children[0]
        self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

        child = child.children[0]
        self.env.assertTrue(child.name == "Filter")

        child = child.children[0]
        self.env.assertTrue("All Node Scan" in child.name)

        # validate that 'a' == 'b'
        result = self.graph.query(q).result_set
        self.env.assertTrue(len(result) == 1)
        for row in result:
            self.env.assertTrue(row[0] == row[1])

        #-----------------------------------------------------------------------

        # traverse from 'a' to a none existing node
        q = """MATCH (a{v:1})-[*0]->(b{v:2}) RETURN a, b"""
        plan = self.graph.explain(q)

        root = plan.structured_plan
        self.env.assertTrue(root.name == "Results")

        child = root.children[0]
        self.env.assertTrue(child.name == "Project")

        child = child.children[0]
        self.env.assertTrue(child.name == "Filter")

        child = child.children[0]
        self.env.assertTrue("Conditional Variable Length Traverse" in child.name)

        child = child.children[0]
        self.env.assertTrue(child.name == "Filter")

        child = child.children[0]
        self.env.assertTrue("All Node Scan" in child.name)

        # validate 'b' wasn't reached
        result = self.graph.query(q).result_set
        self.env.assertTrue(len(result) == 0)

        #-----------------------------------------------------------------------

        # build expected ordered result-set
        q = """MATCH (a) RETURN a ORDER BY a"""
        expected = self.graph.query(q).result_set

        # return named path with a 0 length edge
        q = """MATCH p = ()-[*0]->() RETURN p ORDER BY p"""
        result = self.graph.query(q).result_set

        # validate result sets
        self.env.assertTrue(len(result) == len(expected))
        for i in range(0, len(result)):
            a = expected[i][0]
            path = result[i][0]
            b = path.nodes()[0]
            self.env.assertTrue(a == b)

        #-----------------------------------------------------------------------

        q = """MATCH (a) RETURN a ORDER BY a"""
        expected = self.graph.query(q).result_set

        # get 0 length edge
        # returning a variable length edge returns a path
        # similar to MATCH ()-[e*0..2]->() return e
        q = """MATCH (a)-[e*0]->(b) RETURN e ORDER BY e"""
        result = self.graph.query(q).result_set
        for i in range(0, len(result)):
            a = expected[i][0]
            p = result[i][0]
            b = p.nodes()[0]
            self.env.assertTrue(len(p.edges()) == 0)
            self.env.assertTrue(len(p.nodes()) == 1)
            self.env.assertTrue(a == b)

        # graph: (:A{v:1})-[:R{x:1}]->(:B{v:2})-[:R{x:2}]->(:C{v:3})
        q = """MATCH (c:C) RETURN c"""
        expected = self.graph.query(q).result_set

        # make sure 'c' is reachable
        q = """MATCH (a)-[*0]->(b)-[]->(c:C) RETURN c"""
        result = self.graph.query(q).result_set
        self.env.assertTrue(result == expected)

