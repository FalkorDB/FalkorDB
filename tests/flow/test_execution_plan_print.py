from common import *

GRAPH_KEY = "execution_plan_print"

class test_execution_plan_print():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_KEY)

        # create key
        self.graph.query("RETURN 1")
    
    # 'conditional traverse' after a 'label scan' should not print
    # the label scanned
    def test01_conditional_traverse(self):
        # empty graph (first test) --> order of traversal stays as in query (A --> B).
        plan = str(self.graph.explain("MATCH (n:A:B) RETURN n"))

        # label A is scanned
        self.env.assertIn("Node By Label Scan | (n:A)", plan)

        # conditional traverse prints only B
        self.env.assertIn("Conditional Traverse | (n:B)->(n:B)", plan)
    
    # 'expand into' should not print the label scanned
    def test02_expand_into(self):
        plan = str(self.graph.explain("MATCH (n:A:B:C) RETURN n"))

        # label A is scanned
        self.env.assertIn("Node By Label Scan | (n:A)", plan)

        # expand_into does not print A
        self.env.assertIn("Expand Into | (n:B:C)", plan)
    
    # Make sure the 'conditional traverse' and 'expand into' operations
    # which do not come after a 'label scan' print all labels
    def test03_operations_not_after_scan(self):
        plan = str(self.graph.explain("match p=(n:A:B)-[*]-(m:C:D) RETURN p"))

        # A is scanned
        self.env.assertIn("Node By Label Scan | (n:A)", plan)

        # B is printed alone in 'conditional traverse'
        self.env.assertIn("Conditional Traverse | (n:B)->(n:B)", plan)

        # 'conditional variable length traverse' shouldn't print labels
        # as it does not enforce them
        self.env.assertIn("Conditional Variable Length Traverse | (n)-[@anon_0*1..INF]->(m)", plan)

        # 'expand into' prints labels C and D
        self.env.assertIn("Expand Into | (m:C:D)->(m:C:D)", plan)

