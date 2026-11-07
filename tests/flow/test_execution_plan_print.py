from common import *

GRAPH_KEY = "execution_plan_print"

class test_execution_plan_print():
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env()
        cls.graph = cls.db.select_graph(GRAPH_KEY)

        # create key
        cls.graph.query("RETURN 1")
    
    # 'conditional traverse' after a 'label scan' should not print
    # the label scanned
    def test01_conditional_traverse(self):
        # empty graph (first test) --> order of traversal stays as in query (A --> B).
        plan = str(self.graph.explain("MATCH (n:A:B) RETURN n"))

        # label A is scanned
        assert "Node By Label Scan | (n:A:B)" in plan
    
    # 'expand into' should not print the label scanned
    def test02_expand_into(self):
        plan = str(self.graph.explain("MATCH (n:A:B:C) RETURN n"))

        # label A is scanned
        assert "Node By Label Scan | (n:A:B:C)" in plan
    
    # Make sure the 'conditional traverse' and 'expand into' operations
    # which do not come after a 'label scan' print all labels
    def test03_operations_not_after_scan(self):
        plan = str(self.graph.explain("match p=(n:A:B)-[*]-(m:C:D) RETURN p"))

        # A is scanned
        assert "Node By Label Scan | (n:A:B)" in plan

        # 'conditional variable length traverse' shouldn't print labels
        # as it does not enforce them
        assert "Conditional Variable Length Traverse | (n)-[_anon_0*1..INF]->(m)" in plan

