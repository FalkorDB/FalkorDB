from common import *

GRAPH_ID = "issue_636"

class testIssue636():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_repro_crash(self):
        # 1. MERGE setup
        q1 = "MERGE (:label8) MERGE (:label2{})<-[:reltype5]-(node_0{})<-[:reltype7]-({})"
        self.graph.query(q1)

        # 2. Variable-length path query 1 (with WHERE)
        q2 = "MATCH (node_0:label8{})<-[*..]-(node_0:label9) WHERE node_0.prop7 = [ FALSE ] RETURN *"
        # This used to crash (SIGSEGV). Now it should return empty result or success.
        res2 = self.graph.query(q2)
        self.env.assertEquals(len(res2.result_set), 0)

        # 3. Variable-length path query 2
        q3 = "MATCH (node_0:label8{})<-[*..]-(node_0:label9) RETURN *"
        # This used to crash (SIGSEGV). Now it should return empty result or success.
        res3 = self.graph.query(q3)
        self.env.assertEquals(len(res3.result_set), 0)

    def test02_minimal_repro(self):
        # Minimal repro from PR #1679
        self.graph.query("CREATE (:A), (:B)<-[:R0]-()<-[:R1]-()")
        q = "MATCH (n:A)<-[*]-(n:Z) RETURN 1"
        res = self.graph.query(q)
        self.env.assertEquals(len(res.result_set), 0)
