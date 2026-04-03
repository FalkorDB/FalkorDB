from common import *

GRAPH_ID = "G"

# tests different variations of expand-into operation
# the expand-into operation is used when a traversal pattern say
# (a)-[:R]->()-[:R]->(z) has both of it ends already resolved e.g.
# MATCH (a), (z) WITH a,z MATCH (a)-[:R]->()-[:R]->(z) return z

class testExpandInto():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    # test expand into single hop no multi-edge
    # (:A)-[:R]->(:B)
    def test01_single_hop_no_multi_edge(self):
        # create graph
        query = "CREATE (:A)-[:R {v:1}]->(:B)"
        self.graph.query(query)

        # make sure (a) is connected to (b) via a 'R' edge
        query = "MATCH (a:A)-[]->(b:B) WITH a,b MATCH (a)-[:R]->(b) RETURN count(a)"
        plan = str(self.graph.explain(query))
        result = self.graph.query(query)
        self.env.assertIn("Expand Into", plan)
        self.env.assertEquals(1, result.result_set[0][0])

        query = "MATCH (a:A)-[]->(b:B) WITH a,b MATCH (a)-[e:R]->(b) RETURN e.v"
        plan = str(self.graph.explain(query))
        result = self.graph.query(query)
        self.env.assertIn("Expand Into", plan)
        self.env.assertEquals(1, result.result_set[0][0])


    # test expand into single hop multi-edge
    # (:A)-[:R]->(:B), (:A)-[:R]->(:B)
    def test02_single_hop_multi_edge(self):
        self.graph.delete()

        # create graph
        query = "CREATE (a:A)-[:R {v:1}]->(b:B), (a)-[:R {v:2}]->(b)"
        self.graph.query(query)

        # make sure (a) is connected to (b) via a 'R' edge
        # there are multiple edges of type 'R' connecting the two
        query = "MATCH (a:A)-[]->(b:B) WITH a,b MATCH (a)-[e:R]->(b) RETURN count(e)"
        plan = str(self.graph.explain(query))
        result = self.graph.query(query)
        self.env.assertIn("Expand Into", plan)
        self.env.assertEquals(2, result.result_set[0][0])

        query = "MATCH (a:A)-[]->(b:B) WITH a,b MATCH (a)-[e:R]->(b) RETURN e.v ORDER BY e.v"
        plan = str(self.graph.explain(query))
        result = self.graph.query(query)
        self.env.assertIn("Expand Into", plan)
        self.env.assertEquals(1, result.result_set[0][0])
        self.env.assertEquals(2, result.result_set[1][0])


    # test expand into multiple hops with no multi-edge
    # (:A)-[:R]->()-[:R]->(:B)
    def test03_multi_hop_no_multi_edge(self):
        self.graph.delete()

        # create graph
        query = "CREATE (:A)-[:R]->()-[:R]->(:B)"
        self.graph.query(query)

        # make sure (a) is connected to (b) via a 'R' edge
        # expand-into inspects the result of (F*R*ADJ)[a,b]
        query = "MATCH (a:A)-[*]->(b:B) WITH a,b MATCH (a)-[:R]->()-[]->(b) RETURN count(a)"
        plan = str(self.graph.explain(query))
        result = self.graph.query(query)
        self.env.assertIn("Expand Into", plan)
        self.env.assertEquals(1, result.result_set[0][0])

    # test expand into multiple hops with multi-edge
    # (a:A)-[:R]->(i)-[:R]->(b:B)
    # (a:A)-[:R]->(i)-[:R]->(b:B)
    def test04_multi_hop_multi_edge(self):
        self.graph.delete()

        # create graph
        query = "CREATE (a:A), (b:B), (i), (a)-[:R]->(i)-[:R]->(b), (a)-[:R]->(i)-[:R]->(b)"
        self.graph.query(query)

        # make sure (a) is connected to (b) via a 'R' edge
        # there are multiple ways to reach (b) from (a)
        query = "MATCH (a:A)-[*]->(b:B) WITH a,b MATCH (a)-[:R]->()-[]->(b) RETURN count(1)"
        plan = str(self.graph.explain(query))
        result = self.graph.query(query)
        self.env.assertIn("Expand Into", plan)
        self.env.assertEquals(4, result.result_set[0][0])

    def test05_no_hop_multi_label(self):
        self.graph.delete()

        # create graph
        query = "CREATE (:A), (:A:B), (:A:B:C)"
        self.graph.query(query)

        # make sure expand into is utilize
        queries = ["MATCH (a:A:B:C) RETURN count(a)",
                   "MATCH (a:A:C:B) RETURN count(a)",
                   "MATCH (a:B:A:C) RETURN count(a)",
                   "MATCH (a:B:C:A) RETURN count(a)",
                   "MATCH (a:C:A:B) RETURN count(a)",
                   "MATCH (a:C:B:A) RETURN count(a)"]

        for q in queries:
            plan = str(self.graph.explain(q))
            result = self.graph.query(q)
            self.env.assertIn("Label Scan", plan)
            self.env.assertIn("Expand Into", plan)
            self.env.assertEquals(1, result.result_set[0][0])

    def test06_expand_into_reset_crash(self):
        # crash: https://github.com/FalkorDB/FalkorDB/issues/1231

        # run on an empty graph
        self.graph.delete()

        # populate graph
        q = """
        UNWIND range(0, 30) AS x
        CREATE (d:Item { id: 1 })<-[:FromItem]-(de:ItemElement)<-[:ItemElement]-(e:Thing),
               (d)<-[:FromItem]-(:ItemElement)-[:SubCategory]->(sc:SubCategory),
               (de)-[:TextAssociated]->(t),
               (de)-[:CrossItem]->(:Relation)-[:FromItem]->(:Item),
               (d)<-[:FromItem]-(:ItemElement)-[:SubCategory]->(sc:SubCategory),
               (de)-[:TextAssociated]->(t),
               (de)-[:CrossItem]->(:Relation)-[:FromItem]->(:Item),
               (d)<-[:FromItem]-(:ItemElement)-[:SubCategory]->(sc:SubCategory),
               (de)-[:TextAssociated]->(t),
               (de)-[:CrossItem]->(:Relation)-[:FromItem]->(:Item),
               (d)<-[:FromItem]-(:ItemElement)-[:SubCategory]->(sc:SubCategory),
               (de)-[:TextAssociated]->(t),
               (de)-[:CrossItem]->(:Relation)-[:FromItem]->(:Item)"""

        self.graph.query(q)


        # used to crash due to uninitialized memory
        for i in range(0, 10):
            q = """
            MATCH (d:Item {id: 1})<-[:FromItem]-(de:ItemElement)
            OPTIONAL MATCH (d)<-[:FromItem]-(de:ItemElement)-[:SubCategory]->(sc:SubCategory)
            WHERE (de:Component)
            RETURN 2"""

            self.graph.query(q)

