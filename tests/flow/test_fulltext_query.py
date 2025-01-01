from common import *
from index_utils import *

GRAPH_ID = "query"

class testFulltextIndexQuery():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        self.graph.query("CALL db.idx.fulltext.createNodeIndex('L1', 'v')")
        self.graph.query("CALL db.idx.fulltext.createNodeIndex({label: 'L2', stopwords: ['redis', 'world'] }, 'v')")
        self.graph.query("CALL db.idx.fulltext.createNodeIndex('L3', { field: 'v1', weight: 1 }, { field: 'v2', weight: 2 })")
        self.graph.query("CALL db.idx.fulltext.createNodeIndex('L4', { field: 'v', phonetic: 'dm:en' })")
        self.graph.query("CALL db.idx.fulltext.createNodeIndex('L5', { field: 'v', nostem: true })")
        self.graph.query("CREATE FULLTEXT INDEX FOR ()-[e:E]-() on (e.name)") 
        wait_for_indices_to_sync(self.graph)

        n0 = Node(labels="L1", properties={"v": 'hello redis world'})
        n1 = Node(labels="L2", properties={"v": 'hello redis world'})
        n2 = Node(labels="L3", properties={"v1": 'hello world', "v2": 'hello redis'})
        n3 = Node(labels="L3", properties={"v1": 'hello redis', "v2": 'hello world'})
        n4 = Node(labels="L4", properties={"v": 'felix'})
        n5 = Node(labels="L5", properties={"v": 'there are seven words in this sentence'})
        self.graph.query(f"CREATE {n0}, {n1}, {n2}, {n3}, {n4}, {n5}")
        # create relationships
        e0 = Edge(n5, "E", n0, properties={"name": "just another nice relationship"})
        e1 = Edge(n5, "E", n0, properties={"name": "a nice place to be"})
        e2 = Edge(n5, "E1", n0, properties={"name": "don't find me please, I'm not full text indexed"})
        self.graph.query(f"CREATE {e0}, {e1}, {e2}")

    # full-text query
    def test01_fulltext_query(self):
        expected_result = self.graph.query("MATCH (n:L1) RETURN n")
        # fulltext query L1 for hello 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L1', 'hello')")
        self.env.assertEquals(result.result_set[0][0], expected_result.result_set[0][0])

        # fulltext query L1 for redis 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L1', 'redis')")
        self.env.assertEquals(result.result_set[0][0], expected_result.result_set[0][0])

        # fulltext query L1 for world 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L1', 'world')")
        self.env.assertEquals(result.result_set[0][0], expected_result.result_set[0][0])

        expected_result = self.graph.query("MATCH (n:L2) RETURN n")

        # fulltext query L2 for hello 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L2', 'hello')")
        self.env.assertEquals(result.result_set[0][0], expected_result.result_set[0][0])

        # fulltext query L2 for redis 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L2', 'redis')")
        self.env.assertEquals(result.result_set, [])

        # fulltext query L2 for world 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L2', 'world')")
        self.env.assertEquals(result.result_set, [])

        # fulltext query L3 for redis and document that contains redis in v2 is scored higher than document contains redis in v1
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L3', 'redis') YIELD node, score RETURN node, score ORDER BY score DESC")
        self.env.assertEquals(result.result_set[0][0].properties["v2"], "hello redis")
        self.env.assertEquals(result.result_set[1][0].properties["v1"], "hello redis")
        self.env.assertGreater(result.result_set[0][1], result.result_set[1][1])

        expected_result = self.graph.query("MATCH (n:L4 {v:'felix'}) RETURN n")

        # fulltext query L4 for phelix
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L4', 'phelix')")
        self.env.assertEquals(result.result_set[0][0], expected_result.result_set[0][0])

        expected_result = self.graph.query("MATCH (n:L5) RETURN n")

        # fulltext query L5 for 'words' which exists in the document
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L5', 'words')")
        self.env.assertEquals(result.result_set[0][0], expected_result.result_set[0][0])

        # fulltext query L5 for 'word' nostem did not removed 's' from 'words'
        # as such no results are expected
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L5', 'word')")
        self.env.assertEquals(result.result_set, [])

        # full text query on a relationship E1 (not indexed)
        result = self.graph.query("CALL db.idx.fulltext.queryRelationships('E1', 'please')")
        self.env.assertEquals(result.result_set, [])

        # full text query on a relationship E (indexed) 'nice' appears in two relationships
        result = self.graph.query("CALL db.idx.fulltext.queryRelationships('E', 'nice')")
        self.env.assertEquals(len(result.result_set), 2)
        actual = [x[0].properties["name"] for x in result.result_set]
        actual.sort()
        expected = ["a nice place to be", "just another nice relationship"]
        self.env.assertEquals(actual, expected)
      


