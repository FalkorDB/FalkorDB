from common import *
from index_utils import *

GRAPH_ID = "vecsim"

class testVecsim():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = Graph(self.conn, GRAPH_ID)

        self.populate()
        self.create_indicies()

    def populate(self):
        # introduce Person nodes
        n = 1000 # number of Person nodes
        q = """UNWIND range(0, $n) AS i
               CREATE (:Person {embeddings: vecf32([i,i])})"""

        self.graph.query(q, params={'n': n})

        # introduce self referencing edges
        q = """MATCH (p:Person)
               WITH vecf32([-ID(p), -ID(p)]) AS embeddings, p
               CREATE (p)-[:Points {embeddings: embeddings}]->(p)"""
        
        self.graph.query(q)

    def create_indicies(self):
        # index nodes
        # create vector index over Person:embeddings
        self.graph.create_node_vector_index("Person", "embeddings", dim=2, similarity_function="euclidean")

        # index edges
        # create vector index over Points::embeddings
        self.graph.create_edge_vector_index("Points", "embeddings", dim=2, similarity_function="euclidean")

        # wait for indices to be become operational
        wait_for_indices_to_sync(self.graph)

    def test01_locate_similar_nodes(self):
        k = 3
        x = 50
        y = 50

        result = self.graph.query_node_vector_index("Person", "embeddings", k,
                                                    [x,y]).result_set

        assert len(result) == 3
        for row in result:
            embeddings = row[0].properties['embeddings']
            self.env.assertLess(abs(embeddings[0] - x), k)
            self.env.assertLess(abs(embeddings[1] - y), k)

    def test02_locate_similar_edges(self):
        k = 3
        x = -50
        y = -50
        result = self.graph.query_edge_vector_index("Points", "embeddings", k,
                                                    [x,y]).result_set

        assert len(result) == 3
        for row in result:
            embeddings = row[0].properties['embeddings']
            self.env.assertLess(abs(embeddings[0] - x), k)
            self.env.assertLess(abs(embeddings[1] - y), k)
