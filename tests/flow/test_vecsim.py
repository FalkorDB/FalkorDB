from common import *
from index_utils import wait_for_indices_to_sync

GRAPH_ID = "vecsim"

class testVecsim():
    def __init__(self):
        self.env = Env(decodeResponses=True)
        self.conn = self.env.getConnection()
        self.graph = Graph(self.conn, GRAPH_ID)

        self.populate()
        self.create_indicies()

    def populate(self):
        # introduce Person nodes
        n = 1000 # number of Person nodes
        q = """UNWIND range(0, $n) AS i
               CREATE (:Person {embeddings: vector32f([i,i])})"""

        self.graph.query(q, params={'n': n})

        # introduce self referencing edges
        q = """MATCH (p:Person)
               WITH vector32f([-ID(p), -ID(p)]) AS embeddings, p
               CREATE (p)-[:Points {embeddings: embeddings}]->(p)"""
        
        self.graph.query(q)

    def create_indicies(self):
        # index nodes
        # create vector index over Person:embeddings
        q = """CALL db.idx.vector.createIndex(
            {type:'NODE', label:'Person', attribute:'embeddings', dim:2})"""

        self.graph.query(q)

        # index edges
        # create vector index over Points::embeddings
        q = """CALL db.idx.vector.createIndex(
            {type:'RELATIONSHIP', label:'Points', attribute:'embeddings', dim:2})"""

        self.graph.query(q)

        # wait for indices to be become operational
        wait_for_indices_to_sync(self.graph)

    def test01_locate_similar_nodes(self):
        q = """CALL db.idx.vector.knn({
            type: 'NODE',
            label: 'Person',
            attribute: 'embeddings',
            query_vector: vector32f([$x,$y]),
            k:$k
            })
            YIELD entity
            RETURN entity.embeddings"""

        k = 3
        x = 50
        y = 50
        params = {'x': x, 'y': y, 'k': k}
        result = self.graph.query(q, params=params).result_set

        assert len(result) == 3
        for row in result:
            embeddings = row[0]
            self.env.assertLess(abs(embeddings[0] - x), k)
            self.env.assertLess(abs(embeddings[1] - y), k)

    def test02_locate_similar_edges(self):
        q = """CALL db.idx.vector.knn({
            type: 'RELATIONSHIP',
            label: 'Points',
            attribute: 'embeddings',
            query_vector: vector32f([$x,$y]),
            k:$k
            })
            YIELD entity
            RETURN entity.embeddings"""

        k = 3
        x = -50
        y = -50
        params = {'x': x, 'y': y, 'k': k}
        result = self.graph.query(q, params=params).result_set

        assert len(result) == 3
        for row in result:
            embeddings = row[0]
            self.env.assertLess(abs(embeddings[0] - x), k)
            self.env.assertLess(abs(embeddings[1] - y), k)

