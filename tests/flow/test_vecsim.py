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
               CREATE (:Person {embeddings: vecf32([i,i]), embedding2: vecf32([i,(i + 1) * 2])})"""

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

        self.graph.create_node_vector_index("Person", "embedding2", dim=2, similarity_function="cosine")

        # index edges
        # create vector index over Points::embeddings
        self.graph.create_edge_vector_index("Points", "embeddings", dim=2, similarity_function="euclidean")

        # wait for indices to be become operational
        wait_for_indices_to_sync(self.graph)

    def test01_vector_distance(self):
        # compute euclidean distance between two vectors
        qs = {"""RETURN vec.euclideanDistance(vecf32($a), vecf32($b)) AS dist""": 1.414,
              """RETURN vec.cosineDistance(vecf32($a), vecf32($b)) AS dist""": 0.008}
              

        for q, d in qs.items():
            # distance between NULL and NULL should be NULL
            # distance between NULL and vector should be NULL
            # distance between vector and NULL should be NULL
            inputs = [ (None, None), (None, [1,1]), ([1,1], None) ]
            for a, b in inputs:
                params = {'a': a, 'b': b}
                distance = self.graph.ro_query(q, params=params).result_set[0][0]
                self.env.assertIsNone(distance)

            # distance between same vectors should be 0
            params = {'a': [1,1], 'b': [1,1]}
            distance = self.graph.ro_query(q, params=params).result_set[0][0]
            distance = round(distance, 3)
            self.env.assertEqual(distance, 0)

            # distance between [1,2] and [2,3] should be d
            params = {'a': [1,2], 'b': [2,3]}
            distance = self.graph.ro_query(q, params=params).result_set[0][0]
            distance = round(distance, 3)
            self.env.assertEqual(distance, d)

            # distance between [1,1] and [2,2,3] should fail
            # dimension mismatch
            inputs = [([1,1], [2,2,3]), ([2,2,3], [1,1]) ]
            for a, b in inputs:
                try:
                    params = {'a': [1,1], 'b': [2,2,3]}
                    distance = self.graph.ro_query(q, params=params).result_set[0][0]
                    self.env.assertFalse("Expected query to fail")
                except Exception as e:
                    self.env.assertContains("Vector dimension mismatch", str(e))

            # distance between incompatible types should fail
            inputs = [([1,1], "foo"), ("foo", [1,1]) ]
            for a, b in inputs:
                try:
                    params = {'a': a, 'b': b}
                    distance = self.graph.ro_query(q, params=params).result_set[0][0]
                    self.env.assertFalse("Expected query to fail")
                except Exception as e:
                    self.env.assertContains("Type mismatch", str(e))

    def test02_locate_similar_nodes(self):
        k = 3
        x = 50
        y = 50

        result = query_node_vector_index(self.graph, "Person", "embeddings", k,
                                         [x,y]).result_set

        assert len(result) == 3
        for row in result:
            embeddings = row[0].properties['embeddings']
            self.env.assertLess(abs(embeddings[0] - x), k)
            self.env.assertLess(abs(embeddings[1] - y), k)

    def test03_locate_similar_edges(self):
        k = 3
        x = -50
        y = -50
        result = query_edge_vector_index(self.graph, "Points", "embeddings", k,
                                         [x,y]).result_set

        assert len(result) == 3
        for row in result:
            embeddings = row[0].properties['embeddings']
            self.env.assertLess(abs(embeddings[0] - x), k)
            self.env.assertLess(abs(embeddings[1] - y), k)

    def test04_vecsim_result_order(self):
        # entities returned from vecsim should be sorted by distance
        # starting with the closest entity and increasing distance

        x = 50  # query vector x
        y = 50  # query vector y
        k = 100 # number of results to return

        q = """WITH vecf32($q) as v
               CALL db.idx.vector.queryNodes('Person', 'embeddings', $k, v)
               YIELD node, score
               RETURN score, vec.euclideanDistance(node.embeddings, v) AS dist"""
        res = self.graph.ro_query(q, params={'k':k, 'q': [x, y]}).result_set

        prev_score = float('-inf')
        for row in res:
            score, dist = row
            self.env.assertEqual(round(score, 3), round(dist, 3))
            self.env.assertGreaterEqual(score, prev_score)
            prev_score = score

        q = """WITH vecf32($q) as v
               CALL db.idx.vector.queryNodes('Person', 'embedding2', $k, v)
               YIELD node, score
               RETURN score, vec.cosineDistance(node.embedding2, v) AS dist"""
        res = self.graph.ro_query(q, params={'k':k, 'q': [x, y]}).result_set

        prev_score = float('-inf')
        for row in res:
            score, dist = row
            self.env.assertEqual(round(score, 3), round(dist, 3))
            self.env.assertGreaterEqual(score, prev_score)
            prev_score = score

    def test05_not_enough_results(self):
        # ask for more results than exist in the index
        # should return all results
        x = 50
        y = 50
        k = 1000000

        q = """WITH vecf32($q) as v
                CALL db.idx.vector.queryNodes('Person', 'embeddings', $k, v)
                YIELD node
                RETURN count(node) AS cnt"""

        count = self.graph.ro_query(q, params={'k':k, 'q': [x, y]}).result_set[0][0]
        self.env.assertEqual(count, 1001)

    def test06_validate_arguments(self):
        # validate arguments
        # first arugment must be a string
        # second argument must be a string
        # third argument must be a positive integer > 0
        # fourth argument must be a vector

        q = "CALL db.idx.vector.queryNodes($lbl, $attr, $k, vecf32($q))"

        params = [
                    {'lbl': 2, 'attr': 'embeddings', 'k': 10, 'q': [1,1]},
                    {'lbl': 'Person', 'attr':2, 'k': 10, 'q': [1,1]},
                    {'lbl': 'Person', 'attr': 'embeddings', 'k': '10', 'q': [1,1]},
                    {'lbl': 'Person', 'attr': 'embeddings', 'k': -10, 'q': [1,1]}]

        for p in params:
            try:
                self.graph.ro_query(q, params=p)
                self.env.assertFalse("Expected query to fail")
            except Exception as e:
                self.env.assertContains("Invalid arguments for procedure", str(e))

        p = {'lbl': 'Person', 'attr': 'embeddings', 'k': 10, 'q': False}
        q = "CALL db.idx.vector.queryNodes($lbl, $attr, $k, $q)"
        try:
            self.graph.ro_query(q, params=p)
            self.env.assertFalse("Expected query to fail")
        except Exception as e:
            self.env.assertContains("Invalid arguments for procedure", str(e))

    def test07_mismatch_vector_dim(self):
        # try to query a vector index using a query vector with mismatched dimension

        k = 10
        q = [1,2,3] # query vector with dimension 3, should be 2

        # query node vector index
        try:
            result = query_node_vector_index(self.graph, "Person", "embeddings", k, q).result_set
            self.env.assertFalse("Expected query to fail")
        except Exception as e:
            self.env.assertEqual("Vector dimension mismatch, expected 2 but got 3", str(e))

        # query edge vector index
        try:
            result = query_edge_vector_index(self.graph, "Points", "embeddings", k, q).result_set
            self.env.assertFalse("Expected query to fail")
        except Exception as e:
            self.env.assertEqual("Vector dimension mismatch, expected 2 but got 3", str(e))

