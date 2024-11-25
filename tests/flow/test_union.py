from common import *

GRAPH_ID = "union_test"

class testUnion(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        # Construct a graph with the form:
        # (v1)-[:E1]->(v2)-[:E2]->(v3)
        node_props = ['v1', 'v2', 'v3']

        nodes = {}
        for idx, v in enumerate(node_props):
            node = Node(alias=f"n{idx}", labels="L", properties={"v": v})
            nodes[v] = node

        e0 = Edge(nodes['v1'], "E1", nodes['v2'], properties={"v": "v1_v2"})
        e1 = Edge(nodes['v2'], "E2", nodes['v3'], properties={"v": "v2_v3"})

        nodes_str = [str(node) for node in nodes.values()]
        self.graph.query(f"CREATE {','.join(nodes_str)}, {e0}, {e1}")

    def test01_union(self):
        q = """RETURN 1 as one UNION ALL RETURN 1 as one"""
        result = self.graph.query(q)
        # Expecting 2 identical records.
        self.env.assertEquals(len(result.result_set), 2)
        expected_result = [[1],
                           [1]]
        self.env.assertEquals(result.result_set, expected_result)

        q = """RETURN 1 as one UNION RETURN 1 as one"""
        result = self.graph.query(q)
        # Expecting a single record, duplicate removed.
        self.env.assertEquals(len(result.result_set), 1)
        expected_result = [[1]]
        self.env.assertEquals(result.result_set, expected_result)

        q = """MATCH a = () return length(a) AS len UNION ALL MATCH b = () RETURN length(b) AS len"""
        result = self.graph.query(q)
        # 3 records from each sub-query, coresponding to each path matched.
        self.env.assertEquals(len(result.result_set), 6)

    def test02_invalid_union(self):
        try:
            # projection must be exactly the same.
            q = """RETURN 1 as one UNION RETURN 1 as two"""
            self.graph.query(q)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    # Performing UNION with the same left and right side should
    # produce the same result as evaluating just one side.
    def test03_union_deduplication(self):
        non_union_query = """MATCH (a)-[]->(b) RETURN a.v, b.v ORDER BY a.v, b.v"""
        non_union_result = self.graph.query(non_union_query)

        union_query = """MATCH (a)-[]->(b) RETURN a.v, b.v ORDER BY a.v, b.v
                         UNION
                         MATCH (a)-[]->(b) RETURN a.v, b.v ORDER BY a.v, b.v"""
        union_result = self.graph.query(union_query)
        self.env.assertEquals(union_result.result_set, non_union_result.result_set)

    # A syntax error should be raised on edge alias reuse in one side of a union.
    def test04_union_invalid_reused_edge(self):
        try:
            query = """MATCH ()-[e]->()-[e]->() RETURN e
                       UNION
                       MATCH ()-[e]->() RETURN e"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    # An edge alias appearing on both sides of a UNION is expected.
    def test05_union_valid_reused_edge(self):
        query = """MATCH ()-[e]->() RETURN e.v ORDER BY e.v
                   UNION
                   MATCH ()-[e]->() RETURN e.v ORDER BY e.v
                   UNION
                   MATCH ()-[e]->() RETURN e.v ORDER BY e.v"""
        result = self.graph.query(query)

        expected_result = [["v1_v2"],
                           ["v2_v3"]]
        self.env.assertEquals(result.result_set, expected_result)

    # Union should be capable of collating nodes and edges in a single column.
    def test06_union_nodes_with_edges(self):
        query = """MATCH ()-[e]->() RETURN e
                   UNION
                   MATCH (e) RETURN e"""
        union_result = self.graph.query(query)

        # All 3 nodes and 2 edges should be returned.
        self.env.assertEquals(len(union_result.result_set), 5)

        query = """MATCH ()-[e]->() RETURN e
                   UNION ALL
                   MATCH (e) RETURN e"""
        union_all_result = self.graph.query(query)

        # The same results should be produced regardless of whether ALL is specified.
        self.env.assertEquals(union_result.result_set, union_all_result.result_set)

    # Union should function properly when one of its subqueries is ordered
    # and the other is not.
    def test07_union_with_partial_ordering(self):
        query = """UNWIND range(1, 2) AS v RETURN v ORDER BY v DESC
                   UNION
                   UNWIND range(1, 3) AS v RETURN v"""
        result = self.graph.query(query)
        expected_result = [[2],
                           [1],
                           [3]]
        self.env.assertEquals(result.result_set, expected_result)

        # The results should not be modified when variables are aliased
        query = """UNWIND range(1, 2) AS a RETURN a AS b ORDER BY a DESC
                   UNION
                   UNWIND range(1, 3) AS b RETURN b"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        query = """UNWIND range(1, 2) AS a RETURN a AS b ORDER BY b DESC
                   UNION
                   UNWIND range(1, 3) AS b RETURN b"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

    def test08_union_with_index_scan(self):
        query = """UNWIND range(10,20) AS i 
                   CREATE (n:N {v:tostring(i)})-[:R]->(m:M {v:tostring(i+1)})"""
        self.graph.query(query)
        query = """CREATE INDEX ON :N(v)"""
        self.graph.query(query)

        # test MATCH and CREATE
        query = """MATCH (n:N {v:'10'})-[:R]->(m:M) RETURN m.v AS p
                   UNION
                   MATCH (s:M {v:'12'}) CREATE (:N {v:'10'})-[:R]->(s) RETURN s.v AS p"""
        result = self.graph.query(query)
        expected_result = [['11'],['12']]
        self.env.assertEquals(result.result_set, expected_result)

        # test MATCH, CREATE and MERGE
        query = """MATCH (n:N {v:'10'})-[:R]->(:M) RETURN n.v AS p 
                   UNION 
                   MATCH (s:M {v:'12'}) CREATE (:N {v:'10'})-[:R]->(s) RETURN s.v AS p 
                   UNION 
                   MERGE(x:N {v:'15'})-[:R]->(:M {v:'18'}) RETURN x.v AS p"""
        result = self.graph.query(query)
        expected_result = [['10'],['12'],['15']]
        self.env.assertEquals(result.result_set, expected_result)

    def test09_union_write_read(self):
        # test when we have a read operation followed by a write operation
        # that checks we don't crash when iterating a matrix that was updated
        self.graph.delete()
        self.graph.query("CREATE (:N), (:N:M)")

        query = """MERGE ()-[:R]->(:N) MERGE (:N:M) RETURN 0 AS n0 UNION MATCH (:N) RETURN 0 AS n0"""

        result = self.graph.query(query)
        self.env.assertEquals(len(result.result_set), 1)
        self.env.assertEquals(result.result_set, [[0]])
        self.env.assertEquals(result.nodes_created, 2)