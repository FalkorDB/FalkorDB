from common import *
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from demo import QueryInfo

GRAPH_ID = "path"

class testPath(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def path_to_string(self, path):
        str_path = ", ".join([str(obj) for obj in path])
        return str_path

    def setUp(self):
        self.env.flush()
    
    def test_simple_path(self):
        node0 = Node(node_id=0, labels="L1")
        node1 = Node(node_id=1, labels="L1")
        node2 = Node(node_id=2, labels="L1")
        edge01 = Edge(node0, "R1", node1, edge_id=0, properties={'value': 1})
        edge12 = Edge(node1, "R1", node2, edge_id=1, properties={'value': 2})

        self.graph.add_node(node0)
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        self.graph.add_edge(edge01)
        self.graph.add_edge(edge12)

        self.graph.flush()

        # Rewrite the edges with IDs instead of node values to match how they are returned.
        edge01 = Edge(0, "R1", 1, edge_id=0, properties={'value': 1})
        edge12 = Edge(1, "R1", 2, edge_id=1, properties={'value': 2})

        path01 = Path.new_empty_path().add_node(node0).add_edge(edge01).add_node(node1)
        path12 = Path.new_empty_path().add_node(node1).add_edge(edge12).add_node(node2)
        expected_results = [[path01], [path12]]

        query = "MATCH p=(:L1)-[:R1]->(:L1) RETURN p"
        query_info = QueryInfo(query = query, description="Tests simple paths", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(self.graph.query(query), query_info)

    def test_variable_length_path(self):
        node0 = Node(node_id=0, labels="L1")
        node1 = Node(node_id=1, labels="L1")
        node2 = Node(node_id=2, labels="L1")
        edge01 = Edge(node0, "R1", node1, edge_id=0, properties={'value': 1})
        edge12 = Edge(node1, "R1", node2, edge_id=1, properties={'value': 2})

        self.graph.add_node(node0)
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        self.graph.add_edge(edge01)
        self.graph.add_edge(edge12)

        self.graph.flush()

        # Rewrite the edges with IDs instead of node values to match how they are returned.
        edge01 = Edge(0, "R1", 1, edge_id=0, properties={'value': 1})
        edge12 = Edge(1, "R1", 2, edge_id=1, properties={'value': 2})

        path01 = Path.new_empty_path().add_node(node0).add_edge(edge01).add_node(node1)
        path12 = Path.new_empty_path().add_node(node1).add_edge(edge12).add_node(node2)
        path02 = Path.new_empty_path().add_node(node0).add_edge(edge01).add_node(node1).add_edge(edge12).add_node(node2)
        expected_results=[[path01], [path12], [path02]]

        query = "MATCH p=(:L1)-[:R1*]->(:L1) RETURN p"
        query_info = QueryInfo(query = query, description="Tests variable length paths", expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(self.graph.query(query), query_info)

    def test_bi_directional_path(self):
        node0 = Node(node_id=0, labels="L1")
        node1 = Node(node_id=1, labels="L1")
        node2 = Node(node_id=2, labels="L1")
        edge01 = Edge(node0, "R1", node1, edge_id=0, properties={'value': 1})
        edge12 = Edge(node1, "R1", node2, edge_id=1, properties={'value': 2})

        self.graph.add_node(node0)
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        self.graph.add_edge(edge01)
        self.graph.add_edge(edge12)

        self.graph.flush()

        # Rewrite the edges with IDs instead of node values to match how they are returned.
        edge01 = Edge(0, "R1", 1, edge_id=0, properties={'value': 1})
        edge12 = Edge(1, "R1", 2, edge_id=1, properties={'value': 2})
        # Reverse direction edges which are not part of the graph. Read only values.
        edge10 = Edge(1, "R1", 0, edge_id=0, properties={'value': 1})
        edge21 = Edge(2, "R1", 1, edge_id=1, properties={'value': 2})

        path010 = Path.new_empty_path().add_node(node0).add_edge(edge01).add_node(node1).add_edge(edge10).add_node(node0)
        path0121 = Path.new_empty_path().add_node(node0).add_edge(edge01).add_node(node1).add_edge(edge12) \
                                        .add_node(node2).add_edge(edge21).add_node(node1)
        path01210 = Path.new_empty_path().add_node(node0).add_edge(edge01).add_node(node1).add_edge(edge12) \
                                         .add_node(node2).add_edge(edge21).add_node(node1).add_edge(edge10).add_node(node0)
        path121 = Path.new_empty_path().add_node(node1).add_edge(edge12).add_node(node2).add_edge(edge21).add_node(node1)
        path1210 = Path.new_empty_path().add_node(node1).add_edge(edge12).add_node(node2).add_edge(edge21) \
                                        .add_node(node1).add_edge(edge10).add_node(node0)
        expected_results = [[path010], [path0121], [path01210], [path121], [path1210]]

        query = "MATCH p=(:L1)-[:R1*]->(:L1)<-[:R1*]-() RETURN p"

        query_info = QueryInfo(query=query, description="Tests bi directional variable length paths",
                               expected_result=expected_results)
        self._assert_resultset_and_expected_mutually_included(self.graph.query(query), query_info)

    def test_bi_directional_path_functions(self):
        node0 = Node(node_id=0, labels="L1")
        node1 = Node(node_id=1, labels="L1")
        node2 = Node(node_id=2, labels="L1")
        edge01 = Edge(node0, "R1", node1, edge_id=0, properties={'value': 1})
        edge12 = Edge(node1, "R1", node2, edge_id=1, properties={'value': 2})

        self.graph.add_node(node0)
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        self.graph.add_edge(edge01)
        self.graph.add_edge(edge12)

        self.graph.flush()

        # Rewrite the edges with IDs instead of node values to match how they are returned.
        edge01 = Edge(0, "R1", 1, edge_id=0, properties={'value': 1})
        edge12 = Edge(1, "R1", 2, edge_id=1, properties={'value': 2})
        # Reverse direction edges which are not part of the graph. Read only values.
        edge10 = Edge(1, "R1", 0, edge_id = 0 , properties={'value':1})
        edge21 = Edge(2, "R1", 1, edge_id = 1 , properties={'value':2})

        expected_results=[[[node0, node1, node0], [edge01, edge10], 2],
                        [[node0, node1, node2, node1], [edge01, edge12, edge21], 3], 
                        [[node0, node1, node2, node1, node0], [edge01, edge12, edge21, edge10], 4], 
                        [[node1, node2, node1], [edge12, edge21], 2], 
                        [[node1, node2, node1, node0], [edge12, edge21, edge10], 3]]

        query = "MATCH p=(:L1)-[:R1*]->(:L1)<-[:R1*]-() RETURN nodes(p), relationships(p), length(p)"

        query_info = QueryInfo(query = query, description="Tests path functions over bi directional variable length paths", \
                                        expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(self.graph.query(query), query_info)

    def test_zero_length_path(self):
        node0 = Node(node_id=0, labels="L1")
        node1 = Node(node_id=1, labels="L2")
        edge01 = Edge(node0, "R1", node1, edge_id=0, properties={'value': 1})

        self.graph.add_node(node0)
        self.graph.add_node(node1)
        self.graph.add_edge(edge01)

        self.graph.flush()

        path01 = Path.new_empty_path().add_node(node0).add_edge(edge01).add_node(node1)
        expected_results=[[path01]]

        query = "MATCH p=(:L1)-[*0..]->()-[]->(:L2) RETURN p"

        query_info = QueryInfo(query = query, description="Tests path with zero length variable length paths", \
                                        expected_result = expected_results)
        self._assert_resultset_and_expected_mutually_included(self.graph.query(query), query_info)

    def test_path_comparison(self):
        node0 = Node(node_id=0, labels="L1")
        node1 = Node(node_id=1, labels="L1")
        node2 = Node(node_id=2, labels="L1")
        edge01 = Edge(node0, "R1", node1, edge_id=0, properties={'value': 1})
        edge12 = Edge(node1, "R1", node2, edge_id=1, properties={'value': 2})

        self.graph.add_node(node0)
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        self.graph.add_edge(edge01)
        self.graph.add_edge(edge12)

        self.graph.flush()

        # Rewrite the edges with IDs instead of node values to match how they are returned.
        edge01 = Edge(0, "R1", 1, edge_id=0, properties={'value': 1})
        edge12 = Edge(1, "R1", 2, edge_id=1, properties={'value': 2})

        path01 = Path.new_empty_path().add_node(node0).add_edge(edge01).add_node(node1)
        path12 = Path.new_empty_path().add_node(node1).add_edge(edge12).add_node(node2)

        # Test a path equality filter
        query = "MATCH p1 = (:L1)-[:R1]->(:L1) MATCH p2 = (:L1)-[:R1]->(:L1) WHERE p1 = p2 RETURN p1"
        expected_results = [[path01],
                            [path12]]

        query_info = QueryInfo(query=query, description="Test path equality", expected_result=expected_results)
        self._assert_resultset_and_expected_mutually_included(self.graph.query(query), query_info)

        # Test a path inequality filter
        query = "MATCH p1 = (:L1)-[:R1]->(:L1) MATCH p2 = (:L1)-[:R1]->(:L1) WHERE p1 <> p2 RETURN DISTINCT p1, p2"
        expected_results = [[path01, path12],
                            [path12, path01]]

        query_info = QueryInfo(query=query, description="Test path inequality", expected_result=expected_results)
        self._assert_resultset_and_expected_mutually_included(self.graph.query(query), query_info)

    # Test property accesses against non-identifier entities.
    def test_path_property_access(self):
        node0 = Node(node_id=0, labels="L1", properties={'value': 1})
        node1 = Node(node_id=1, labels="L1", properties={'value': 2})
        edge01 = Edge(node0, "R1", node1, edge_id=0)

        self.graph.add_node(node0)
        self.graph.add_node(node1)
        self.graph.add_edge(edge01)

        self.graph.flush()

        # Test access of pre-existing properties along a path.
        query = """MATCH p=()-[]->() RETURN nodes(p)[0].value, nodes(p)[1].value"""
        result = self.graph.query(query)
        expected_result = [[1, 2]]
        self.env.assertEqual(result.result_set, expected_result)

        # Test access of properties introduced by the query.
        query = """MATCH p=(a)-[]->() SET a.newval = 'new' RETURN nodes(p)[0].value, nodes(p)[0].newval"""
        result = self.graph.query(query)
        expected_result = [[1, 'new']]
        self.env.assertEqual(result.result_set, expected_result)

    # Test path deletion
    def test_path_deletion(self):
        # Test delete empty path
        query = """CREATE (a:X), (b:Y)"""
        self.graph.query(query)
        query = """MATCH p = (a:X)-[r:R]-(b:Y) DELETE p"""
        result = self.graph.query(query)
        expected_result = []
        self.env.assertEquals(result.result_set, expected_result)
        query = """MATCH (a:X) DELETE a"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_deleted, 1)
        query = """MATCH (b:Y) DELETE b"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_deleted, 1)

        # Test delete empty path
        query = """MATCH p = (a:X)-[r:R]-(b:Y) DELETE p"""
        result = self.graph.query(query)
        expected_result = []
        self.env.assertEquals(result.result_set, expected_result)

        # Test delete simple path
        query = """CREATE (a:X), (b:Y)"""
        self.graph.query(query)
        query = """MATCH (a:X), (b:Y) create (a)-[r:R]->(b)"""
        self.graph.query(query)
        query = """MATCH p = (a:X)-[r:R]-(b:Y) DELETE p"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_deleted, 2)
        self.env.assertEquals(result.relationships_deleted, 1)

        # Test delete 2 nodes, 2 relationships
        query = """CREATE (a:X), (b:Y)"""
        self.graph.query(query)
        query = """MATCH (a:X), (b:Y) create (a)-[r:R]->(b)"""
        self.graph.query(query)
        query = """MATCH (a:X), (b:Y) create (a)<-[r:R]-(b)"""
        self.graph.query(query)
        query = """MATCH p = (a:X)-[r:R]-(b:Y) DELETE p"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_deleted, 2)
        self.env.assertEquals(result.relationships_deleted, 2)

        # Test delete multiple paths
        query = """CREATE (a:X), (b:Y), (c:Z), (d:W)"""
        self.graph.query(query)
        query = """MATCH (a:X), (b:Y) create (a:X)-[r:R]->(b:Y)"""
        self.graph.query(query)
        query = """MATCH (a:X), (c:Z) create (a:X)-[r:R]->(c:Z)"""
        self.graph.query(query)
        query = """MATCH (c:Z), (d:W) create (a:X)-[r:R]->(c:Z)"""
        self.graph.query(query)
        query = """MATCH p = (n)-[r:R]-(m) DELETE p"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_deleted, 4)
        self.env.assertEquals(result.relationships_deleted, 3)

        # Test delete path length 3
        query = """CREATE (a:X), (b:Y), (c:Z)"""
        self.graph.query(query)
        query = """MATCH (a:X), (b:Y), (c:Z) create (a:X)-[r1:R1]->(b:Y)-[r2:R2]->(c:Z)"""
        self.graph.query(query)
        query = """MATCH p = (n)<-[r1:R1]-(m)-[r2:R2]->(o) DELETE p"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_deleted, 0)
        self.env.assertEquals(result.relationships_deleted, 0)
        query = """MATCH p = (n)-[r1:R1]-(m)-[r2:R2]-(o) DELETE p"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_deleted, 3)
        self.env.assertEquals(result.relationships_deleted, 2)
        
        # Test delete nodes, edges and path
        query = """CREATE (a)-[b:B]->(c)"""
        self.graph.query(query)
        query = """MATCH p = (d)-[e]-(f) DELETE d,e,f,p"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_deleted, 2)
        self.env.assertEquals(result.relationships_deleted, 1)

        # Test delete nodes
        query = """CREATE (a)-[b:B]->(c)"""
        self.graph.query(query)
        query = """MATCH p = (d)-[e]-(f) DELETE d,p"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_deleted, 2)
        self.env.assertEquals(result.relationships_deleted, 1)

         # Test delete path duplicated match
        query = """CREATE (a)-[b:B]->(c)"""
        self.graph.query(query)
        query = """MATCH p = (d)-[e]-(f) MATCH q = (g)-[h]-(i) DELETE p,q"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_deleted, 2)
        self.env.assertEquals(result.relationships_deleted, 1)
