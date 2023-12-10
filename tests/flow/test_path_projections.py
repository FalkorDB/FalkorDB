from common import *

nodes        =  {}
GRAPH_ID     =  "path_projections"


class testPathProjections():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        # Construct a graph with the form:
        # (v0)-[:E]->(v1)-[:E]->(v2)-[:E]->(v3), (v0)-[:E]->(v4)

        global nodes
        edges = []
        for v in range(0, 5):
            node = Node(alias=f"n{v}", labels="L", properties={"v": v})
            nodes[v] = node

        edges.append(Edge(nodes[0], "E", nodes[1], properties={"connects": "01"}))
        edges.append(Edge(nodes[1], "E", nodes[2], properties={"connects": "12"}))
        edges.append(Edge(nodes[2], "E", nodes[3], properties={"connects": "23"}))
        edges.append(Edge(nodes[0], "E", nodes[4], properties={"connects": "04"}))

        nodes_str = [str(n) for n in nodes.values()]
        edges_str = [str(e) for e in edges]

        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")

    def test01_single_source_projection(self):
        query = """MATCH (a {v: 0}) WITH (a)-[]->() AS paths
                   UNWIND paths as path
                   RETURN nodes(path) AS nodes ORDER BY nodes"""
        actual_result = self.graph.query(query)
        # The nodes on Node 0's two outgoing paths should be returned
        traversal01 = [nodes[0], nodes[1]]
        traversal04 = [nodes[0], nodes[4]]
        expected_result = [[traversal01],
                           [traversal04]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test02_multi_source_projection(self):
        query = """MATCH (a) WITH (a)-[]->() AS paths WHERE a.v < 2
                   UNWIND paths as path
                   RETURN nodes(path) AS nodes ORDER BY nodes"""
        actual_result = self.graph.query(query)
        traversal01 = [nodes[0], nodes[1]]
        traversal04 = [nodes[0], nodes[4]]
        traversal12 = [nodes[1], nodes[2]]
        expected_result = [[traversal01],
                           [traversal04],
                           [traversal12]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test03_multiple_projections(self):
        query = """MATCH (a {v: 1}) WITH (a)-[]->() AS p1, (a)<-[]-() AS p2
                   UNWIND p1 AS n1 UNWIND p2 AS n2
                   RETURN nodes(n1) AS nodes, nodes(n2) ORDER BY nodes"""
        actual_result = self.graph.query(query)
        traversal = [[nodes[1], nodes[2]], [nodes[1], nodes[0]]]
        expected_result = [traversal]
        self.env.assertEqual(actual_result.result_set, expected_result)

        plan = str(self.graph.explain(query))
        self.env.assertEquals(2, plan.count("Apply"))

    def test04_variable_length_projection(self):
        query = """MATCH (a {v: 1}) WITH (a)-[*]->({v: 3}) AS paths
                   UNWIND paths as path
                   RETURN nodes(path) AS nodes ORDER BY nodes"""
        actual_result = self.graph.query(query)
        traversal = [nodes[1], nodes[2], nodes[3]]
        expected_result = [[traversal]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test05_no_bound_variables_projection(self):
        query = """MATCH (a {v: 1}) WITH a, ({v: 2})-[]->({v: 3}) AS paths
                   UNWIND paths as path
                   RETURN a, nodes(path) AS nodes ORDER BY nodes"""
        actual_result = self.graph.query(query)
        traversal = [nodes[2], nodes[3]]
        expected_result = [[nodes[1], traversal]]
        self.env.assertEqual(actual_result.result_set, expected_result)

    def test06_nested_traversal(self):
        query = """MATCH (a {v: 1}) WITH a, [({v: 2})-[]->({v: 3})] AS path_arr
                   UNWIND path_arr as paths
                   UNWIND paths AS path
                   RETURN a, nodes(path) AS nodes ORDER BY nodes"""
        actual_result = self.graph.query(query)
        traversal = [nodes[2], nodes[3]]
        expected_result = [[nodes[1], traversal]]
        self.env.assertEqual(actual_result.result_set, expected_result)
