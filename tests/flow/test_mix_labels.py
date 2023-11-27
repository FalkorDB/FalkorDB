from common import *

male = ["Roi", "Alon", "Omri"]
female = ["Hila", "Lucy"]


class testGraphMixLabelsFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("G")
        self.populate_graph()

    def populate_graph(self):
        nodes = {}
         # Create entities
        
        for m in male:
            node = Node(alias=m, labels="male", properties={"name": m})
            nodes[m] = node
        
        for f in female:
            node = Node(alias=f, labels="female", properties={"name": f})
            nodes[f] = node

        edges = []
        for n in nodes:
            for m in nodes:
                if n == m: continue
                edges.append(Edge(nodes[n], "knows", nodes[m]))

        nodes_str = [str(n) for n in nodes.values()]
        edges_str = [str(e) for e in edges]
        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")

    # Connect a single node to all other nodes.
    def test_male_to_all(self):
        query = """MATCH (m:male)-[:knows]->(t) RETURN m,t ORDER BY m.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), (len(male) * (len(male + female)-1)))
    
    def test_male_to_male(self):
        query = """MATCH (m:male)-[:knows]->(t:male) RETURN m,t ORDER BY m.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), (len(male) * (len(male)-1)))
    
    def test_male_to_female(self):
        query = """MATCH (m:male)-[:knows]->(t:female) RETURN m,t ORDER BY m.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), (len(male) * len(female)))
    
    def test_female_to_all(self):
        query = """MATCH (f:female)-[:knows]->(t) RETURN f,t ORDER BY f.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), (len(female) * (len(male + female)-1)))

    def test_female_to_male(self):
        query = """MATCH (f:female)-[:knows]->(t:male) RETURN f,t ORDER BY f.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), (len(female) * len(male)))
    
    def test_female_to_female(self):
        query = """MATCH (f:female)-[:knows]->(t:female) RETURN f,t ORDER BY f.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), (len(female) * (len(female)-1)))
    
    def test_all_to_female(self):
        query = """MATCH (f)-[:knows]->(t:female) RETURN f,t ORDER BY f.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), (len(male) * len(female)) + (len(female) * (len(female)-1)))

    def test_all_to_male(self):
        query = """MATCH (f)-[:knows]->(t:male) RETURN f,t ORDER BY f.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), (len(male) * (len(male)-1)) + len(female) * len(male))
    
    def test_all_to_all(self):
        query = """MATCH (f)-[:knows]->(t) RETURN f,t ORDER BY f.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), (len(male+female) * (len(male+female)-1)))
