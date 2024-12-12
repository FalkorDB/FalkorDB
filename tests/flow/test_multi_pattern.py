from common import *

GRAPH_ID = "multi_pattern"
people = ["Roi", "Alon", "Ailon", "Boaz", "Tal", "Omri", "Ori"]


class testGraphMultiPatternQueryFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
         # Create entities
        nodes = []
        for p in people:
            nodes.append(Node(labels="person", properties={"name": p}))

        self.graph.query(f"CREATE {','.join(map(str, nodes))}")

    # Connect a single node to all other nodes.
    def test01_connect_node_to_rest(self):
        query = """MATCH(r:person {name:"Roi"}), (f:person) WHERE f.name <> r.name CREATE (r)-[:friend]->(f) RETURN count(f)"""
        actual_result = self.graph.query(query)
        friend_count = actual_result.result_set[0][0]
        self.env.assertEquals(friend_count, 6)
        self.env.assertEquals(actual_result.relationships_created, 6)

    def test02_verify_cartesian_product_streams_reset(self):
        # See https://github.com/RedisGraph/RedisGraph/issues/249
        # Forevery outgoing edge, we expect len(people) to be matched.
        expected_resultset_size = 6 * len(people)
        queries = ["""MATCH (r:person {name:"Roi"})-[]->(f), (x) RETURN f, x""",
                   """MATCH (r:person {name:"Roi"})-[]->(f), (x) RETURN f, x""",
                   """MATCH (x), (r:person {name:"Roi"})-[]->(f) RETURN f, x""",
                   """MATCH (r:person {name:"Roi"})-[]->(f) MATCH (x) RETURN f, x""",
                   """MATCH (x) MATCH (r:person {name:"Roi"})-[]->(f) RETURN f, x"""]
        for q in queries:
            actual_result = self.graph.query(q)
            records_count = len(actual_result.result_set)
            self.env.assertEquals(records_count, expected_resultset_size)

        # crash from fuzzer
        q = """MATCH (n0:X), (n1:X), (:X) WHERE n0.v = n1.v OR 0 <> NULL DELETE n0 , n1 RETURN *"""
        self.graph.query(q)

    # Connect every node to every node.
    def test03_create_fully_connected_graph(self):
        query = """MATCH(a:person), (b:person) WHERE a.name <> b.name CREATE (a)-[f:friend]->(b) RETURN count(f)"""
        actual_result = self.graph.query(query)
        friend_count = actual_result.result_set[0][0]
        self.env.assertEquals(friend_count, 42)
        self.env.assertEquals(actual_result.relationships_created, 42)
    
    # Perform a cartesian product of 3 sets.
    def test04_cartesian_product(self):
        queries = {"""MATCH (a), (b), (c) RETURN count(a)""": 343,
                   """MATCH (a) MATCH (b), (c) RETURN count(a)""": 343,
                   """MATCH (a), (b) MATCH (c) RETURN count(a)""": 343,
                   """MATCH (a) MATCH (b) MATCH (c) RETURN count(a)""": 343,
                   """MATCH (a) OPTIONAL MATCH ({n0:0}), () RETURN count(a)""": 7}

        for q, c in queries.items():
            actual_result = self.graph.query(q)
            friend_count = actual_result.result_set[0][0]
            self.env.assertEquals(friend_count, c)

    def test06_multiple_create_clauses(self):
        queries = ["""CREATE (:a {v:1}), (:b {v:2, z:3}), (:c), (:a)-[:r0 {k:9}]->(:b), (:c)-[:r1]->(:d)""",
                   """CREATE (:a {v:1}) CREATE (:b {v:2, z:3}) CREATE (:c) CREATE (:a)-[:r0 {k:9}]->(:b) CREATE (:c)-[:r1]->(:d)""",
                   """CREATE (:a {v:1}), (:b {v:2, z:3}) CREATE (:c), (:a)-[:r0 {k:9}]->(:b) CREATE (:c)-[:r1]->(:d)"""]
        for q in queries:
            actual_result = self.graph.query(q)
            self.env.assertEquals(actual_result.relationships_created, 2)
            self.env.assertEquals(actual_result.properties_set, 4)
            self.env.assertEquals(actual_result.nodes_created, 7)
