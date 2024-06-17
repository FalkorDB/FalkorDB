from common import *

GRAPH_ID = "G"

class testGraphCreationFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_create_return(self):
        query = """CREATE (a:person {name:'A'}), (b:person {name:'B'})"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_created, 2)

        query = """MATCH (src:person) CREATE (src)-[e:knows]->(dest {name:'C'}) RETURN src,e,dest ORDER BY ID(src) DESC LIMIT 1"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.relationships_created, 2)
        self.env.assertEquals(len(result.result_set), 1)
        self.env.assertEquals(result.result_set[0][0].properties['name'], 'B')

    def test02_create_from_prop(self):
        query = """MATCH (p:person)-[e:knows]->() CREATE (c:clone {doublename: p.name + toLower(p.name), source_of: TYPE(e)}) RETURN c.doublename, c.source_of ORDER BY c.doublename"""
        result = self.graph.query(query)
        expected_result = [['Aa', 'knows'], ['Bb', 'knows']]

        self.env.assertEquals(result.labels_added, 1)
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.properties_set, 4)
        self.env.assertEquals(result.result_set, expected_result)

    def test03_create_from_projection(self):
        query = """UNWIND [10,20,30] AS x CREATE (p:person {age:x}) RETURN p.age ORDER BY p.age"""
        result = self.graph.query(query)
        expected_result = [[10], [20], [30]]
        self.env.assertEquals(result.nodes_created, 3)
        self.env.assertEquals(result.properties_set, 3)
        self.env.assertEquals(result.result_set, expected_result)

        query = """UNWIND ['Vancouver', 'Portland', 'Calgary'] AS city CREATE (p:person {birthplace: city}) RETURN p.birthplace ORDER BY p.birthplace"""
        result = self.graph.query(query)
        expected_result = [['Calgary'], ['Portland'], ['Vancouver']]
        self.env.assertEquals(result.nodes_created, 3)
        self.env.assertEquals(result.properties_set, 3)
        self.env.assertEquals(result.result_set, expected_result)

    def test04_create_with_null_properties(self):
        query = """CREATE (a:L {v1: NULL, v2: 'prop'}) RETURN a"""
        result = self.graph.query(query)
        node = Node(labels="L", properties={"v2": "prop"})
        expected_result = [[node]]

        self.env.assertEquals(result.labels_added, 1)
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.properties_set, 1)
        self.env.assertEquals(result.result_set, expected_result)

        # Create 2 new nodes, one with no properties and one with a property 'v'
        query = """CREATE (:M), (:M {v: 1})"""
        self.graph.query(query)

        # Verify that a MATCH...CREATE accesses the property correctly.
        query = """MATCH (m:M) WITH m ORDER BY m.v DESC CREATE ({v: m.v})"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.properties_set, 1)

    def test05_create_with_property_reference(self):
        # Queries that reference properties before they have been created should emit an error.
        try:
            query = """CREATE (a {val: 2}), (b {val: a.val})"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("'a' not defined", str(e))

    def test06_create_project_volatile_value(self):
        # The path e is volatile; verify that it can be projected after entity creation.
        query = """MATCH ()-[e*]->() CREATE (:L) WITH e RETURN 5"""
        result = self.graph.query(query)
        expected_result = [[5], [5]]

        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.result_set, expected_result)

        query = """UNWIND [1, 2] AS val WITH collect(val) AS arr CREATE (:L) RETURN arr"""
        result = self.graph.query(query)
        expected_result = [[[1, 2]]]

        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.result_set, expected_result)

    # Fail when a property is a complex type nested within an array type
    def test07_create_invalid_complex_type_in_array(self):
        # Test combinations of invalid types with nested and top-level arrays
        # Invalid types are NULL, maps, nodes, edges, and paths
        queries = ["CREATE (a), (b) SET a.v = [b]",
                   "CREATE (a {v: ['str', [1, NULL]]})",
                   "CREATE (a {v: [[{k: 'v'}]]})"]

        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("Property values can only be of primitive types or arrays of primitive types", str(e))

    # test creating a node with multiple attributes with the same name
    # expecting node with single attribute 'name' with the last mentioned value 'B'
    def test08_create_node_with_2_attr_same_name(self):
        query = """CREATE (a:N {name:'A', name:'B'})"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.properties_set, 1)

        query = """MATCH (a:N) RETURN a.name"""
        result = self.graph.query(query)
        expected_result = [['B']]
        self.env.assertEquals(result.result_set, expected_result)

    # test creating a node with some alias, and then creating an edge that touches that node
    # "variable redeclared" error should return only if the relation alias was already declared
    # also, a node cannot be redeclared unless it is used to create new edges in a pattern
    def test09_create_use_alias_in_many_clauses(self):
        query = """CREATE (n1:Node1) CREATE (n2:Node1) CREATE (n1)-[r:Rel1]->(n2)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.relationships_created, 1)

        # This is a valid query, even though 'n1' is redeclared in the CREATE clause
        query = """MATCH (n1:Node1) CREATE (n1)-[r:Rel1]->(n2)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.relationships_created, 2)

        queries = ["CREATE (n1)-[r:Rel1]->(n2) CREATE (n2)-[r:Rel1]->(n1)",
                   "CREATE (n1)-[r:Rel1]->(n2), (n2)-[r:Rel1]->(n1)",
                   "CREATE (n1)-[r:Rel1]->(n2) CREATE (n2)-[r2:Rel1]->(n3), (n3)-[r:Rel1]->(n2)"]

        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except redis.exceptions.ResponseError as e:
                self.env.assertContains("The bound edge 'r' can't be redeclared in a CREATE clause", str(e))

        query = "MATCH (r) CREATE (r)"
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("The bound variable 'r' can't be redeclared in a CREATE clause", str(e))

    # test creating queries with matching relationship type :R|R
    # the results can't report duplicates
    def test10_match_duplicated_reltype(self):
        query = """CREATE (a:A)-[r1:R1]->(b:B), (a:A)-[r2:R2]->(b:B)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.relationships_created, 2)
        self.env.assertEquals(result.labels_added, 2)

        # search for same relationship multiple times, should report one relationship
        queries = ["MATCH (a:A)-[r:R1]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R1|R1]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R1|R1|R1|R1]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R2]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R2|R2]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R2|R2|R3]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R2|R2|R3|R4|R4]->(b:B) RETURN count(r)",
                   ]
        for query in queries:
            result = self.graph.query(query)
            expected_result = [[1]]
            self.env.assertEquals(result.result_set, expected_result)

        # search for two relationship multiple times, should report two relationships
        queries = ["MATCH (a:A)-[r:R1|R2]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R1|R2|R1|R2]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R1|R1|R2|R2]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R1|R2|R2|R2]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R2|R1|R1|R1]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R2|R2|R1|R1]->(b:B) RETURN count(r)",
                   ]
        for query in queries:
            result = self.graph.query(query)
            expected_result = [[2]]
            self.env.assertEquals(result.result_set, expected_result)

        # These queries should not report results
        queries = ["MATCH (a:A)<-[r:R1|R2]-(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R3]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R3|R3]->(b:B) RETURN count(r)",
                   "MATCH (a:A)-[r:R3|R3|R4|R4]->(b:B) RETURN count(r)",
                   ]
        for query in queries:
            result = self.graph.query(query)
            expected_result = [[0]]
            self.env.assertEquals(result.result_set, expected_result)
