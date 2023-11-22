from common import *

graph1 = None
graph2 = None
graph3 = None


class testReturnDistinctFlow1(FlowTestsBase):

    def __init__(self):
        self.env, self.db = Env()
        global graph1
        graph1 = self.db.select_graph("G1")
        self.populate_graph()

    def populate_graph(self):
        global graph1
        graph1.query("CREATE (:PARENT {name: 'Stevie'})")
        graph1.query("CREATE (:PARENT {name: 'Mike'})")
        graph1.query("CREATE (:PARENT {name: 'James'})")
        graph1.query("CREATE (:PARENT {name: 'Rich'})")
        graph1.query("MATCH (p:PARENT {name: 'Stevie'}) CREATE (p)-[:HAS]->(c:CHILD {name: 'child1'})")
        graph1.query("MATCH (p:PARENT {name: 'Stevie'}) CREATE (p)-[:HAS]->(c:CHILD {name: 'child2'})")
        graph1.query("MATCH (p:PARENT {name: 'Stevie'}) CREATE (p)-[:HAS]->(c:CHILD {name: 'child3'})")
        graph1.query("MATCH (p:PARENT {name: 'Mike'}) CREATE (p)-[:HAS]->(c:CHILD {name: 'child4'})")
        graph1.query("MATCH (p:PARENT {name: 'James'}) CREATE (p)-[:HAS]->(c:CHILD {name: 'child5'})")
        graph1.query("MATCH (p:PARENT {name: 'James'}) CREATE (p)-[:HAS]->(c:CHILD {name: 'child6'})")

    def test_distinct_optimization(self):
        global graph1
        # Make sure we do not omit distinct when performain none aggregated projection.
        execution_plan = str(graph1.explain("MATCH (n) RETURN DISTINCT n.name, n.age"))
        self.env.assertIn("Distinct", execution_plan)

        # Distinct should be omitted when performain aggregation.
        execution_plan = str(graph1.explain("MATCH (n) RETURN DISTINCT n.name, max(n.age)"))
        self.env.assertNotIn("Distinct", execution_plan)

    def test_issue_395_scenario(self):
        global graph1
        # all
        result = graph1.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN p.name")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Stevie'], ['Stevie'], ['Mike'], ['James'], ['James']])

        # order
        result = graph1.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN p.name ORDER BY p.name")
        self.env.assertEqual(result.result_set, [['James'], ['James'], ['Mike'], ['Stevie'], ['Stevie'], ['Stevie']])

        # limit
        result = graph1.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN p.name LIMIT 2")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Stevie']])

        # order+limit
        result = graph1.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN p.name ORDER BY p.name LIMIT 2")
        self.env.assertEqual(result.result_set, [['James'], ['James']])

        # all+distinct
        result = graph1.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN DISTINCT p.name")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Mike'], ['James']])

        # order+distinct
        result = graph1.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN DISTINCT p.name ORDER BY p.name")
        self.env.assertEqual(result.result_set, [['James'], ['Mike'], ['Stevie']])

        # limit+distinct
        result = graph1.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN DISTINCT p.name LIMIT 2")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Mike']])

        # order+limit+distinct
        result = graph1.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN DISTINCT p.name ORDER BY p.name LIMIT 2")
        self.env.assertEqual(result.result_set, [['James'], ['Mike']])

    def test_distinct_with_order(self):
        # The results of DISTINCT should not be affected by the values in the ORDER BY clause
        result = graph1.query("MATCH (p:PARENT)-[:HAS]->(c:CHILD) RETURN DISTINCT p.name ORDER BY c.name")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Mike'], ['James']])

        result = graph1.query("UNWIND range(0,3) AS a UNWIND range(4,7) AS b WITH DISTINCT a ORDER BY b RETURN a ORDER BY a DESC")
        self.env.assertEqual(result.result_set, [[3], [2], [1], [0]])


class testReturnDistinctFlow2(FlowTestsBase):

    def __init__(self):
        self.env, self.db = Env()
        global graph2
        graph2 = self.db.select_graph("G2")
        self.populate_graph()

    def populate_graph(self):
        global graph2
        create_query = """
            CREATE
                (s:PARENT {name: 'Stevie'}),
                (m:PARENT {name: 'Mike'}),
                (j:PARENT {name: 'James'}),
                (r:PARENT {name: 'Rich'}),
                (s)-[:HAS]->(c1:CHILD {name: 'child1'}),
                (s)-[:HAS]->(c2:CHILD {name: 'child2'}),
                (s)-[:HAS]->(c3:CHILD {name: 'child3'}),
                (m)-[:HAS]->(c4:CHILD {name: 'child4'}),
                (j)-[:HAS]->(c5:CHILD {name: 'child5'}),
                (j)-[:HAS]->(c6:CHILD {name: 'child6'})"""
        graph2.query(create_query)

    def test_issue_395_scenario_2(self):
        global graph2
        # all
        result = graph2.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN p.name")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Stevie'], ['Stevie'], ['Mike'], ['James'], ['James']])

        # order
        result = graph2.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN p.name ORDER BY p.name")
        self.env.assertEqual(result.result_set, [['James'], ['James'], ['Mike'], ['Stevie'], ['Stevie'], ['Stevie']])

        # limit
        result = graph2.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN p.name LIMIT 2")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Stevie']])

        # order+limit
        result = graph2.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN p.name ORDER BY p.name DESC LIMIT 2")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Stevie']])

        # all+distinct
        result = graph2.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN DISTINCT p.name")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Mike'], ['James']])

        # order+distinct
        result = graph2.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN DISTINCT p.name ORDER BY p.name DESC")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Mike'], ['James']])

        # limit+distinct
        result = graph2.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN DISTINCT p.name LIMIT 2")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Mike']])

        # order+limit+distinct
        result = graph2.query("MATCH (p:PARENT)-[:HAS]->(:CHILD) RETURN DISTINCT p.name ORDER BY p.name DESC LIMIT 2")
        self.env.assertEqual(result.result_set, [['Stevie'], ['Mike']])

class testDistinct(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        global graph3
        graph3 = self.db.select_graph("G3")
        self.populate_graph()

    def populate_graph(self):
        global graph3
        a = Node()
        b = Node()
        c = Node()
        graph3.add_node(a)
        graph3.add_node(b)
        graph3.add_node(c)
        graph3.add_edge(Edge(a, "know", b))
        graph3.add_edge(Edge(a, "know", b))
        graph3.add_edge(Edge(a, "know", c))
        graph3.commit()

    def test_unwind_count_distinct(self):
        global graph3
        query = """UNWIND [1, 2, 2, "a", "a", null] as x RETURN count(distinct x)"""
        actual_result = graph3.query(query)
        expected_result = [[3]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test_match_count_distinct(self):
        global graph3
        query = """MATCH (a)-[]->(x) RETURN count(distinct x)"""
        actual_result = graph3.query(query)
        expected_result = [[2]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test_collect_distinct(self):
        global graph3
        query = "UNWIND ['a', 'a', null, 1, 2, 2, 3, 3, 3] AS x RETURN collect(distinct x)"
        actual_result = graph3.query(query)
        expected_result = [[['a', 1, 2, 3]]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test_distinct_path(self):
        global graph3
        # Create duplicate paths using a Cartesian Product, collapse into 1 column,
        # and unique the paths.
        query = """MATCH p1 = ()-[]->(), p2 = ()-[]->() UNWIND [p1, p2] AS a RETURN DISTINCT a"""
        actual_result = graph3.query(query)
        # Only three paths should be returned, one for each edge.
        self.env.assertEquals(len(actual_result.result_set), 3)

    def test_distinct_multiple_nulls(self):
        global graph3
        # DISTINCT should remove multiple null values.
        query = """UNWIND [null, null, null] AS x RETURN DISTINCT x"""
        actual_result = graph3.query(query)
        expected_result = [[None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test_distinct_union(self):
        # UNION performs implicit distinct, following query has 2 branches coming into a JOIN op
        # followed by an implicit distinct operation, once the left branch will be depleted
        # records coming in from the right branch will have different length mapping
        # then the previous records, yet distinct should only check for uniques of projected values
        # and ignore intermidate values such as 'n' and 'z'

        # no aggregations
        query = "MATCH (n) WITH n AS n RETURN 1 UNION MATCH (n), (z) WHERE ID(n) = ID(z) RETURN 1"
        actual_result = graph3.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # left hand side performs aggregations
        query = "MATCH (n) WITH n AS n RETURN max(1) AS one UNION MATCH (n), (z) WHERE ID(n) = ID(z) RETURN 1 AS one"
        actual_result = graph3.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # right hand side performs aggregations
        query = "MATCH (n) WITH n AS n RETURN 1 AS one UNION MATCH (n), (z) WHERE ID(n) = ID(z) RETURN max(1) AS one"
        actual_result = graph3.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # both ends perform aggregations
        query = "MATCH (n) WITH n AS n RETURN max(1) AS one UNION MATCH (n), (z) WHERE ID(n) = ID(z) RETURN min(1) AS one"
        actual_result = graph3.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # aggregation with explicit group key
        query = "MATCH (n) WITH n AS n RETURN 2 as key, max(1) AS one UNION MATCH (n), (z) WHERE ID(n) = ID(z) RETURN 2 as key, min(1) AS one"
        actual_result = graph3.query(query)
        expected_result = [[2, 1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

