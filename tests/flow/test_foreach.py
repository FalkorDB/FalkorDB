from common import *
from collections import Counter

GRAPH_ID = "foreach"

class testForeachFlow():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def get_res_and_assertEquals(self, query, expected_result):
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)
        return actual_result

    def test01_literal_list(self):
        """check that FOREACH with a literal list works properly"""

        # graph is empty
        res = self.graph.query("FOREACH(i in range(0, 4) | CREATE (n:N {v: i}))")

        # 5 nodes should have been created, and 5 properties set
        self.env.assertEquals(res.nodes_created, 5)
        self.env.assertEquals(res.properties_set, 5)

        # validate that the correct nodes have been created
        self.get_res_and_assertEquals("MATCH (n) RETURN n.v ORDER BY n.v",
                                      [[0], [1], [2], [3], [4]])

        # validate MERGE (without creation) and SET
        res = self.graph.query(
            'FOREACH(i in range(0, 4) | MERGE (n:N {v: i}) SET n.v=i^2)'
            )

        # make sure no nodes were created, and 3 properties were set
        self.env.assertEquals(res.nodes_created, 0)
        self.env.assertEquals(res.properties_set, 3)

        # validate that the update is correct
        self.get_res_and_assertEquals("MATCH (n) RETURN n.v ORDER BY n.v",
                                      [[0], [1], [4], [9], [16]])
        
        # validate MERGE creation
        res = self.graph.query(
            'FOREACH(i in range(0, 4) | MERGE (m:M {v: i}))'
        )

        # 5 nodes should have been created, and 5 properties set
        self.env.assertEquals(res.nodes_created, 5)
        self.env.assertEquals(res.properties_set, 5)

        # delete the entities (the four nodes)
        res = self.graph.query("""
        FOREACH(i in range(0, 4) |
            MERGE (m:M {v: i}) DELETE m
        )""")

        # validate the deletion
        self.env.assertEquals(res.nodes_deleted, 5)

        # remove the properties and labels of the nodes using REMOVE
        res = self.graph.query("""
        MATCH (n:N)
        WITH collect(n) as ns
        FOREACH(n in ns |
            REMOVE n.v, n:N
        )""")

        # validate removal
        self.env.assertEquals(res.properties_removed, 5)
        self.env.assertEquals(res.labels_removed, 5)
        res = self.graph.query("MATCH (n) return labels(n), n.v")
        self.env.assertEquals(res.result_set, 
                    [[[], None], [[], None],[[], None], [[], None], [[], None]])

        # embedded foreach
        res = self.graph.query("""
        FOREACH(i in [0, 1, 2, 3, 4] |
            FOREACH(j in [1, 2] |
                CREATE (n:N)
                SET n.i=i, n.j=j
            )
        )"""
        )

        # validate the actions of the query
        self.env.assertEquals(res.nodes_created, 10)
        self.env.assertEquals(res.properties_set, 20)

    def test02_aliased_list(self):
        """tests that FOREACH with an aliased list works properly"""

        # clear db
        self.graph.delete()

        res = self.graph.query(
        "CYPHER li = [0, 1, 2, 3, 4] FOREACH(i in $li | CREATE (n:N {v: i}))"
        )

        # 5 nodes should have been created, and 5 properties set
        self.env.assertEquals(res.nodes_created, 5)
        self.env.assertEquals(res.properties_set, 5)

        # validate that the correct nodes have been created
        self.get_res_and_assertEquals("MATCH (n) RETURN n.v ORDER BY n.v",
                                      [[0], [1], [2], [3], [4]])

        # validate MERGE (without creation) and SET
        res = self.graph.query("""
        CYPHER li = [0, 1, 2, 3, 4]
        FOREACH(i in $li |
            MERGE (n:N {v: i}) SET n.v=i^2
        )"""
        )

        # make sure no nodes were created, and 3 properties were set
        self.env.assertEquals(res.nodes_created, 0)
        self.env.assertEquals(res.properties_set, 3)

        # validate that the update is correct
        self.get_res_and_assertEquals("MATCH (n) RETURN n.v ORDER BY n.v",
                                      [[0], [1], [4], [9], [16]])

        # validate MERGE creation
        res = self.graph.query("""
        CYPHER li = [0, 1, 2, 3, 4]
        FOREACH(i in $li |
            MERGE (m:M {v: i})
        )"""
        )

        # 5 nodes should have been created, and 5 properties set
        self.env.assertEquals(res.nodes_created, 5)
        self.env.assertEquals(res.properties_set, 5)

        # delete the entities (the four nodes)
        res = self.graph.query("""
        CYPHER li = [0, 1, 2, 3, 4]
        FOREACH(i in $li |
            MERGE (m:M {v: i}) DELETE m
        )"""
        )

        # validate the deletion
        self.env.assertEquals(res.nodes_deleted, 5)
        
        # remove the properties of the nodes using remove
        res = self.graph.query("""
        MATCH (n:N)
        WITH collect(n) as ns
        FOREACH(n in ns |
            REMOVE n.v, n:N
        )"""
        )

        # validate removal
        self.env.assertEquals(res.properties_removed, 5)
        self.env.assertEquals(res.labels_removed, 5)
        res = self.graph.query("MATCH (n) return labels(n), n.v")
        self.env.assertEquals(res.result_set, 
                    [[[], None], [[], None],[[], None], [[], None], [[], None]])

        # embedded foreach
        res = self.graph.query("""
        CYPHER li = [0, 1, 2, 3, 4]
        FOREACH(i in $li |
            FOREACH(j in [1, 2] |
                CREATE (n:N)
                SET n.i=i, n.j=j
            )
        )"""
        )

        # validate the actions of the query
        self.env.assertEquals(res.nodes_created, 10)
        self.env.assertEquals(res.properties_set, 20)

    def test03_case(self):
        """tests a CASE WHEN THEN ELSE"""

        self.graph.delete()

        # perform a conditional query using a CASE expression
        res = self.graph.query("""
        FOREACH(do_perform IN CASE WHEN true THEN [1] ELSE [] END |
            CREATE (:N)
        )"""
        )

        # make sure a node was created
        self.env.assertEquals(res.nodes_created, 1)

        # same case with a negative test-clause
        res = self.graph.query("""
        FOREACH(do_perform IN CASE WHEN false THEN [1] ELSE [] END |
            CREATE (:N)
        )"""
        )

        # make sure a node was not created
        self.env.assertEquals(res.nodes_created, 0)

    def test04_tie_with_foreach(self):
        """test the tieing of different segments with FOREACH"""

        self.graph.delete()

        # populate graph
        self.graph.query(
            "CYPHER li = [0, 1, 2, 3, 4] FOREACH(i in $li | CREATE (n:N {v: i}))"
        )

        # send a query that demands the tieing of segments containing foreach
        query = """
        MATCH (n:N) WITH collect(n) as ns
        FOREACH(n in ns |
            CREATE (:N {v: n.v})
        )
        WITH ns
        FOREACH(n in ns |
            CREATE (:N {v: n.v^10})
        )
        """

        res = self.graph.query(query)

        # 5 + 5 = 10 nodes created
        self.env.assertEquals(res.nodes_created, 10)

    def test05_multiple_records(self):
        """validate that multiple records are passed to Foreach appropriately.
        namely, the Foreach clause should run once for every record passed to it
        """

        self.graph.delete()

        # create 5 nodes
        self.graph.query("UNWIND range(0, 4) as val CREATE (n:N {v: val})")

        # execute a FOREACH clause for every node matched
        res = self.graph.query("""
        MATCH (n:N)
        FOREACH(i in [1] |
            CREATE (m:M {v: 2*n.v})
        )"""
        )
        self.env.assertEquals(res.nodes_created, 5)
        self.env.assertEquals(res.properties_set, 5)

        # validate that the creation is correct
        self.get_res_and_assertEquals("MATCH (m:M) RETURN m.v ORDER BY m.v",
                                      [[0], [2], [4], [6], [8]])

    def test06_field_access(self):
        """validate that Foreach accesses fields correctly"""

        self.graph.delete()

        # create two nodes with list properties
        self.graph.query("CREATE (:N {li: [1, 2, 3, 4]}), (:M {li: [1, 2, 3, 4]})")

        # run a Foreach clause for every node, accessing its list
        res = self.graph.query("""MATCH (n) FOREACH(i in n.li |
                           CREATE (t:TEMP {v: i}))""")

        # validate the creation
        self.get_res_and_assertEquals("""MATCH (t:TEMP) RETURN t.v, count(t.v)
                                         ORDER BY t.v""",
                                      [[1, 2], [2, 2], [3, 2], [4, 2]])

    def test07_midfail(self):
        """mid-evaluation failure (memory free'd appropriately)"""

        self.graph.delete()

        try:
            self.graph.query("FOREACH(i in [1, 2, 0, 3] | CREATE (n:N {v: 1/i}))")
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("Division by zero", str(e))

        # assure that no nodes are left (undo-log was applied correctly)
        res = self.graph.query("MATCH (n) return count(n)")
        self.env.assertEquals(res.result_set[0][0], 0)

        # collect may not be used inside the list expression of FOREACH
        try:
            self.graph.query("""
            MATCH (n:N)
            FOREACH(n in collect(n) |
                CREATE (m:M {v: n.v})
            )"""
            )
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("Invalid use of aggregating function 'collect'", str(e))

    def test08_complex(self):
        """complicate things up"""

        #-----------------------------------------------------------------------
        # addressing node properties in the record sent to Foreach
        #-----------------------------------------------------------------------

        # create a single node
        self.graph.query("CREATE (:N {v: 1, name: 'RAZ', li: [1, 2, 3, 4]})")

        query = """
        MATCH (n:N {v: 1, name: toUpper('raz')})
        FOREACH(x in n.li |
            CREATE (m:M {v: n.v, name: toLower(n.name), li: [x]})
        )
        """

        res = self.graph.query(query)

        # validate the query statistics
        self.env.assertEquals(res.nodes_created, 4)
        self.env.assertEquals(res.properties_set, 12)
        self.env.assertEquals(res.nodes_deleted, 0)

        # validate that the end state is correct
        # there should be one node in the graph, with label N, 
        # properties: {v: 1, name: 'RAZ', li: [1,2,3,4]
        # and 4 nodes with label M
        # and properties: {v: 1, name: 'raz'} and li from 1 to 4
        res = self.graph.query("MATCH (n:N) return n")
        n = Node(labels='N', properties={'v': 1, 'name': 'RAZ', 'li': [1, 2, 3, 4]})
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0], n)

        res = self.graph.query("MATCH (m:M) return m")
        self.env.assertEquals(len(res.result_set), 4)
        for i in range(len(res.result_set)):
            m = Node(labels= 'M', properties={'v': 1, 'name': 'raz', 'li': [i+1]})
            self.env.assertEquals(res.result_set[i][0], m)

        # clear nodes with label M
        res = self.graph.query("MATCH (m:M) DELETE m")
        self.env.assertEquals(res.nodes_deleted, 4)

        #-----------------------------------------------------------------------
        # triple embedding of a FOREACH clause, referencing node properties
        #-----------------------------------------------------------------------

        query = """
        MATCH (n:N)
        WITH collect(n) as ns
        FOREACH(n in ns |
            FOREACH(x in n.li |
                FOREACH(do_action IN CASE WHEN x > 2 THEN [1] ELSE [] END |
                    CREATE (t:TEMP {v: 2 * x + n.v, name: toLower(n.name), li: n.li + [x]})
                )
            )
        )
        """

        res = self.graph.query(query)

        # validate the query statistics
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(res.properties_set, 6)
        self.env.assertEquals(res.nodes_deleted, 0)

        # validate that the correct nodes were created
        n1 = Node(labels = 'TEMP',
                  properties={'v': 7, 'name': 'raz', 'li': [1, 2, 3, 4, 3]})
        n2 = Node(labels = 'TEMP',
                  properties={'v': 9, 'name': 'raz', 'li': [1, 2, 3, 4, 4]})
        res = self.get_res_and_assertEquals("""MATCH (t:TEMP) RETURN t ORDER BY
                                            t.v""",
                                            [[n1], [n2]])

        #-----------------------------------------------------------------------
        # triple embedded Foreach clause followed by reading and writing clauses
        #-----------------------------------------------------------------------

        self.graph.delete()

        # create a single node
        self.graph.query("CREATE (:N {v: 1, name: 'RAZ', li: [1, 2, 3, 4]})")

        query = """
        MATCH (n:N)
        WITH collect(n) as ns, toLower(n.name) as name
        FOREACH(n in ns |
            FOREACH(x in n.li |
                FOREACH(do_action IN CASE WHEN x > 2 THEN [1] ELSE [] END |
                    CREATE (t:TEMP {v: x, li: n.li + x, name: name})
                )
            )
        )
        WITH ns, name
        MATCH (t: TEMP)
        SET t.li = t.li + toUpper(name)
        WITH ns, t
        FOREACH(n in ns | CREATE (:TEMP))
        """

        res = self.graph.query(query)

        # validate the correctness of the new state
        self.env.assertEquals(res.nodes_created, 4)
        self.env.assertEquals(res.properties_set, 8)
        self.env.assertEquals(res.nodes_deleted, 0)

        # validate that the altered state is correct
        res = self.graph.query("MATCH (t:TEMP) RETURN t ORDER BY t.v ASC")
        t1 = Node(labels='TEMP', properties={'v': 3,
                                            'li': [1, 2, 3, 4, 3, 'RAZ'],
                                            'name': 'raz'})
        t2 = Node(labels='TEMP', properties={'v': 4,
                                            'li': [1, 2, 3, 4, 4, 'RAZ'],
                                            'name': 'raz'})
        t3 = Node(labels='TEMP')
        t4 = Node(labels='TEMP')
        nodes = [t1, t2, t3, t4]
        for i in range(4):
            self.env.assertEquals(res.result_set[i][0], nodes[i])

        # delete newly created nodes
        self.graph.query("MATCH (t:TEMP) DELETE t")

    def test09_clause_order(self):
        """a WITH clause must appear between FOREACH and a reading clause
        (MATCH, UNWIND, CALL)"""
        try:
            self.graph.query("FOREACH(i in [1] | CREATE (:M)) MATCH (m:M) RETURN m")
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("A WITH clause is required to introduce MATCH after an updating clause.", str(e))

        try:
            self.graph.query("""
            FOREACH(i in [1] |
                CREATE (:M)
            )
            UNWIND [1, 2, 3] as x
            RETURN x"""
            )
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("A WITH clause is required to introduce UNWIND after an updating clause.", str(e))

        try:
            self.graph.query("FOREACH(i in [1] | CREATE (:M)) CALL db.labels() YIELD\
                label RETURN label")
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("A WITH clause is required to introduce CALL after an updating clause.", str(e))

    def test10_edge_manipulation(self):
        """validate that edge (relationship) manipulation (creation, deletion update)
        operations work correctly when embedded in a FOREACH clause"""

        self.graph.delete()

        # ----------------------------------------------------------------------
        # create an edge between 2 nodes
        # ----------------------------------------------------------------------

        # create two nodes and an edge between them
        res = self.graph.query("""
        CREATE (n1:N {v: 1}), (n2:N {v: 2})
        FOREACH(i in [1] |
            CREATE (n1)-[:R]->(n2)
        )"""
        )
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(res.relationships_created, 1)

        # ----------------------------------------------------------------------
        # delete an edge
        # ----------------------------------------------------------------------
        
        # delete the edge using FOREACH
        res = self.graph.query("MATCH (n1)-[r]-(n2) FOREACH(i in [1] | DELETE r)")
        self.env.assertEquals(res.nodes_deleted, 0)
        self.env.assertEquals(res.relationships_deleted, 1)

        # ----------------------------------------------------------------------
        # delete a node that has an edge. Validate that the edge is deleted as
        # well when the node is deleted
        # ----------------------------------------------------------------------

        # create an edge between the two existing nodes
        res = self.graph.query("""
        MERGE (n1:N {v: 1})
        MERGE (n2:N {v: 2})
        FOREACH(i in [1] |
            CREATE (n1)-[:R]->(n2)
        )"""
        )
        self.env.assertEquals(res.nodes_created, 0)
        self.env.assertEquals(res.relationships_created, 1)

        # delete the nodes with FOREACH
        res = self.graph.query("MATCH (n) FOREACH(i in [1] | DELETE n)")
        self.env.assertEquals(res.nodes_deleted, 2)
        self.env.assertEquals(res.relationships_deleted, 1)

    def test11_path_reference(self):
        """reference a named path in the FOREACH clause"""

        self.graph.delete()

        # create some data
        res = self.graph.query(
            """
            CREATE (:Actor{id:'Marketing'}),
                   (:Actor{id:'vietld'}),
                   (:Actor{id:'phuongdd'})
            """
        )

        # make sure three nodes were created
        self.env.assertEquals(res.nodes_created, 3)

        # reference named paths defined in the outer scope, in the list exp
        res = self.graph.query(
            """
            MATCH path = (u:Actor), (g:Actor { id: 'Marketing' })
            WHERE u.id IN ['vietld', 'phuongdd']
            FOREACH (n IN nodes(path) |
                CREATE (n)-[:belong]->(g)
            )
            """
        )

        # make sure two relationships were created
        self.env.assertEquals(res.relationships_created, 2)

        # reference named paths defined in the outer scope, in the list exp
        res = self.graph.query(
            """
            MATCH path = (u:Actor), (g:Actor { id: 'Marketing' })
            WHERE u.id = 'vietld' OR u.id = 'phuongdd'
            FOREACH (n IN CASE WHEN u.id = 'vietld' THEN nodes(path) ELSE [] END |
                CREATE (n)-[:belong]->(g)
            )
            """
        )

        # make sure one relationship was created
        self.env.assertEquals(res.relationships_created, 1)

        # reference named paths defined in the outer scope, in the body
        res = self.graph.query(
            """
            MATCH path = (u: Actor)
            FOREACH (n IN CASE WHEN u.id = 'vietld' THEN nodes(path) ELSE [] END |
                MERGE (a:Actor {id: nodes(path)[0].id})
                SET a.c = 1
            )
            """
        )

        # make sure no nodes were created, and one property was set
        self.env.assertEquals(res.nodes_created, 0)
        self.env.assertEquals(res.properties_set, 1)

        # reference named paths defined in the body
        res = self.graph.query(
            """
            FOREACH (n in [1] |
                MERGE p = (:Actor {id: 'vietld'})
                FOREACH (node in nodes(p) |
                    MERGE (a:Actor {id: 'vietld'})
                    SET a.c = 2
                )
            )
            """
        )

        # make sure no nodes were created, and one property was set
        self.env.assertEquals(res.nodes_created, 0)
        self.env.assertEquals(res.properties_set, 1)

    def test12_passes_empty_record(self):
        """Tests that when FOREACH is the first clause in the query, it passes
        on an empty record so that the rest of the execution-plan is executed
        as well"""

        self.graph.delete()

        # create a node with label `N`
        res = self.graph.query("CREATE (:N)")
        self.env.assertEquals(res.nodes_created, 1)

        # execute a query in which FOREACH is the first clause, and it is
        # followed by a scan that finds this node
        res = self.graph.query(
            """
            FOREACH(i in [1] |
                MERGE (:N)
            )
            WITH 1 AS x
            MATCH (n:N)
            RETURN n
            """
        )

        # assure that no nodes were created, and one node was returned
        self.env.assertEquals(res.nodes_created, 0)
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N'))

    def test13_empty_list(self):
        """Tests that given an empty list, Foreach passes no records"""

        res = self.graph.query("FOREACH(n IN [] | CREATE (:N))")

        # no nodes should have been created
        self.env.assertEquals(res.nodes_created, 0)

    def test14_unbound_list_var(self):
        """Tests that given an unbound list-var, an error is raised"""

        try:
            self.graph.query("FOREACH(n in li | CREATE (:N))")
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("'li' not defined", str(e))

        # same check, when the list-var is the same as the list expression
        try:
            self.graph.query("FOREACH(n in n | CREATE (:N))")
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("'n' not defined", str(e))

    def test15_foreach_and_index_scan(self):
        query = """UNWIND range(1,9) AS i 
                   CREATE (n:N {v:i})-[:R]->(m:M {v:(i+1)})"""
        self.graph.query(query)

        query = """CREATE INDEX FOR (n:N) ON (n.v)"""
        self.graph.query(query)

        query = """MATCH p=(n:N)-[:R]->(:M) WHERE n.v <= 5 
                   FOREACH( n in nodes(p) | SET n.x = 3) RETURN count(n.v)"""
        result = self.graph.query(query)
        expected_result = [[5]]
        self.env.assertEquals(result.result_set, expected_result)

    def test16_accumulate_updates(self):
        """Tests that updates in between cycles of body execution are treated
        correctly"""

        self.graph.delete()

        # create a node with property `v` initialized with value 1
        res = self.graph.query("CREATE (:N {v: 1})")
        self.env.assertEquals(res.nodes_created, 1)

        query = """
        CYPHER li=[0, 1, 2, 3]
        MATCH (n)
        FOREACH(x in $li |
            SET n.v = n.v + x
        )
        RETURN n.v
        """

        res = self.graph.query(query)
        # check that the `v` property of the node is now 1 + 0 + 1 + 2 + 3 = 7
        self.env.assertEquals(res.result_set[0][0], 7)

    def test17_bound_variables(self):
        """Tests that foreach bound variables correctly"""

        self.graph.delete()

        res = self.graph.query("""CREATE (:M), (:N {name: "a"}), (:N {name: "b"}), (:N {name: "c"})""")
        self.env.assertEquals(res.nodes_created, 4)

        query = """
        MATCH (n:N)
        WITH COLLECT(n) as ns
        MATCH (m:M)
        WITH m, ns
        FOREACH (n IN ns |
            MERGE (m)-[:R]->(n)
        )
        """

        res = self.graph.query(query)
        self.env.assertEquals(res.nodes_created, 0)
        self.env.assertEquals(res.relationships_created, 3)

    def test18_foreach_within_conditional_traverse(self):
        """Tests that a FOREACH clause can be used within a conditional traversal"""

        self.graph.delete()

        res = self.graph.query("""CREATE ()""")
        self.env.assertEquals(res.nodes_created, 1)

        query = """
        FOREACH (n IN [] |
            MERGE ()
        )
        WITH *
        MATCH ()--()
        RETURN 0
        """

        try:
            self.graph.query(query)
            self.env.assertTrue(True)
        except:
            self.env.assertTrue(False)
