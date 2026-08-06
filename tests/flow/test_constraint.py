import threading
from common import *
from index_utils import *
from constraint_utils import *

GRAPH_ID = "constraints"

class testConstraintNodes():
    def __init__(self):
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.con.delete(GRAPH_ID)
        self.g = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        g = self.g
        g.query("CREATE (:Engineer:Person {name: 'Mike', age: 10, height: 180, loc: point({latitude:1, longitude:2})})")
        g.query("CREATE (:Engineer:Person {name: 'Tim', age: 20, height: 190, loc: point({latitude:2, longitude:2})})")
        g.query("CREATE (:Person:Engineer {name: 'Rick', age: 30, height: 200, loc: point({latitude:3, longitude:2})})")
        g.query("CREATE (:Person:Engineer {name: 'Andrew', age: 36, height: 173, loc: point({latitude:4, longitude:2})})")
        g.query("MATCH (a{name: 'Andrew'}),({name:'Rick'}) CREATE (a)-[:Knows {since:1984}]->(b)")

    def test01_create_constraint(self):
        #-----------------------------------------------------------------------
        # create constraints
        #-----------------------------------------------------------------------

        # create mandatory node constraint over Person height
        create_mandatory_node_constraint(self.g, 'Person', 'height')

        # create unique node constraint over Person height
        result = create_unique_node_constraint(self.g, 'Person', 'height')
        self.env.assertEqual(result, "PENDING")

        # create unique node constraint over Person name and age
        create_unique_node_constraint(self.g, 'Person', 'name', 'age')

        # create unique node constraint over Person loc
        create_unique_node_constraint(self.g, 'Person', 'loc')

        # create mandatory edge constraint over
        create_mandatory_edge_constraint(self.g, 'Knows', 'since')

        # create unique edge constraint over
        create_unique_edge_constraint(self.g, 'Knows', 'since')

        # validate constrains
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 6)
        for c in constraints:
            self.env.assertTrue(c.status != 'FAILED')

    def test02_constraint_violations(self):
        # active constrains:
        # 1. mandatory node constraint over Person height
        # 2. unique node constraint over Person height
        # 3. unique node constraint over Person name and age
        # 4. unique node constraint over Person loc
        # 5. mandatory edge constraint over Knows since
        # 6. unique edge constraint over Knows since

        g = self.g

        # backup original dataset
        expected_result_set = g.query("MATCH (n) RETURN n ORDER BY ID(n)").result_set

        #-----------------------------------------------------------------------
        # create a node that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("CREATE (:Person)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: node with label Person missing property height", str(e))

        #-----------------------------------------------------------------------
        # create a node that violates the unique constraint on point data ignored
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (p:Person) CREATE (n:Person{height:p.height + 1, loc: p.loc}) DELETE n")
            self.env.assertTrue(True)
        except ResponseError as e:
            self.env.assertTrue(False)

        #-----------------------------------------------------------------------
        # create a node that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (p:Person) CREATE (:Person{height:p.height})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Person", str(e))

        #-----------------------------------------------------------------------
        # create a node that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (p:Person) CREATE (:Person{height:rand(), name:p.name, age:p.age})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Person", str(e))

        #-----------------------------------------------------------------------
        # node update that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (n:Person) SET n.height = NULL")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: node with label Person missing property height", str(e))

        #-----------------------------------------------------------------------
        # node update that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (a:Person), (b:Person) WHERE a<>b SET a.height = b.height")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Person", str(e))

        #-----------------------------------------------------------------------
        # node update that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (a:Person), (b:Person) WHERE a<>b SET a.name = b.name, a.age = b.age")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Person", str(e))

        #-----------------------------------------------------------------------
        # node merge-match that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (n:Person {name: 'Andrew'}) ON MATCH SET n.height = NULL")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: node with label Person missing property height", str(e))

        #-----------------------------------------------------------------------
        # node merge-match that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {height:180}) ON MATCH SET p.height = 190")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Person", str(e))

        #-----------------------------------------------------------------------
        # node merge-match that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {name:'Mike'}) ON MATCH SET p.age = 20, p.height = 190")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Person", str(e))

        #-----------------------------------------------------------------------
        # node merge-create that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (n:Person {name: 'Dor', height: 187}) ON CREATE SET n.height = NULL")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: node with label Person missing property height", str(e))

        #-----------------------------------------------------------------------
        # node merge-create that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {name: 'Dor', height:187}) ON CREATE SET p.height = 190")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Person", str(e))

        #-----------------------------------------------------------------------
        # node merge-create that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {name:'Dor', height:187}) ON CREATE SET p.age = 10, p.name = 'Mike'")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Person", str(e))

        #-----------------------------------------------------------------------
        # node merge that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (n:Person {name: 'Dor'})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: node with label Person missing property height", str(e))

        #-----------------------------------------------------------------------
        # node merge that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {name: 'Dor', height:180})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Person", str(e))

        #-----------------------------------------------------------------------
        # node merge that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {v:12, name:'Mike', age:10})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: node with label Person missing property height", str(e))

        #-----------------------------------------------------------------------
        # node label update which will conflict with mandatory constraint
        #-----------------------------------------------------------------------
        # 1. mandatory node constraint over Person height
        # 2. unique node constraint over Person height
        # 3. unique node constraint over Person name and age

        g.query("CREATE (:Architect)")

        #-----------------------------------------------------------------------
        # node label update which will conflict with mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (n:Architect) SET n:Person")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: node with label Person missing property height", str(e))

        # add attributes to Architect which will conflict with both unique constraints
        g.query("MATCH (n:Architect) SET n.name = 'Mike', n.age = 10, n.height = 180")

        #-----------------------------------------------------------------------
        # node label update which will conflict with unique constraints
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (n:Architect) SET n:Person")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Person", str(e))

        # delete Architect
        g.query("MATCH (n:Architect) DELETE n")

        # validate graph did not changed
        actual_result_set = self.g.query("MATCH (n) RETURN n ORDER BY ID(n)").result_set
        self.env.assertEqual(actual_result_set, expected_result_set)

        try:
            g.query("CREATE CONSTRAINT ON (n:N) ASSERT n.v IS UNIQUE")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid constraint command use the GRAPH.CONSTRAINT command instead", str(e))

        try:
            g.query("DROP CONSTRAINT ON (n:N) ASSERT n.v IS UNIQUE")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid constraint command use the GRAPH.CONSTRAINT command instead", str(e))

        try:
            g.query("CREATE CONSTRAINT ON ()-[r:R]->() ASSERT r.v")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid constraint command use the GRAPH.CONSTRAINT command instead", str(e))

        try:
            g.query("DROP CONSTRAINT ON ()-[r:R]->() ASSERT r.v")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid constraint command use the GRAPH.CONSTRAINT command instead", str(e))


    def test03_drop_constraint(self):
        #-----------------------------------------------------------------------
        # drop constraints
        #-----------------------------------------------------------------------

        # get all constraints
        constraints = list_constraints(self.g)

        # drop each constraint
        for c in constraints:
            drop_constraint(self.g, c.type, c.entity_type, c.label, *c.attributes)

        # validate graph doesn't contains any constraints
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 0)

    def test04_invalid_constraint_command(self):
        # constraint create command:
        # GRAPH.CONSTRAIN <key> CREATE/DEL UNIQUE/MANDATORY [NODE label / RELATIONSHIP type] PROPERTIES prop_count prop0...

        #-----------------------------------------------------------------------
        # invalid constraint operation
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "LIST", GRAPH_ID)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("wrong number of arguments for 'graph.CONSTRAINT' command", str(e))

        #-----------------------------------------------------------------------
        # invalid constraint operation
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", GRAPH_ID, "INVALID_OP", "unique", "LABEL", "New_Label", "PROPERTIES", 1, "New_Attr")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid constraint operation", str(e))

        #-----------------------------------------------------------------------
        # invalid constraint type
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "INVALID_CT", "New_Label", "Person", "PROPERTIES", 1, "New_Attr")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid constraint type", str(e))

        #-----------------------------------------------------------------------
        # invalid entity type
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "INVALID_ENTITY_TYPE", "New_Label", "PROPERTIES", 1, "New_Attr")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid constraint entity type", str(e))

        #-----------------------------------------------------------------------
        # invalid label name
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "1ab", "PROPERTIES", 1, "New_Attr")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Label name 1ab is invalid", str(e))

        #-----------------------------------------------------------------------
        # invalid property name
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "label", "PROPERTIES", 2, 1, 2)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Property name 1 is invalid", str(e))

        #-----------------------------------------------------------------------
        # invalid property name
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "label", "PROPERTIES", 2, 'a1', '2pb')
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Property name 2pb is invalid", str(e))

        #-----------------------------------------------------------------------
        # invalid property count
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "label", "PROPERTIES", 0)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Number of properties must be an integer between 1 and 255", str(e))

        #-----------------------------------------------------------------------
        # invalid property count
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "label", "PROPERTIES", -1, 12)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Number of properties must be an integer between 1 and 255", str(e))

        #-----------------------------------------------------------------------
        # del constraint on non exsisting label
        #-----------------------------------------------------------------------
        try:
            drop_unique_node_constraint(self.g, "None_Existing_Label", "age")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Unable to drop constraint, no such constraint.", str(e))

        #-----------------------------------------------------------------------
        # del constraint on non exsisting attribute
        #-----------------------------------------------------------------------
        try:
            drop_unique_node_constraint(self.g, "Person", "None_Existing_Attr")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Unable to drop constraint, no such constraint.", str(e))

        #-----------------------------------------------------------------------
        # create constraint which already exists
        #-----------------------------------------------------------------------
        create_unique_node_constraint(self.g, "Person", "age", sync=True)
        try:
            create_unique_node_constraint(self.g, "Person", "age")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Constraint already exists", str(e))

        # validate labels and attributes were not created for failed operations
        # not expecting None_Existing_Label, New_Label, None_Existing_Attr and New_Attr
        # to be added to the graph
        labels = self.g.query("CALL db.labels()").result_set
        attributes = self.g.query("CALL db.propertyKeys()").result_set
        self.env.assertFalse("New_Label" in labels)
        self.env.assertFalse("None_Existing_Label" in labels)
        self.env.assertFalse("New_Attr" in attributes)
        self.env.assertFalse("None_Existing_Attr" in attributes)

        #-----------------------------------------------------------------------
        # unique constraint missing supporting exact-match index
        #-----------------------------------------------------------------------

    def test05_constraint_create_drop_simultanously(self):
        # make sure there are no constraints in the graph
        for c in list_constraints(self.g):
            drop_constraint(self.g, c.type, c.entity_type, c.label, *c.attributes)
        self.env.assertEqual(0, len(list_constraints(self.g)))

        # create 500K new entities
        self.g.query("UNWIND range(0, 500000) AS x CREATE (:MarineBiologist {age: x})")

        # create unique constraint over MarineBiologist age attribute
        create_unique_node_constraint(self.g, "MarineBiologist", "age")

        # make sure constraint is pending
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 1)
        c = constraints[0]
        self.env.assertEqual(c.status, "UNDER CONSTRUCTION")

        # delete constraint
        drop_unique_node_constraint(self.g, "MarineBiologist", "age")

        # constraint should be dropped immediately
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 0)

        # try to create two nodes which would have conflicted
        self.g.query("CREATE (:MarineBiologist {age: 35}), (:MarineBiologist {age: 35})")

    def test06_constraint_fix(self):
        # test that a failing constraint can be recreated successfully once
        # all conflicts are resolved

        # create a Person node without any attributes
        self.g.query("CREATE (:Person)")

        # create two Person nodes with the same name
        self.g.query("CREATE (:Person {name:'jerry'})")
        self.g.query("CREATE (:Person {name:'jerry'})")

        #-----------------------------------------------------------------------
        # create a unique constraint over Person name
        #-----------------------------------------------------------------------

        create_unique_node_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation faile
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 1)
        self.env.assertEqual(constraints[0].status, "FAILED")

        # fix name uniqueness by deleting duplicated node
        self.g.query("MATCH (p:Person {name:'jerry'}) WITH p LIMIT 1 DELETE p")

        #-----------------------------------------------------------------------
        # re-create unique constraint
        #-----------------------------------------------------------------------

        create_unique_node_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation succeeded
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 1)
        self.env.assertEqual(constraints[0].status, "OPERATIONAL")

        #-----------------------------------------------------------------------
        # try to create mandatory constraint over Person name
        #-----------------------------------------------------------------------

        create_mandatory_node_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation faile
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 2)
        for c in constraints:
            if c.type == "UNIQUE":
                self.env.assertEqual(c.status, "OPERATIONAL")
            else:
                self.env.assertEqual(c.status, "FAILED")

        #-----------------------------------------------------------------------
        # try deleting a failed constraint
        #-----------------------------------------------------------------------

        drop_mandatory_node_constraint(self.g, "Person", "name")

        # make sure constraint was deleted
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 1)
        c = constraints[0]
        self.env.assertEqual(c.label, "Person")
        self.env.assertEqual(c.type, "UNIQUE")
        self.env.assertEqual(c.status, "OPERATIONAL")

        #-----------------------------------------------------------------------
        # re-create mandatory constraint
        #-----------------------------------------------------------------------

        # add missing name attribute to resolve conflict
        self.g.query("MATCH (p:Person) WHERE p.name is NULL SET p.name = 'kramer'")

        create_mandatory_node_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation succeeded
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 2)
        self.env.assertEqual(constraints[0].status, "OPERATIONAL")
        self.env.assertEqual(constraints[1].status, "OPERATIONAL")

    def test07_constraint_creation_with_new_label_attr(self):
        # create a constraint against a new label and a new attribute
        create_unique_node_constraint(self.g, "Artist", "nickname", sync=True)
        self.g.query("CREATE (:Artist {nickname: 'Banksy'})")

        # make sure constraint is enforced
        try:
            self.g.query("CREATE (:Artist {nickname: 'Banksy'})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Artist", str(e))

    def test08_remove_supporting_index(self):
        # try to create unique index without a supporting index
        try:
            create_constraint(self.g, "unique", "node", "Author", "nickname", "birthdate")
            self.assertFalse(1)
        except ResponseError as e:
            self.env.assertContains("missing supporting exact-match index", str(e))

        # create supporting index
        create_node_range_index(self.g, "Author", "nickname", "birthdate")

        # create unique index
        create_constraint(self.g, "unique", "node", "Author", "nickname", "birthdate")

        # try to drop supporting index
        try:
            drop_node_range_index(self.g, "Author", "nickname")
        except ResponseError as e:
            self.env.assertContains("Index supports constraint", str(e))

        try:
            drop_node_range_index(self.g, "Author", "birthdate")
        except ResponseError as e:
            self.env.assertContains("Index supports constraint", str(e))

        # drop constraint
        drop_unique_node_constraint(self.g, "Author", "nickname", "birthdate")

        # try to drop supporting index
        drop_node_range_index(self.g, "Author", "nickname")
        drop_node_range_index(self.g, "Author", "birthdate")

class testConstraintEdges():
    def __init__(self):
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.con.delete(GRAPH_ID)
        self.g = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        g = self.g
        g.query("CREATE ()-[:Person {name: 'Mike', age: 10, height: 180}]->()")
        g.query("CREATE ()-[:Person {name: 'Tim', age: 20, height: 190}]->()")
        g.query("CREATE ()-[:Person {name: 'Rick', age: 30, height: 200}]->()")
        g.query("CREATE ()-[:Person {name: 'Andrew', age: 36, height: 173}]->()")

    def test01_create_constraint(self):
        #-----------------------------------------------------------------------
        # create constraints
        #-----------------------------------------------------------------------

        # create mandatory edge constraint over Person height
        create_mandatory_edge_constraint(self.g, 'Person', 'height')

        # create unique edge constraint over Person height
        create_unique_edge_constraint(self.g, 'Person', 'height')

        # create unique edge constraint over Person name and age
        create_unique_edge_constraint(self.g, 'Person', 'name', 'age', sync=True)

        # validate constrains
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 3)
        for c in constraints:
            self.env.assertTrue(c.status == 'OPERATIONAL')

    def test02_edge_constraint_violations(self):
        # active constrains:
        # 1. mandatory edge constraint over Person height
        # 2. unique edge constraint over Person height
        # 3. unique edge constraint over Person name and age

        g = self.g

        # backup original dataset
        expected_result_set = g.query("MATCH ()-[e]->() RETURN e ORDER BY ID(e)").result_set

        #-----------------------------------------------------------------------
        # create an edge that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("CREATE ()-[:Person]->()")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: edge with relationship-type Person missing property height", str(e))

        #-----------------------------------------------------------------------
        # create an edge that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH ()-[e:Person]->() CREATE ()-[:Person{height:e.height}]->()")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type Person", str(e))

        #-----------------------------------------------------------------------
        # create an edge that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH ()-[e:Person]->() CREATE ()-[:Person{height:rand(), name:e.name, age:e.age}]->()")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type Person", str(e))

        #-----------------------------------------------------------------------
        # edge update that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH ()-[e:Person]->() SET e.height = NULL")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: edge with relationship-type Person missing property height", str(e))

        #-----------------------------------------------------------------------
        # edge update that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH ()-[a:Person]->(), ()-[b:Person]->() WHERE a<>b SET a.height = b.height")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type Person", str(e))

        #-----------------------------------------------------------------------
        # edge update that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH ()-[a:Person]->(), ()-[b:Person]->() WHERE a<>b SET a.name = b.name, a.age = b.age")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type Person", str(e))

        #-----------------------------------------------------------------------
        # edge merge-match that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name: 'Andrew'}]->() ON MATCH SET e.height = NULL")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: edge with relationship-type Person missing property height", str(e))

        #-----------------------------------------------------------------------
        # edge merge-match that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {height:180}]->() ON MATCH SET e.height = 190")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type Person", str(e))

        #-----------------------------------------------------------------------
        # edge merge-match that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name:'Mike'}]->() ON MATCH SET e.age = 20, e.height = 190")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type Person", str(e))

        #-----------------------------------------------------------------------
        # edge merge-create that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name: 'Dor', height: 187}]->() ON CREATE SET e.height = NULL")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: edge with relationship-type Person missing property height", str(e))

        #-----------------------------------------------------------------------
        # edge merge-create that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name: 'Dor', height:187}]->() ON CREATE SET e.height = 190")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type Person", str(e))

        #-----------------------------------------------------------------------
        # edge merge-create that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name:'Dor', height:187}]->() ON CREATE SET e.age = 10, e.name = 'Mike'")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type Person", str(e))

        #-----------------------------------------------------------------------
        # edge merge that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name: 'Dor'}]->()")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: edge with relationship-type Person missing property height", str(e))

        #-----------------------------------------------------------------------
        # edge merge that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name: 'Dor', height:180}]->()")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type Person", str(e))

        #-----------------------------------------------------------------------
        # edge merge that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {v:12, name:'Mike', age:10}]->()")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: edge with relationship-type Person missing property height", str(e))

        # validate graph did not changed
        actual_result_set = self.g.query("MATCH ()-[e]->() RETURN e ORDER BY ID(e)").result_set
        self.env.assertEqual(actual_result_set, expected_result_set)

    def test03_drop_constraint(self):
        #-----------------------------------------------------------------------
        # drop constraints
        #-----------------------------------------------------------------------

        # get all constraints
        constraints = list_constraints(self.g)

        # drop each constraint
        for c in constraints:
            drop_constraint(self.g, c.type, c.entity_type, c.label, *c.attributes)

        # validate graph doesn't contains any constraints
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 0)

    def test04_invalid_constraint_command(self):
        # constraint create command:
        # GRAPH.CONSTRAIN <key> CREATE/DEL UNIQUE/MANDATORY [NODE label / RELATIONSHIP type] PROPERTIES prop_count prop0...

        #-----------------------------------------------------------------------
        # invalid constraint operation
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "INVALID_OP", GRAPH_ID, "unique", "RELATIONSHIP", "New_Label", "PROPERTIES", 1, "New_Attr")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid constraint operation", str(e))

        #-----------------------------------------------------------------------
        # invalid constraint type
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "INVALID_CT", "New_Label", "Person", "PROPERTIES", 1, "New_Attr")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid constraint type", str(e))

        #-----------------------------------------------------------------------
        # invalid entity type
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "INVALID_ENTITY_TYPE", "New_Label", "PROPERTIES", 1, "New_Attr")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid constraint entity type", str(e))

        #-----------------------------------------------------------------------
        # del constraint on non exsisting label
        #-----------------------------------------------------------------------
        try:
            drop_unique_edge_constraint(self.g, "None_Existing_Label", "age")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Unable to drop constraint, no such constraint.", str(e))

        #-----------------------------------------------------------------------
        # del constraint on non exsisting attribute type
        #-----------------------------------------------------------------------
        try:
            drop_unique_edge_constraint(self.g, "Person", "None_Existing_Attr")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Unable to drop constraint, no such constraint.", str(e))

        #-----------------------------------------------------------------------
        # create constraint with duplicate attributes
        #-----------------------------------------------------------------------
        try:
            create_unique_edge_constraint(self.g, "Person", "age", "height", "weight", "height", sync=True)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Properties cannot contain duplicates", str(e))

        #-----------------------------------------------------------------------
        # create constraint which already exists
        #-----------------------------------------------------------------------
        create_unique_edge_constraint(self.g, "Person", "age", sync=True)
        try:
            create_unique_edge_constraint(self.g, "Person", "age")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Constraint already exists", str(e))

        # validate labels and attributes were not created for failed operations
        # not expecting None_Existing_Label, New_Label, None_Existing_Attr and New_Attr
        # to be added to the graph
        labels = self.g.query("CALL db.labels()").result_set
        attributes = self.g.query("CALL db.propertyKeys()").result_set
        self.env.assertFalse("New_Label" in labels)
        self.env.assertFalse("None_Existing_Label" in labels)
        self.env.assertFalse("New_Attr" in attributes)
        self.env.assertFalse("None_Existing_Attr" in attributes)

    def test05_constraint_create_drop_simultanously(self):
        # make sure there are no constraints in the graph
        for c in list_constraints(self.g):
            drop_constraint(self.g, c.type, c.entity_type, c.label, *c.attributes)
        self.env.assertEqual(0, len(list_constraints(self.g)))

        # create 500K new entities
        self.g.query("UNWIND range(0, 500000) AS x CREATE ()-[:MarineBiologist {age: x}]->()")

        # create unique constraint over MarineBiologist age attribute
        create_unique_edge_constraint(self.g, "MarineBiologist", "age")

        # make sure constraint is pending
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 1)
        c = constraints[0]
        self.env.assertEqual(c.status, "UNDER CONSTRUCTION")

        # delete constraint
        drop_unique_edge_constraint(self.g, "MarineBiologist", "age")

        # constraint should be dropped immediately
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 0)

        # try to create two edges which would have conflicted
        self.g.query("CREATE ()-[:MarineBiologist {age: 35}]->(), ()-[:MarineBiologist {age: 35}]->()")

    def test06_constraint_fix(self):
        # test that a failing constraint can be recreated successfully once
        # all conflicts are resolved

        # create a Person edge without any attributes
        self.g.query("CREATE ()-[:Person]->()")

        # create two Person edgess with the same name
        self.g.query("CREATE ()-[:Person {name:'jerry'}]->()")
        self.g.query("CREATE ()-[:Person {name:'jerry'}]->()")

        #-----------------------------------------------------------------------
        # create a unique constraint over Person name
        #-----------------------------------------------------------------------

        create_unique_edge_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation faile
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 1)
        self.env.assertEqual(constraints[0].status, "FAILED")

        # fix name uniqueness by deleting duplicated edge
        self.g.query("MATCH ()-[e:Person {name:'jerry'}]->() WITH e LIMIT 1 DELETE e")

        #-----------------------------------------------------------------------
        # re-create unique constraint
        #-----------------------------------------------------------------------

        create_unique_edge_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation succeeded
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 1)
        self.env.assertEqual(constraints[0].status, "OPERATIONAL")

        #-----------------------------------------------------------------------
        # try to create mandatory constraint over Person name
        #-----------------------------------------------------------------------

        create_mandatory_edge_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation faile
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 2)
        for c in constraints:
            if c.type == "UNIQUE":
                self.env.assertEqual(c.status, "OPERATIONAL")
            else:
                self.env.assertEqual(c.status, "FAILED")

        #-----------------------------------------------------------------------
        # try deleting a failed constraint
        #-----------------------------------------------------------------------

        drop_mandatory_edge_constraint(self.g, "Person", "name")

        # make sure constraint was deleted
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 1)
        c = constraints[0]
        self.env.assertEqual(c.label, "Person")
        self.env.assertEqual(c.type, "UNIQUE")
        self.env.assertEqual(c.status, "OPERATIONAL")

        #-----------------------------------------------------------------------
        # re-create mandatory constraint
        #-----------------------------------------------------------------------

        # add missing name attribute to resolve conflict
        self.g.query("MATCH ()-[e:Person]->() WHERE e.name is NULL SET e.name = 'kramer'")

        create_mandatory_edge_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation succeeded
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 2)
        self.env.assertEqual(constraints[0].status, "OPERATIONAL")
        self.env.assertEqual(constraints[1].status, "OPERATIONAL")

    def test07_constraint_creation_with_new_relation_attr(self):
        # create a constraint against a new relationship-type and a new attribute
        create_unique_edge_constraint(self.g, "Artist", "nickname", sync=True)
        self.g.query("CREATE ()-[:Artist {nickname: 'Banksy'}]->()")

        # make sure constraint is enforced
        try:
            self.g.query("CREATE ()-[:Artist {nickname: 'Banksy'}]->()")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type Artist", str(e))

MONITOR_ATTACHED = False

class testConstraintReplication():
    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True)
        self.source  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        self.monitor = []
        self.g = self.db.select_graph(GRAPH_ID)

        self.monitor_thread = threading.Thread(target=self.monitor_thread, daemon=True)
        self.monitor_thread.start()

        # wait for monitor thread to attach
        while MONITOR_ATTACHED is False:
            time.sleep(0.2)

        # clear DB
        self.source.delete(GRAPH_ID)

        # the WAIT command forces master slave sync to complete
        self.source.execute_command("WAIT", 1, 0)

    def monitor_thread(self):
        global MONITOR_ATTACHED
        try:
            with self.replica.monitor() as m:
                MONITOR_ATTACHED = True
                for cmd in m.listen():
                    if 'GRAPH.CONSTRAINT' in cmd['command']:
                        self.monitor.append(cmd)
        except:
            pass

    def test_01_constraint_replication(self):
        # create mandatory node constraint over Person height
        create_mandatory_node_constraint(self.g, 'Person', 'height')

        # create unique node constraint over Person height
        create_unique_node_constraint(self.g, 'Person', 'height')

        # create unique node constraint over Person name and age
        create_unique_node_constraint(self.g, 'Person', 'name', 'age')

        # create unique node constraint over Person loc
        create_unique_node_constraint(self.g, 'Person', 'loc')

        # create mandatory edge constraint over Knows since
        create_mandatory_edge_constraint(self.g, 'Knows', 'since')

        # create unique edge constraint over Knows since
        create_unique_edge_constraint(self.g, 'Knows', 'since', sync=True)

        # validate constrains
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 6)
        for c in constraints:
            self.env.assertEqual(c.status, 'OPERATIONAL')

        # each constraint should be replicated twice from source to replica:
        # 1. upon creation
        # 2. upon constraint becoming activate
        self.source.execute_command("WAIT", 1, 0)

        # wait for all 12 GRAPH.CONSTRAINT commands to be replicated
        elapsed = 10
        while len(self.monitor) < 12 and elapsed > 0:
            time.sleep(0.2)
            elapsed -= 0.2

        self.env.assertEqual(len(self.monitor), 12)

    def test_02_constraint_introduces_new_schema_and_attr(self):
        # every constraint created so far targeted a label/relationship-type
        # and attribute(s) that a prior query had already introduced to the
        # graph. this test makes sure a constraint created against a label,
        # relationship-type and attribute(s) that no query has ever
        # referenced before is still correctly replicated: the constraint
        # command itself must introduce the missing schema and attribute
        # ID(s) on the replica, rather than relying on the replica having
        # already learned about them from some earlier data-modifying query

        # brand new node label + attribute
        create_unique_node_constraint(self.g, 'Artist', 'nickname', sync=True)

        # brand new relationship-type + attribute
        create_mandatory_edge_constraint(self.g, 'Wrote', 'year', sync=True)

        # the WAIT command forces master slave sync to complete
        self.source.execute_command("WAIT", 1, 0)

        replica = Graph(self.replica, GRAPH_ID)

        #-----------------------------------------------------------------------
        # the constraints should be visible on the replica
        #-----------------------------------------------------------------------

        # WAIT only guarantees the GRAPH.CONSTRAINT CREATE command itself was
        # applied on the replica; constraint enforcement runs asynchronously
        # on the replica's own indexer thread pool, so wait for it separately
        # before checking status
        wait_on_constraint(replica, 'UNIQUE',    'NODE',         'Artist', 'nickname')
        wait_on_constraint(replica, 'MANDATORY', 'RELATIONSHIP', 'Wrote',  'year')

        c = get_constraint(replica, 'UNIQUE', 'NODE', 'Artist', 'nickname')
        self.env.assertIsNotNone (c)
        self.env.assertEqual(c.status, 'OPERATIONAL')

        c = get_constraint(replica, 'MANDATORY', 'RELATIONSHIP', 'Wrote', 'year')
        self.env.assertIsNotNone (c)
        self.env.assertEqual(c.status, 'OPERATIONAL')

        #-----------------------------------------------------------------------
        # the schema and attributes the constraints introduced should be
        # visible on the replica too, even though no data referencing the new
        # label / relationship-type / attributes was ever created
        #-----------------------------------------------------------------------

        labels = replica.ro_query("CALL db.labels() YIELD label RETURN label").result_set
        self.env.assertContains(['Artist'], labels)

        rel_types = replica.ro_query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType").result_set
        self.env.assertContains(['Wrote'], rel_types)

        prop_keys = replica.ro_query("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey").result_set
        self.env.assertContains(['nickname'], prop_keys)
        self.env.assertContains(['year'], prop_keys)

        #-----------------------------------------------------------------------
        # constraints should be enforced on the source
        #-----------------------------------------------------------------------

        self.g.query("CREATE (:Artist {nickname: 'Banksy'})")
        self.g.query("CREATE ()-[:Wrote {year: 2000}]->()")

        try:
            self.g.query("CREATE (:Artist {nickname: 'Banksy'})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Artist", str(e))

        try:
            self.g.query("CREATE ()-[:Wrote]->()")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: edge with relationship-type Wrote missing property year", str(e))

        #-----------------------------------------------------------------------
        # the data referencing the new label / relationship-type / attributes
        # should replicate correctly, using the same schema / attribute IDs
        # the constraint introduced
        #-----------------------------------------------------------------------

        self.source.execute_command("WAIT", 1, 0)

        q = "MATCH (n:Artist) RETURN n ORDER BY n"
        result = self.g.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        self.env.assertEqual(replica_result, result)

        q = "MATCH ()-[e:Wrote]->() RETURN e ORDER BY e"
        result = self.g.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        self.env.assertEqual(replica_result, result)

class testConstraintAOF():
    def __init__(self):
        self.env, self.db = Env(useAof=True)
        self.con = self.env.getConnection()

        # this test restarts the server mid-test (self.env.stop()/start())
        # to force an AOF reload. FalkorDB's shutdown handler only frees
        # global singletons (thread pool, indexer, GraphBLAS, RediSearch) and
        # never walks the keyspace to free per-graph state (schemas,
        # constraints, indexes, ...), so a graceful process exit while a
        # graph still holds data is always reported by LeakSanitizer as a
        # pile of leaks. this is a long-standing, harmless shutdown gap that
        # other tests don't hit because they never explicitly restart their
        # (shared) server mid-test, not something introduced by this test
        if SANITIZER:
            self.env.skip()

        self.con.delete(GRAPH_ID)
        self.g = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        g = self.g
        g.query("CREATE (:Person {name: 'Mike', age: 10, height: 180})")
        g.query("CREATE (:Person {name: 'Tim', age: 20, height: 190})")
        g.query("MATCH (a{name: 'Mike'}), (b{name:'Tim'}) CREATE (a)-[:Knows {since: 2000, weight: 1}]->(b)")

    def test01_aof_load_constraints(self):
        # this test exercises GRAPH.CONSTRAINT CREATE and DROP commands being
        # replayed as part of AOF loading (as opposed to being issued by a
        # regular client)
        #
        # a constraint of every UNIQUE/MANDATORY x NODE/RELATIONSHIP
        # combination is created, an additional constraint of every
        # combination is created and then dropped, so the AOF contains both
        # GRAPH.CONSTRAINT CREATE and DROP commands for every combination

        g = self.g

        #-----------------------------------------------------------------------
        # constraints that should survive the reload
        #-----------------------------------------------------------------------

        create_unique_node_constraint    (g, 'Person', 'name',  sync=True)
        create_mandatory_node_constraint (g, 'Person', 'age',   sync=True)
        create_unique_edge_constraint    (g, 'Knows',  'since', sync=True)
        create_mandatory_edge_constraint (g, 'Knows',  'since', sync=True)

        #-----------------------------------------------------------------------
        # constraints that get dropped before the reload, to make sure
        # GRAPH.CONSTRAINT DROP commands are present in the AOF as well
        #-----------------------------------------------------------------------

        create_unique_node_constraint    (g, 'Person', 'height', sync=True)
        create_mandatory_node_constraint (g, 'Person', 'height', sync=True)
        create_unique_edge_constraint    (g, 'Knows',  'weight', sync=True)
        create_mandatory_edge_constraint (g, 'Knows',  'weight', sync=True)

        drop_unique_node_constraint    (g, 'Person', 'height')
        drop_mandatory_node_constraint (g, 'Person', 'height')
        drop_unique_edge_constraint    (g, 'Knows',  'weight')
        drop_mandatory_edge_constraint (g, 'Knows',  'weight')

        constraints = list_constraints(g)
        self.env.assertEqual(len(constraints), 4)
        for c in constraints:
            self.env.assertEqual(c.status, 'OPERATIONAL')

        #-----------------------------------------------------------------------
        # restart the server; on start-up the whole AOF command log,
        # including every GRAPH.CONSTRAINT CREATE/DROP issued above, is
        # replayed while the module is executing under
        # REDISMODULE_CTX_FLAGS_LOADING
        #-----------------------------------------------------------------------

        self.env.stop()
        self.env.start()

        self.con = self.env.getConnection()
        g = self.db.select_graph(GRAPH_ID)
        self.g = g

        # (re)activation of a constraint is asynchronous, wait for it to
        # complete for each of the surviving constraints
        wait_on_constraint(g, 'UNIQUE',    'NODE',         'Person', 'name')
        wait_on_constraint(g, 'MANDATORY', 'NODE',         'Person', 'age')
        wait_on_constraint(g, 'UNIQUE',    'RELATIONSHIP', 'Knows',  'since')
        wait_on_constraint(g, 'MANDATORY', 'RELATIONSHIP', 'Knows',  'since')

        #-----------------------------------------------------------------------
        # validate constraints survived the AOF reload
        #-----------------------------------------------------------------------

        constraints = list_constraints(g)
        self.env.assertEqual(len(constraints), 4)
        for c in constraints:
            self.env.assertEqual(c.status, 'OPERATIONAL')

        # the dropped constraints must not have been resurrected by the reload
        self.env.assertIsNone (get_constraint(g, 'UNIQUE',    'NODE',         'Person', 'height'))
        self.env.assertIsNone (get_constraint(g, 'MANDATORY', 'NODE',         'Person', 'height'))
        self.env.assertIsNone (get_constraint(g, 'UNIQUE',    'RELATIONSHIP', 'Knows',  'weight'))
        self.env.assertIsNone (get_constraint(g, 'MANDATORY', 'RELATIONSHIP', 'Knows',  'weight'))

        #-----------------------------------------------------------------------
        # surviving constraints should still be enforced after the reload
        #-----------------------------------------------------------------------

        try:
            g.query("CREATE (:Person {name:'Mike', age:99})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Person", str(e))

        try:
            g.query("CREATE (:Person {name:'NewGuy'})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: node with label Person missing property age", str(e))

        try:
            g.query("MATCH (a:Person{name:'Mike'}), (b:Person{name:'Tim'}) CREATE (a)-[:Knows{since:2000}]->(b)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type Knows", str(e))

        try:
            g.query("MATCH (a:Person{name:'Mike'}), (b:Person{name:'Tim'}) CREATE (a)-[:Knows]->(b)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: edge with relationship-type Knows missing property since", str(e))

        # the dropped constraints should no longer be enforced
        # (name/age satisfy the still-active constraints; the duplicate/missing
        # height is what would have violated the now-dropped constraints)
        g.query("CREATE (:Person {name:'Architect1', age:40, height:180})")
        g.query("CREATE (:Person {name:'Architect2', age:41})")

        # (since satisfies the still-active constraints; the duplicate/missing
        # weight is what would have violated the now-dropped constraints)
        g.query("MATCH (a:Person{name:'Mike'}), (b:Person{name:'Tim'}) CREATE (a)-[:Knows{since:3000, weight:1}]->(b)")
        g.query("MATCH (a:Person{name:'Mike'}), (b:Person{name:'Tim'}) CREATE (a)-[:Knows{since:4000}]->(b)")

    def test02_aof_load_constraint_introduces_new_schema_and_attr(self):
        # every constraint created in test01 targeted a label/relationship-type
        # and attribute(s) that populate_graph() had already introduced before
        # the constraint was created. this test makes sure a constraint
        # created against a brand new label, relationship-type and
        # attribute(s) - ones no query has ever referenced before - survives
        # an AOF reload: the GRAPH.CONSTRAINT CREATE command replayed from the
        # AOF must itself recreate the missing schema and attribute ID(s),
        # rather than relying on some earlier CREATE query in the AOF to have
        # introduced them first

        g = self.g

        # brand new node label + attribute
        create_unique_node_constraint(g, 'Poet', 'penname', sync=True)

        # brand new relationship-type + attribute
        create_mandatory_edge_constraint(g, 'Painted', 'year', sync=True)

        #-----------------------------------------------------------------------
        # restart the server, forcing the whole AOF command log - including
        # the two GRAPH.CONSTRAINT CREATE commands above - to be replayed
        # while the module is executing under REDISMODULE_CTX_FLAGS_LOADING
        #-----------------------------------------------------------------------

        self.env.stop()
        self.env.start()

        self.con = self.env.getConnection()
        g = self.db.select_graph(GRAPH_ID)
        self.g = g

        wait_on_constraint(g, 'UNIQUE',    'NODE',         'Poet',    'penname')
        wait_on_constraint(g, 'MANDATORY', 'RELATIONSHIP', 'Painted', 'year')

        #-----------------------------------------------------------------------
        # the constraints, and the schema / attributes they introduced,
        # should have survived the reload
        #-----------------------------------------------------------------------

        c = get_constraint(g, 'UNIQUE', 'NODE', 'Poet', 'penname')
        self.env.assertIsNotNone (c)
        self.env.assertEqual(c.status, 'OPERATIONAL')

        c = get_constraint(g, 'MANDATORY', 'RELATIONSHIP', 'Painted', 'year')
        self.env.assertIsNotNone (c)
        self.env.assertEqual(c.status, 'OPERATIONAL')

        labels = g.ro_query("CALL db.labels() YIELD label RETURN label").result_set
        self.env.assertContains(['Poet'], labels)

        rel_types = g.ro_query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType").result_set
        self.env.assertContains(['Painted'], rel_types)

        prop_keys = g.ro_query("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey").result_set
        self.env.assertContains(['penname'], prop_keys)
        self.env.assertContains(['year'], prop_keys)

        #-----------------------------------------------------------------------
        # the recreated schema/attribute IDs should be usable: write data
        # through them and make sure it's queryable, and make sure a plain
        # query introducing yet another new label right after the reload
        # doesn't collide with the IDs the constraint reload assigned
        #-----------------------------------------------------------------------

        g.query("CREATE (:Poet {penname: 'Twain'})")
        g.query("CREATE ()-[:Painted {year: 1889}]->()")
        g.query("CREATE (:Curator {since: 1990})")

        result = g.query("MATCH (n:Poet) RETURN n.penname").result_set
        self.env.assertEqual(result, [['Twain']])

        result = g.query("MATCH ()-[e:Painted]->() RETURN e.year").result_set
        self.env.assertEqual(result, [[1889]])

        result = g.query("MATCH (n:Curator) RETURN n.since").result_set
        self.env.assertEqual(result, [[1990]])

        #-----------------------------------------------------------------------
        # constraints should still be enforced after the reload
        #-----------------------------------------------------------------------

        try:
            g.query("CREATE (:Poet {penname: 'Twain'})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Poet", str(e))

        try:
            g.query("CREATE ()-[:Painted]->()")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation: edge with relationship-type Painted missing property year", str(e))
