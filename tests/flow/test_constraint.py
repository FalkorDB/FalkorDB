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
        create_unique_node_constraint(self.g, 'Person', 'height')

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

    def test09_whitespace_crash(self):
        create_node_range_index(self.g, "Author", "nickname")

        create_constraint(self.g, "unique", "node", "Author", "nickname")

        self.g.query("CREATE (:Author {nickname: 'abcd'})")

        self.g.query("CREATE (:Author {nickname: '   abcd   '})")

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
