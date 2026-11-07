import threading
from common import *
from index_utils import *
from constraint_utils import *

GRAPH_ID = "constraints"

class testConstraintNodes():
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env()
        cls.con = cls.env.getConnection()
        cls.con.delete(GRAPH_ID)
        cls.g = cls.db.select_graph(GRAPH_ID)
        cls.populate_graph()

    @classmethod
    def populate_graph(cls):
        g = cls.g
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
        assert result == "PENDING"

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
        assert len(constraints) == 6
        for c in constraints:
            assert c.status != 'FAILED'

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
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: node with label Person missing property height" in str(e)

        #-----------------------------------------------------------------------
        # create a node that violates the unique constraint on point data ignored
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (p:Person) CREATE (n:Person{height:p.height + 1, loc: p.loc}) DELETE n")
            assert True
        except ResponseError as e:
            assert False

        #-----------------------------------------------------------------------
        # create a node that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (p:Person) CREATE (:Person{height:p.height})")
            assert False
        except ResponseError as e:
            assert "unique constraint violation on node of type Person" in str(e)

        #-----------------------------------------------------------------------
        # create a node that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (p:Person) CREATE (:Person{height:rand(), name:p.name, age:p.age})")
            assert False
        except ResponseError as e:
            assert "unique constraint violation on node of type Person" in str(e)

        #-----------------------------------------------------------------------
        # node update that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (n:Person) SET n.height = NULL")
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: node with label Person missing property height" in str(e)

        #-----------------------------------------------------------------------
        # node update that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (a:Person), (b:Person) WHERE a<>b SET a.height = b.height")
            assert False
        except ResponseError as e:
            assert "unique constraint violation on node of type Person" in str(e)

        #-----------------------------------------------------------------------
        # node update that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (a:Person), (b:Person) WHERE a<>b SET a.name = b.name, a.age = b.age")
            assert False
        except ResponseError as e:
            assert "unique constraint violation on node of type Person" in str(e)

        #-----------------------------------------------------------------------
        # node merge-match that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (n:Person {name: 'Andrew'}) ON MATCH SET n.height = NULL")
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: node with label Person missing property height" in str(e)

        #-----------------------------------------------------------------------
        # node merge-match that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {height:180}) ON MATCH SET p.height = 190")
            assert False
        except ResponseError as e:
            assert "unique constraint violation on node of type Person" in str(e)

        #-----------------------------------------------------------------------
        # node merge-match that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {name:'Mike'}) ON MATCH SET p.age = 20, p.height = 190")
            assert False
        except ResponseError as e:
            assert "unique constraint violation on node of type Person" in str(e)

        #-----------------------------------------------------------------------
        # node merge-create that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (n:Person {name: 'Dor', height: 187}) ON CREATE SET n.height = NULL")
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: node with label Person missing property height" in str(e)

        #-----------------------------------------------------------------------
        # node merge-create that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {name: 'Dor', height:187}) ON CREATE SET p.height = 190")
            assert False
        except ResponseError as e:
            assert "unique constraint violation on node of type Person" in str(e)

        #-----------------------------------------------------------------------
        # node merge-create that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {name:'Dor', height:187}) ON CREATE SET p.age = 10, p.name = 'Mike'")
            assert False
        except ResponseError as e:
            assert "unique constraint violation on node of type Person" in str(e)

        #-----------------------------------------------------------------------
        # node merge that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (n:Person {name: 'Dor'})")
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: node with label Person missing property height" in str(e)

        #-----------------------------------------------------------------------
        # node merge that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {name: 'Dor', height:180})")
            assert False
        except ResponseError as e:
            assert "unique constraint violation on node of type Person" in str(e)

        #-----------------------------------------------------------------------
        # node merge that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE (p:Person {v:12, name:'Mike', age:10})")
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: node with label Person missing property height" in str(e)

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
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: node with label Person missing property height" in str(e)

        # add attributes to Architect which will conflict with both unique constraints
        g.query("MATCH (n:Architect) SET n.name = 'Mike', n.age = 10, n.height = 180")

        #-----------------------------------------------------------------------
        # node label update which will conflict with unique constraints
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH (n:Architect) SET n:Person")
            assert False
        except ResponseError as e:
            assert "unique constraint violation on node of type Person" in str(e)

        # delete Architect
        g.query("MATCH (n:Architect) DELETE n")

        # validate graph did not changed
        actual_result_set = self.g.query("MATCH (n) RETURN n ORDER BY ID(n)").result_set
        assert actual_result_set == expected_result_set

        try:
            g.query("CREATE CONSTRAINT ON (n:N) ASSERT n.v IS UNIQUE")
            assert False
        except ResponseError as e:
            assert "Invalid constraint command use the GRAPH.CONSTRAINT command instead" in str(e)

        try:
            g.query("DROP CONSTRAINT ON (n:N) ASSERT n.v IS UNIQUE")
            assert False
        except ResponseError as e:
            assert "Invalid constraint command use the GRAPH.CONSTRAINT command instead" in str(e)

        try:
            g.query("CREATE CONSTRAINT ON ()-[r:R]->() ASSERT r.v")
            assert False
        except ResponseError as e:
            assert "Invalid constraint command use the GRAPH.CONSTRAINT command instead" in str(e)

        try:
            g.query("DROP CONSTRAINT ON ()-[r:R]->() ASSERT r.v")
            assert False
        except ResponseError as e:
            assert "Invalid constraint command use the GRAPH.CONSTRAINT command instead" in str(e)


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
        assert len(constraints) == 0

    def test04_invalid_constraint_command(self):
        # constraint create command:
        # GRAPH.CONSTRAIN <key> CREATE/DEL UNIQUE/MANDATORY [NODE label / RELATIONSHIP type] PROPERTIES prop_count prop0...

        #-----------------------------------------------------------------------
        # invalid constraint operation
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "LIST", GRAPH_ID)
            assert False
        except ResponseError as e:
            assert "wrong number of arguments for 'graph.CONSTRAINT' command" in str(e)

        #-----------------------------------------------------------------------
        # invalid constraint operation
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", GRAPH_ID, "INVALID_OP", "unique", "LABEL", "New_Label", "PROPERTIES", 1, "New_Attr")
            assert False
        except ResponseError as e:
            assert "Invalid constraint operation" in str(e)

        #-----------------------------------------------------------------------
        # invalid constraint type
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "INVALID_CT", "New_Label", "Person", "PROPERTIES", 1, "New_Attr")
            assert False
        except ResponseError as e:
            assert "Invalid constraint type" in str(e)

        #-----------------------------------------------------------------------
        # invalid entity type
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "INVALID_ENTITY_TYPE", "New_Label", "PROPERTIES", 1, "New_Attr")
            assert False
        except ResponseError as e:
            assert "Invalid constraint entity type" in str(e)

        #-----------------------------------------------------------------------
        # invalid label name
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "1ab", "PROPERTIES", 1, "New_Attr")
            assert False
        except ResponseError as e:
            assert "Label name 1ab is invalid" in str(e)

        #-----------------------------------------------------------------------
        # invalid property name
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "label", "PROPERTIES", 2, 1, 2)
            assert False
        except ResponseError as e:
            assert "Property name 1 is invalid" in str(e)

        #-----------------------------------------------------------------------
        # invalid property name
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "label", "PROPERTIES", 2, 'a1', '2pb')
            assert False
        except ResponseError as e:
            assert "Property name 2pb is invalid" in str(e)

        #-----------------------------------------------------------------------
        # invalid property count
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "label", "PROPERTIES", 0)
            assert False
        except ResponseError as e:
            assert "Number of properties must be an integer between 1 and 255" in str(e)

        #-----------------------------------------------------------------------
        # invalid property count
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "label", "PROPERTIES", -1, 12)
            assert False
        except ResponseError as e:
            assert "Number of properties must be an integer between 1 and 255" in str(e)

        #-----------------------------------------------------------------------
        # del constraint on non exsisting label
        #-----------------------------------------------------------------------
        try:
            drop_unique_node_constraint(self.g, "None_Existing_Label", "age")
            assert False
        except ResponseError as e:
            assert "Unable to drop constraint, no such constraint." in str(e)

        #-----------------------------------------------------------------------
        # del constraint on non exsisting attribute
        #-----------------------------------------------------------------------
        try:
            drop_unique_node_constraint(self.g, "Person", "None_Existing_Attr")
            assert False
        except ResponseError as e:
            assert "Unable to drop constraint, no such constraint." in str(e)

        #-----------------------------------------------------------------------
        # create constraint which already exists
        #-----------------------------------------------------------------------
        create_unique_node_constraint(self.g, "Person", "age", sync=True)
        try:
            create_unique_node_constraint(self.g, "Person", "age")
            assert False
        except ResponseError as e:
            assert "Constraint already exists" in str(e)

        # validate labels and attributes were not created for failed operations
        # not expecting None_Existing_Label, New_Label, None_Existing_Attr and New_Attr
        # to be added to the graph
        labels = self.g.query("CALL db.labels()").result_set
        attributes = self.g.query("CALL db.propertyKeys()").result_set
        assert not "New_Label" in labels
        assert not "None_Existing_Label" in labels
        assert not "New_Attr" in attributes
        assert not "None_Existing_Attr" in attributes

        #-----------------------------------------------------------------------
        # unique constraint missing supporting exact-match index
        #-----------------------------------------------------------------------

    def test05_constraint_create_drop_simultanously(self):
        # make sure there are no constraints in the graph
        for c in list_constraints(self.g):
            drop_constraint(self.g, c.type, c.entity_type, c.label, *c.attributes)
        assert 0 == len(list_constraints(self.g))

        # create 500K new entities
        self.g.query("UNWIND range(0, 500000) AS x CREATE (:MarineBiologist {age: x})")

        # create unique constraint over MarineBiologist age attribute
        create_unique_node_constraint(self.g, "MarineBiologist", "age")

        # make sure constraint is pending
        constraints = list_constraints(self.g)
        assert len(constraints) == 1
        c = constraints[0]
        assert c.status == "UNDER CONSTRUCTION"

        # delete constraint
        drop_unique_node_constraint(self.g, "MarineBiologist", "age")

        # constraint should be dropped immediately
        constraints = list_constraints(self.g)
        assert len(constraints) == 0

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
        assert len(constraints) == 1
        assert constraints[0].status == "FAILED"

        # fix name uniqueness by deleting duplicated node
        self.g.query("MATCH (p:Person {name:'jerry'}) WITH p LIMIT 1 DELETE p")

        #-----------------------------------------------------------------------
        # re-create unique constraint
        #-----------------------------------------------------------------------

        create_unique_node_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation succeeded
        constraints = list_constraints(self.g)
        assert len(constraints) == 1
        assert constraints[0].status == "OPERATIONAL"

        #-----------------------------------------------------------------------
        # try to create mandatory constraint over Person name
        #-----------------------------------------------------------------------

        create_mandatory_node_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation faile
        constraints = list_constraints(self.g)
        assert len(constraints) == 2
        for c in constraints:
            if c.type == "UNIQUE":
                assert c.status == "OPERATIONAL"
            else:
                assert c.status == "FAILED"

        #-----------------------------------------------------------------------
        # try deleting a failed constraint
        #-----------------------------------------------------------------------

        drop_mandatory_node_constraint(self.g, "Person", "name")

        # make sure constraint was deleted
        constraints = list_constraints(self.g)
        assert len(constraints) == 1
        c = constraints[0]
        assert c.label == "Person"
        assert c.type == "UNIQUE"
        assert c.status == "OPERATIONAL"

        #-----------------------------------------------------------------------
        # re-create mandatory constraint
        #-----------------------------------------------------------------------

        # add missing name attribute to resolve conflict
        self.g.query("MATCH (p:Person) WHERE p.name is NULL SET p.name = 'kramer'")

        create_mandatory_node_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation succeeded
        constraints = list_constraints(self.g)
        assert len(constraints) == 2
        assert constraints[0].status == "OPERATIONAL"
        assert constraints[1].status == "OPERATIONAL"

    def test07_constraint_creation_with_new_label_attr(self):
        # create a constraint against a new label and a new attribute
        create_unique_node_constraint(self.g, "Artist", "nickname", sync=True)
        self.g.query("CREATE (:Artist {nickname: 'Banksy'})")

        # make sure constraint is enforced
        try:
            self.g.query("CREATE (:Artist {nickname: 'Banksy'})")
            assert False
        except ResponseError as e:
            assert "unique constraint violation on node of type Artist" in str(e)

    def test08_remove_supporting_index(self):
        # try to create unique index without a supporting index
        try:
            create_constraint(self.g, "unique", "node", "Author", "nickname", "birthdate")
            assert not 1
        except ResponseError as e:
            assert "missing supporting exact-match index" in str(e)

        # create supporting index
        create_node_range_index(self.g, "Author", "nickname", "birthdate")

        # create unique index
        create_constraint(self.g, "unique", "node", "Author", "nickname", "birthdate")

        # try to drop supporting index
        try:
            drop_node_range_index(self.g, "Author", "nickname")
        except ResponseError as e:
            assert "Index supports constraint" in str(e)

        try:
            drop_node_range_index(self.g, "Author", "birthdate")
        except ResponseError as e:
            assert "Index supports constraint" in str(e)

        # drop constraint
        drop_unique_node_constraint(self.g, "Author", "nickname", "birthdate")

        # try to drop supporting index
        drop_node_range_index(self.g, "Author", "nickname")
        drop_node_range_index(self.g, "Author", "birthdate")

class testConstraintEdges():
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env()
        cls.con = cls.env.getConnection()
        cls.con.delete(GRAPH_ID)
        cls.g = cls.db.select_graph(GRAPH_ID)
        cls.populate_graph()

    @classmethod
    def populate_graph(cls):
        g = cls.g
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
        assert len(constraints) == 3
        for c in constraints:
            assert c.status == 'OPERATIONAL'

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
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: edge with relationship-type Person missing property height" in str(e)

        #-----------------------------------------------------------------------
        # create an edge that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH ()-[e:Person]->() CREATE ()-[:Person{height:e.height}]->()")
            assert False
        except ResponseError as e:
            assert "unique constraint violation, on edge of relationship-type Person" in str(e)

        #-----------------------------------------------------------------------
        # create an edge that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH ()-[e:Person]->() CREATE ()-[:Person{height:rand(), name:e.name, age:e.age}]->()")
            assert False
        except ResponseError as e:
            assert "unique constraint violation, on edge of relationship-type Person" in str(e)

        #-----------------------------------------------------------------------
        # edge update that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH ()-[e:Person]->() SET e.height = NULL")
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: edge with relationship-type Person missing property height" in str(e)

        #-----------------------------------------------------------------------
        # edge update that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH ()-[a:Person]->(), ()-[b:Person]->() WHERE a<>b SET a.height = b.height")
            assert False
        except ResponseError as e:
            assert "unique constraint violation, on edge of relationship-type Person" in str(e)

        #-----------------------------------------------------------------------
        # edge update that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MATCH ()-[a:Person]->(), ()-[b:Person]->() WHERE a<>b SET a.name = b.name, a.age = b.age")
            assert False
        except ResponseError as e:
            assert "unique constraint violation, on edge of relationship-type Person" in str(e)

        #-----------------------------------------------------------------------
        # edge merge-match that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name: 'Andrew'}]->() ON MATCH SET e.height = NULL")
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: edge with relationship-type Person missing property height" in str(e)

        #-----------------------------------------------------------------------
        # edge merge-match that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {height:180}]->() ON MATCH SET e.height = 190")
            assert False
        except ResponseError as e:
            assert "unique constraint violation, on edge of relationship-type Person" in str(e)

        #-----------------------------------------------------------------------
        # edge merge-match that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name:'Mike'}]->() ON MATCH SET e.age = 20, e.height = 190")
            assert False
        except ResponseError as e:
            assert "unique constraint violation, on edge of relationship-type Person" in str(e)

        #-----------------------------------------------------------------------
        # edge merge-create that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name: 'Dor', height: 187}]->() ON CREATE SET e.height = NULL")
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: edge with relationship-type Person missing property height" in str(e)

        #-----------------------------------------------------------------------
        # edge merge-create that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name: 'Dor', height:187}]->() ON CREATE SET e.height = 190")
            assert False
        except ResponseError as e:
            assert "unique constraint violation, on edge of relationship-type Person" in str(e)

        #-----------------------------------------------------------------------
        # edge merge-create that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name:'Dor', height:187}]->() ON CREATE SET e.age = 10, e.name = 'Mike'")
            assert False
        except ResponseError as e:
            assert "unique constraint violation, on edge of relationship-type Person" in str(e)

        #-----------------------------------------------------------------------
        # edge merge that violates the mandatory constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name: 'Dor'}]->()")
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: edge with relationship-type Person missing property height" in str(e)

        #-----------------------------------------------------------------------
        # edge merge that violates the unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {name: 'Dor', height:180}]->()")
            assert False
        except ResponseError as e:
            assert "unique constraint violation, on edge of relationship-type Person" in str(e)

        #-----------------------------------------------------------------------
        # edge merge that violates the composite unique constraint
        #-----------------------------------------------------------------------

        try:
            g.query("MERGE ()-[e:Person {v:12, name:'Mike', age:10}]->()")
            assert False
        except ResponseError as e:
            assert "mandatory constraint violation: edge with relationship-type Person missing property height" in str(e)

        # validate graph did not changed
        actual_result_set = self.g.query("MATCH ()-[e]->() RETURN e ORDER BY ID(e)").result_set
        assert actual_result_set == expected_result_set

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
        assert len(constraints) == 0

    def test04_invalid_constraint_command(self):
        # constraint create command:
        # GRAPH.CONSTRAIN <key> CREATE/DEL UNIQUE/MANDATORY [NODE label / RELATIONSHIP type] PROPERTIES prop_count prop0...

        #-----------------------------------------------------------------------
        # invalid constraint operation
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "INVALID_OP", GRAPH_ID, "unique", "RELATIONSHIP", "New_Label", "PROPERTIES", 1, "New_Attr")
            assert False
        except ResponseError as e:
            assert "Invalid constraint operation" in str(e)

        #-----------------------------------------------------------------------
        # invalid constraint type
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "INVALID_CT", "New_Label", "Person", "PROPERTIES", 1, "New_Attr")
            assert False
        except ResponseError as e:
            assert "Invalid constraint type" in str(e)

        #-----------------------------------------------------------------------
        # invalid entity type
        #-----------------------------------------------------------------------
        try:
            self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "INVALID_ENTITY_TYPE", "New_Label", "PROPERTIES", 1, "New_Attr")
            assert False
        except ResponseError as e:
            assert "Invalid constraint entity type" in str(e)

        #-----------------------------------------------------------------------
        # del constraint on non exsisting label
        #-----------------------------------------------------------------------
        try:
            drop_unique_edge_constraint(self.g, "None_Existing_Label", "age")
            assert False
        except ResponseError as e:
            assert "Unable to drop constraint, no such constraint." in str(e)

        #-----------------------------------------------------------------------
        # del constraint on non exsisting attribute type
        #-----------------------------------------------------------------------
        try:
            drop_unique_edge_constraint(self.g, "Person", "None_Existing_Attr")
            assert False
        except ResponseError as e:
            assert "Unable to drop constraint, no such constraint." in str(e)

        #-----------------------------------------------------------------------
        # create constraint with duplicate attributes
        #-----------------------------------------------------------------------
        try:
            create_unique_edge_constraint(self.g, "Person", "age", "height", "weight", "height", sync=True)
            assert False
        except ResponseError as e:
            assert "Properties cannot contain duplicates" in str(e)

        #-----------------------------------------------------------------------
        # create constraint which already exists
        #-----------------------------------------------------------------------
        create_unique_edge_constraint(self.g, "Person", "age", sync=True)
        try:
            create_unique_edge_constraint(self.g, "Person", "age")
            assert False
        except ResponseError as e:
            assert "Constraint already exists" in str(e)

        # validate labels and attributes were not created for failed operations
        # not expecting None_Existing_Label, New_Label, None_Existing_Attr and New_Attr
        # to be added to the graph
        labels = self.g.query("CALL db.labels()").result_set
        attributes = self.g.query("CALL db.propertyKeys()").result_set
        assert not "New_Label" in labels
        assert not "None_Existing_Label" in labels
        assert not "New_Attr" in attributes
        assert not "None_Existing_Attr" in attributes

    def test05_constraint_create_drop_simultanously(self):
        # make sure there are no constraints in the graph
        for c in list_constraints(self.g):
            drop_constraint(self.g, c.type, c.entity_type, c.label, *c.attributes)
        assert 0 == len(list_constraints(self.g))

        # create 500K new entities
        self.g.query("UNWIND range(0, 500000) AS x CREATE ()-[:MarineBiologist {age: x}]->()")

        # create unique constraint over MarineBiologist age attribute
        create_unique_edge_constraint(self.g, "MarineBiologist", "age")

        # make sure constraint is pending
        constraints = list_constraints(self.g)
        assert len(constraints) == 1
        c = constraints[0]
        assert c.status == "UNDER CONSTRUCTION"

        # delete constraint
        drop_unique_edge_constraint(self.g, "MarineBiologist", "age")

        # constraint should be dropped immediately
        constraints = list_constraints(self.g)
        assert len(constraints) == 0

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
        assert len(constraints) == 1
        assert constraints[0].status == "FAILED"

        # fix name uniqueness by deleting duplicated edge
        self.g.query("MATCH ()-[e:Person {name:'jerry'}]->() WITH e LIMIT 1 DELETE e")

        #-----------------------------------------------------------------------
        # re-create unique constraint
        #-----------------------------------------------------------------------

        create_unique_edge_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation succeeded
        constraints = list_constraints(self.g)
        assert len(constraints) == 1
        assert constraints[0].status == "OPERATIONAL"

        #-----------------------------------------------------------------------
        # try to create mandatory constraint over Person name
        #-----------------------------------------------------------------------

        create_mandatory_edge_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation faile
        constraints = list_constraints(self.g)
        assert len(constraints) == 2
        for c in constraints:
            if c.type == "UNIQUE":
                assert c.status == "OPERATIONAL"
            else:
                assert c.status == "FAILED"

        #-----------------------------------------------------------------------
        # try deleting a failed constraint
        #-----------------------------------------------------------------------

        drop_mandatory_edge_constraint(self.g, "Person", "name")

        # make sure constraint was deleted
        constraints = list_constraints(self.g)
        assert len(constraints) == 1
        c = constraints[0]
        assert c.label == "Person"
        assert c.type == "UNIQUE"
        assert c.status == "OPERATIONAL"

        #-----------------------------------------------------------------------
        # re-create mandatory constraint
        #-----------------------------------------------------------------------

        # add missing name attribute to resolve conflict
        self.g.query("MATCH ()-[e:Person]->() WHERE e.name is NULL SET e.name = 'kramer'")

        create_mandatory_edge_constraint(self.g, "Person", "name", sync=True)

        # make sure constraint creation succeeded
        constraints = list_constraints(self.g)
        assert len(constraints) == 2
        assert constraints[0].status == "OPERATIONAL"
        assert constraints[1].status == "OPERATIONAL"

    def test07_constraint_creation_with_new_relation_attr(self):
        # create a constraint against a new relationship-type and a new attribute
        create_unique_edge_constraint(self.g, "Artist", "nickname", sync=True)
        self.g.query("CREATE ()-[:Artist {nickname: 'Banksy'}]->()")

        # make sure constraint is enforced
        try:
            self.g.query("CREATE ()-[:Artist {nickname: 'Banksy'}]->()")
            assert False
        except ResponseError as e:
            assert "unique constraint violation, on edge of relationship-type Artist" in str(e)

MONITOR_ATTACHED = False

class testConstraintReplication():
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env(env='oss', useSlaves=True)
        cls.source  = cls.env.getConnection()
        cls.replica = cls.env.getSlaveConnection()
        cls.monitor = []
        cls.g = cls.db.select_graph(GRAPH_ID)

        cls.monitor_thread = threading.Thread(target=cls.monitor_thread, daemon=True)
        cls.monitor_thread.start()

        # wait for monitor thread to attach
        while MONITOR_ATTACHED is False:
            time.sleep(0.2)

        # clear DB
        cls.source.delete(GRAPH_ID)

        # the WAIT command forces master slave sync to complete
        cls.source.execute_command("WAIT", 1, 0)

    @classmethod
    def monitor_thread(cls):
        global MONITOR_ATTACHED
        try:
            with cls.replica.monitor() as m:
                MONITOR_ATTACHED = True
                for cmd in m.listen():
                    if 'GRAPH.CONSTRAINT' in cmd['command']:
                        cls.monitor.append(cmd)
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
        assert len(constraints) == 6
        for c in constraints:
            assert c.status == 'OPERATIONAL'

        # each constraint should be replicated twice from source to replica:
        # 1. upon creation
        # 2. upon constraint becoming activate
        self.source.execute_command("WAIT", 1, 0)

        # wait for all 12 GRAPH.CONSTRAINT commands to be replicated
        elapsed = 10
        while len(self.monitor) < 12 and elapsed > 0:
            time.sleep(0.2)
            elapsed -= 0.2

        assert len(self.monitor) == 12
