import time
import asyncio
import threading
from common import *
from index_utils import *
from constraint_utils import *
from long_read_utils import *
from falkordb.asyncio import FalkorDB as AsyncFalkorDB
from redis.asyncio import Redis as AsyncRedis

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
        # property count is read as C's string2ll: no sign, no leading zeros
        #-----------------------------------------------------------------------
        for count in ["+1", "01", " 1", "1x"]:
            try:
                self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", "NODE", "label", "PROPERTIES", count, "New_Attr")
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Number of properties must be an integer between 1 and 255", str(e))

        #-----------------------------------------------------------------------
        # the entity type is NODE or RELATIONSHIP, as in C
        #-----------------------------------------------------------------------
        for entity in ["LABEL", "EDGE"]:
            try:
                self.con.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, "MANDATORY", entity, "New_Label", "PROPERTIES", 1, "New_Attr")
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Invalid constraint entity type", str(e))

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

    def test09_constraint_enforced_on_removed_and_readded_label(self):
        # A label that is removed and re-added in the same query is still
        # carried by the node at commit time, so constraints on it must be
        # enforced. Before the fix for #2777 the pending add and the pending
        # remove both sat in the transaction's bookkeeping and the remove won,
        # so the node looked unlabelled to the constraint check and violations
        # were silently let through.

        #-----------------------------------------------------------------------
        # unique constraint
        #-----------------------------------------------------------------------
        create_unique_node_constraint(self.g, "Rejoin", "v", sync=True)
        self.g.query("CREATE (:Rejoin {v: 1})")

        # duplicate created in the SAME query that removes and re-adds the
        # constrained label must still be rejected
        try:
            self.g.query("MATCH (n:Rejoin {v: 1}) REMOVE n:Rejoin SET n:Rejoin CREATE (:Rejoin {v: 1})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Rejoin", str(e))

        # the rejected query must not have left anything behind
        self.env.assertEqual(self.g.query("MATCH (n:Rejoin) RETURN count(n)").result_set[0][0], 1)

        # a node that re-acquires the label must also collide with an existing
        # value it is updated into
        self.g.query("CREATE (:Rejoin {v: 2})")
        try:
            self.g.query("MATCH (n:Rejoin {v: 2}) REMOVE n:Rejoin SET n:Rejoin SET n.v = 1")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type Rejoin", str(e))
        self.env.assertEqual(self.g.query("MATCH (n:Rejoin {v: 2}) RETURN count(n)").result_set[0][0], 1)

        # genuinely dropping the label frees the value — the constraint must
        # not be over-enforced
        result = self.g.query("MATCH (n:Rejoin {v: 1}) REMOVE n:Rejoin CREATE (:Rejoin {v: 1})")
        self.env.assertEqual(result.labels_removed, 1)
        self.env.assertEqual(result.nodes_created, 1)
        self.env.assertEqual(self.g.query("MATCH (n:Rejoin) RETURN count(n)").result_set[0][0], 2)

        #-----------------------------------------------------------------------
        # mandatory constraint
        #-----------------------------------------------------------------------
        create_mandatory_node_constraint(self.g, "Mandate", "p", sync=True)
        self.g.query("CREATE (:Mandate {p: 1})")

        # dropping the mandatory property while the label is removed and
        # re-added must be rejected — the node still ends up labelled
        try:
            self.g.query("MATCH (n:Mandate) REMOVE n:Mandate SET n:Mandate SET n.p = NULL")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("mandatory constraint violation", str(e))
        self.env.assertEqual(self.g.query("MATCH (n:Mandate) RETURN n.p").result_set[0][0], 1)

        # dropping the label for real releases the node from the constraint
        result = self.g.query("MATCH (n:Mandate) REMOVE n:Mandate SET n.p = NULL")
        self.env.assertEqual(result.labels_removed, 1)
        self.env.assertEqual(self.g.query("MATCH (n:Mandate) RETURN count(n)").result_set[0][0], 0)

    def test10_constraint_in_multi(self):
        # a client's GRAPH.CONSTRAINT runs on a worker with the client blocked,
        # and a client inside MULTI can't be blocked: there the command must
        # run inline and answer in the EXEC reply
        create = constraint_cmd("CREATE", GRAPH_ID, "MANDATORY", "NODE", "InMulti", "v")
        drop   = constraint_cmd("DROP",   GRAPH_ID, "MANDATORY", "NODE", "InMulti", "v")

        def exec_in_multi(cmd):
            pipe = self.con.pipeline(transaction=True)
            pipe.execute_command(*cmd)
            return [r.decode() if isinstance(r, bytes) else r for r in pipe.execute()]

        self.env.assertEqual(exec_in_multi(create), ["PENDING"])
        wait_on_constraint(self.g, "MANDATORY", "NODE", "InMulti", "v")
        c = get_constraint(self.g, "MANDATORY", "NODE", "InMulti", "v")
        self.env.assertEqual(c.status, "OPERATIONAL")

        self.env.assertEqual(exec_in_multi(drop), ["OK"])
        self.env.assertEqual(get_constraint(self.g, "MANDATORY", "NODE", "InMulti", "v"), None)

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

# a composite UNIQUE constraint is vacuously satisfied when ANY of the
# constrained properties is NULL / absent: the composite key is unknown, and an
# unknown key can not be proven to collide with another. see issue #2778
COMPOSITE_NULL_GRAPH_ID = "composite_unique_nulls"

class testCompositeUniqueConstraintNulls():
    def __init__(self):
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.con.delete(COMPOSITE_NULL_GRAPH_ID)
        self.g = self.db.select_graph(COMPOSITE_NULL_GRAPH_ID)

    def test01_partial_nodes_accepted(self):
        g = self.g

        create_unique_node_constraint(g, "P", "a", "b", sync=True)
        c = get_constraint(g, "UNIQUE", "NODE", "P", "a", "b")
        self.env.assertEqual(c.status, "OPERATIONAL")

        # two nodes agreeing on 'a', both missing 'b'
        g.query("CREATE (:P {a: 1})")
        g.query("CREATE (:P {a: 1})")

        # mirrored: two nodes agreeing on 'b', both missing 'a'
        g.query("CREATE (:P {b: 7})")
        g.query("CREATE (:P {b: 7})")

        # an explicit NULL is equivalent to an absent property
        g.query("CREATE (:P {a: 1, b: NULL})")

        self.env.assertEqual(g.query("MATCH (n:P) RETURN count(n)").result_set[0][0], 5)

    def test02_full_duplicate_still_rejected(self):
        g = self.g

        # every constrained property present -> the key is known and enforced
        g.query("CREATE (:P {a: 5, b: 6})")

        try:
            g.query("CREATE (:P {a: 5, b: 6})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type P", str(e))

        # a key sharing only one component is still distinct
        g.query("CREATE (:P {a: 5, b: 7})")

        self.env.assertEqual(g.query("MATCH (n:P) RETURN count(n)").result_set[0][0], 7)

    def test03_completing_a_partial_key_is_enforced(self):
        g = self.g

        # this node's key is unknown, so it is accepted
        g.query("CREATE (:P {a: 100, tag: 'partial'})")

        # filling in the missing property would produce a real duplicate
        g.query("CREATE (:P {a: 100, b: 200})")
        try:
            g.query("MATCH (n:P {tag: 'partial'}) SET n.b = 200")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type P", str(e))

        # completing it to a distinct key is fine, and dropping the property
        # makes the key unknown again
        g.query("MATCH (n:P {tag: 'partial'}) SET n.b = 201")
        g.query("MATCH (n:P {tag: 'partial'}) REMOVE n.b")

    def test04_single_property_constraint_unchanged(self):
        # for a single property "any null" and "all null" coincide
        g = self.g

        create_unique_node_constraint(g, "S", "a", sync=True)
        c = get_constraint(g, "UNIQUE", "NODE", "S", "a")
        self.env.assertEqual(c.status, "OPERATIONAL")

        g.query("CREATE (:S {a: 1})")

        try:
            g.query("CREATE (:S {a: 1})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation on node of type S", str(e))

        g.query("CREATE (:S {a: 2})")

        # nodes missing the constrained property do not participate
        g.query("CREATE (:S {z: 1})")
        g.query("CREATE (:S {z: 2})")

        self.env.assertEqual(g.query("MATCH (n:S) RETURN count(n)").result_set[0][0], 4)

    def test05_constraint_creation_over_existing_partial_data(self):
        g = self.g

        # partial entities must not block the constraint from becoming
        # operational...
        g.query("CREATE (:E {a: 1}), (:E {a: 1}), (:E {b: 2}), (:E {b: 2}), (:E {a: 3, b: 3})")
        create_unique_node_constraint(g, "E", "a", "b", sync=True)
        c = get_constraint(g, "UNIQUE", "NODE", "E", "a", "b")
        self.env.assertEqual(c.status, "OPERATIONAL")

        # ...but genuine duplicates must
        g.query("CREATE (:D {a: 1, b: 1}), (:D {a: 1, b: 1})")
        create_unique_node_constraint(g, "D", "a", "b", sync=True)
        c = get_constraint(g, "UNIQUE", "NODE", "D", "a", "b")
        self.env.assertEqual(c.status, "FAILED")

    def test06_partial_edges_accepted(self):
        g = self.g

        create_unique_edge_constraint(g, "R", "a", "b", sync=True)
        c = get_constraint(g, "UNIQUE", "RELATIONSHIP", "R", "a", "b")
        self.env.assertEqual(c.status, "OPERATIONAL")

        g.query("CREATE (:N {i: 1}), (:N {i: 2})")

        # two edges agreeing on 'a', both missing 'b'
        g.query("MATCH (x:N {i:1}), (y:N {i:2}) CREATE (x)-[:R {a: 1}]->(y)")
        g.query("MATCH (x:N {i:1}), (y:N {i:2}) CREATE (x)-[:R {a: 1}]->(y)")

        # a fully specified key is still enforced
        g.query("MATCH (x:N {i:1}), (y:N {i:2}) CREATE (x)-[:R {a: 2, b: 2}]->(y)")
        try:
            g.query("MATCH (x:N {i:1}), (y:N {i:2}) CREATE (x)-[:R {a: 2, b: 2}]->(y)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation, on edge of relationship-type R", str(e))

        self.env.assertEqual(g.query("MATCH ()-[r:R]->() RETURN count(r)").result_set[0][0], 3)

BULK_UNIQUE_GRAPH_ID = "unique_bulk_writes"

class testUniqueConstraintBulkWrites():
    """UNIQUE enforcement for write queries that touch many entities at once:
    duplicates among the query's own entities and against existing ones are
    caught, and enforcement is linear in the label size (it used to rescan the
    whole label for every affected entity)."""
    def __init__(self):
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.con.delete(BULK_UNIQUE_GRAPH_ID)
        self.g = self.db.select_graph(BULK_UNIQUE_GRAPH_ID)

    def _rejects(self, q, msg):
        try:
            self.g.query(q)
            self.env.assertTrue(False, message=q)
        except ResponseError as e:
            self.env.assertContains(msg, str(e))

    def _assert_operational(self):
        # a constraint that is not OPERATIONAL enforces nothing, which would
        # make every rejection check below meaningless
        constraints = list_constraints(self.g)
        self.env.assertGreaterEqual(len(constraints), 1)
        for c in constraints:
            self.env.assertEqual(c.status, "OPERATIONAL")

    def test01_nodes(self):
        g = self.g
        create_unique_node_constraint(g, "B", "v", sync=True)
        self._assert_operational()

        # 20k nodes in one query: seconds-to-minutes when enforcement was
        # quadratic, tens of milliseconds now
        start = time.time()
        g.query("UNWIND range(1, 20000) AS x CREATE (:B {v: x, id: x})")
        self.env.assertLess(time.time() - start, 10)

        msg = "unique constraint violation on node of type B"
        # duplicate within the query's own nodes
        self._rejects("UNWIND [-1, -2, -1] AS x CREATE (:B {v: x})", msg)
        # duplicate of an existing node, among many new ones
        self._rejects("UNWIND range(20001, 21000) + [500] AS x CREATE (:B {v: x})", msg)
        # an existing node updated to another existing node's value
        self._rejects("MATCH (n:B {v: 1}) SET n.v = 20000", msg)
        # the rejected SET changed nothing
        self.env.assertEqual(g.query("MATCH (n:B {id: 1}) RETURN n.v").result_set, [[1]])
        self.env.assertEqual(g.query("MATCH (n:B {v: 20000}) RETURN count(n)").result_set[0][0], 1)
        # swapping two values in one query leaves no duplicate
        g.query("MATCH (a:B {v: 1}), (b:B {v: 2}) SET a.v = 2, b.v = 1")
        self.env.assertEqual(g.query("MATCH (n:B {id: 1}) RETURN n.v").result_set, [[2]])
        self.env.assertEqual(g.query("MATCH (n:B {id: 2}) RETURN n.v").result_set, [[1]])
        self.env.assertEqual(g.query("MATCH (n:B) RETURN count(n)").result_set[0][0], 20000)

    def test02_edges(self):
        g = self.g
        create_unique_edge_constraint(g, "T", "v", sync=True)
        self._assert_operational()
        g.query("CREATE (:A), (:Z)")

        start = time.time()
        g.query("MATCH (a:A), (z:Z) UNWIND range(1, 20000) AS x CREATE (a)-[:T {v: x, id: x}]->(z)")
        self.env.assertLess(time.time() - start, 10)

        msg = "unique constraint violation, on edge of relationship-type T"
        self._rejects("MATCH (a:A), (z:Z) UNWIND [-1, -2, -1] AS x CREATE (a)-[:T {v: x}]->(z)", msg)
        self._rejects("MATCH (a:A), (z:Z) UNWIND range(20001, 21000) + [500] AS x CREATE (a)-[:T {v: x}]->(z)", msg)
        self._rejects("MATCH ()-[e:T {v: 1}]->() SET e.v = 20000", msg)
        # the rejected SET changed nothing
        self.env.assertEqual(g.query("MATCH ()-[e:T {id: 1}]->() RETURN e.v").result_set, [[1]])
        self.env.assertEqual(g.query("MATCH ()-[e:T {v: 20000}]->() RETURN count(e)").result_set[0][0], 1)
        self.env.assertEqual(g.query("MATCH ()-[e:T]->() RETURN count(e)").result_set[0][0], 20000)

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
                    if 'GRAPH.EFFECT' in cmd['command']:
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

        # Six constraints, not six anything-else. The number that used to be
        # here was 12: v2 replicated each `GRAPH.CONSTRAINT` *twice* — once on
        # creation and once more as the signal that validation had finished,
        # because the command had no way to carry a status. v3 carries the
        # status in the announcement, so the repeat is not a signal any more.
        #
        # Each of these is announced once because this graph is empty, so
        # validation runs inline on the main thread and the status is settled
        # before the command returns. Above the async threshold there are two
        # announcements — UNDER CONSTRUCTION, then the settled status — and
        # that is `testEffectsV3_03_ConstraintConvergence`, which asserts both
        # of them are CREATE_CONSTRAINT records rather than merely two effects.
        constraints = list_constraints(self.g)
        self.env.assertEqual(len(constraints), 6)
        for c in constraints:
            self.env.assertEqual(c.status, 'OPERATIONAL')

        # What is worth asserting here is *which command* carries a constraint to
        # a replica, and that the replica converges — not how many payloads went
        # past.
        #
        # Counting them was wrong twice over. A GRAPH.EFFECT payload is binary,
        # so MONITOR cannot tell a constraint's effect from a node-create's, and
        # the count is not six anyway: each `create_unique_*` helper builds a
        # supporting index first, which is another effect. Measured, these six
        # creates put eleven effects on the wire, so `>= 6` was passing with five
        # to spare and functioning as a sleep.
        #
        # `test_effects_ddl.py` pins the per-shape announcement counts, where
        # the payloads are read off the wire rather than counted through
        # MONITOR.
        self.source.execute_command("WAIT", 1, 0)

        replica_g = Graph(self.replica, GRAPH_ID)
        deadline = time.time() + 30
        while time.time() < deadline:
            self.source.execute_command("WAIT", 1, 0)
            try:
                cs = list_constraints(replica_g)
            except ResponseError as e:
                # The replica can answer before it has applied the effect that
                # creates the graph; keep polling until the deadline.
                if "empty key" not in str(e):
                    raise
                cs = []
            if len(cs) == 6 and all(c.status == 'OPERATIONAL' for c in cs):
                break
            time.sleep(0.25)

        replica_constraints = list_constraints(replica_g)
        self.env.assertEqual(len(replica_constraints), 6)
        for c in replica_constraints:
            self.env.assertEqual(c.status, 'OPERATIONAL')

        # And the mechanism: effects carried them.
        self.env.assertGreater(len(self.monitor), 0)

        # That no verbatim GRAPH.CONSTRAINT was replayed is asked of the
        # replica's own command counters rather than of MONITOR, so it does not
        # depend on what the filter above happens to collect — a filter that
        # only keeps GRAPH.EFFECT would make a MONITOR-based check of this
        # vacuously true, which reads like coverage and is not.
        stats = self.replica.execute_command("INFO", "commandstats")
        if not isinstance(stats, dict):
            stats = {
                k: v
                for k, v in (
                    line.split(':', 1) for line in str(stats).splitlines() if ':' in line
                )
            }
        replayed = [k for k in stats if 'constraint' in k.lower()]
        self.env.assertEqual(replayed, [],
            message=f"the replica executed a verbatim constraint command: {replayed}")

    def test_02_async_validation_reaches_operational_on_replica(self):
        # Regression guard for the pause/role re-check added in #2371.
        #
        # Constraint validation above the async threshold runs on a spawned
        # thread that escalates to writer. That escalation is deliberately NOT
        # re-authorized (`QuerySession::begin_preauthorized`), because the thread
        # runs on the replica too -- applying the master's replicated
        # GRAPH.CONSTRAINT -- where a read-only check would reject it and strand
        # the constraint at UNDER CONSTRUCTION while the master reports
        # OPERATIONAL. Nothing else covers the async path with a replica.
        #
        # Needs > 10_000 entities: below that `Graph::create_constraint`
        # validates synchronously on the main thread and never spawns.
        lbl = 'AsyncValidated'
        self.g.query(f"UNWIND range(1, 10500) AS x CREATE (:{lbl} {{v: x}})")
        self.source.execute_command("WAIT", 1, 0)

        create_mandatory_node_constraint(self.g, lbl, 'v', sync=True)

        replica_g = Graph(self.replica, GRAPH_ID)
        deadline = time.time() + 30
        c = None
        while time.time() < deadline:
            self.source.execute_command("WAIT", 1, 0)
            c = get_constraint(replica_g, 'MANDATORY', 'NODE', lbl, 'v')
            if c is not None and c.status != 'UNDER CONSTRUCTION':
                break
            time.sleep(0.25)

        master_c = get_constraint(self.g, 'MANDATORY', 'NODE', lbl, 'v')
        self.env.assertEqual(master_c.status, 'OPERATIONAL')
        self.env.assertIsNotNone(c)
        self.env.assertEqual(c.status, 'OPERATIONAL')


class testConstraintSchemaRegistration():
    """A constraint's label and properties have to be interned before it is stored.

       The constraint itself holds names, so nothing at runtime needs the ids — but the
       RDB stores it by attribute id (`encode_constraint_block` resolves each property
       with `position(..).unwrap_or(0)`), so a property no entity has ever used is
       persisted as id 0 and read back as whatever attribute 0 happens to be. That is
       #2749, and it is silent: the constraint keeps enforcing, on the wrong property.

       Not reachable through UNIQUE, which needs a supporting range index first and so
       interns the property on the way. MANDATORY on an empty label is the case."""

    def __init__(self):
        # enableDebugCommand: this reloads via DEBUG RELOAD, which redis refuses
        # by default.
        self.env, self.db = Env(env='oss', enableDebugCommand=True)
        self.con = self.env.getConnection()

    def test01_a_constraint_on_an_unused_property_survives_a_reload(self):
        g = self.db.select_graph("constraint_schema_registration")

        # `a` is interned first, so it is attribute id 0 — the id an unresolved
        # property falls back to. `Q` ends up an empty label, so MANDATORY on it
        # validates trivially and reaches OPERATIONAL, which is what gets encoded.
        g.query("CREATE (:P {a: 1})")
        g.query("CREATE (:Q {b: 1})")
        g.query("MATCH (n:Q) DELETE n")

        create_mandatory_node_constraint(g, "Q", "z", sync=True)

        # Interned by the create, not by any entity: no node has ever had `z`.
        keys = [r[0] for r in g.query("CALL db.propertyKeys()").result_set]
        self.env.assertContains("z", keys)

        before = g.query("CALL db.constraints() YIELD label, properties").result_set
        self.env.assertEqual(before, [["Q", ["z"]]])

        self.con.execute_command("DEBUG", "RELOAD")

        # Without the registration this comes back as ["a"] — attribute id 0.
        after = g.query("CALL db.constraints() YIELD label, properties").result_set
        self.env.assertEqual(after, before)

    def test02_a_refused_create_interns_nothing(self):
        g = self.db.select_graph("constraint_schema_refused")
        g.query("CREATE (:R {c: 1})")

        before = sorted(r[0] for r in g.query("CALL db.propertyKeys()").result_set)

        # UNIQUE without a supporting range index is refused. The registration
        # runs only after the create succeeds, so the name must not leak in.
        try:
            self.con.execute_command(
                "GRAPH.CONSTRAINT", "CREATE", "constraint_schema_refused",
                "UNIQUE", "NODE", "R", "PROPERTIES", "1", "neverseen")
            self.env.assertTrue(False, message="UNIQUE without an index must be refused")
        except ResponseError as e:
            self.env.assertContains("missing supporting exact-match index", str(e))

        after = sorted(r[0] for r in g.query("CALL db.propertyKeys()").result_set)
        self.env.assertEqual(after, before)


UNIQUE_LOOKUP_GRAPH_ID = "unique_constraint_lookup"

class testUniqueConstraintLookup():
    """A UNIQUE constraint is enforced by looking each written entity up in the
       supporting index, not by scanning the label for every written entity (#3330:
       the scan made a bulk insert quadratic, ~26s for 10k nodes over 10k). The index
       lags the commit being checked, so these cover the cases where what it holds and
       what the graph holds differ."""

    def __init__(self):
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.con.delete(UNIQUE_LOOKUP_GRAPH_ID)
        self.g = self.db.select_graph(UNIQUE_LOOKUP_GRAPH_ID)

    def _violates(self, q, msg="unique constraint violation on node of type U"):
        try:
            self.g.query(q)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains(msg, str(e))

    def _count(self, q):
        return self.g.query(q).result_set[0][0]

    def test01_bulk_insert_is_linear(self):
        g = self.g
        create_unique_node_constraint(g, "U", "id", sync=True)

        g.query("UNWIND range(1, 10000) AS i CREATE (:U {id: i})")
        start = time.time()
        g.query("UNWIND range(10001, 20000) AS i CREATE (:U {id: i})")
        self.env.assertLess(time.time() - start, 10)

        # a duplicate of an existing node, and within the batch, still fail
        self._violates("CREATE (:U {id: 7})")
        self._violates("UNWIND [30000, 30000] AS i CREATE (:U {id: i})")
        self.env.assertEqual(self._count("MATCH (n:U) RETURN count(n)"), 20000)

    def test02_set_onto_an_existing_value(self):
        self._violates("MATCH (n:U {id: 1}) SET n.id = 2")
        # swapping two values is fine: both are written, neither collides
        self.g.query("MATCH (a:U {id: 1}), (b:U {id: 2}) SET a.id = 2, b.id = 1")
        self.env.assertEqual(self._count("MATCH (n:U {id: 1}) RETURN count(n)"), 1)
        self.env.assertEqual(self._count("MATCH (n:U {id: 2}) RETURN count(n)"), 1)

    def test03_freed_value_is_reusable_in_the_same_query(self):
        # the index still holds the deleted / unlabelled node
        self.g.query("MATCH (n:U {id: 3}) DELETE n CREATE (:U {id: 3})")
        self.g.query("MATCH (n:U {id: 4}) REMOVE n:U CREATE (:U {id: 4})")
        # re-adding the label collides with the node that took the value
        self._violates("MATCH (n {id: 4}) WHERE NOT n:U SET n:U")
        self.env.assertEqual(self._count("MATCH (n:U {id: 4}) RETURN count(n)"), 1)

    def test04_earlier_commit_in_the_same_query(self):
        # the MATCH forces the first CREATE to commit before the second
        self._violates("CREATE (:U {id: 40000}) WITH 1 AS x "
                       "MATCH (m:U {id: 40000}) CREATE (:U {id: 40000})")
        self.env.assertEqual(self._count("MATCH (n:U {id: 40000}) RETURN count(n)"), 0)

    def test05_values_the_index_cannot_match(self):
        g = self.g
        # strings (NUL included) and booleans go through the index; arrays,
        # large integers, NaN and the empty string are checked by a scan
        for v in ["'abc'", "'a\\u0000b'", "'\\u0000'", "true", "[1, 2]",
                  "1152921504606846976", "0.0/0.0", "''"]:
            g.query(f"CREATE (:U {{id: {v}}})")
            self._violates(f"CREATE (:U {{id: {v}}})")

    def test06_composite(self):
        g = self.g
        create_unique_node_constraint(g, "C", "a", "b", sync=True)
        g.query("UNWIND range(1, 100) AS i CREATE (:C {a: i, b: 'x'})")
        g.query("CREATE (:C {a: 1, b: 'y'})")
        self._violates("CREATE (:C {a: 2, b: 'x'})",
                       "unique constraint violation on node of type C")

    def test07_edges(self):
        g = self.g
        create_unique_edge_constraint(g, "R", "id", sync=True)
        msg = "unique constraint violation, on edge of relationship-type R"
        g.query("CREATE (a:E), (b:E) WITH a, b UNWIND range(1, 1000) AS i "
                "CREATE (a)-[:R {id: i}]->(b)")
        self._violates("MATCH (a:E) WITH a LIMIT 1 CREATE (a)-[:R {id: 5}]->(a)", msg)
        self._violates("MATCH ()-[r:R {id: 1}]->() SET r.id = 2", msg)
        g.query("MATCH ()-[r:R {id: 1}]->() DELETE r WITH 1 AS x "
                "MATCH (a:E) WITH a LIMIT 1 CREATE (a)-[:R {id: 1}]->(a)")
        self.env.assertEqual(self._count("MATCH ()-[r:R]->() RETURN count(r)"), 1000)

    def test08_composite_bulk_insert_is_linear(self):
        # every node shares `tenant`, so its lookup returns the whole label for
        # each inserted node; the check has to read the `id` lookup instead
        g = self.g
        create_unique_node_constraint(g, "T", "tenant", "id", sync=True)
        msg = "unique constraint violation on node of type T"

        g.query("UNWIND range(1, 10000) AS i CREATE (:T {tenant: 1, id: i})")
        start = time.time()
        g.query("UNWIND range(10001, 20000) AS i CREATE (:T {tenant: 1, id: i})")
        self.env.assertLess(time.time() - start, 10)

        self._violates("CREATE (:T {tenant: 1, id: 7})", msg)
        self._violates("UNWIND [30000, 30000] AS i CREATE (:T {tenant: 1, id: i})", msg)
        g.query("CREATE (:T {tenant: 2, id: 7})")
        self.env.assertEqual(self._count("MATCH (n:T) RETURN count(n)"), 20001)

    def test09_composite_bulk_insert_with_string_ids_is_linear(self):
        # every node shares `tenant`; a string `id` is looked up by no native
        # column today, so each inserted node's only lookup is the tenant's,
        # which has to be read once per check, not once per node
        g = self.g
        create_unique_node_constraint(g, "S", "tenant", "id", sync=True)
        msg = "unique constraint violation on node of type S"

        g.query("UNWIND range(1, 10000) AS i CREATE (:S {tenant: 1, id: 'u' + toString(i)})")
        start = time.time()
        g.query("UNWIND range(10001, 20000) AS i CREATE (:S {tenant: 1, id: 'u' + toString(i)})")
        self.env.assertLess(time.time() - start, 10)

        self._violates("CREATE (:S {tenant: 1, id: 'u7'})", msg)
        g.query("CREATE (:S {tenant: 2, id: 'u7'})")
        self.env.assertEqual(self._count("MATCH (n:S) RETURN count(n)"), 20001)

# GRAPH.CONSTRAINT CREATE / DROP need both the GIL and the graph WRITE lock.
# run inline on the main thread, the command waited for the write lock there,
# and the whole server (PING included) froze for as long as a long read held
# the READ lock (#3781). the command now runs on a worker thread and waits for
# the write lock in GIL-yielding slices, so the main thread keeps running while
# it waits - and can change state the command relies on.
#
# every test keeps a few long reads holding the READ lock, sends the command
# behind them, and checks either that the server stayed responsive, or that a
# change made on the main thread during the wait (CLIENT PAUSE, DEL, a second
# CREATE) is re-checked before the command mutates the graph and replicates.

WRITE_LOCK_GRAPH_ID = "constraint_write_lock"
WRITE_LOCK_RENAMED_ID = WRITE_LOCK_GRAPH_ID + "_renamed"

# target duration of one lock-holding read
READ_TARGET_SEC = 3.0

CREATE_UNIQUE    = constraint_cmd("CREATE", WRITE_LOCK_GRAPH_ID, "UNIQUE",    "NODE", "P", "v")
CREATE_MANDATORY = constraint_cmd("CREATE", WRITE_LOCK_GRAPH_ID, "MANDATORY", "NODE", "P", "v")
DROP_UNIQUE      = constraint_cmd("DROP",   WRITE_LOCK_GRAPH_ID, "UNIQUE",    "NODE", "P", "v")

# shared harness: runs constraint commands behind calibrated long reads
class ConstraintBehindLongReads():
    def __init__(self, **env_args):
        # enough worker threads that the reads + the commands never queue on a
        # small runner
        self.env, self.db = Env(moduleArgs="THREAD_COUNT 8", **env_args)

        # timing assertions are unreliable under instrumentation
        if SANITIZER:
            self.env.skip()

        self.conn  = self.env.getConnection()
        self.graph = self.db.select_graph(WRITE_LOCK_GRAPH_ID)

        self.graph.query("CREATE ()")  # a read needs an existing graph
        n, self.read_dur = calibrate_long_read(self.graph, READ_TARGET_SEC)
        self.long_read = long_read_query(n)
        self.env.debugPrint(f"long read: n={n} dur={self.read_dur:.2f}s",
                            force=True)

    def setUp(self):
        self.conn.delete(WRITE_LOCK_GRAPH_ID, WRITE_LOCK_RENAMED_ID)
        with redis.Redis(host=self.env.host, port=self.env.port, db=1) as db1:
            db1.delete(WRITE_LOCK_GRAPH_ID)

    def _populate(self, unique_index=False, without_v=False):
        self.graph.query("UNWIND range(1, 50000) AS x CREATE (:P {v:x})")
        if without_v:
            # MANDATORY enforcement of v ends FAILED. keeps the validation
            # thread's re-announcement out of the pause test, which is about
            # the command's own GRAPH.EFFECT
            self.graph.query("CREATE (:P)")
        if unique_index:
            self.graph.query("CREATE INDEX FOR (p:P) ON (p.v)")
            wait_for_indices_to_sync(self.graph)

    # keep 3 long reads holding the READ lock, send `cmds` concurrently behind
    # them, optionally run `action` on another connection while they wait, and
    # (with probe=True) sample PING throughout.
    #
    # new reads stop before `action`: the ones in flight keep holding the lock,
    # so the commands are still waiting when the action lands, and no new read
    # hits a deleted / paused graph.
    #
    # `after` runs on the same connection as `action` once the commands have
    # replied; what it returns is kept in self.after_result.
    #
    # returns (replies, latencies, worst_ping)
    def _behind_long_reads(self, cmds, action=None, probe=False, after=None):
        host, port = self.env.host, self.env.port

        async def run():
            pool = async_pool(self.env, 64, socket_timeout=120)
            g = AsyncFalkorDB(connection_pool=pool).select_graph(WRITE_LOCK_GRAPH_ID)
            # one client per command, plus one to run `action`
            clients = [AsyncRedis(host=host, port=port, decode_responses=True,
                                  socket_timeout=120) for _ in cmds + [None]]
            admin = clients.pop()
            loop = asyncio.get_running_loop()

            ping = PingProbe(port, host=host) if probe else None
            if ping:
                await ping.start()

            reads = ReadLockHolder(g, self.long_read)
            reads.start()
            await asyncio.sleep(0.5)  # the reads hold the read lock

            async def send(client, cmd):
                t0 = loop.time()
                try:
                    reply = await client.execute_command(*cmd)
                except Exception as e:
                    reply = e
                return reply, loop.time() - t0
            cmd_tasks = []
            for client, cmd in zip(clients, cmds):
                cmd_tasks.append(asyncio.create_task(send(client, cmd)))
                await asyncio.sleep(0.05)  # keep the send order deterministic

            if action is not None:
                await asyncio.sleep(0.5)  # the commands are waiting
                await reads.stop()
                await action(admin)
            results = await asyncio.gather(*cmd_tasks)
            self.after_result = await after(admin) if after is not None else None

            await asyncio.sleep(0.3)  # keep probing a little past the replies
            worst = await ping.stop() if ping else 0.0
            read_errors = await reads.drain()

            for c in clients + [admin]:
                await c.aclose()
            await pool.aclose()
            return results, read_errors, worst

        results, read_errors, worst = asyncio.run(run())

        # the reads exist only to hold the lock; an error means the setup broke
        # (or the server died), which would make the assertions vacuous
        self.env.assertEqual(read_errors, [], message=str(read_errors[:3]))

        replies   = [r for r, _ in results]
        latencies = [l for _, l in results]
        self.env.debugPrint(f"replies={replies!r} latencies="
                f"{[round(l, 2) for l in latencies]} "
                f"worst_ping={worst*1000:.1f}ms", force=True)
        return replies, latencies, worst

    # run `cmd` behind the reads: it must reply `expected`, having really
    # waited for the read lock, and no PING may stall for a large part of that
    # wait (with the freeze, the worst PING ~= the command's latency)
    def _assert_waits_without_freezing(self, cmd, expected):
        replies, latencies, worst = self._behind_long_reads([cmd], probe=True)
        self.env.assertEqual(replies[0], expected)
        self.env.assertGreater(latencies[0], 0.3 * self.read_dur)
        self.env.assertLess(worst, 0.5 * latencies[0])

    # run `cmd` behind the reads with `action` landing while it waits: it must
    # fail with an error containing `expected_err`
    def _assert_aborted_by(self, cmd, action, expected_err):
        replies, _, _ = self._behind_long_reads([cmd], action=action)
        self.env.assertTrue(isinstance(replies[0], ResponseError),
                            message=repr(replies[0]))
        self.env.assertContains(expected_err, str(replies[0]))

    def _constraint_count(self, graph=None):
        if graph is None:
            graph = self.graph
        res = graph.ro_query("CALL db.constraints() YIELD type RETURN count(1)")
        return res.result_set[0][0]

    def _assert_operational(self, ct_type):
        wait_on_constraint(self.graph, ct_type, 'NODE', 'P', 'v')
        c = get_constraint(self.graph, ct_type, 'NODE', 'P', 'v')
        self.env.assertEqual(c.status, 'OPERATIONAL')

class testConstraintWriteLock(ConstraintBehindLongReads):
    def __init__(self):
        super().__init__()

    def test_01_create_unique_stays_responsive(self):
        self._populate(unique_index=True)
        self._assert_waits_without_freezing(CREATE_UNIQUE, "PENDING")
        self._assert_operational('UNIQUE')

    def test_02_create_mandatory_stays_responsive(self):
        self._populate()
        self._assert_waits_without_freezing(CREATE_MANDATORY, "PENDING")
        self._assert_operational('MANDATORY')

    def test_03_drop_stays_responsive(self):
        self._populate(unique_index=True)
        create_unique_node_constraint(self.graph, 'P', 'v', sync=True)
        self._assert_waits_without_freezing(DROP_UNIQUE, "OK")
        self.env.assertEqual(self._constraint_count(), 0)

    def test_04_duplicate_create_during_wait(self):
        # two identical CREATEs queue behind the same reads. the second must be
        # rejected once it holds the write lock, without disturbing the first
        self._populate(unique_index=True)
        replies, _, _ = self._behind_long_reads([CREATE_UNIQUE, CREATE_UNIQUE])

        errs = [r for r in replies if isinstance(r, ResponseError)]
        self.env.assertEqual(replies.count("PENDING"), 1)
        self.env.assertEqual(len(errs), 1)
        self.env.assertContains("Constraint already exists", str(errs[0]))

        self._assert_operational('UNIQUE')
        self.env.assertEqual(self._constraint_count(), 1)

        # the graph still accepts writes and enforces the constraint
        self.graph.query("CREATE (:P {v:-1})")
        try:
            self.graph.query("CREATE (:P {v:-1})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation", str(e))

    def test_05_watch_sees_the_commit(self):
        # a client WATCHes the graph while CREATE waits: CREATE's commit
        # modifies the key, so that client's EXEC after CREATE replies must
        # abort
        self._populate()
        async def watch(admin):
            await admin.execute_command("WATCH", WRITE_LOCK_GRAPH_ID)
        async def exec_after(admin):
            await admin.execute_command("MULTI")
            await admin.execute_command("PING")
            return await admin.execute_command("EXEC")

        replies, _, _ = self._behind_long_reads([CREATE_MANDATORY],
                action=watch, after=exec_after)
        self.env.assertEqual(replies[0], "PENDING")
        self.env.assertEqual(self.after_result, None)

# a server of its own with AOF on: a GRAPH.EFFECT replicated at the wrong time
# either crashes the server (inside a pause) or is replayed by DEBUG LOADAOF.
# no constraint is ever created here, so the AOF holds no enforcement
# re-announcements, which aren't re-checked against a deleted graph either
class testConstraintWriteLockStateChanges(ConstraintBehindLongReads):
    def __init__(self):
        super().__init__(enableDebugCommand=True)
        self.conn.config_set("appendonly", "yes")
        self._wait_for_aof_rewrite()

    def _wait_for_aof_rewrite(self, timeout=30):
        # turning AOF on starts a background rewrite; a rewrite can be
        # scheduled and not yet started, so wait for both
        deadline = time.time() + timeout
        while True:
            p = self.conn.info("persistence")
            if not (p.get("aof_rewrite_in_progress", 0) or
                    p.get("aof_rewrite_scheduled", 0)):
                return
            self.env.assertLess(time.time(), deadline,
                                message="AOF rewrite did not finish")
            time.sleep(0.1)

    def test_01_client_pause_during_wait(self):
        # a CLIENT PAUSE WRITE window opens while CREATE waits. CREATE must not
        # replicate GRAPH.EFFECT inside it: with AOF on, the server asserts on
        # propagation during a pause and crashes
        self._populate(without_v=True)

        # long enough to cover the whole wait; lifted right after CREATE replies
        pause_ms = int((3 * self.read_dur + 10) * 1000)
        async def pause(admin):
            await admin.execute_command("CLIENT", "PAUSE", pause_ms, "WRITE")

        try:
            self._assert_aborted_by(CREATE_MANDATORY, pause,
                                    "replica traffic is currently paused")
        finally:
            self.conn.execute_command("CLIENT", "UNPAUSE")

        self.env.assertTrue(self.conn.ping())
        self.env.assertEqual(self._constraint_count(), 0)

    # `action` deletes the graph's key, or moves the graph off it, while
    # CREATE waits. CREATE announces under the key it was sent to, and a
    # GRAPH.EFFECT creates a missing graph, so replicating there would bring
    # back an empty graph at that key on a replica (or an AOF replay) and
    # apply the constraint to it. CREATE must be refused, and replaying the
    # AOF must neither crash nor bring back a graph at the key.
    # `check_live` runs between the two, on the state in memory.
    def _assert_refused_when_key_loses_graph(self, action, check_live=None):
        self._populate()
        self._assert_aborted_by(CREATE_MANDATORY, action,
                                "graph was deleted or replaced")
        self.env.assertEqual(self.conn.exists(WRITE_LOCK_GRAPH_ID), 0)
        if check_live is not None:
            check_live()

        self.conn.execute_command("DEBUG", "LOADAOF")
        self.env.assertEqual(self.conn.exists(WRITE_LOCK_GRAPH_ID), 0)

    def test_02_graph_deleted_during_wait(self):
        async def delete(admin):
            await admin.delete(WRITE_LOCK_GRAPH_ID)

        self._assert_refused_when_key_loses_graph(delete)

    def test_03_graph_renamed_during_wait(self):
        async def rename(admin):
            await admin.rename(WRITE_LOCK_GRAPH_ID, WRITE_LOCK_RENAMED_ID)

        # the refused CREATE did not apply the constraint to the graph, now
        # at the new key: not in memory, and not after the AOF replay
        renamed = self.db.select_graph(WRITE_LOCK_RENAMED_ID)
        no_constraint = lambda: self.env.assertEqual(
            self._constraint_count(renamed), 0)
        self._assert_refused_when_key_loses_graph(rename, check_live=no_constraint)
        self.env.assertEqual(self.conn.exists(WRITE_LOCK_RENAMED_ID), 1)
        no_constraint()

    def test_04_graph_moved_during_wait(self):
        # MOVE doesn't go through the module's registry, so only the keyspace
        # can tell that the key CREATE was sent to no longer holds the graph
        async def move(admin):
            await admin.move(WRITE_LOCK_GRAPH_ID, 1)

        self._assert_refused_when_key_loses_graph(move)
        # the graph lives on in the other database
        with redis.Redis(host=self.env.host, port=self.env.port, db=1) as db1:
            self.env.assertEqual(db1.exists(WRITE_LOCK_GRAPH_ID), 1)

# a constraint on a large label is validated in the background: the settle reads
# one version of the graph and publishes its verdict on whatever version is
# current by then. a write that committed in between was not looked at by the
# settle, and was not enforced either while only OPERATIONAL constraints bound
# writes — so the constraint went OPERATIONAL over data that violates it (17 of
# 20 tries on a million-node label; never on C, which enforces a constraint from
# the moment it is created)
class testConstraintSettleRace():
    def __init__(self):
        self.env, self.db = Env()
        self.g = self.db.select_graph("constraint_settle_race")
        # above the size a constraint is validated inline at
        self.g.query("UNWIND range(0, 499999) AS x CREATE (:L {v: x})")
        create_node_range_index(self.g, "L", "v", sync=True)

    def _race(self, ct_type, create, violating_write, violations):
        con = self.env.getConnection()
        for _ in range(5):
            create(self.g, "L", "v")
            try:
                # refused, unless it gets in before the constraint is created
                self.g.query(violating_write)
            except ResponseError as e:
                self.env.assertContains("constraint violation", str(e))
            wait_on_constraint(self.g, ct_type, "NODE", "L", "v")
            c = get_constraint(self.g, ct_type, "NODE", "L", "v")
            bad = self.g.ro_query(violations).result_set[0][0]
            # OPERATIONAL means the data satisfies it
            self.env.assertFalse(c.status == "OPERATIONAL" and bad > 0,
                                 message=f"{ct_type} is OPERATIONAL over {bad} violating nodes")
            drop_constraint(self.g, ct_type, "NODE", "L", "v")
            con.execute_command("GRAPH.QUERY", "constraint_settle_race",
                                "MATCH (n:L) WHERE n.w = 1 DELETE n")

    def test01_mandatory(self):
        self._race("MANDATORY", create_mandatory_node_constraint,
                   "CREATE (:L {w: 1})",
                   "MATCH (n:L) WHERE n.v IS NULL RETURN count(n)")

    def test02_unique(self):
        self._race("UNIQUE", create_unique_node_constraint,
                   "CREATE (:L {v: 0, w: 1})",
                   "MATCH (n:L) WITH n.v AS v, count(*) AS c WHERE c > 1 RETURN count(v)")
