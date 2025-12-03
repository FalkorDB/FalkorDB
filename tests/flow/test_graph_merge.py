import re
from common import *
from index_utils import *

GRAPH_ID = "merge_1"

class testGraphMergeFlow():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    # Create a single node without any labels or properties.
    def test01_single_node_with_label(self):
        query = """MERGE (robert:Critic)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 1)
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.properties_set, 0)

    # Retry to create an existing entity.
    def test02_existing_single_node_with_label(self):
        query = """MERGE (robert:Critic)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 0)

    # Create a single node with two properties and no labels.
    def test03_single_node_with_properties(self):
        query = """MERGE (charlie { name: 'Charlie Sheen', age: 10 })"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.properties_set, 2)

    # Retry to create an existing entity.
    def test04_existing_single_node_with_properties(self):
        query = """MERGE (charlie { name: 'Charlie Sheen', age: 10 })"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 0)

    # Create a single node with both label and property.
    def test05_single_node_both_label_and_property(self):
        query = """MERGE (michael:Person { name: 'Michael Douglas' })"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 1)
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.properties_set, 1)

    # Retry to create an existing entity.
    def test06_existing_single_node_both_label_and_property(self):
        query = """MERGE (michael:Person { name: 'Michael Douglas' })"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 0)

    # Create a single edge and additional two nodes.
    def test07_merge_on_relationship(self):
        query = """MERGE (charlie:ACTOR)-[r:ACTED_IN]->(wallStreet:MOVIE)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 2)
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.properties_set, 0)
        self.env.assertEquals(result.relationships_created, 1)

    # Retry to create a single edge and additional two nodes.
    def test08_existing_merge_on_relationship(self):
        query = """MERGE (charlie:ACTOR)-[r:ACTED_IN]->(wallStreet:MOVIE)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 0)
        self.env.assertEquals(result.relationships_created, 0)

    # Update existing entity
    def test09_update_existing_node(self):
        query = """MERGE (charlie { name: 'Charlie Sheen' }) SET charlie.age = 11, charlie.lastname='Sheen' """
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 2)
        self.env.assertEquals(result.relationships_created, 0)

        query = """MATCH (charlie { name: 'Charlie Sheen' }) RETURN charlie.age, charlie.name, charlie.lastname"""
        actual_result = self.graph.query(query)
        expected_result = [[11, 'Charlie Sheen', 'Sheen']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Update new entity
    def test10_update_new_node(self):
        query = """MERGE (tamara:ACTOR { name: 'tamara tunie' }) SET tamara.age = 59, tamara.name = 'Tamara Tunie' """
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.properties_set, 3)
        self.env.assertEquals(result.relationships_created, 0)

        query = """MATCH (tamara:ACTOR { name: 'Tamara Tunie' }) RETURN tamara.name, tamara.age"""
        actual_result = self.graph.query(query)
        expected_result = [['Tamara Tunie', 59]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Create a single edge and additional two nodes.
    def test11_update_new_relationship(self):
        query = """MERGE (franklin:ACTOR { name: 'Franklin Cover' })-[r:ACTED_IN {rate:5.7}]->(almostHeroes:MOVIE) SET r.date=1998, r.rate=5.8"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.properties_set, 4)
        self.env.assertEquals(result.relationships_created, 1)

    # Update existing relation
    def test12_update_existing_edge(self):
        query = """MERGE (franklin:ACTOR { name: 'Franklin Cover' })-[r:ACTED_IN {rate:5.8, date:1998}]->(almostHeroes:MOVIE) SET r.date=1998, r.rate=5.9"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 1)
        self.env.assertEquals(result.properties_removed, 1)
        self.env.assertEquals(result.relationships_created, 0)

        query = """MATCH (franklin:ACTOR { name: 'Franklin Cover' })-[r:ACTED_IN {rate:5.9, date:1998}]->(almostHeroes:MOVIE) RETURN franklin.name, franklin.age, r.rate, r.date"""
        actual_result = self.graph.query(query)
        expected_result = [['Franklin Cover', None, 5.9, 1998]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Update multiple nodes
    def test13_update_multiple_nodes(self):
        query = """CREATE (:person {age:31}),(:person {age:31}),(:person {age:31}),(:person {age:31})"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 1)
        self.env.assertEquals(result.nodes_created, 4)
        self.env.assertEquals(result.properties_set, 4)

        query = """MERGE (p:person {age:31}) SET p.newprop=100"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 4)

        query = """MATCH (p:person) RETURN p.age, p.newprop"""
        actual_result = self.graph.query(query)
        expected_result = [[31, 100],
                           [31, 100],
                           [31, 100],
                           [31, 100]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Update multiple nodes
    def test14_merge_unbounded_pattern(self):
        query = """MERGE (p:person {age:31})-[:owns]->(d:dog {name:'max'})"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 1)
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.properties_set, 2)
        self.env.assertEquals(result.relationships_created, 1)

        # Although person with age 31 and dog with the name max exists,
        # specified pattern doesn't exists, as a result the entire pattern
        # will be created, if we were to support MATCH MERGE 'p' and 'd'
        # would probably be defined in the MATCH clause, as a result they're
        # bounded and won't be duplicated.
        query = """MERGE (p:person {age:31})-[:owns]->(d:dog {name:'max'})-[:eats]->(f:food {name:'Royal Canin'})"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 1)
        self.env.assertEquals(result.nodes_created, 3)
        self.env.assertEquals(result.properties_set, 3)
        self.env.assertEquals(result.relationships_created, 2)

    # Add node that matches pre-existing index
    def test15_merge_indexed_entity(self):
        # Create index
        create_node_range_index(self.graph, 'person', 'age', sync=True)

        count_query = """MATCH (p:person) WHERE p.age > 0 RETURN COUNT(p)"""
        result = self.graph.query(count_query)
        original_count = result.result_set[0][0]

        # Add one new person
        merge_query = """MERGE (p:person {age:40})"""
        result = self.graph.query(merge_query)
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.properties_set, 1)
        # Verify that one indexed node has been added
        result = self.graph.query(count_query)
        updated_count = result.result_set[0][0]
        self.env.assertEquals(updated_count, original_count+1)

        # Perform another merge that does not create an entity
        result = self.graph.query(merge_query)
        self.env.assertEquals(result.nodes_created, 0)

        # Verify that indexed node count is unchanged
        result = self.graph.query(count_query)
        updated_count = result.result_set[0][0]
        self.env.assertEquals(updated_count, original_count+1)

    # Update nodes based on non-constant inlined properties
    def test16_merge_dynamic_properties(self):
        # Create and verify a new node
        query = """MERGE (q:dyn {name: toUpper('abcde')}) RETURN q.name"""
        expected = [['ABCDE']]

        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 1)
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.properties_set, 1)

        self.env.assertEquals(result.result_set, expected)

        # Repeat the query and verify that no changes were introduced
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 0)

        # Verify that MATCH...MERGE on the same entity does not introduce changes
        query = """MATCH (q {name: 'ABCDE'}) MERGE (r {name: q.name}) RETURN r.name"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 0)
        self.env.assertEquals(result.result_set, expected)

    def test17_complex_merge_queries(self):
        self.graph.delete()
        # Beginning with an empty graph
        # Create a new pattern
        query = """MERGE (a:Person {name: 'a'}) MERGE (b:Person {name: 'b'}) MERGE (a)-[e:FRIEND {val: 1}]->(b) RETURN a.name, e.val, b.name"""
        result = self.graph.query(query)
        expected = [['a', 1, 'b']]

        # Verify the results
        self.env.assertEquals(result.labels_added, 1)
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.relationships_created, 1)
        self.env.assertEquals(result.properties_set, 3)
        self.env.assertEquals(result.result_set, expected)

        # Repeat the query and verify that no changes were introduced
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.relationships_created, 0)
        self.env.assertEquals(result.properties_set, 0)
        self.env.assertEquals(result.result_set, expected)

        # Verify that these entities are accessed properly with MATCH...MERGE queries
        query = """MATCH (a:Person {name: 'a'}), (b:Person {name: 'b'}) MERGE (a)-[e:FRIEND {val: 1}]->(b) RETURN a.name, e.val, b.name"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.relationships_created, 0)
        self.env.assertEquals(result.properties_set, 0)
        self.env.assertEquals(result.result_set, expected)

        # Verify that we can bind entities properly in variable-length traversals
        query = """MATCH (a)-[*]->(b) MERGE (a)-[e:FRIEND {val: 1}]->(b) RETURN a.name, e.val, b.name"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.relationships_created, 0)
        self.env.assertEquals(result.properties_set, 0)
        self.env.assertEquals(result.result_set, expected)

        # Verify UNWIND...MERGE does not recreate existing entities
        query = """UNWIND ['a', 'b'] AS names MERGE (a:Person {name: names}) RETURN a.name"""
        expected = [['b'], ['a']]

        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.relationships_created, 0)
        self.env.assertEquals(result.properties_set, 0)
        self.env.assertEquals(result.result_set, expected)

        # Merging entities from an UNWIND list
        query = """UNWIND ['a', 'b', 'c'] AS names MERGE (a:Person {name: names}) ON MATCH SET a.set_by = 'match' ON CREATE SET a.set_by = 'create' RETURN a.name, a.set_by ORDER BY a.name"""
        expected = [['a', 'match'],
                    ['b', 'match'],
                    ['c', 'create']]

        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.properties_set, 4)
        self.env.assertEquals(result.result_set, expected)

        # Connect 'c' to both 'a' and 'b' via a Friend relation
        # One thing to note here is that both `c` and `x` are bounded, which means
        # our current merge distinct validation inspect the created edge only using its relationship, properties and bounded
        # nodes! as such the first created edge is different from the second one (due to changes in the destination node).
        query = """MATCH (c:Person {name: 'c'}) MATCH (x:Person) WHERE x.name in ['a', 'b'] WITH c, x MERGE(c)-[:FRIEND]->(x)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 0)
        self.env.assertEquals(result.relationships_created, 2)

        # Verify function calls in MERGE do not recreate existing entities
        query = """UNWIND ['A', 'B'] AS names MERGE (a:Person {name: toLower(names)}) RETURN a.name"""
        expected = [['b'], ['a']]

        result = self.graph.query(query)
        self.env.assertEquals(result.labels_added, 0)
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.relationships_created, 0)
        self.env.assertEquals(result.properties_set, 0)
        self.env.assertEquals(result.result_set, expected)

        query = """MERGE (a:Person {name: 'a'}) ON MATCH SET a.set_by = 'match' ON CREATE SET a.set_by = 'create' MERGE (b:Clone {name: a.name + '_clone'}) ON MATCH SET b.set_by = 'match' ON CREATE SET b.set_by = 'create' RETURN a.name, a.set_by, b.name, b.set_by"""
        result = self.graph.query(query)
        expected = [['a', 'match', 'a_clone', 'create']]

        # Verify the results
        self.env.assertEquals(result.labels_added, 1)
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.properties_set, 2)
        self.env.assertEquals(result.result_set, expected)

    def test18_merge_unique_creations(self):
        # Create a new pattern with non-unique entities.
        query = """UNWIND ['newprop1', 'newprop2'] AS x MERGE ({v:x})-[:e]->(n {v:'newprop1'})"""
        result = self.graph.query(query)

        # Verify that every entity was created in both executions.
        self.env.assertEquals(result.nodes_created, 4)
        self.env.assertEquals(result.relationships_created, 2)
        self.env.assertEquals(result.properties_set, 4)

        # Repeat the query.
        result = self.graph.query(query)

        # Verify that no data was modified.
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.relationships_created, 0)
        self.env.assertEquals(result.properties_set, 0)

    def test19_merge_dependency(self):
        self.graph.delete()

        # Starting with an empty graph.
        # Create 2 nodes and connect them to one another.
        query = """MERGE (a:Person {name: 'a'}) MERGE (b:Person {name: 'b'}) MERGE (a)-[:FRIEND]->(b) MERGE (b)-[:FRIEND]->(a)"""
        result = self.graph.query(query)

        # Verify that every entity was created.
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.relationships_created, 2)
        self.env.assertEquals(result.properties_set, 2)

        # Repeat the query.
        result = self.graph.query(query)

        # Verify that no data was modified.
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.relationships_created, 0)
        self.env.assertEquals(result.properties_set, 0)

    def test20_merge_edge_dependency(self):
        self.graph.delete()

        # Starting with an empty graph.
        # Make sure the pattern ()-[]->()-[]->()-[]->() exists.
        query = """MERGE (a {v:1}) MERGE (b {v:2}) MERGE (a)-[:KNOWS]->(b) MERGE ()-[:KNOWS]->()-[:KNOWS]->()"""
        result = self.graph.query(query)

        # Verify that every entity was created.
        self.env.assertEquals(result.nodes_created, 5)
        self.env.assertEquals(result.relationships_created, 3)
        self.env.assertEquals(result.properties_set, 2)

        # Repeat the query.
        result = self.graph.query(query)

        # Verify that no data was modified.
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.relationships_created, 0)
        self.env.assertEquals(result.properties_set, 0)

    def test21_merge_scan(self):
        # Starting with an empty graph.
        # All node scan should see created nodes.
        self.graph.delete()
        query = """MERGE (a {v:1}) WITH a MATCH (n) MERGE (n)-[:KNOWS]->(m)"""
        result = self.graph.query(query)

        # Verify that every entity was created.
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.relationships_created, 1)
        self.env.assertEquals(result.properties_set, 1)

        # Starting with an empty graph.
        # Label scan should see created nodes.
        self.graph.delete()
        query = """MERGE (a:L {v:1}) WITH a MATCH (n:L) MERGE (n)-[:KNOWS]->(m)"""
        result = self.graph.query(query)

        # Verify that every entity was created.
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.relationships_created, 1)
        self.env.assertEquals(result.properties_set, 1)

    def test22_merge_label_scan(self):
        # Starting with an empty graph.
        # Make sure the pattern ()-[]->()-[]->()-[]->() exists.
        self.graph.delete()
        query = """MERGE (a {v:1}) MERGE (b {v:2}) MERGE (a)-[:KNOWS]->(b) WITH a AS c, b AS d MATCH (c)-[:KNOWS]->(d) MERGE (c)-[:LIKES]->(d)"""
        result = self.graph.query(query)

        # Verify that every entity was created.
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.relationships_created, 2)
        self.env.assertEquals(result.properties_set, 2)

        # Repeat the query.
        result = self.graph.query(query)

        # Verify that no data was modified.
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.relationships_created, 0)
        self.env.assertEquals(result.properties_set, 0)

    def test23_merge_var_traverse(self):
        # Starting with an empty graph.
        # Make sure the pattern ()-[]->()-[]->()-[]->() exists.
        self.graph.delete()
        query = """MERGE (a {v:1}) MERGE (b {v:2}) MERGE (a)-[:KNOWS]->(b) WITH a AS c, b AS d MATCH (c)-[:KNOWS*]->(d) MERGE (c)-[:LIKES]->(d)"""
        result = self.graph.query(query)

        # Verify that every entity was created.
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.relationships_created, 2)
        self.env.assertEquals(result.properties_set, 2)

        # Repeat the query.
        result = self.graph.query(query)

        # Verify that no data was modified.
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.relationships_created, 0)
        self.env.assertEquals(result.properties_set, 0)

    def test24_merge_merge_delete(self):
        # Merge followed by an additional merge and ending with a deletion
        # which doesn't have any data to operate on,
        # this used to trigger force lock release, as the delete didn't tried to acquire/release the lock
        self.graph.delete()
        query = """MERGE (user:User {name:'Sceat'}) WITH user UNWIND [1,2,3] AS sessionHash MERGE (user)-[:HAS_SESSION]->(newSession:Session {hash:sessionHash}) WITH DISTINCT user, collect(newSession.hash) as newSessionHash MATCH (user)-->(s:Session) WHERE NOT s.hash IN newSessionHash DELETE s"""
        result = self.graph.query(query)

        # Verify that every entity was created.
        self.env.assertEquals(result.nodes_created, 4)
        self.env.assertEquals(result.properties_set, 4)
        self.env.assertEquals(result.relationships_created, 3)

        # Repeat the query.
        result = self.graph.query(query)

        # Verify that no data was modified.
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 0)
        self.env.assertEquals(result.relationships_created, 0)

    def test25_merge_with_where(self):
        # Index the "L:prop) combination so that the MERGE tree will not have a filter op.
        create_node_range_index(self.graph, 'L', 'prop', sync=True)

        query = """MERGE (n:L {prop:1}) WITH n WHERE n.prop < 1 RETURN n.prop"""
        result = self.graph.query(query)
        plan = str(self.graph.explain(query))

        # Verify that the Filter op follows a Project op.
        self.env.assertTrue(re.search('Project\s+Filter', plan))

        # Verify that there is no Filter op after the Merge op.
        self.env.assertFalse(re.search('Merge\s+Filter', plan))

        # Verify that the entity was created and no results were returned.
        self.env.assertEquals(result.nodes_created, 1)
        self.env.assertEquals(result.properties_set, 1)

        # Repeat the query.
        result = self.graph.query(query)

        # Verify that no data was modified and no results were returned.
        self.env.assertEquals(result.nodes_created, 0)
        self.env.assertEquals(result.properties_set, 0)

    def test26_merge_set_invalid_property(self):
        self.graph.delete()
        query = """MATCH p=() MERGE () ON MATCH SET p.prop4 = 5"""
        result = self.graph.query(query)
        self.env.assertEquals(result.properties_set, 0)

    def test27_merge_create_invalid_entity(self):
        try:
            # Try to create a node with an invalid NULL property.
            query = """MERGE (n {v: NULL})"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            self.env.assertIn("Cannot merge node using null property value", str(e))
            pass

        # Verify that no entities were created.
        query = """MATCH (a) RETURN a"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, [])

        try:
            # Try to merge a node with a self-referential property.
            query = """MERGE (a:L {v: a.v})"""
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            self.env.assertIn("Cannot merge node using null property value", str(e))

    def test28_merge_reset_label_scan(self):
        # Starting with an empty graph.
        # Create 2 nodes and connect them to one another.
        self.graph.delete()
        query = """MERGE (a:Person {name: 'a'}) MERGE (b:Person {name: 'b'}) MERGE (a)-[:FRIEND]->(b)"""
        result = self.graph.query(query)

        # Verify that every entity was created.
        self.env.assertEquals(result.nodes_created, 2)
        self.env.assertEquals(result.relationships_created, 1)
        self.env.assertEquals(result.properties_set, 2)

        # Issue a query that forces a Cartesian Product to reset a Node by Label Scan
        # after a Merge operation has freed that scan once.
        query = """MERGE (a:Person {name: 'a'})-[:FRIEND]->() WITH NULL AS a MATCH (n1:Person), (n2:Person) MERGE (:NEW)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_created, 1)

    def test29_merge_resue(self):
        query = """
        CREATE (m:L1 {v: 'abc'})
        CREATE (u:L2 {v: 'x'})
        CREATE (n:L2 {v: 'y'})
        CREATE (:L2 {v: 'y'})
        CREATE (u)-[:R]->(m), (u)-[:R]->(m)
        CREATE (n)-[:R]->(m), (n)-[:R]->(m)"""
        self.graph.query(query)

        query = """
        MERGE (m:L1 {v: 'abc'})
        SET m.v = 'abcd'
        WITH m
        MATCH (u:L2 {v: 'x'})
        MATCH (n:L2 {v: 'y'})
        MERGE (u)-[:R]->(m)<-[:R]-(n)
        RETURN m.v, u.v, n.v"""

        res = self.graph.query(query)
        self.env.assertEquals(res.nodes_created, 0)
        self.env.assertEquals(res.relationships_created, 2)
        self.env.assertEquals(res.result_set, [['abcd', 'x', 'y'],['abcd', 'x', 'y']])

    def test30_record_clone_under_merge(self):
        # the following operations
        # 1. node label scan
        # 2. all node scan
        # 3. node by id seek
        # hold the child record clone and enrich it before returning to parent
        # if the parent is eager then these operation delete the child record
        # this can lead to free values that is used in other operation
        # this tests check that this operations using deep clone

        # Create data
        self.graph.query("""CREATE (:A {name:"A"}), (:B {id:"B"}), (:B {id:"B"})""")

        expected = {"name": "A", "id": "C"}
        
        # node label scan under merge
        query = """UNWIND [{name: "A", id: "C"}] AS x
                   MATCH (i:A {name:x.name})
                   WITH *
                   MATCH (m:B {id:"B"})
                   MERGE (m)-[:R]->(i)
                   RETURN x"""
        res = self.graph.query(query)
        self.env.assertEquals(res.result_set[0][0], expected)

        # all node scan under merge
        query = """UNWIND [{name: "A", id: "C"}] AS x
                   MATCH (i:A {name:x.name})
                   WITH *
                   MATCH (m)
                   MERGE (m)-[:R]->(i)
                   RETURN x"""
        res = self.graph.query(query)
        self.env.assertEquals(res.result_set[0][0], expected)

        # node by id seek under merge
        query = """UNWIND [{name: "A", id: "C"}] AS x
                   MATCH (i:A {name:x.name})
                   WITH *
                   MATCH (m)
                   WHERE id(m) > 0
                   MERGE (m)-[:R]->(i)
                   RETURN x"""
        res = self.graph.query(query)
        self.env.assertEquals(res.result_set[0][0], expected)

    def test31_alias_multiple_definition(self):
        # Redefinition of an alias by depicting L2 as a label of a
        # should raise an exception
        query = """MERGE ()-[:R2]->(a:L1)-[:R1]->(a:L2) RETURN *"""
        try:
            self.graph.explain(query)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("can't be redeclared in a MERGE clause" in str(e))

    def test32_reset_op(self):
        # MERGE operation register a reset function validate that it works as expected
        res = self.graph.query("CREATE (a:A), (b:B)")
        self.env.assertEquals(res.nodes_created, 2)

        res = self.graph.query("MATCH (a:A), (b:B) SET a:X MERGE (c:C) MERGE (d:D)")
        self.env.assertEquals(res.nodes_created, 2)
    
    def test33_merge_create_reserve_id(self):
        # MERGE and CREATE node id reservation should be done only if new node is created
        # ensure that only 21 nodes are created
        res = self.graph.query("UNWIND range(0, 10) AS i CREATE (:A {id: i}) MERGE (:B {id: i % 10})")
        self.env.assertEquals(res.nodes_created, 21)

        # ensure that only 11 nodes are created and no crash
        res = self.graph.query("UNWIND range(0, 10) AS i CREATE (:A {id: i}) MERGE (:B {id: i % 10})")
        self.env.assertEquals(res.nodes_created, 11)

    def test34_merge_handle_duplicates(self):
        # duplicates scheduled for creation should be matched
        # consider the following:
        # UNWIND [{a:1, b:1}, {a:1, b:2}] AS x
        # MERGE (n {v:x.a})
        # ON CREATE SET n.created = true
        # ON MATCH  SET n.matched = true
        #
        # given an empty graph the first record {a:1, b:1} will create the node
        # the second record {a:1, b:2} will be detected as a duplicate
        # MERGE need to match the second record applying the ON MATCH directive
        # on it and include it in its output

        q = """UNWIND [{a:1, b:1}, {a:1, b:2}, {a:1, b:3}] AS x
               MERGE (n {v:x.a})
               ON CREATE SET n.created = true
               ON MATCH  SET n.matched = true
               RETURN n.v, n.created, n.matched"""

        res = self.graph.query(q).result_set
        self.env.assertEquals(len(res), 3)
        for row in res:
            self.env.assertEquals(row[0], 1)
            self.env.assertEquals(row[1], True)
            self.env.assertEquals(row[2], True)

    def test35_inquery_rel_intro(self):
        # make sure relationship types introduced within a query
        # are visible to the merge clause once it is done commiting

        # in this query B is firstly created
        # then on the second time MERGE ()-[:B]->()
        # is called, the pattern is considered duplicated and as such
        # need to be matched

        # start with an empty graph
        self.graph.delete()

        q = """MERGE ()-[:A]->()
               WITH * MATCH (x)
               MERGE ()-[:B]->()
               ON MATCH SET x = {}
               RETURN count(1)"""

        res = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(res, 2)

        # clear the graph
        self.graph.delete()

        # this time the second merge pattern will use ExpandInto
        q = """MERGE (a)-[:A]->(b)
               WITH * MATCH (x)
               MERGE (a)-[:B]->(b)
               ON MATCH SET x = {}
               RETURN count(1)"""

        res = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(res, 2)

    def test36_reset_index(self):
        self.graph.delete()

        self.graph.query("CREATE INDEX FOR (n:N) ON(n.v)")

        res = self.graph.query("""UNWIND range(1, 10) AS x
                            MERGE (cv:N {v: toString(x % 2)})
                            ON CREATE SET
                              cv.x = x""")
        self.env.assertEquals(res.nodes_created, 2)
