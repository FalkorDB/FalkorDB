from common import *
from index_utils import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../demo/social')
import social_utils

GRAPH_ID = social_utils.graph_name


class testIndexScanFlow():
    def __init__(self):
        self.env, self.db = Env()

    def setUp(self):
        redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        social_utils.populate_graph(redis_con, self.graph)
        self.build_indices()

    def tearDown(self):
        self.graph.delete()

    def build_indices(self):
        self.graph.create_node_range_index('person', 'age')
        self.graph.create_node_range_index('country', 'name')
        wait_for_indices_to_sync(self.graph)

    # Validate that Cartesian products using index and label scans succeed
    def test01_cartesian_product_mixed_scans(self):
        query = "MATCH (p:person), (c:country) WHERE p.age > 0 RETURN p.age, c.name ORDER BY p.age, c.name"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertIn('Label Scan', plan)
        indexed_result = self.graph.query(query)

        query = "MATCH (p:person), (c:country) RETURN p.age, c.name ORDER BY p.age, c.name"
        plan = str(self.graph.explain(query))
        self.env.assertNotIn('Node By Index Scan', plan)
        self.env.assertIn('Label Scan', plan)
        unindexed_result = self.graph.query(query)

        self.env.assertEquals(indexed_result.result_set, unindexed_result.result_set)

    # Validate that Cartesian products using just index scans succeed
    def test02_cartesian_product_index_scans_only(self):
        query = "MATCH (p:person), (c:country) WHERE p.age > 0 AND c.name > '' RETURN p.age, c.name ORDER BY p.age, c.name"
        plan = str(self.graph.explain(query))
        # The two streams should both use index scans
        self.env.assertEquals(plan.count('Node By Index Scan'), 2)
        self.env.assertNotIn('Label Scan', plan)
        indexed_result = self.graph.query(query)

        query = "MATCH (p:person), (c:country) RETURN p.age, c.name ORDER BY p.age, c.name"
        plan = str(self.graph.explain(query))
        self.env.assertNotIn('Node By Index Scan', plan)
        self.env.assertIn('Label Scan', plan)
        unindexed_result = self.graph.query(query)

        self.env.assertEquals(indexed_result.result_set, unindexed_result.result_set)

    # Validate that the appropriate bounds are respected when a Cartesian product uses the same index in two streams
    def test03_cartesian_product_reused_index(self):
        create_node_range_index(self.graph, 'person', 'name', sync=True)
        query = "MATCH (a:person {name: 'Omri Traub'}), (b:person) WHERE b.age <= 30 RETURN a.name, b.name ORDER BY a.name, b.name"
        plan = str(self.graph.explain(query))
        # The two streams should both use index scans
        self.env.assertEquals(plan.count('Node By Index Scan'), 2)
        self.env.assertNotIn('Label Scan', plan)


        expected_result = [['Omri Traub', 'Gal Derriere'],
                           ['Omri Traub', 'Lucy Yanfital']]
        result = self.graph.query(query)

        self.env.assertEquals(result.result_set, expected_result)

    # Validate index utilization when filtering on a numeric field with the `IN` keyword.
    def test04_test_in_operator_numerics(self):
        # Validate the transformation of IN to multiple OR expressions.
        query = "MATCH (p:person) WHERE p.age IN [1,2,3] RETURN p"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)

        # Validate that nested arrays are not scanned in index.
        query = "MATCH (p:person) WHERE p.age IN [[1,2],3] RETURN p"
        plan = str(self.graph.explain(query))
        self.env.assertNotIn('Node By Index Scan', plan)
        self.env.assertIn('Label Scan', plan)

        # Validate the transformation of IN to multiple OR, over a range.
        query = "MATCH (p:person) WHERE p.age IN range(0,30) RETURN p.name ORDER BY p.name"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = [['Gal Derriere'], ['Lucy Yanfital']]
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

         # Validate the transformation of IN to empty index iterator.
        query = "MATCH (p:person) WHERE p.age IN [] RETURN p.name"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = []
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Validate the transformation of IN OR IN to empty index iterators.
        query = "MATCH (p:person) WHERE p.age IN [] OR p.age IN [] RETURN p.name"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = []
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Validate the transformation of multiple IN filters.
        query = "MATCH (p:person) WHERE p.age IN [26, 27, 30] OR p.age IN [33, 34, 35] RETURN p.name ORDER BY p.age"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = [['Gal Derriere'], ['Lucy Yanfital'], ['Omri Traub'], ['Noam Nativ']]
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Validate the transformation of multiple IN filters.
        query = "MATCH (p:person) WHERE p.age IN [26, 27, 30] OR p.age IN [33, 34, 35] OR p.age IN [] RETURN p.name ORDER BY p.age"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = [['Gal Derriere'], ['Lucy Yanfital'], ['Omri Traub'], ['Noam Nativ']]
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Validate the transformation of IN filters 1 not on attribute.
        query = """MATCH (p:person)
                   WHERE p.age IN [33, 34, 35]
                   RETURN ID(p)"""
        ids = self.graph.ro_query(query).result_set
        ids = [x[0] for x in ids]

        query = """MATCH (p:person)
                   WHERE id(p) IN $ids AND p.age IN [33, 34, 35]
                   RETURN p.name ORDER BY p.age"""

        plan = str(self.graph.explain(query, params={'ids': ids}))
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = [['Omri Traub'], ['Noam Nativ']]
        result = self.graph.ro_query(query, params={'ids': ids})
        self.env.assertEquals(result.result_set, expected_result)

    # Validate index utilization when filtering on string fields with the `IN` keyword.
    def test05_test_in_operator_string_props(self):
        # Build an index on the name property.
        create_node_range_index(self.graph, 'person', 'name', sync=True)
        # Validate the transformation of IN to multiple OR expressions over string properties.
        query = "MATCH (p:person) WHERE p.name IN ['Gal Derriere', 'Lucy Yanfital'] RETURN p.name ORDER BY p.name"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertNotIn('Label Scan', plan)

        expected_result = [['Gal Derriere'], ['Lucy Yanfital']]
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Combine numeric and string filters specified by IN.
        query = "MATCH (p:person) WHERE p.name IN ['Gal Derriere', 'Lucy Yanfital'] AND p.age in [30] RETURN p.name ORDER BY p.name"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertNotIn('Label Scan', plan)

        expected_result = [['Lucy Yanfital']]
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

         # Validate an empty index on IN with multiple indexes
        query = "MATCH (p:person) WHERE p.name IN [] OR p.age IN [] RETURN p.name"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = []
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Combine IN filters with other relational filters.
        query = "MATCH (p:person) WHERE p.name IN ['Gal Derriere', 'Lucy Yanfital'] AND p.name < 'H' RETURN p.name ORDER BY p.name"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertNotIn('Label Scan', plan)

        expected_result = [['Gal Derriere']]
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        query = "MATCH (p:person) WHERE p.name IN ['Gal Derriere', 'Lucy Yanfital'] OR p.age = 33 RETURN p.name ORDER BY p.name"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertNotIn('Label Scan', plan)

        expected_result = [['Gal Derriere'], ['Lucy Yanfital'], ['Omri Traub']]
        result = self.graph.query(query)
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

    # ',' is the default separator for tag indices
    # we've updated our separator to '\0' this test verifies issue 696:
    # https://github.com/RedisGraph/RedisGraph/issues/696
    def test06_tag_separator(self):
        # Create a single node with a long string property, introduce a comma as part of the string.
        query = """CREATE (:Node{value:"A ValuePartition is a pattern that describes a restricted set of classes from which a property can be associated. The parent class is used in restrictions, and the covering axiom means that only members of the subclasses may be used as values."})"""
        self.graph.query(query)

        # Index property.
        create_node_range_index(self.graph, 'Node', 'value', sync=True)

        # Make sure node is returned by index scan.
        query = """MATCH (a:Node{value:"A ValuePartition is a pattern that describes a restricted set of classes from which a property can be associated. The parent class is used in restrictions, and the covering axiom means that only members of the subclasses may be used as values."}) RETURN a"""
        plan = str(self.graph.explain(query))
        result_set = self.graph.query(query).result_set
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertEqual(len(result_set), 1)

    def test07_index_scan_and_id(self):
        self.graph.delete()
        self.graph.query("UNWIND range(0, 9) AS i CREATE (a:person {age: i})")

        query_result = create_node_range_index(self.graph, 'person', 'age', sync=True)
        self.env.assertEqual(1, query_result.indices_created)

        query = """MATCH (n:person)
                   WHERE id(n)>=7 AND n.age<9
                   RETURN id(n) ORDER BY n.age"""
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = self.graph.query(query)

        self.env.assertEqual(2, len(query_result.result_set))
        expected_result = [[7], [8]]
        self.env.assertEquals(expected_result, query_result.result_set)

    # Validate placement of index scans and filter ops when not all filters can be replaced.
    def test08_index_scan_multiple_filters(self):
        query = "MATCH (p:person) WHERE p.age = 30 AND NOT EXISTS(p.fakeprop) RETURN p.name"
        plan = str(self.graph.explain(query))
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertNotIn('Label Scan', plan)
        self.env.assertIn('Filter', plan)

        query_result = self.graph.query(query)
        expected_result = ["Lucy Yanfital"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

    def test09_index_scan_with_params(self):
        query = "MATCH (p:person) WHERE p.age = $age RETURN p.name"
        params = {'age': 30}
        plan = str(self.graph.explain(query, params=params))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(query, params=params)
        expected_result = ["Lucy Yanfital"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

    def test10_index_scan_with_param_array(self):
        query = "MATCH (p:person) WHERE p.age in $ages RETURN p.name"
        params = {'ages': [30]}
        plan = str(self.graph.explain(query, params=params))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(query, params=params)
        expected_result = ["Lucy Yanfital"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

    def test11_single_index_multiple_scans(self):
        query = "MERGE (p1:person {age: 40}) MERGE (p2:person {age: 41})"
        plan = str(self.graph.explain(query))
        # Two index scans should be performed.
        self.env.assertEqual(plan.count("Node By Index Scan"), 2)

        query_result = self.graph.query(query)
        # Two new nodes should be created.
        self.env.assertEquals(query_result.nodes_created, 2)

    def test12_remove_scans_before_index(self):
        query = "MATCH (a:person {age: 32})-[]->(b) WHERE (b:person)-[]->(a) RETURN a"
        plan = str(self.graph.explain(query))
        # One index scan should be performed.
        self.env.assertEqual(plan.count("Node By Index Scan"), 1)

    def test13_point_index_scan(self):
        # create index
        create_node_range_index(self.graph, 'restaurant', 'location', sync=True)

        # create restaurant
        q = "CREATE (:restaurant {location: point({latitude:30.27822306, longitude:-97.75134723})})"
        self.graph.query(q)

        # locate other restaurants within a 1000m radius
        q = """MATCH (r:restaurant)
        WHERE distance(r.location, point({latitude:30.27822306, longitude:-97.75134723})) < 1000
        RETURN r"""

        # make sure index is used
        plan = str(self.graph.explain(q))
        self.env.assertIn("Node By Index Scan", plan)

        # refine query from '<' to '<='
        q = """MATCH (r:restaurant)
        WHERE distance(r.location, point({latitude:30.27822306, longitude:-97.75134723})) <= 1000
        RETURN r"""

        # make sure index is used
        plan = str(self.graph.explain(q))
        self.env.assertIn("Node By Index Scan", plan)

        # index should NOT be used when searching for points outside of a circle
        # testing operand: '>', '>=' and '='
        q = """MATCH (r:restaurant)
        WHERE distance(r.location, point({latitude:30.27822306, longitude:-97.75134723})) > 1000
        RETURN r"""

        # make sure index is NOT used
        plan = str(self.graph.explain(q))
        self.env.assertNotIn("Node By Index Scan", plan)

        q = """MATCH (r:restaurant)
        WHERE distance(r.location, point({latitude:30.27822306, longitude:-97.75134723})) >= 1000
        RETURN r"""

        # make sure index is NOT used
        plan = str(self.graph.explain(q))
        self.env.assertNotIn("Node By Index Scan", plan)

        q = """MATCH (r:restaurant)
        WHERE distance(r.location, point({latitude:30.27822306, longitude:-97.75134723})) = 1000
        RETURN r"""

        # make sure index is NOT used
        plan = str(self.graph.explain(q))
        self.env.assertNotIn("Node By Index Scan", plan)

    def test14_index_scan_utilize_array(self):
        # Querying indexed properties using IN a constant array should utilize indexes.
        query = "MATCH (a:person) WHERE a.age IN [34, 33] RETURN a.name ORDER BY a.name"
        plan = str(self.graph.explain(query))
        # One index scan should be performed.
        self.env.assertEqual(plan.count("Node By Index Scan"), 1)
        query_result = self.graph.query(query)
        expected_result = [["Noam Nativ"],
                           ["Omri Traub"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Querying indexed properties using IN a generated array should utilize indexes.
        query = "MATCH (a:person) WHERE a.age IN range(33, 34) RETURN a.name ORDER BY a.name"
        plan = str(self.graph.explain(query))
        # One index scan should be performed.
        self.env.assertEqual(plan.count("Node By Index Scan"), 1)
        query_result = self.graph.query(query)
        expected_result = [["Noam Nativ"],
                           ["Omri Traub"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Querying indexed properties using IN a non-constant array should not utilize indexes.
        query = "MATCH (a:person)-[]->(b) WHERE a.age IN b.arr RETURN a"
        plan = str(self.graph.explain(query))
        # No index scans should be performed.
        self.env.assertEqual(plan.count("Label Scan"), 1)
        self.env.assertEqual(plan.count("Node By Index Scan"), 0)

        # Querying indexed properties using IN a with array with stop word.
        query = "CREATE (:country { name: 'a' })"
        self.graph.query(query)

        query = "MATCH (a:country) WHERE a.name IN ['a'] RETURN a.name ORDER BY a.name"
        plan = str(self.graph.explain(query))
        # One index scan should be performed.
        self.env.assertEqual(plan.count("Node By Index Scan"), 1)
        query_result = self.graph.query(query)
        expected_result = [['a']]
        self.env.assertEquals(query_result.result_set, expected_result)

        query = "MATCH (a:country { name: 'a' }) DELETE a"
        self.graph.query(query)

    # Test fulltext result scoring
    def test15_fulltext_result_scoring(self):
        g = Graph(self.env.getConnection(), 'fulltext_scoring')

        # create full-text index over label 'L', attribute 'v'
        create_node_fulltext_index(g, 'L', 'v', sync=True)

        # introduce 2 nodes
        g.query("create (:L {v:'hello world hello'})")
        g.query("create (:L {v:'hello world hello world'})")

        # query nodes using fulltext search
        q = """CALL db.idx.fulltext.queryNodes('L', 'hello world') YIELD node, score
               RETURN node.v, score
               ORDER BY score"""
        res = g.query(q)
        actual = res.result_set
        expected = [['hello world hello', 1.5], ['hello world hello world', 2]]
        self.env.assertEqual(expected, actual)

    def test16_runtime_index_utilization(self):
        # find all person nodes with age in the range 33-37
        # current age (x) should be resolved at runtime
        # index query should be constructed for each age value
        q = """UNWIND range(33, 37) AS x
        MATCH (p:person {age:x})
        RETURN p.name
        ORDER BY p.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["Noam Nativ"], ["Omri Traub"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # similar to the query above, only this time the filter is specified
        # by an OR condition
        q = """WITH 33 AS min, 34 AS max 
        MATCH (p:person)
        WHERE p.age = min OR p.age = max
        RETURN p.name
        ORDER BY p.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["Noam Nativ"], ["Omri Traub"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # find all person nodes with age equals 33 'x'
        # 'x' value is known only at runtime
        q = """WITH 33 AS x
        MATCH (p:person {age:x})
        RETURN p.name
        ORDER BY p.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["Omri Traub"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # find all person nodes with age equals x + 1
        # the expression x+1 is evaluated to the constant 33 only at runtime
        # expecting index query to be constructed at runtime
        q = """WITH 32 AS x
        MATCH (p:person)
        WHERE p.age = (x + 1)
        RETURN p.name
        ORDER BY p.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["Omri Traub"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # same idea as previous query only we've switched the position of the
        # operands, queried entity (p.age) is now on the right hand side of the
        # filter, expecting the same behavior
        q = """WITH 32 AS x
        MATCH (p:person)
        WHERE (x + 1) = p.age
        RETURN p.name
        ORDER BY p.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["Omri Traub"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # find all person nodes 'b' with age greater than node 'a'
        # a's age value is determined only at runtime
        # expecting index to be used to resolve 'b' nodes, index query should be
        # constructed at runtime
        q = """MATCH (a:person {name:'Omri Traub'})
        WITH a AS a
        MATCH (b:person)
        WHERE b.age > a.age
        RETURN b.name
        ORDER BY b.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["Noam Nativ"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # same idea as previous query, only this time we've switched filter
        # operands position, queries entity is on the right hand side
        q = """MATCH (a:person {name: 'Omri Traub'})
        WITH a AS a
        MATCH (b:person)
        WHERE a.age < b.age
        RETURN b.name
        ORDER BY b.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["Noam Nativ"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # check that the value is evaluated before sending it to index query
        q = """MATCH (b:person)
        WHERE b.age = rand()*0 + 32
        RETURN b.name
        ORDER BY b.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [['Ailon Velger'], ['Alon Fital'], ['Ori Laslo'], ['Roi Lipman'], ['Tal Doron']]
        self.env.assertEquals(query_result.result_set, expected_result)

        # check that the value is evaluated before sending it to index query
        q = """MATCH (a:person)
        WHERE a.age = toInteger('32')
        RETURN a.name
        ORDER BY a.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        self.env.assertEquals(query_result.result_set, expected_result)

        # TODO: The following query uses the "Value Hash Join" where it would be
        # better to use "Index Scan"
        q = """UNWIND range(33, 37) AS x MATCH (a:person {age:x}), (b:person {age:x}) RETURN a.name, b.name ORDER BY a.name, b.name"""

    def test17_runtime_index_utilization_array_values(self):
        # when constructing an index query at runtime it is possible to encounter
        # none indexable values e.g. Array, in which case the index will still be
        # utilize, producing every entity which was indexed with a none indexable value
        # to which the index scan operation will have to apply the original filter

        # create person nodes with array value for their 'age' attribute
        q = """CREATE (:person {age:[36], name:'leonard'}), (:person {age:[34], name:['maynard']})"""
        self.graph.query(q)

        # find all person nodes with age value of [36]
        q = """WITH [36] AS age MATCH (a:person {age:age}) RETURN a.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["leonard"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # find all person nodes with age > [33]
        q = """WITH [33] AS age MATCH (a:person) WHERE a.age > age RETURN a.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["leonard"], [["maynard"]]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # combine indexable value with none-indexable value index query
        q = """WITH [33] AS age, 'leonard' AS name MATCH (a:person) WHERE a.age >= age AND a.name = name RETURN a.name"""
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)
        query_result = self.graph.query(q)
        expected_result = [["leonard"]]
        self.env.assertEquals(query_result.result_set, expected_result)

    # test for https://github.com/RedisGraph/RedisGraph/issues/1980
    def test18_index_scan_inside_apply(self):
        create_node_range_index(self.graph, 'L1', 'id', sync=True)
        self.graph.query("UNWIND range(1, 5) AS v CREATE (:L1 {id: v})")
        result = self.graph.query("UNWIND range(1, 5) AS id OPTIONAL MATCH (u:L1{id: 5}) RETURN u.id")

        expected_result = [[5], [5], [5], [5], [5]]
        self.env.assertEquals(result.result_set, expected_result)

    def test19_index_scan_numeric_accuracy(self):
        create_node_range_index(self.graph, 'L1', 'id', sync=True)
        create_node_range_index(self.graph, 'L2', 'id1', 'id2', sync=True)
        self.graph.query("""UNWIND range(1, 5) AS v
                            CREATE (:L1 {id: 990000000262240068 + v})"""
        )

        self.graph.query("""UNWIND range(1, 5) AS v
                            CREATE (:L2 {id1: 990000000262240068 + v, id2: 990000000262240068 - v})"""
        )

        # test index search
        result = self.graph.query("""MATCH (u:L1{id: 990000000262240069})
                                     RETURN u.id"""
        )
        expected_result = [[990000000262240069]]
        self.env.assertEquals(result.result_set, expected_result)

        # test index search from child
        result = self.graph.query("""MATCH (u:L1)
                                     WITH min(u.id) AS id
                                     MATCH (u:L1{id: id})
                                     RETURN u.id"""
        )
        expected_result = [[990000000262240069]]
        self.env.assertEquals(result.result_set, expected_result)

        # test index search with or
        result = self.graph.query("""MATCH (u:L1)
                                     WHERE u.id = 990000000262240069 OR
                                           u.id = 990000000262240070
                                     RETURN u.id
                                     ORDER BY u.id"""
        )
        expected_result = [[990000000262240069], [990000000262240070]]
        self.env.assertEquals(result.result_set, expected_result)

        # test resetting index scan operation
        result = self.graph.query("""MATCH (u1:L1), (u2:L1)
                                     WHERE u1.id = 990000000262240069 AND
                                     (u2.id = 990000000262240070 OR u2.id = 990000000262240071)
                                     RETURN u1.id, u2.id
                                     ORDER BY u1.id, u2.id""")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEquals(result.result_set, expected_result)

        # test resetting index scan operation when using the consume from child function
        result = self.graph.query("""MATCH (u:L1)
                                     WITH min(u.id) as id
                                     MATCH (u1:L1), (u2:L1)
                                     WHERE u1.id = 990000000262240069 AND
                                     (u2.id = 990000000262240070 OR u2.id = 990000000262240071)
                                     RETURN u1.id, u2.id
                                     ORDER BY u1.id, u2.id""")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEquals(result.result_set, expected_result)

        # test resetting index scan operation when rebuild index is required
        result = self.graph.query("""MATCH (u:L1)
                                     WITH min(u.id) as id
                                     MATCH (u1:L1), (u2:L1)
                                     WHERE u1.id = id AND
                                     (u2.id = 990000000262240070 OR u2.id = 990000000262240071)
                                     RETURN u1.id, u2.id
                                     ORDER BY u1.id, u2.id""")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEquals(result.result_set, expected_result)

        # test index scan with 2 different attributes
        result = self.graph.query("""MATCH (u:L2)
                                     WHERE u.id1 = 990000000262240069 AND
                                           u.id2 = 990000000262240067
                                     RETURN u.id1, u.id2""")
        expected_result = [[990000000262240069, 990000000262240067]]
        self.env.assertEquals(result.result_set, expected_result)

    def test20_index_scan_stopwords(self):
        #-----------------------------------------------------------------------
        # create indices
        #-----------------------------------------------------------------------

        # create exact match index over User id
        create_node_range_index(self.graph, 'User', 'id', sync=True)
        # create a fulltext index over User id
        create_node_fulltext_index(self.graph, 'User', 'id', sync=True)

        #-----------------------------------------------------------------------
        # create node
        #-----------------------------------------------------------------------

        # create a User node with a RediSearch stopword as the id attribute
        self.graph.query("CREATE (:User {id:'not'})")

        #-----------------------------------------------------------------------
        # query indices
        #-----------------------------------------------------------------------

        # query exact match index for user
        # expecting node to return as stopwords are not enforced
        result = self.graph.query("MATCH (u:User {id: 'not'}) RETURN u")
        user = Node(labels='User', properties={'id': 'not'})
        self.env.assertEquals(result.result_set[0][0], user)

        # query fulltext index for user
        # expecting no results as stopwords are enforced
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('User', 'stop')")
        self.env.assertEquals(result.result_set, [])
    
    def test21_invalid_distance_query(self):
        # create exact match index over User id
        create_node_range_index(self.graph, 'User', 'loc', sync=True)
        
        # create a node
        self.graph.query("CREATE (:User {loc:point({latitude:40.4, longitude:30.3})})")

        # invalid query
        try:
            self.graph.query("MATCH (u:User) WHERE distance(point({latitude:40.5, longitude: 30.4}, u.loc)) < 20000 RETURN u")
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("Received 1 arguments to function 'distance', expected at least 2", str(e))

    def test_22_pickup_on_index_creation(self):
        g = Graph(self.env.getConnection(), 'late_index_creation')

        # create graph
        g.query("RETURN 1")

        # issue query which has to potential to utilize an index
        # this query is going to be cached
        q = "MATCH (n:N) WHERE n.v = 1 RETURN n"
        plan = str(g.explain(q))

        # expecting no index scan operation, as we've yet to create an index
        self.env.assertNotIn('Node By Index Scan', plan)

        # create an index
        resultset = create_node_range_index(g, 'N', 'v', sync=True)
        self.env.assertEqual(1, resultset.indices_created)

        # re-issue the same query
        q = "MATCH (n:N) WHERE n.v = 1 RETURN n"
        plan = str(g.explain(q))

        # expecting an index scan operation
        self.env.assertIn('Node By Index Scan', plan)

    def test_23_do_not_utilize_index_(self):
        # create graph
        self.graph.query("RETURN 1")

        # issue query which not utilize an index
        q = "MATCH (n:N) WHERE id(n) IN [0] RETURN n"
        plan = str(self.graph.explain(q))

        # expecting no index scan operation
        self.env.assertNotIn('Node By Index Scan', plan)

        # create an index
        resultset = create_node_range_index(self.graph, 'N', 'v', sync=True)
        self.env.assertEqual(1, resultset.indices_created)

        # re-issue the same query
        q = "MATCH (n:N) WHERE id(n) IN [0] RETURN n"
        plan = str(self.graph.explain(q))

        # expecting an no index scan operation
        self.env.assertNotIn('Node By Index Scan', plan)

    def test_24_multitype_index(self):
        # create index with multiple types
        # 1. RANGE
        # 2. FULLTEXT
        create_node_range_index(self.graph, 'person', 'name')
        create_node_fulltext_index(self.graph, 'person', 'name', sync=True)

        # make sure we can search index using both range and fulltext

        # search using range
        q = "MATCH (p:person) WHERE p.name = 'Ailon Velger' RETURN p.name"
        plan = str(self.graph.explain(q))
        self.env.assertIn('Node By Index Scan', plan)

        result = self.graph.query(q).result_set
        self.env.assertEqual(len(result), 1)

        # search usign fulltext
        q = """CALL db.idx.fulltext.queryNodes('person', 'A*') YIELD node
               WITH node.name AS name
               ORDER BY name
               RETURN collect(name)"""
        names = self.graph.query(q).result_set[0][0]
        self.env.assertIn('Alon Fital', names)
        self.env.assertIn('Ailon Velger', names)
        self.env.assertIn('Boaz Arad', names)
        self.env.assertIn('Valerie Abigail Arad', names)

    def test_25_unescaped_string(self):
        # make sure range index doesn't alter strings

        self.graph.create_node_range_index("Page", "url")

        # create node with escapable string
        url = "http://mapper.acme.com/?ll=35.33977,-111.64513&z=13&t=R&marker0=37.17417%2C-113.32611%2CHurricane%2C%20Utah&marker1=33.34950%2C-110.99262%2CTop%20of%20the%20World\\%2C%20AZ&marker2=36.14666%2C-111.03209%2CCoal%20Mine%20Canyon\\%2C%20AZ&marker3=34.72585%2C-111.55126%2CApache%20Maid%20Mountain&marker4=35.34640%2C-111.67849%2CSan%20Francisco%20Peaks\\%2C%20Flagstaff&marker5=33.46420%2C-113.36047%2CCourthouse%20Rock\\%2C%20Maricopa%20County%20AZ&marker6=35.64082%2C-115.35916%2CRoach\\%2C%20Nevada&marker7=37.25276%2C-113.36773%2CSilver%20Reef&marker8=35.33001%2C-111.64627%2CDoyle%20Peak\\%2C%20Arizona&marker9=34.20086%2C-112.15294%2CBumble%20Bee\\%2C%20AZ"
        params = {'url': url}
        res = self.graph.query("CREATE (:Page {url: $url})", params)
        self.env.assertEqual(res.nodes_created, 1)

        # make sure we're able to locate node using index scan
        res = self.graph.query("MATCH (p:Page {url: $url}) RETURN p.url", params)
        self.env.assertEqual(url, res.result_set[0][0])

    def test_26_index_scan_with_other_filters(self):
        # make sure index is utilized when the compared value is not a trivial
        # expression e.g. p.name = metadata.age

        queries = ["""UNWIND [{age: 30}, {age: 31}] AS metadata
                      MATCH (p:person)
                      WHERE p.age = metadata.age
                      RETURN p.name""",
                    
                    # different property name
                    """UNWIND [{my_age: 30}, {my_age: 31}] AS metadata
                      MATCH (p:person)
                      WHERE p.age = metadata.my_age
                      RETURN p.name""",

                   # same as previous query only reversed filter operands
                   """UNWIND [{age: 30}, {age: 31}] AS metadata
                      MATCH (p:person)
                      WHERE metadata.age = p.age
                      RETURN p.name""",

                   """UNWIND [ [-1, 30], [0, 31] ] AS metadata
                      MATCH (p:person)
                      WHERE p.age = metadata[1]
                      RETURN p.name""",

                   # same as previous query only reversed filter operands
                   """UNWIND [ [-1, 30], [0, 31] ] AS metadata
                      MATCH (p:person)
                      WHERE metadata[1] = p.age
                      RETURN p.name""",

                   """UNWIND [ [-1, 30], [0, 31] ] AS metadata
                      MATCH (p:person)
                      WHERE p.age = metadata[0] + metadata[1]
                      RETURN p.name""",

                   # same as previous query only reversed filter operands
                   """UNWIND [ [-1, 30], [0, 31] ] AS metadata
                      MATCH (p:person)
                      WHERE  metadata[0] + metadata[1] = p.age
                      RETURN p.name""",

                   """UNWIND [{age: 30}, {age: 31}] AS metadata
                      MATCH (p:person)
                      WHERE p.age = metadata.age + metadata.age
                      RETURN p.name""",

                   # same as previous query only reversed filter operands
                   """UNWIND [{age: 30}, {age: 31}] AS metadata
                      MATCH (p:person)
                      WHERE metadata.age + metadata.age = p.age
                      RETURN p.name"""]

        for q in queries:
            plan = self.graph.explain(q)
            self.env.assertIn('Node By Index Scan', plan)

    def test_27_multi_index_scan(self):
        # make sure multiple index scans are utilized

        queries = ["""UNWIND [{age: 30}, {age: 31}] AS metadata
                      MATCH (a:person), (b:person)
                      WHERE a.age = metadata.age AND b.age = metadata.age + 1
                      RETURN a.name, b.name""",

                    # different property name
                    """UNWIND [{my_age: 30}, {my_age: 31}] AS metadata
                      MATCH (a:person), (b:person)
                      WHERE a.age = metadata.my_age AND b.age = metadata.my_age
                      RETURN a.name, b.name""",

                   # same as previous query only reversed filter operands
                   """UNWIND [{age: 30}, {age: 31}] AS metadata
                      MATCH (a:person), (b:person)
                      WHERE metadata.my_age = a.age AND metadata.my_age = b.age
                      RETURN a.name, b.name""",

                   """UNWIND [ [-1, 30], [0, 31] ] AS metadata
                      MATCH (a:person), (b:person)
                      WHERE a.age = metadata[1] AND b.age = metadata[0]
                      RETURN a.name, b.name""",

                   # same as previous query only reversed filter operands
                   """UNWIND [ [-1, 30], [0, 31] ] AS metadata
                      MATCH (a:person), (b:person)
                      WHERE metadata[1] = a.age AND metadata[0] = b.age
                      RETURN a.name, b.name""",

                   """UNWIND [ [-1, 30], [0, 31] ] AS metadata
                      MATCH (a:person), (b:person)
                      WHERE a.age = metadata[0] + metadata[1] AND
                            b.age = metadata[1] - metadata[0]
                      RETURN a.name, b.name""",

                   # same as previous query only reversed filter operands
                   """UNWIND [ [-1, 30], [0, 31] ] AS metadata
                      MATCH (a:person), (b:person)
                      WHERE metadata[0] + metadata[1] = a.age AND
                            metadata[1] - metadata[0] = b.age
                      RETURN a.name, b.name""",

                   """UNWIND [{age: 30}, {age: 31}] AS metadata
                      MATCH (a:person), (b:person)
                      WHERE a.age = metadata.age + metadata.age AND
                            b.age = metadata.age + metadata.age
                      RETURN a.name, b.name""",

                   # same as previous query only reversed filter operands
                   """UNWIND [{age: 30}, {age: 31}] AS metadata
                      MATCH (a:person), (b:person)
                      WHERE metadata.age + metadata.age = a.age AND
                            metadata.age + metadata.age = b.age
                      RETURN a.name, b.name"""]

        for q in queries:
            plan = self.graph.explain(q)
            self.env.assertIn('Node By Index Scan', plan)

