from common import *
from index_utils import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../demo/social')
import social_utils

redis_graph = None


class testIndexScanFlow():
    def __init__(self):
        self.env = Env(decodeResponses=True)

    def setUp(self):
        global redis_graph
        redis_con = self.env.getConnection()
        redis_graph = Graph(redis_con, social_utils.graph_name)
        social_utils.populate_graph(redis_con, redis_graph)
        self.build_indices()

    def tearDown(self):
        self.env.cmd('flushall')

    def build_indices(self):
        redis_graph.query("CREATE INDEX ON :person(age)")
        redis_graph.query("CREATE INDEX ON :country(name)")
        wait_for_indices_to_sync(redis_graph)

    # Validate that Cartesian products using index and label scans succeed
    def test01_cartesian_product_mixed_scans(self):
        query = "MATCH (p:person), (c:country) WHERE p.age > 0 RETURN p.age, c.name ORDER BY p.age, c.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertIn('Label Scan', plan)
        indexed_result = redis_graph.query(query)

        query = "MATCH (p:person), (c:country) RETURN p.age, c.name ORDER BY p.age, c.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertNotIn('Node By Index Scan', plan)
        self.env.assertIn('Label Scan', plan)
        unindexed_result = redis_graph.query(query)

        self.env.assertEquals(indexed_result.result_set, unindexed_result.result_set)

    # Validate that Cartesian products using just index scans succeed
    def test02_cartesian_product_index_scans_only(self):
        query = "MATCH (p:person), (c:country) WHERE p.age > 0 AND c.name > '' RETURN p.age, c.name ORDER BY p.age, c.name"
        plan = redis_graph.execution_plan(query)
        # The two streams should both use index scans
        self.env.assertEquals(plan.count('Node By Index Scan'), 2)
        self.env.assertNotIn('Label Scan', plan)
        indexed_result = redis_graph.query(query)

        query = "MATCH (p:person), (c:country) RETURN p.age, c.name ORDER BY p.age, c.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertNotIn('Node By Index Scan', plan)
        self.env.assertIn('Label Scan', plan)
        unindexed_result = redis_graph.query(query)

        self.env.assertEquals(indexed_result.result_set, unindexed_result.result_set)

    # Validate that the appropriate bounds are respected when a Cartesian product uses the same index in two streams
    def test03_cartesian_product_reused_index(self):
        create_node_range_index(redis_graph, 'person', 'name', sync=True)
        query = "MATCH (a:person {name: 'Omri Traub'}), (b:person) WHERE b.age <= 30 RETURN a.name, b.name ORDER BY a.name, b.name"
        plan = redis_graph.execution_plan(query)
        # The two streams should both use index scans
        self.env.assertEquals(plan.count('Node By Index Scan'), 2)
        self.env.assertNotIn('Label Scan', plan)


        expected_result = [['Omri Traub', 'Gal Derriere'],
                           ['Omri Traub', 'Lucy Yanfital']]
        result = redis_graph.query(query)

        self.env.assertEquals(result.result_set, expected_result)

    # Validate index utilization when filtering on a numeric field with the `IN` keyword.
    def test04_test_in_operator_numerics(self):
        # Validate the transformation of IN to multiple OR expressions.
        query = "MATCH (p:person) WHERE p.age IN [1,2,3] RETURN p"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)

        # Validate that nested arrays are not scanned in index.
        query = "MATCH (p:person) WHERE p.age IN [[1,2],3] RETURN p"
        plan = redis_graph.execution_plan(query)
        self.env.assertNotIn('Node By Index Scan', plan)
        self.env.assertIn('Label Scan', plan)

        # Validate the transformation of IN to multiple OR, over a range.
        query = "MATCH (p:person) WHERE p.age IN range(0,30) RETURN p.name ORDER BY p.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = [['Gal Derriere'], ['Lucy Yanfital']]
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

         # Validate the transformation of IN to empty index iterator.
        query = "MATCH (p:person) WHERE p.age IN [] RETURN p.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = []
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Validate the transformation of IN OR IN to empty index iterators.
        query = "MATCH (p:person) WHERE p.age IN [] OR p.age IN [] RETURN p.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = []
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Validate the transformation of multiple IN filters.
        query = "MATCH (p:person) WHERE p.age IN [26, 27, 30] OR p.age IN [33, 34, 35] RETURN p.name ORDER BY p.age"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = [['Gal Derriere'], ['Lucy Yanfital'], ['Omri Traub'], ['Noam Nativ']]
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Validate the transformation of multiple IN filters.
        query = "MATCH (p:person) WHERE p.age IN [26, 27, 30] OR p.age IN [33, 34, 35] OR p.age IN [] RETURN p.name ORDER BY p.age"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = [['Gal Derriere'], ['Lucy Yanfital'], ['Omri Traub'], ['Noam Nativ']]
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Validate the transformation of IN filters 1 not on attribute.
        query = "MATCH (p:person) WHERE id(p) IN [18, 26] AND p.age IN [33, 34, 35] RETURN p.name ORDER BY p.age"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = [['Omri Traub'], ['Noam Nativ']]
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

    # Validate index utilization when filtering on string fields with the `IN` keyword.
    def test05_test_in_operator_string_props(self):
        # Build an index on the name property.
        create_node_range_index(redis_graph, 'person', 'name', sync=True)
        # Validate the transformation of IN to multiple OR expressions over string properties.
        query = "MATCH (p:person) WHERE p.name IN ['Gal Derriere', 'Lucy Yanfital'] RETURN p.name ORDER BY p.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertNotIn('Label Scan', plan)

        expected_result = [['Gal Derriere'], ['Lucy Yanfital']]
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Combine numeric and string filters specified by IN.
        query = "MATCH (p:person) WHERE p.name IN ['Gal Derriere', 'Lucy Yanfital'] AND p.age in [30] RETURN p.name ORDER BY p.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertNotIn('Label Scan', plan)

        expected_result = [['Lucy Yanfital']]
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

         # Validate an empty index on IN with multiple indexes
        query = "MATCH (p:person) WHERE p.name IN [] OR p.age IN [] RETURN p.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)

        expected_result = []
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        # Combine IN filters with other relational filters.
        query = "MATCH (p:person) WHERE p.name IN ['Gal Derriere', 'Lucy Yanfital'] AND p.name < 'H' RETURN p.name ORDER BY p.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertNotIn('Label Scan', plan)

        expected_result = [['Gal Derriere']]
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

        query = "MATCH (p:person) WHERE p.name IN ['Gal Derriere', 'Lucy Yanfital'] OR p.age = 33 RETURN p.name ORDER BY p.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertNotIn('Label Scan', plan)

        expected_result = [['Gal Derriere'], ['Lucy Yanfital'], ['Omri Traub']]
        result = redis_graph.query(query)
        result = redis_graph.query(query)
        self.env.assertEquals(result.result_set, expected_result)

    # ',' is the default separator for tag indices
    # we've updated our separator to '\0' this test verifies issue 696:
    # https://github.com/RedisGraph/RedisGraph/issues/696
    def test06_tag_separator(self):
        redis_con = self.env.getConnection()
        redis_graph = Graph(redis_con, "G")

        # Create a single node with a long string property, introduce a comma as part of the string.
        query = """CREATE (:Node{value:"A ValuePartition is a pattern that describes a restricted set of classes from which a property can be associated. The parent class is used in restrictions, and the covering axiom means that only members of the subclasses may be used as values."})"""
        redis_graph.query(query)

        # Index property.
        create_node_range_index(redis_graph, 'Node', 'value', sync=True)

        # Make sure node is returned by index scan.
        query = """MATCH (a:Node{value:"A ValuePartition is a pattern that describes a restricted set of classes from which a property can be associated. The parent class is used in restrictions, and the covering axiom means that only members of the subclasses may be used as values."}) RETURN a"""
        plan = redis_graph.execution_plan(query)
        result_set = redis_graph.query(query).result_set
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertEqual(len(result_set), 1)

    def test07_index_scan_and_id(self):
        redis_con = self.env.getConnection()
        redis_graph = Graph(redis_con, "G")
        nodes=[]
        for i in range(10):
            node = Node(node_id=i, label='person', properties={'age':i})
            nodes.append(node)
            redis_graph.add_node(node)
            redis_graph.flush()
        
        query_result = create_node_range_index(redis_graph, 'person', 'age', sync=True)
        self.env.assertEqual(1, query_result.indices_created)

        query = """MATCH (n:person) WHERE id(n)>=7 AND n.age<9 RETURN n ORDER BY n.age"""
        plan = redis_graph.execution_plan(query)
        query_result = redis_graph.query(query)
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertIn('Filter', plan)
        query_result = redis_graph.query(query)

        self.env.assertEqual(2, len(query_result.result_set))
        expected_result = [[nodes[7]], [nodes[8]]]
        self.env.assertEquals(expected_result, query_result.result_set)

    # Validate placement of index scans and filter ops when not all filters can be replaced.
    def test08_index_scan_multiple_filters(self):
        query = "MATCH (p:person) WHERE p.age = 30 AND NOT EXISTS(p.fakeprop) RETURN p.name"
        plan = redis_graph.execution_plan(query)
        self.env.assertIn('Node By Index Scan', plan)
        self.env.assertNotIn('Label Scan', plan)
        self.env.assertIn('Filter', plan)

        query_result = redis_graph.query(query)
        expected_result = ["Lucy Yanfital"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

    def test09_index_scan_with_params(self):
        query = "MATCH (p:person) WHERE p.age = $age RETURN p.name"
        params = {'age': 30}
        plan = redis_graph.execution_plan(query, params=params)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(query, params=params)
        expected_result = ["Lucy Yanfital"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

    def test10_index_scan_with_param_array(self):
        query = "MATCH (p:person) WHERE p.age in $ages RETURN p.name"
        params = {'ages': [30]}
        plan = redis_graph.execution_plan(query, params=params)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(query, params=params)
        expected_result = ["Lucy Yanfital"]
        self.env.assertEquals(query_result.result_set[0], expected_result)

    def test11_single_index_multiple_scans(self):
        query = "MERGE (p1:person {age: 40}) MERGE (p2:person {age: 41})"
        plan = redis_graph.execution_plan(query)
        # Two index scans should be performed.
        self.env.assertEqual(plan.count("Node By Index Scan"), 2)

        query_result = redis_graph.query(query)
        # Two new nodes should be created.
        self.env.assertEquals(query_result.nodes_created, 2)

    def test12_remove_scans_before_index(self):
        query = "MATCH (a:person {age: 32})-[]->(b) WHERE (b:person)-[]->(a) RETURN a"
        plan = redis_graph.execution_plan(query)
        # One index scan should be performed.
        self.env.assertEqual(plan.count("Node By Index Scan"), 1)

    def test13_point_index_scan(self):
        # create index
        create_node_range_index(redis_graph, 'restaurant', 'location', sync=True)

        # create restaurant
        q = "CREATE (:restaurant {location: point({latitude:30.27822306, longitude:-97.75134723})})"
        redis_graph.query(q)

        # locate other restaurants within a 1000m radius
        q = """MATCH (r:restaurant)
        WHERE distance(r.location, point({latitude:30.27822306, longitude:-97.75134723})) < 1000
        RETURN r"""

        # make sure index is used
        plan = redis_graph.execution_plan(q)
        self.env.assertIn("Node By Index Scan", plan)

        # refine query from '<' to '<='
        q = """MATCH (r:restaurant)
        WHERE distance(r.location, point({latitude:30.27822306, longitude:-97.75134723})) <= 1000
        RETURN r"""

        # make sure index is used
        plan = redis_graph.execution_plan(q)
        self.env.assertIn("Node By Index Scan", plan)

        # index should NOT be used when searching for points outside of a circle
        # testing operand: '>', '>=' and '='
        q = """MATCH (r:restaurant)
        WHERE distance(r.location, point({latitude:30.27822306, longitude:-97.75134723})) > 1000
        RETURN r"""

        # make sure index is NOT used
        plan = redis_graph.execution_plan(q)
        self.env.assertNotIn("Node By Index Scan", plan)

        q = """MATCH (r:restaurant)
        WHERE distance(r.location, point({latitude:30.27822306, longitude:-97.75134723})) >= 1000
        RETURN r"""

        # make sure index is NOT used
        plan = redis_graph.execution_plan(q)
        self.env.assertNotIn("Node By Index Scan", plan)

        q = """MATCH (r:restaurant)
        WHERE distance(r.location, point({latitude:30.27822306, longitude:-97.75134723})) = 1000
        RETURN r"""

        # make sure index is NOT used
        plan = redis_graph.execution_plan(q)
        self.env.assertNotIn("Node By Index Scan", plan)

    def test14_index_scan_utilize_array(self):
        # Querying indexed properties using IN a constant array should utilize indexes.
        query = "MATCH (a:person) WHERE a.age IN [34, 33] RETURN a.name ORDER BY a.name"
        plan = redis_graph.execution_plan(query)
        # One index scan should be performed.
        self.env.assertEqual(plan.count("Node By Index Scan"), 1)
        query_result = redis_graph.query(query)
        expected_result = [["Noam Nativ"],
                           ["Omri Traub"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Querying indexed properties using IN a generated array should utilize indexes.
        query = "MATCH (a:person) WHERE a.age IN range(33, 34) RETURN a.name ORDER BY a.name"
        plan = redis_graph.execution_plan(query)
        # One index scan should be performed.
        self.env.assertEqual(plan.count("Node By Index Scan"), 1)
        query_result = redis_graph.query(query)
        expected_result = [["Noam Nativ"],
                           ["Omri Traub"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Querying indexed properties using IN a non-constant array should not utilize indexes.
        query = "MATCH (a:person)-[]->(b) WHERE a.age IN b.arr RETURN a"
        plan = redis_graph.execution_plan(query)
        # No index scans should be performed.
        self.env.assertEqual(plan.count("Label Scan"), 1)
        self.env.assertEqual(plan.count("Node By Index Scan"), 0)

        # Querying indexed properties using IN a with array with stop word.
        query = "CREATE (:country { name: 'a' })"
        redis_graph.query(query)

        query = "MATCH (a:country) WHERE a.name IN ['a'] RETURN a.name ORDER BY a.name"
        plan = redis_graph.execution_plan(query)
        # One index scan should be performed.
        self.env.assertEqual(plan.count("Node By Index Scan"), 1)
        query_result = redis_graph.query(query)
        expected_result = [['a']]
        self.env.assertEquals(query_result.result_set, expected_result)

        query = "MATCH (a:country { name: 'a' }) DELETE a"
        redis_graph.query(query)

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
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [["Noam Nativ"], ["Omri Traub"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # similar to the query above, only this time the filter is specified
        # by an OR condition
        q = """WITH 33 AS min, 34 AS max 
        MATCH (p:person)
        WHERE p.age = min OR p.age = max
        RETURN p.name
        ORDER BY p.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [["Noam Nativ"], ["Omri Traub"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # find all person nodes with age equals 33 'x'
        # 'x' value is known only at runtime
        q = """WITH 33 AS x
        MATCH (p:person {age:x})
        RETURN p.name
        ORDER BY p.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
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
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
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
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
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
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
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
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [["Noam Nativ"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # check that the value is evaluated before sending it to index query
        q = """MATCH (b:person)
        WHERE b.age = rand()*0 + 32
        RETURN b.name
        ORDER BY b.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [['Ailon Velger'], ['Alon Fital'], ['Ori Laslo'], ['Roi Lipman'], ['Tal Doron']]
        self.env.assertEquals(query_result.result_set, expected_result)

        # check that the value is evaluated before sending it to index query
        q = """MATCH (a:person)
        WHERE a.age = toInteger('32')
        RETURN a.name
        ORDER BY a.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
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
        redis_graph.query(q)

        # find all person nodes with age value of [36]
        q = """WITH [36] AS age MATCH (a:person {age:age}) RETURN a.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [["leonard"]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # find all person nodes with age > [33]
        q = """WITH [33] AS age MATCH (a:person) WHERE a.age > age RETURN a.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [["leonard"], [["maynard"]]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # combine indexable value with none-indexable value index query
        q = """WITH [33] AS age, 'leonard' AS name MATCH (a:person) WHERE a.age >= age AND a.name = name RETURN a.name"""
        plan = redis_graph.execution_plan(q)
        self.env.assertIn('Node By Index Scan', plan)
        query_result = redis_graph.query(q)
        expected_result = [["leonard"]]
        self.env.assertEquals(query_result.result_set, expected_result)

    # test for https://github.com/RedisGraph/RedisGraph/issues/1980
    def test18_index_scan_inside_apply(self):
        redis_graph = Graph(self.env.getConnection(), 'g')

        create_node_range_index(redis_graph, 'L1', 'id', sync=True)
        redis_graph.query("UNWIND range(1, 5) AS v CREATE (:L1 {id: v})")
        result = redis_graph.query("UNWIND range(1, 5) AS id OPTIONAL MATCH (u:L1{id: 5}) RETURN u.id")

        expected_result = [[5], [5], [5], [5], [5]]
        self.env.assertEquals(result.result_set, expected_result)

    def test19_index_scan_numeric_accuracy(self):
        redis_graph = Graph(self.env.getConnection(), 'large_index_values')

        create_node_range_index(redis_graph, 'L1', 'id', sync=True)
        create_node_range_index(redis_graph, 'L2', 'id1', 'id2', sync=True)
        redis_graph.query("UNWIND range(1, 5) AS v CREATE (:L1 {id: 990000000262240068 + v})")
        redis_graph.query("UNWIND range(1, 5) AS v CREATE (:L2 {id1: 990000000262240068 + v, id2: 990000000262240068 - v})")

        # test index search
        result = redis_graph.query("MATCH (u:L1{id: 990000000262240069}) RETURN u.id")
        expected_result = [[990000000262240069]]
        self.env.assertEquals(result.result_set, expected_result)

        # test index search from child
        result = redis_graph.query("MATCH (u:L1) WITH min(u.id) as id MATCH (u:L1{id: id}) RETURN u.id")
        expected_result = [[990000000262240069]]
        self.env.assertEquals(result.result_set, expected_result)

        # test index search with or
        result = redis_graph.query("MATCH (u:L1) WHERE u.id = 990000000262240069 OR u.id = 990000000262240070 RETURN u.id ORDER BY u.id")
        expected_result = [[990000000262240069], [990000000262240070]]
        self.env.assertEquals(result.result_set, expected_result)

        # test resetting index scan operation
        result = redis_graph.query("MATCH (u1:L1), (u2:L1) WHERE u1.id = 990000000262240069 AND (u2.id = 990000000262240070 OR u2.id = 990000000262240071) RETURN u1.id, u2.id ORDER BY u1.id, u2.id")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEquals(result.result_set, expected_result)

        # test resetting index scan operation when using the consume from child function
        result = redis_graph.query("MATCH (u:L1) WITH min(u.id) as id MATCH (u1:L1), (u2:L1) WHERE u1.id = 990000000262240069 AND (u2.id = 990000000262240070 OR u2.id = 990000000262240071) RETURN u1.id, u2.id ORDER BY u1.id, u2.id")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEquals(result.result_set, expected_result)

        # test resetting index scan operation when rebuild index is required
        result = redis_graph.query("MATCH (u:L1) WITH min(u.id) as id MATCH (u1:L1), (u2:L1) WHERE u1.id = id AND (u2.id = 990000000262240070 OR u2.id = 990000000262240071) RETURN u1.id, u2.id ORDER BY u1.id, u2.id")
        expected_result = [[990000000262240069, 990000000262240070], [990000000262240069, 990000000262240071]]
        self.env.assertEquals(result.result_set, expected_result)

        # test index scan with 2 different attributes
        result = redis_graph.query("MATCH (u:L2) WHERE u.id1 = 990000000262240069 AND u.id2 = 990000000262240067 RETURN u.id1, u.id2")
        expected_result = [[990000000262240069, 990000000262240067]]
        self.env.assertEquals(result.result_set, expected_result)

    def test20_index_scan_stopwords(self):
        redis_graph = Graph(self.env.getConnection(), 'stopword')

        #-----------------------------------------------------------------------
        # create indices
        #-----------------------------------------------------------------------

        # create exact match index over User id
        create_node_range_index(redis_graph, 'User', 'id', sync=True)
        # create a fulltext index over User id
        create_node_fulltext_index(redis_graph, 'User', 'id', sync=True)

        #-----------------------------------------------------------------------
        # create node
        #-----------------------------------------------------------------------

        # create a User node with a RediSearch stopword as the id attribute
        user = Node(label='User', properties={'id': 'not'})
        redis_graph.add_node(user)
        redis_graph.commit()

        #-----------------------------------------------------------------------
        # query indices
        #-----------------------------------------------------------------------

        # query exact match index for user
        # expecting node to return as stopwords are not enforced
        result = redis_graph.query("MATCH (u:User {id: 'not'}) RETURN u")
        self.env.assertEquals(result.result_set[0][0], user)

        # query fulltext index for user
        # expecting no results as stopwords are enforced
        result = redis_graph.query("CALL db.idx.fulltext.queryNodes('User', 'stop')")
        self.env.assertEquals(result.result_set, [])
    
    def test21_invalid_distance_query(self):
        redis_graph = Graph(self.env.getConnection(), 'invalid_distance')

        # create exact match index over User id
        create_node_range_index(redis_graph, 'User', 'loc', sync=True)
        
        # create a node
        redis_graph.query("CREATE (:User {loc:point({latitude:40.4, longitude:30.3})})")

        # invalid query
        try:
            redis_graph.query("MATCH (u:User) WHERE distance(point({latitude:40.5, longitude: 30.4}, u.loc)) < 20000 RETURN u")
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
        plan = g.execution_plan(q)

        # expecting no index scan operation, as we've yet to create an index
        self.env.assertNotIn('Node By Index Scan', plan)

        # create an index
        resultset = create_node_range_index(g, 'N', 'v', sync=True)
        self.env.assertEqual(1, resultset.indices_created)

        # re-issue the same query
        q = "MATCH (n:N) WHERE n.v = 1 RETURN n"
        plan = g.execution_plan(q)

        # expecting an index scan operation
        self.env.assertIn('Node By Index Scan', plan)

    def test_23_do_not_utilize_index_(self):
        g = Graph(self.env.getConnection(), 'late_index_creation')

        # create graph
        g.query("RETURN 1")

        # issue query which not utilize an index
        q = "MATCH (n:N) WHERE id(n) IN [0] RETURN n"
        plan = g.execution_plan(q)

        # expecting no index scan operation
        self.env.assertNotIn('Node By Index Scan', plan)

        # create an index
        resultset = create_node_range_index(g, 'N', 'v', sync=True)
        self.env.assertEqual(1, resultset.indices_created)

        # re-issue the same query
        q = "MATCH (n:N) WHERE id(n) IN [0] RETURN n"
        plan = g.execution_plan(q)

        # expecting an no index scan operation
        self.env.assertNotIn('Node By Index Scan', plan)
