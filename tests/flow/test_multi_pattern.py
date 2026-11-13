from common import *

GRAPH_ID = "multi_pattern"
people = ["Roi", "Alon", "Ailon", "Boaz", "Tal", "Omri", "Ori"]


class testGraphMultiPatternQueryFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
         # Create entities
        nodes = []
        for p in people:
            nodes.append(Node(labels="person", properties={"name": p}))

        self.graph.query(f"CREATE {','.join(map(str, nodes))}")

    # Connect a single node to all other nodes.
    def test01_connect_node_to_rest(self):
        query = """MATCH(r:person {name:"Roi"}), (f:person) WHERE f.name <> r.name CREATE (r)-[:friend]->(f) RETURN count(f)"""
        actual_result = self.graph.query(query)
        friend_count = actual_result.result_set[0][0]
        self.env.assertEqual(friend_count, 6)
        self.env.assertEqual(actual_result.relationships_created, 6)

    def test02_verify_cartesian_product_streams_reset(self):
        # See https://github.com/RedisGraph/RedisGraph/issues/249
        # Forevery outgoing edge, we expect len(people) to be matched.
        expected_resultset_size = 6 * len(people)
        queries = ["""MATCH (r:person {name:"Roi"})-[]->(f), (x) RETURN f, x""",
                   """MATCH (x), (r:person {name:"Roi"})-[]->(f) RETURN f, x""",
                   """MATCH (r:person {name:"Roi"})-[]->(f) MATCH (x) RETURN f, x""",
                   """MATCH (x) MATCH (r:person {name:"Roi"})-[]->(f) RETURN f, x"""]
        for q in queries:
            actual_result = self.graph.query(q)
            records_count = len(actual_result.result_set)
            self.env.assertEqual(records_count, expected_resultset_size)


    def test03_reset_nested_cartesian_product(self):
        # here's the plan for the following query:
        #
        # Results | Records produced: 0, Execution time: 0.010041 ms"
        #     Project | Records produced: 0, Execution time: 0.012334 ms"
        #         Delete | Records produced: 0, Execution time: 2134.899041 ms"
        #             Cartesian Product | Records produced: 0, Execution time: 0.012709 ms"
        #                 Node By Label Scan | (@anon_0:NoneExistingLbl) | Records produced: 0, Execution time: 0.004791 ms"
        #                 Filter | Records produced: 0, Execution time: 0.000000 ms"
        #                     Cartesian Product | Records produced: 0, Execution time: 0.000000 ms"
        #                         Node By Label Scan | (n0:NoneExistingLbl) | Records produced: 0, Execution time: 0.000000 ms"
        #                         Node By Label Scan | (n1:NoneExistingLbl) | Records produced: 0, Execution time: 0.000000 ms"

        # as can be seen the first Cartesian Product will not be able to generate
        # any records from the Node By Label Scan operation
        # as a result Delete used to issue a propagate reset up the op chain
        # causing the second Cartesian Product to reset while it has never been
        # initialized, which caused a crash
        #
        # this test validate that such scenarios won't crash

        q = """MATCH (n0:NoneExistingLbl), (n1:NoneExistingLbl), (:NoneExistingLbl)
               WHERE n0.v = n1.v OR 0 <> NULL
               DELETE n0 , n1
               RETURN *"""
        self.graph.query(q)

        # we won't be here is the server had crashed

    # Connect every node to every node.
    def test04_create_fully_connected_graph(self):
        query = """MATCH(a:person), (b:person) WHERE a.name <> b.name CREATE (a)-[f:friend]->(b) RETURN count(f)"""
        actual_result = self.graph.query(query)
        friend_count = actual_result.result_set[0][0]
        self.env.assertEqual(friend_count, 42)
        self.env.assertEqual(actual_result.relationships_created, 42)
    
    # Perform a cartesian product of 3 sets.
    def test05_cartesian_product(self):
        queries = {"""MATCH (a), (b), (c) RETURN count(a)""": 343,
                   """MATCH (a) MATCH (b), (c) RETURN count(a)""": 343,
                   """MATCH (a), (b) MATCH (c) RETURN count(a)""": 343,
                   """MATCH (a) MATCH (b) MATCH (c) RETURN count(a)""": 343,
                   """MATCH (a) OPTIONAL MATCH ({n0:0}), () RETURN count(a)""": 7}

        for q, c in queries.items():
            actual_result = self.graph.query(q)
            friend_count = actual_result.result_set[0][0]
            self.env.assertEqual(friend_count, c)

    def test06_multiple_create_clauses(self):
        queries = ["""CREATE (:a {v:1}), (:b {v:2, z:3}), (:c), (:a)-[:r0 {k:9}]->(:b), (:c)-[:r1]->(:d)""",
                   """CREATE (:a {v:1}) CREATE (:b {v:2, z:3}) CREATE (:c) CREATE (:a)-[:r0 {k:9}]->(:b) CREATE (:c)-[:r1]->(:d)""",
                   """CREATE (:a {v:1}), (:b {v:2, z:3}) CREATE (:c), (:a)-[:r0 {k:9}]->(:b) CREATE (:c)-[:r1]->(:d)"""]
        for q in queries:
            actual_result = self.graph.query(q)
            self.env.assertEqual(actual_result.relationships_created, 2)
            self.env.assertEqual(actual_result.properties_set, 4)
            self.env.assertEqual(actual_result.nodes_created, 7)

    # A cartesian product must stream its right-hand branches rather than
    # cross-multiplying them into one materialized set: over N nodes a 4-way
    # product holds 3N rows, not N^3. Materializing the product happened inside
    # a single next() call, so the query neither honored its deadline nor
    # bounded its memory — a 1 ms timeout over 1001 nodes ran for minutes and
    # allocated ~20 GB.
    def test07_multi_branch_cartesian_product_is_streamed(self):
        g = self.db.select_graph("multi_pattern_stream")
        g.query("UNWIND range(0, 1000) AS x CREATE ({v: x})")

        q = "MATCH (a), (b), (c), (d) RETURN *"

        start = time.time()
        try:
            # 1001^4 rows: the query must hit its deadline, not run to completion
            g.query(q, timeout=1)
            self.env.assertTrue(False)
        except ResponseError as error:
            self.env.assertContains("Query timed out", str(error))
        # Generous bound: streaming gives up in single-digit milliseconds, while
        # materializing the branches took over two minutes.
        self.env.assertLess(time.time() - start, 10)

        # Nor may LIMIT pay for the product before emitting a row. Each branch is
        # still materialized in full (3 x 1001 rows), but only the combinations
        # actually consumed are ever merged, so one row costs one merge instead
        # of 1001^3 of them.
        start = time.time()
        res = g.query(q + " LIMIT 1")
        self.env.assertEqual(len(res.result_set), 1)
        self.env.assertLess(time.time() - start, 10)

        g.delete()


    def test08_inline_filter_after_unwind_waits_for_its_node(self):
        # The best endpoint of the chain is `t` (filtered and labelled), but
        # the chain hangs off the UNWIND, so no scan binds `t`. The inline
        # filter `{id:100}` must still wait for the hop that binds `t` (#2556).
        g = self.db.select_graph("multi_pattern_unwind")
        g.query("""CREATE (p:Person {id:1}), (f:Person {id:2}), (p)-[:KNOWS]->(f),
                          (m1:Post {id:10}), (m2:Post {id:11}),
                          (m1)-[:HAS_CREATOR]->(f), (m2)-[:HAS_CREATOR]->(f),
                          (t1:Tag {id:100}), (t2:Tag {id:200}),
                          (m1)-[:HAS_TAG]->(t1), (m1)-[:HAS_TAG]->(t2), (m2)-[:HAS_TAG]->(t1)""")

        for head in ["MATCH (person:Person {id:1})-[:KNOWS]-(friend) WITH collect(friend) AS friends",
                     "MATCH (friend:Person {id:2}) WITH collect(friend) AS friends"]:
            res = g.query(f"""{head}
                              UNWIND friends AS f
                              MATCH (f)<-[:HAS_CREATOR]-(post:Post),
                                    (post)-[:HAS_TAG]->(t:Tag {{id:100}}),
                                    (post)-[:HAS_TAG]->(tag:Tag)
                              RETURN count(post)""")
            self.env.assertEqual(res.result_set, [[3]], depth=1)

        res = g.query("""MATCH (person:Person {id:1})-[:KNOWS]-(friend)
                         WITH collect(friend) AS friends
                         UNWIND friends AS f
                         MATCH (f)<-[:HAS_CREATOR]-(post:Post), (post)-[:HAS_TAG]->(t:Tag {id:100})
                         MATCH (post)-[:HAS_CREATOR]->(z:Person)
                         RETURN count(post)""")
        self.env.assertEqual(res.result_set, [[2]])
        g.delete()
