from common import *
from index_utils import *

GRAPH_ID = "query"

class testFulltextIndexQuery():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        self.graph.query("CREATE FULLTEXT INDEX FOR (n:L1) ON (n.v)")
        self.graph.query("CREATE FULLTEXT INDEX FOR (n:L2) ON (n.v) OPTIONS {stopwords: ['redis', 'world']}")
        self.graph.query("CREATE FULLTEXT INDEX FOR (n:L3) ON (n.v1) OPTIONS {weight: 1}")
        self.graph.query("CREATE FULLTEXT INDEX FOR (n:L3) ON (n.v2) OPTIONS {weight: 2}")
        self.graph.query("CREATE FULLTEXT INDEX FOR (n:L4) ON (n.v) OPTIONS {phonetic: 'dm:en'}")
        self.graph.query("CREATE FULLTEXT INDEX FOR (n:L5) ON (n.v) OPTIONS {nostem: true}")

        # create full text index on relationship type E inedxing property 'name'
        self.graph.query("CREATE FULLTEXT INDEX FOR ()-[e:E]-() on (e.name)") 

        wait_for_indices_to_sync(self.graph)

        n0 = Node(labels="L1", properties={"v": 'hello redis world'})
        n1 = Node(labels="L2", properties={"v": 'hello redis world'})
        n2 = Node(labels="L3", properties={"v1": 'hello world', "v2": 'hello redis'})
        n3 = Node(labels="L3", properties={"v1": 'hello redis', "v2": 'hello world'})
        n4 = Node(labels="L4", properties={"v": 'felix'})
        n5 = Node(labels="L5", properties={"v": 'there are seven words in this sentence'})

        # introduce a number of relationships of type E and E1
        e0 = Edge(n5, "E", n0, properties={"name": "just another nice relationship", "relation_id": "e0"})
        e1 = Edge(n5, "E", n0, properties={"name": "a nice place to be"})
        e2 = Edge(n5, "E1", n0, properties={"name": "don't find me please, I'm not full text indexed"})

        # create the nodes and relationships
        self.graph.query(f"CREATE {n0}, {n1}, {n2}, {n3}, {n4}, {n5}, {e0}, {e1}, {e2}")

    # test full-text query on nodes
    def test01_fulltext_node_query(self):
        expected_result = self.graph.query("MATCH (n:L1) RETURN n")
        # fulltext query L1 for hello 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L1', 'hello')")
        self.env.assertEqual(result.result_set[0][0], expected_result.result_set[0][0])

        # fulltext query L1 for redis 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L1', 'redis')")
        self.env.assertEqual(result.result_set[0][0], expected_result.result_set[0][0])

        # fulltext query L1 for world 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L1', 'world')")
        self.env.assertEqual(result.result_set[0][0], expected_result.result_set[0][0])

        expected_result = self.graph.query("MATCH (n:L2) RETURN n")

        # fulltext query L2 for hello 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L2', 'hello')")
        self.env.assertEqual(result.result_set[0][0], expected_result.result_set[0][0])

        # fulltext query L2 for redis 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L2', 'redis')")
        self.env.assertEqual(result.result_set, [])

        # fulltext query L2 for world 
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L2', 'world')")
        self.env.assertEqual(result.result_set, [])

        # fulltext query L3 for redis and document that contains redis in v2 is scored higher than document contains redis in v1
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L3', 'redis') YIELD node, score RETURN node, score ORDER BY score DESC")
        self.env.assertEqual(result.result_set[0][0].properties["v2"], "hello redis")
        self.env.assertEqual(result.result_set[1][0].properties["v1"], "hello redis")
        self.env.assertGreater(result.result_set[0][1], result.result_set[1][1])

        expected_result = self.graph.query("MATCH (n:L4 {v:'felix'}) RETURN n")

        # fulltext query L4 for phelix
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L4', 'phelix')")
        self.env.assertEqual(result.result_set[0][0], expected_result.result_set[0][0])

        expected_result = self.graph.query("MATCH (n:L5) RETURN n")

        # fulltext query L5 for 'words' which exists in the document
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L5', 'words')")
        self.env.assertEqual(result.result_set[0][0], expected_result.result_set[0][0])

        # fulltext query L5 for 'word' nostem did not removed 's' from 'words'
        # as such no results are expected
        result = self.graph.query("CALL db.idx.fulltext.queryNodes('L5', 'word')")
        self.env.assertEqual(result.result_set, [])

    # test full-text query on edges
    def test02_fulltext_edge_query(self):
        # full text query on a relationship E1 (not indexed)
        result = self.graph.query(
            "CALL db.idx.fulltext.queryRelationships('E1', 'please')"
        )
        self.env.assertEqual(result.result_set, [])

        # full-text query on a relationship E (indexed) 'nice' appears in two relationships
        result = self.graph.query("""
            CALL db.idx.fulltext.queryRelationships('E', 'nice')
            YIELD relationship AS r
            RETURN r.name AS name
            ORDER BY r.name""")

        # expecting two relationships with 'nice' in their name
        self.env.assertEqual(len(result.result_set), 2)
        actual = [row[0] for row in result.result_set]

        expected = ["a nice place to be", "just another nice relationship"]
        self.env.assertEqual(actual, expected)

        # full-text query on an indexed relationship-type that does not return any match
        result = self.graph.query(
            """CALL db.idx.fulltext.queryRelationships('E', 'nonexistent')"""
        )
        self.env.assertEqual(result.result_set, [])

        # full-text query on an indexed relationship-type that returns only
        # a single match
        result = self.graph.query(
            """CALL db.idx.fulltext.queryRelationships('E', 'place')
            YIELD relationship AS r
            RETURN r.name"""
        )
        self.env.assertEqual(len(result.result_set), 1)
        self.env.assertEqual(result.result_set[0][0], "a nice place to be")

    def test03_fulltext_edge_query_with_crud(self):
        # this test make sure the index returns valid results
        # after performing CRUD operations on the indexed entities
              
        # 1. update an indexed edge (relation_id:e0)
        # adding the word 'updated' to its name
        self.graph.query(
            """MATCH ()-[e:E {relation_id:'e0'}]-()
            SET e.name='just another nice relationship that was updated'"""
        )

        # query full-text search edge with 'updated'
        result = self.graph.query(
            """CALL db.idx.fulltext.queryRelationships('E', 'updated')
            YIELD relationship AS r
            RETURN r.relation_id"""
        )

        # verify only one correct result
        self.env.assertEqual(len(result.result_set), 1)
        self.env.assertEqual(result.result_set[0][0], "e0")
        
        # 2. adding a new indexed edge
        self.graph.query(
            f"""MATCH (n:L1), (m:L2)
            CREATE (n)-[e:E {{name: 'new edge'}}]->(m)"""
        )

        # query full-text search edge with 'new'
        result = self.graph.query(
            """CALL db.idx.fulltext.queryRelationships('E', 'new')
            YIELD relationship AS r
            RETURN r.name"""
        )

        # verify only one correct result
        self.env.assertEqual(len(result.result_set), 1)
        self.env.assertEqual(result.result_set[0][0], "new edge")
        
        # 3. deleting an indexed edge
        self.graph.query("MATCH ()-[e:E {name:'new edge'}]-() DELETE e")

        # query full-text search edge with 'new'
        result = self.graph.query(
            "CALL db.idx.fulltext.queryRelationships('E', 'new')"
        )

        # verify no results
        self.env.assertEqual(result.result_set, [])

    # exercise non-canonical YIELD orderings to lock in name-based
    # resolution of the relationship/score slots in the planner rewrite.
    #
    # This test owns its own labels (`Yield`, `YieldW`) and edge type
    # (`Yield`) so it does not depend on data created or mutated by any
    # earlier test in the class.
    def test04_fulltext_query_yield_variants(self):
        # Dedicated indexes — `YieldW` uses per-field weights so the
        # node test below can verify that scoring actually distinguishes
        # the two rows.
        self.graph.query("CREATE FULLTEXT INDEX FOR (n:Yield) ON (n.text)")
        self.graph.query(
            "CREATE FULLTEXT INDEX FOR (n:YieldW) ON (n.a) OPTIONS {weight: 1}"
        )
        self.graph.query(
            "CREATE FULLTEXT INDEX FOR (n:YieldW) ON (n.b) OPTIONS {weight: 2}"
        )
        self.graph.query("CREATE FULLTEXT INDEX FOR ()-[e:Yield]-() ON (e.text)")
        wait_for_indices_to_sync(self.graph)

        # Dedicated dataset (vocabulary deliberately uses generic words
        # so the test isn't entangled with any other suite's fixtures).
        self.graph.query(
            """CREATE
              (a:Yield {text:'apple banana cherry'}),
              (b:Yield {text:'apple cherry'}),
              (a)-[:Yield {text:'apple banana'}]->(b),
              (b)-[:Yield {text:'banana cherry'}]->(a),
              (:YieldW {a:'banana cherry', b:'apple cherry'}),
              (:YieldW {a:'apple cherry', b:'banana cherry'})"""
        )

        # YIELD with reversed canonical order: score first, relationship second.
        # The planner must resolve slots by field name, not list position;
        # otherwise the relationship slot would be bound to the score value.
        result = self.graph.query(
            """CALL db.idx.fulltext.queryRelationships('Yield', 'banana')
            YIELD score, relationship
            RETURN relationship.text AS text, score
            ORDER BY text"""
        )
        self.env.assertEqual(len(result.result_set), 2)
        texts = [row[0] for row in result.result_set]
        self.env.assertEqual(texts, ["apple banana", "banana cherry"])
        # score column must be a number, not a relationship value
        for row in result.result_set:
            self.env.assertTrue(isinstance(row[1], (int, float)))
            self.env.assertGreater(row[1], 0.0)

        # YIELD with AS aliases on both fields, also reversed.
        result = self.graph.query(
            """CALL db.idx.fulltext.queryRelationships('Yield', 'banana')
            YIELD score AS s, relationship AS r
            RETURN r.text AS text, s
            ORDER BY text"""
        )
        self.env.assertEqual(len(result.result_set), 2)

        # Same on the node-fulltext path: YIELD with score first and node
        # aliased. The `b` field has weight 2 vs `a`'s weight 1, so the row
        # whose `b` contains the search term must score higher.
        result = self.graph.query(
            """CALL db.idx.fulltext.queryNodes('YieldW', 'apple')
            YIELD score AS s, node AS n
            RETURN n.b AS b, s
            ORDER BY s DESC"""
        )
        self.env.assertEqual(len(result.result_set), 2)
        self.env.assertEqual(result.result_set[0][0], "apple cherry")
        self.env.assertGreater(result.result_set[0][1], result.result_set[1][1])

        # YIELD without the entity field must be rejected — the scan operator
        # needs an entity slot to bind into.
        try:
            self.graph.query(
                """CALL db.idx.fulltext.queryRelationships('Yield', 'banana')
                YIELD score
                RETURN score"""
            )
            raise AssertionError(
                "Expected error when YIELD omits the 'relationship' field"
            )
        except ResponseError as e:
            self.env.assertContains("requires YIELD of 'relationship'", str(e))

        try:
            self.graph.query(
                """CALL db.idx.fulltext.queryNodes('Yield', 'banana')
                YIELD score
                RETURN score"""
            )
            raise AssertionError(
                "Expected error when YIELD omits the 'node' field"
            )
        except ResponseError as e:
            self.env.assertContains("requires YIELD of 'node'", str(e))

    def test05_fulltext_query_with_limit_and_skip(self):
        # A downstream LIMIT/SKIP lowers the fulltext scan's packing ceiling
        # (limit pushdown) — verify result correctness under those plans.
        self.graph.query("CREATE FULLTEXT INDEX FOR (n:Page) ON (n.text)")
        self.graph.query("CREATE FULLTEXT INDEX FOR ()-[e:Link]-() ON (e.text)")
        wait_for_indices_to_sync(self.graph)

        # 20 matching nodes and 20 matching edges.
        self.graph.query(
            """UNWIND range(0, 19) AS i
            CREATE (a:Page {id: i, text: 'shared keyword page'})
                   -[:Link {id: i, text: 'shared keyword link'}]->(:Page2)"""
        )

        # LIMIT smaller than the match count.
        result = self.graph.query(
            """CALL db.idx.fulltext.queryNodes('Page', 'keyword')
            YIELD node, score
            RETURN node.id, score
            LIMIT 5"""
        )
        self.env.assertEqual(len(result.result_set), 5)

        # SKIP + LIMIT still yields the requested window of distinct rows.
        result = self.graph.query(
            """CALL db.idx.fulltext.queryNodes('Page', 'keyword')
            YIELD node
            RETURN node.id
            SKIP 3 LIMIT 5"""
        )
        self.env.assertEqual(len(result.result_set), 5)
        self.env.assertEqual(len({row[0] for row in result.result_set}), 5)

        # LIMIT larger than the match count returns everything.
        result = self.graph.query(
            """CALL db.idx.fulltext.queryNodes('Page', 'keyword')
            YIELD node
            RETURN node.id
            LIMIT 100"""
        )
        self.env.assertEqual(len(result.result_set), 20)

        # LIMIT 0 returns nothing.
        result = self.graph.query(
            """CALL db.idx.fulltext.queryNodes('Page', 'keyword')
            YIELD node
            RETURN node.id
            LIMIT 0"""
        )
        self.env.assertEqual(len(result.result_set), 0)

        # Same for the edge fulltext scan.
        result = self.graph.query(
            """CALL db.idx.fulltext.queryRelationships('Link', 'keyword')
            YIELD relationship, score
            RETURN relationship.id, score
            LIMIT 5"""
        )
        self.env.assertEqual(len(result.result_set), 5)

        result = self.graph.query(
            """CALL db.idx.fulltext.queryRelationships('Link', 'keyword')
            YIELD relationship
            RETURN relationship.id
            SKIP 3 LIMIT 5"""
        )
        self.env.assertEqual(len(result.result_set), 5)
        self.env.assertEqual(len({row[0] for row in result.result_set}), 5)
