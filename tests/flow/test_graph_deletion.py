from common import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from demo import QueryInfo

GRAPH_ID = "graph_deletion"

class testGraphDeletionFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        nodes = {}
         # Create entities
        people = ["Roi", "Alon", "Ailon", "Boaz", "Tal", "Omri", "Ori"]
        for idx, p in enumerate(people):
            node = Node(alias=f"n_{idx}",labels="person", properties={"name": p})
            nodes[p] = node

        # Fully connected graph
        edges = []
        for src in nodes:
            for dest in nodes:
                if src != dest:
                    edge = Edge(nodes[src], "know", nodes[dest])
                    edges.append(edge)

        # Connect Roi to Alon via another edge type.
        edges.append(Edge(nodes["Roi"], "SameBirthday", nodes["Alon"]))

        nodes_str = [str(n) for n in nodes.values()]
        edges_str = [str(e) for e in edges]
        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")

    # Count how many nodes contains the `name` attribute
    # remove the `name` attribute from some nodes
    # make sure the count updates accordingly,
    # restore `name` attribute from, verify that count returns to its original value.
    def test01_delete_attribute(self):
        # How many nodes contains the 'name' attribute
        query = """MATCH (n) WHERE EXISTS(n.name)=true RETURN count(n)"""
        actual_result = self.graph.query(query)
        nodeCount = actual_result.result_set[0][0]
        self.env.assertEquals(nodeCount, 7)

        # Remove Tal's name attribute.
        query = """MATCH (n) WHERE n.name = 'Tal' SET n.name = NULL"""
        self.graph.query(query)

        # How many nodes contains the 'name' attribute,
        # should reduce by 1 from previous count.
        query = """MATCH (n) WHERE EXISTS(n.name)=true RETURN count(n)"""
        actual_result = self.graph.query(query)
        nodeCount = actual_result.result_set[0][0]
        self.env.assertEquals(nodeCount, 6)

        # Reintroduce Tal's name attribute.
        query = """MATCH (n) WHERE EXISTS(n.name)=false SET n.name = 'Tal'"""
        actual_result = self.graph.query(query)

        # How many nodes contains the 'name' attribute
        query = """MATCH (n) WHERE EXISTS(n.name)=true RETURN count(n)"""
        actual_result = self.graph.query(query)
        nodeCount = actual_result.result_set[0][0]
        self.env.assertEquals(nodeCount, 7)

    # Delete edges pointing into either Boaz or Ori.
    def test02_delete_edges(self):
        query = """MATCH (s:person)-[e:know]->(d:person) WHERE d.name = "Boaz" OR d.name = "Ori" RETURN count(e)"""
        actual_result = self.graph.query(query)
        edge_count = actual_result.result_set[0][0]

        query = """MATCH (s:person)-[e:know]->(d:person) WHERE d.name = "Boaz" OR d.name = "Ori" DELETE e"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.relationships_deleted, edge_count)
        self.env.assertEquals(actual_result.nodes_deleted, 0)

    # Make sure there are no edges going into either Boaz or Ori.
    def test03_verify_edge_deletion(self):
        query = """MATCH (s:person)-[e:know]->(d:person)
                    WHERE d.name = "Boaz" AND d.name = "Ori"
                    RETURN COUNT(s)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 0)

    # Remove 'know' edge connecting Roi to Alon
    # Leaving a single edge of type SameBirthday
    # connecting the two.
    def test04_delete_typed_edge(self):
        query = """MATCH (s:person {name: "Roi"})-[e:know]->(d:person {name: "Alon"})
                   RETURN count(e)"""

        actual_result = self.graph.query(query)
        edge_count = actual_result.result_set[0][0]

        query = """MATCH (s:person {name: "Roi"})-[e:know]->(d:person {name: "Alon"})
                   DELETE e"""

        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.relationships_deleted, edge_count)
        self.env.assertEquals(actual_result.nodes_deleted, 0)

    # Make sure Roi is still connected to Alon
    # via the "SameBirthday" type edge.
    def test05_verify_delete_typed_edge(self):
        query = """MATCH (s:person {name: "Roi"})-[e:SameBirthday]->(d:person {name: "Alon"})
                   RETURN COUNT(s)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), 1)

        query = """MATCH (s:person {name: "Roi"})-[e:know]->(d:person {name: "Alon"})
                   RETURN COUNT(s)"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set[0][0], 0)

    # Remove both Alon and Boaz from the graph.
    def test06_delete_nodes(self):
        rel_count_query = """MATCH (a:person)-[e]->(b:person)
                             WHERE a.name = 'Boaz' OR a.name = 'Alon'
                             OR b.name = 'Boaz' OR b.name = 'Alon'
                             RETURN COUNT(e)"""
        rel_count_result = self.graph.query(rel_count_query)
        # Get the total number of unique edges (incoming and outgoing)
        # connected to Alon and Boaz.
        rel_count = rel_count_result.result_set[0][0]

        query = """MATCH (s:person)
                    WHERE s.name = "Boaz" OR s.name = "Alon"
                    DELETE s"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.relationships_deleted, rel_count)
        self.env.assertEquals(actual_result.nodes_deleted, 2)

    # Make sure Alon and Boaz are not in the graph.
    def test07_get_deleted_nodes(self):
        query = """MATCH (s:person)
                    WHERE s.name = "Boaz" OR s.name = "Alon"
                    RETURN s"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), 0)

    # Make sure Alon and Boaz are the only removed nodes.
    def test08_verify_node_deletion(self):
        query = """MATCH (s:person)
                   RETURN COUNT(s)"""
        actual_result = self.graph.query(query)
        nodeCount = actual_result.result_set[0][0]
        self.env.assertEquals(nodeCount, 5)

    def test09_delete_entire_graph(self):
        # Make sure graph exists.
        query = """MATCH (n) RETURN COUNT(n)"""
        result = self.graph.query(query)
        nodeCount = result.result_set[0][0]
        self.env.assertGreater(nodeCount, 0)

        # Delete graph.
        self.graph.delete()

        # Try to query a deleted graph.
        self.graph.query(query)
        result = self.graph.query(query)
        nodeCount = result.result_set[0][0]
        self.env.assertEquals(nodeCount, 0)

    def test10_bulk_edge_deletion_timing(self):
        # Create large amount of relationships (50000).
        self.graph.query("""UNWIND(range(1, 50000)) as x CREATE ()-[:R]->()""")
        # Delete and benchmark for 300ms.
        query = """MATCH (a)-[e:R]->(b) DELETE e"""
        result = self.graph.query(query)
        query_info = QueryInfo(query = query, description = "Test the execution time for deleting large number of edges")
        self.env.assertEquals(result.relationships_deleted, 50000)

    def test11_delete_entity_type_validation(self):
        # Currently we only support deletion of either nodes, edges or paths

        # Try to delete an integer.
        query = """UNWIND [1] AS x DELETE x"""
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except Exception as error:
            self.env.assertTrue("Delete type mismatch" in str(error))

    def test12_delete_unwind_entity(self):
        self.graph.delete()

        # Create 10 nodes.
        self.graph.query("UNWIND(range(1, 10)) as x CREATE ()")

        # Unwind path nodes.
        query = """MATCH p = () UNWIND nodes(p) AS node DELETE node"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.nodes_deleted, 10)
        self.env.assertEquals(actual_result.relationships_deleted, 0)

        self.graph.query("UNWIND(range(1, 10)) as x CREATE ()")

        # Unwind collected nodes.
        query = """MATCH (n) WITH collect(n) AS nodes UNWIND nodes AS node DELETE node"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.nodes_deleted, 10)
        self.env.assertEquals(actual_result.relationships_deleted, 0)

    def test13_delete_path_elements(self):
        self.graph.query("CREATE ()-[:R]->()")

        # Delete projected
        # Unwind path nodes.
        query = """MATCH p = (src)-[e]->(dest) WITH nodes(p)[0] AS node, relationships(p)[0] as edge DELETE node, edge"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.nodes_deleted, 1)
        self.env.assertEquals(actual_result.relationships_deleted, 1)

    # Verify that variable-length traversals in each direction produce the correct results after deletion.
    def test14_post_deletion_traversal_directions(self):
        nodes = {}
        # Create entities.
        labels = ["Dest", "Src", "Src2"]
        for idx, l in enumerate(labels):
            node = Node(alias=f"n_{idx}", labels=l, properties={"val": idx})
            nodes[l] = node

        edges = [Edge(nodes["Src"], "R", nodes["Dest"]),
                 Edge(nodes["Src2"], "R", nodes["Dest"])]

        nodes_str = [str(n) for n in nodes.values()]
        edges_str = [str(e) for e in edges]
        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")

        # Delete a node.
        query = """MATCH (n:Src2) DELETE n"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.nodes_deleted, 1)
        self.env.assertEquals(actual_result.relationships_deleted, 1)

        query = """MATCH (n1:Src)-[*]->(n2:Dest) RETURN COUNT(*)"""
        actual_result = self.graph.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Perform the same traversal, this time traveling from destination to source.
        query = """MATCH (n1:Src)-[*]->(n2:Dest {val: 0}) RETURN COUNT(*)"""
        actual_result = self.graph.query(query)
        expected_result = [[1]]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test15_update_deleted_entities(self):
        self.graph.delete()
        self.graph.query("CREATE ()-[:R]->()")

        # Attempt to update entities after deleting them.
        query = """MATCH (a)-[e]->(b) DELETE a, b SET a.v = 1, e.v = 2, b.v = 3"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.nodes_deleted, 2)
        self.env.assertEquals(actual_result.relationships_deleted, 1)
        # No properties should be set.
        # (Note that this behavior is left unspecified by Cypher.)
        # self.env.assertEquals(actual_result.properties_set, 0)

        # Validate that the graph is empty.
        query = """MATCH (a) RETURN a"""
        actual_result = self.graph.query(query)
        expected_result = []
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test16_repeated_entity_deletion(self):
        # create 2 nodes cyclically connected by 2 edges
        actual_result = self.graph.query("CREATE (x1:A)-[r:R]->(n2:B)-[t:T]->(x1)")
        self.env.assertEquals(actual_result.nodes_created, 2)
        self.env.assertEquals(actual_result.relationships_created, 2)

        # attempt to repeatedly delete edges
        query = """MATCH ()-[r]-() delete r delete r, r delete r, r"""
        actual_result = self.graph.query(query)
        # 2 edges should be reported as deleted
        self.env.assertEquals(actual_result.relationships_deleted, 2)

        # attempt to repeatedly delete nodes
        query = """MATCH (n) delete n delete n, n delete n, n"""
        actual_result = self.graph.query(query)
        # 2 nodes should be reported as deleted
        self.env.assertEquals(actual_result.nodes_deleted, 2)

    def test17_invalid_deletions(self):
        self.graph.query("CREATE ()")

        # try to delete a value that's not a graph entity
        try:
            query = """DELETE 1"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("DELETE can only be called on nodes, paths and relationships", str(e))

        # try to delete the output of a nonexistent function call
        try:
            query = """DELETE x()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Unknown function 'x'", str(e))

        # try to delete with no child op
        try:
            query = """DELETE rand()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Delete was constructed without a child operation", str(e))

        # try to delete a function return that's not a graph entity
        try:
            query = """MATCH (a) DELETE rand()"""
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Delete type mismatch", str(e))

        # try deleting all scalar types at runtime
        queries = ["WITH 1 AS n DELETE n",
                   "WITH 'str' AS n DELETE n",
                   "WITH true AS n DELETE n",
                   "WITH [] AS n DELETE n",
                   "WITH {} AS n DELETE n"]
        for query in queries:
            try:
                self.graph.query(query)
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Delete type mismatch", str(e))

    def test18_delete_self_edge(self):
        self.graph.query("CREATE (:person{name:'roi',age:32})")
        self.graph.query("CREATE (:person{name:'amit',age:30})")
        self.graph.query("MATCH (a:person) WHERE (a.name = 'roi') DELETE a")

        self.graph.query("CREATE (:person{name:'roi',age:32})")
        self.graph.query("MATCH (a:person), (b:person) WHERE (a.name = 'roi' AND b.name='amit')  CREATE (a)-[:knows]->(a)")
        res = self.graph.query("MATCH (a:person) WHERE (a.name = 'roi') DELETE a")

        self.env.assertEquals(res.nodes_deleted, 1)
        self.env.assertEquals(res.relationships_deleted, 1)

    def test19_random_delete(self):
        # test random graph deletion added as a result of a crash found in Graph_GetNodeEdges
        # when iterating Delta_Matrix of type BOOL with Delta_MatrixTupleIter_next_UINT64
        for i in range(1, 10):
            self.graph.delete()

            query = """UNWIND range(0, 10000) AS x CREATE (src:N {v: x}), (src)-[:R]->(:N), (src)-[:R]->(:N), (src)-[:R]->(:N)"""
            self.graph.query(query)

            query = """MATCH (n:N {v: floor(rand()*100001)}) DELETE n RETURN 1 LIMIT 1"""
            for _ in range(1, 10):
                self.graph.query(query)

    def test20_consecutive_delete_clauses(self):
        """Tests that consecutive `DELETE` clauses are handled correctly."""

        # clean the db
        self.graph.delete()

        # create a graph with 2 nodes, with labels N and M
        self.graph.query("CREATE (n:N) CREATE (m:M)")

        # delete the nodes in 2 consecutive delete clauses
        res = self.graph.query("MATCH p1=(n:N), p2=(m:M) DELETE nodes(p1)[0] \
            DELETE nodes(p2)[0]")

        # validate that the nodes were deleted
        self.env.assertEquals(res.nodes_deleted, 2)

        # create 2 nodes, with the same label N
        self.graph.query("CREATE (:N), (:N)")
        res = self.graph.query("MATCH p=(n:N) DELETE nodes(p)[0] DELETE \
            nodes(p)[0]")

        # validate that the nodes were deleted
        self.env.assertEquals(res.nodes_deleted, 2)

    def test21_not_existed_label(self):
        res = self.graph.query("CREATE (n:Foo:Bar)")
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.labels_added, 2)

        res = self.graph.query("MATCH (n) REMOVE n:Bar")
        self.env.assertEquals(res.labels_removed, 1)

        res = self.graph.query("MATCH (n) REMOVE n:Bar")
        self.env.assertEquals(res.labels_removed, 0)

        res = self.graph.query("MATCH (n:Bar) RETURN count(n)")
        self.env.assertEquals(res.result_set[0][0], 0)

    def test22_delete_reserve_id(self):
        # clean the db
        self.graph.delete()

        res = self.graph.query("UNWIND range(0, 10) AS i CREATE (:A {id: i})")
        self.env.assertEquals(res.nodes_created, 11)

        # expecting IDs to be reused
        res = self.graph.query("MATCH (a:A) DELETE a CREATE (b:A) RETURN id(b)")
        self.env.assertEquals(res.nodes_deleted, 11)
        self.env.assertEquals(res.nodes_created, 11)
        self.env.assertEquals(res.result_set, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

        res = self.graph.query("MATCH (a:A) DELETE a CREATE (b:A) RETURN id(b)")
        self.env.assertEquals(res.nodes_deleted, 11)
        self.env.assertEquals(res.nodes_created, 11)
        self.env.assertEquals(res.result_set, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

        # clean the db
        self.graph.delete()

        res = self.graph.query("UNWIND range(0, 10) AS i CREATE (:A {id: i})")
        self.env.assertEquals(res.nodes_created, 11)

        res = self.graph.query("MATCH (a:A) WITH a, a.id as id DELETE a MERGE (b:A {id: id}) RETURN id(b), b.id")
        self.env.assertEquals(res.nodes_deleted, 11)
        self.env.assertEquals(res.nodes_created, 11)
        self.env.assertEquals(res.result_set, [[10, 0], [9, 1], [8, 2], [7, 3], [6, 4], [5, 5], [4, 6], [3, 7], [2, 8], [1, 9], [0, 10]])

        res = self.graph.query("MATCH (a:A) WITH a, a.id as id DELETE a MERGE (b:A {id: id}) RETURN id(b), b.id")
        self.env.assertEquals(res.nodes_deleted, 11)
        self.env.assertEquals(res.nodes_created, 11)
        self.env.assertEquals(res.result_set, [[10, 10], [9, 9], [8, 8], [7, 7], [6, 6], [5, 5], [4, 4], [3, 3], [2, 2], [1, 1], [0, 0]])

    def test23_delete_edges(self):
        # clean the db
        self.graph.delete()

        # test deleting edges delete the matrix entries correctly
        # GraphBLAS bug fixed in v9.1.0 https://github.com/DrTimothyAldenDavis/GraphBLAS/commit/01a3b746f29ea3bf03e7599b54d5e9a2b5e9dddb
        self.graph.query("UNWIND range(1, 1000000) AS v CREATE (:N {v: v})")

        for i in range(1, 1000):
            self.graph.query("MATCH (n:N) WITH n LIMIT 10000 DELETE n")
            self.graph.query("MATCH (n:N) RETURN n.v LIMIT 1")