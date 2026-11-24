from common import *


class testAccessDelNode():
    def __init__(self):
        GRAPH_ID = "access_del_node"
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_return_deleted_attribute(self):
        # try to return an attribute of a deleted entity

        # create a node
        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 1)

        # retrieve attribute from a deleted node
        q = "MATCH (n) DELETE n RETURN n.v"
        res = self.graph.query(q)
        self.env.assertEqual(res.result_set[0][0], 1)
    
    def test02_return_deleted_node(self):
        # try to return a deleted node
        # expecting node ID and attributes to be returned

        # create a node
        n = Node(labels="A", properties = {'v':1})
        self.graph.query(f"CREATE {n}")

        # return a deleted node
        q = "MATCH (n) DELETE n RETURN n"
        res = self.graph.query(q)
        deleted_node = res.result_set[0][0]

        self.env.assertEqual(n.properties, deleted_node.properties)
        self.env.assertEqual(deleted_node.labels, ['A'])

    def test03_deleted_node_as_argument(self):
        # try to invoke a function on a deleted node

        # create a node
        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 1)

        # invoke function on a deleted node
        q = "MATCH (n) DELETE n RETURN labels(n)"
        res = self.graph.query(q)
        self.env.assertEqual(res.result_set[0][0], ['A'])

        #-----------------------------------------------------------------------

        # create a node
        q = "CREATE (:A {a:1, b:'str', c:[1,2,3]})"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 1)

        # invoke function on a deleted node
        q = "MATCH (n) DELETE n RETURN properties(n)"
        res = self.graph.query(q)
        properties = {'a': 1, 'b': 'str', 'c': [1,2,3]}
        self.env.assertEqual(res.result_set[0][0], properties)

    def test04_update_deleted_node(self):
        # try to update a deleted node

        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 1)

        # update a deleted node
        q = "MATCH (n) DELETE n SET n.v = 2 RETURN n.v"
        res = self.graph.query(q)
        self.env.assertEqual(res.properties_set, 0)
        self.env.assertEqual(res.result_set[0][0], 1)

        #-----------------------------------------------------------------------

        q = "CREATE (:A {v:'value'}), (:B {v:1})"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 2)

        # set existing node to a deleted node
        q = "MATCH (a:A), (b:B) DELETE a SET b = a RETURN b.v"
        res = self.graph.query(q)
        self.env.assertEqual(res.properties_set, 1)
        self.env.assertEqual(res.result_set[0][0], 'value')

        # clear graph
        self.graph.delete()

        #-----------------------------------------------------------------------

        q = "CREATE (:A {a:'value'}), (:B {b:1})"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 2)

        # set existing node to a deleted node
        q = "MATCH (a:A), (b:B) DELETE a SET b += a RETURN b.a, b.b"
        res = self.graph.query(q)
        self.env.assertEqual(res.properties_set, 1)
        self.env.assertEqual(res.result_set[0][0], 'value')
        self.env.assertEqual(res.result_set[0][1], 1)

    def test05_update_deleted_node_lables(self):
        # clear graph
        self.graph.delete()

        # Add label to a deleted node

        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 1)

        # add label to a deleted node
        q = "MATCH (n) DELETE n SET n:A"
        res = self.graph.query(q)
        self.env.assertEqual(res.labels_added, 0)
        self.env.assertEqual(res.nodes_deleted, 1)

        #-----------------------------------------------------------------------

        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 1)

        q = "MATCH (n) DELETE n REMOVE n:A"
        res = self.graph.query(q)
        self.env.assertEqual(res.labels_removed, 0)
        self.env.assertEqual(res.nodes_deleted, 1)

    def test06_merge_using_deleted_node_attr(self):
        # try to merge a node based on a deleted node attribute
        
        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 1)

        q = "MATCH (n) DELETE n MERGE (m {v:n.v+2}) RETURN m.v"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_deleted, 1)
        self.env.assertEqual(res.nodes_created, 1)
        self.env.assertEqual(res.result_set[0][0], 3)

    def test07_dobule_node_delete(self):
        # clear graph
        self.graph.delete()

        # try to delete a deleted node
        # expecting a single delete to be performed
        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 1)

        # delete a deleted node
        q = "MATCH (n) DELETE n WITH n DELETE n"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_deleted, 1)

    def test08_create_edge_to_deleted_node(self):
        # try to create an edge to a deleted node
        # expecting an exception

        q = "CREATE (:A)"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 1)

        try:
            q = "MATCH (n) DELETE n CREATE ()-[:R]->(n)"
            res = self.graph.query(q)
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertEqual(str(e), "Failed to create relationship; endpoint was not found.")

        #-----------------------------------------------------------------------

        q = "CREATE (:A)"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 1)

        try:
            q = "MATCH (n) DELETE n CREATE (n)-[:R]->()"
            res = self.graph.query(q)
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertEqual(str(e), "Failed to create relationship; endpoint was not found.")

    def test09_path_with_deleted_node(self):
        # test path with deleted node
        # create a 3 nodes path (a)->(b)->(c)
        a  = Node(alias="a", labels="A", properties = {'v':'a'})
        b  = Node(alias="b", labels="B", properties = {'v':'b'})
        c  = Node(alias="c", labels="C", properties = {'v':'c'})
        r1 = Edge(a, 'R', b, alias="r1",)
        r2 = Edge(b, 'R', c, alias="r2")
        self.graph.query(f"CREATE {a}, {b}, {c}, {r1}, {r2}")

        # delete the middle node on the path
        q = "MATCH p = (a:A)-[:R]->(b:B)-[:R]->(c:C) DELETE b RETURN nodes(p)"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_deleted, 1)
        nodes = res.result_set[0][0]
        self.env.assertEqual(len(nodes), 3)

        # assert individual nodes
        self.env.assertEqual(nodes[0].properties['v'], 'a')
        self.env.assertContains('A', nodes[0].labels)

        self.env.assertEqual(nodes[1].properties['v'], 'b')
        self.env.assertEqual(nodes[1].labels, ['B'])

        self.env.assertEqual(nodes[2].properties['v'], 'c')
        self.env.assertContains('C', nodes[2].labels)

    def test10_deleted_node_keeps_labels_staged_in_the_same_query(self):
        # #3245. A deleted node keeps everything it had at the DELETE, and that
        # includes labels this query set or removed before it. The snapshot used
        # to read the committed labels, losing the staged ones. C returns no
        # labels at all for a deleted node; keeping them is Rust's rule.
        #
        # `DELETE n` takes the bulk path, `DELETE coalesce(n)` the per-entity one.
        for target in ("n", "coalesce(n)"):
            g = self.db.select_graph(f"del_staged_labels_set_{target}")
            g.query("CREATE (:A {v: 1})")
            res = g.query(
                f"MATCH (n:A) SET n:C DELETE {target} RETURN labels(n), n:C, n:A")
            labels, has_c, has_a = res.result_set[0]
            self.env.assertEqual(sorted(labels), ['A', 'C'], message=target)
            self.env.assertEqual([has_c, has_a], [True, True], message=target)
            self.env.assertEqual(res.nodes_deleted, 1)
            # Nothing staged on the deleted node leaks into the graph.
            self.env.assertEqual(
                g.query("MATCH (n) RETURN count(n)").result_set, [[0]])

            g = self.db.select_graph(f"del_staged_labels_remove_{target}")
            g.query("CREATE (:A:B {v: 1})")
            res = g.query(
                f"MATCH (n:A) REMOVE n:A DELETE {target} RETURN labels(n), n:A, n:B")
            self.env.assertEqual(res.result_set[0], [['B'], False, True], message=target)
            self.env.assertEqual(res.nodes_deleted, 1)
            self.env.assertEqual(
                g.query("MATCH (n) RETURN count(n)").result_set, [[0]])


class testAccessDelEdge():
    def __init__(self):
        GRAPH_ID = "access_del_edge"
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_return_deleted_attribute(self):
        # try to return an attribute of a deleted entity

        # create an edge
        q = "CREATE ()-[:R {v:1}]->()"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 2)
        self.env.assertEqual(res.relationships_created, 1)

        # retrieve attribute from a deleted edge
        q = "MATCH ()-[e]->() DELETE e RETURN e.v"
        res = self.graph.query(q)
        self.env.assertEqual(res.result_set[0][0], 1)
    
    def test02_return_deleted_edge(self):
        # try to return a deleted edge

        # create an edge
        src  = Node(alias="src",  labels="A", properties = {'v':1})
        dest = Node(alias="dest", labels="B", properties = {'v':2})
        e    = Edge(src, 'R', dest, properties = {'v':3})
        self.graph.query(f"CREATE {src}, {dest}, {e}")

        # return a deleted edge
        q = "MATCH ()-[e]->() DELETE e RETURN e"
        res = self.graph.query(q)
        deleted_edge = res.result_set[0][0]

        self.env.assertEqual(e.relation, deleted_edge.relation)
        self.env.assertEqual(e.properties, deleted_edge.properties)

    def test03_deleted_edge_as_argument(self):
        # try to invoke a function on a deleted edge

        # create an edge
        q = "CREATE ()-[:R{v:1}]->()"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 2)
        self.env.assertEqual(res.relationships_created, 1)

        # invoke function on a deleted edge
        q = "MATCH ()-[e]->() DELETE e RETURN type(e)"
        res = self.graph.query(q)
        self.env.assertEqual(res.result_set[0][0], "R")

    def test04_update_deleted_edge(self):
        # try to update a deleted edge

        q = "CREATE ()-[:R{v:1}]->()"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 2)
        self.env.assertEqual(res.relationships_created, 1)

        # update a deleted edge
        q = "MATCH ()-[e]->() DELETE e SET e.v = 2 RETURN e.v"
        res = self.graph.query(q)
        self.env.assertEqual(res.properties_set, 0)
        self.env.assertEqual(res.result_set[0][0], 1)

        #-----------------------------------------------------------------------

        q = "CREATE ()-[:A{v:'value'}]->(), ()-[:B{v:1}]->()"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 4)
        self.env.assertEqual(res.relationships_created, 2)

        # set existing edge to a deleted edge
        q = "MATCH ()-[a:A]->(), ()-[b:B]->() DELETE a SET b = a RETURN b.v"
        res = self.graph.query(q)
        self.env.assertEqual(res.properties_set, 1)
        self.env.assertEqual(res.result_set[0][0], "value")

        # clear graph
        self.graph.delete()

        #-----------------------------------------------------------------------

        q = "CREATE ()-[:A{a:'value'}]->(), ()-[:B{b:1}]->()"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 4)
        self.env.assertEqual(res.relationships_created, 2)

        # set existing edge to a deleted edge
        q = "MATCH ()-[a:A]->(), ()-[b:B]->() DELETE a SET b += a RETURN b.a, b.b"
        res = self.graph.query(q)
        self.env.assertEqual(res.properties_set, 1)
        self.env.assertEqual(res.result_set[0][0], 'value')
        self.env.assertEqual(res.result_set[0][1], 1)

        # clear graph
        self.graph.delete()

    def test05_merge_using_deleted_edge_attr(self):
        # try to merge an edge based on a deleted edge attribute

        q = "CREATE ()-[:R{v:1}]->()"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 2)
        self.env.assertEqual(res.relationships_created, 1)

        q = "MATCH ()-[e]->() DELETE e MERGE (m {v:e.v+2}) RETURN m.v"
        res = self.graph.query(q)
        self.env.assertEqual(res.result_set[0][0], 3)

    def test06_merge_using_deleted_edge_attr(self):
        # try to merge an edge based on a deleted edge attribute
        
        q = "CREATE ()-[:R{v:1}]->()"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 2)
        self.env.assertEqual(res.relationships_created, 1)

        q = "MATCH (a)-[e]->(b) DELETE e MERGE (a)-[:R {v:e.v+2}]->(b) RETURN e.v"
        res = self.graph.query(q)
        self.env.assertEqual(res.result_set[0][0], 1)
        self.env.assertEqual(res.relationships_deleted, 1)
        self.env.assertEqual(res.relationships_created, 1)

    def test07_dobule_edge_delete(self):
        # clear graph
        self.graph.delete()

        # try to delete a deleted edge
        # expecting a single delete to be performed
        q = "CREATE ()-[:R]->()"
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 2)
        self.env.assertEqual(res.relationships_created, 1)

        # delete a deleted edge
        q = "MATCH ()-[e]->() DELETE e WITH e DELETE e"
        res = self.graph.query(q)
        self.env.assertEqual(res.relationships_deleted, 1)

    def test08_path_with_deleted_edge(self):
        # test path with deleted edge
        # create a 3 nodes path (a)->(b)->(c)
        a  = Node(alias="a", labels="A",  properties = {'v':'a'})
        b  = Node(alias="b", labels="B",  properties = {'v':'b'})
        c  = Node(alias="c", labels="C",  properties = {'v':'c'})
        r1 = Edge(a, 'R1', b, alias="r1", properties = {'v':1})
        r2 = Edge(b, 'R2', c, alias="r2", properties = {'v':2})

        self.graph.query(f"CREATE {a}, {b}, {c}, {r1}, {r2}")

        # delete the middle node on the path
        q = "MATCH p = (:A)-[e1:R1]->(:B)-[e2:R2]->(:C) DELETE e1 RETURN relationships(p)"
        res = self.graph.query(q)
        self.env.assertEqual(res.relationships_deleted, 1)
        edges = res.result_set[0][0]
        self.env.assertEqual(len(edges), 2)

        # assert individual edges
        self.env.assertEqual(edges[0].properties['v'], 1)
        self.env.assertEqual(edges[0].relation, 'R1')

        self.env.assertEqual(edges[1].properties['v'], 2)
        self.env.assertEqual(edges[1].relation, 'R2')

    def test09_implicitly_deleted_edge_in_write_only_query(self):
        # an edge removed implicitly by DETACH DELETE must remain readable by
        # later clauses even when the query has no RETURN clause
        self.graph.delete()

        q = """CREATE (a:EdgeCluster)-[r:R]->(a)
               WITH r, a DETACH DELETE a
               WITH startNode(r) AS ec DETACH DELETE ec"""
        res = self.graph.query(q)
        self.env.assertEqual(res.nodes_created, 1)
        self.env.assertEqual(res.relationships_created, 1)
        self.env.assertEqual(res.nodes_deleted, 1)
        self.env.assertEqual(res.relationships_deleted, 1)

        # same shape, but reading the implicitly deleted edge's type and
        # endpoints across two separate nodes
        self.graph.delete()

        q = """CREATE (a:A {v:'a'})-[r:R {v:1}]->(b:B {v:'b'})
               WITH r, a DETACH DELETE a
               WITH startNode(r) AS s, endNode(r) AS e, type(r) AS t, r AS r
               RETURN s.v, e.v, t, r.v"""
        res = self.graph.query(q)
        self.env.assertEqual(res.result_set[0], ['a', 'b', 'R', 1])

