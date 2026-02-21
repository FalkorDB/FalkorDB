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
        self.env.assertEquals(res.nodes_created, 1)

        # retrieve attribute from a deleted node
        q = "MATCH (n) DELETE n RETURN n.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set[0][0], 1)
    
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

        self.env.assertEquals(n.properties, deleted_node.properties)
        self.env.assertEquals(deleted_node.labels, None)

    def test03_deleted_node_as_argument(self):
        # try to invoke a function on a deleted node

        # create a node
        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 1)

        # invoke function on a deleted node
        q = "MATCH (n) DELETE n RETURN labels(n)"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set[0][0], [])

        #-----------------------------------------------------------------------

        # create a node
        q = "CREATE (:A {a:1, b:'str', c:[1,2,3]})"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 1)

        # invoke function on a deleted node
        q = "MATCH (n) DELETE n RETURN properties(n)"
        res = self.graph.query(q)
        properties = {'a': 1, 'b': 'str', 'c': [1,2,3]}
        self.env.assertEquals(res.result_set[0][0], properties)

    def test04_update_deleted_node(self):
        # try to update a deleted node

        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 1)

        # update a deleted node
        q = "MATCH (n) DELETE n SET n.v = 2 RETURN n.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_set, 0)
        self.env.assertEquals(res.result_set[0][0], 1)

        #-----------------------------------------------------------------------

        q = "CREATE (:A {v:'value'}), (:B {v:1})"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 2)

        # set existing node to a deleted node
        q = "MATCH (a:A), (b:B) DELETE a SET b = a RETURN b.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_set, 1)
        self.env.assertEquals(res.result_set[0][0], 'value')

        # clear graph
        self.graph.delete()

        #-----------------------------------------------------------------------

        q = "CREATE (:A {a:'value'}), (:B {b:1})"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 2)

        # set existing node to a deleted node
        q = "MATCH (a:A), (b:B) DELETE a SET b += a RETURN b.a, b.b"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_set, 1)
        self.env.assertEquals(res.result_set[0][0], 'value')
        self.env.assertEquals(res.result_set[0][1], 1)

    def test05_update_deleted_node_lables(self):
        # clear graph
        self.graph.delete()

        # Add label to a deleted node

        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 1)

        # add label to a deleted node
        q = "MATCH (n) DELETE n SET n:A"
        res = self.graph.query(q)
        self.env.assertEquals(res.labels_added, 0)
        self.env.assertEquals(res.nodes_deleted, 1)

        #-----------------------------------------------------------------------

        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 1)

        q = "MATCH (n) DELETE n REMOVE n:A"
        res = self.graph.query(q)
        self.env.assertEquals(res.labels_removed, 0)
        self.env.assertEquals(res.nodes_deleted, 1)

    def test06_merge_using_deleted_node_attr(self):
        # try to merge a node based on a deleted node attribute
        
        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 1)

        q = "MATCH (n) DELETE n MERGE (m {v:n.v+2}) RETURN m.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_deleted, 1)
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.result_set[0][0], 3)

    def test07_dobule_node_delete(self):
        # clear graph
        self.graph.delete()

        # try to delete a deleted node
        # expecting a single delete to be performed
        q = "CREATE (:A {v:1})"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 1)

        # delete a deleted node
        q = "MATCH (n) DELETE n WITH n DELETE n"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_deleted, 1)

    def test08_create_edge_to_deleted_node(self):
        # try to create an edge to a deleted node
        # expecting an exception

        q = "CREATE (:A)"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 1)

        try:
            q = "MATCH (n) DELETE n CREATE ()-[:R]->(n)"
            res = self.graph.query(q)
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertEquals(str(e), "Failed to create relationship; endpoint was not found.")

        #-----------------------------------------------------------------------

        q = "CREATE (:A)"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 1)

        try:
            q = "MATCH (n) DELETE n CREATE (n)-[:R]->()"
            res = self.graph.query(q)
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertEquals(str(e), "Failed to create relationship; endpoint was not found.")

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
        self.env.assertEquals(res.nodes_deleted, 1)
        nodes = res.result_set[0][0]
        self.env.assertEquals(len(nodes), 3)

        # assert individual nodes
        self.env.assertEquals(nodes[0].properties['v'], 'a')
        self.env.assertIn('A', nodes[0].labels);

        self.env.assertEquals(nodes[1].properties['v'], 'b')
        self.env.assertEquals(nodes[1].labels, None)

        self.env.assertEquals(nodes[2].properties['v'], 'c')
        self.env.assertIn('C', nodes[2].labels)

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
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(res.relationships_created, 1)

        # retrieve attribute from a deleted edge
        q = "MATCH ()-[e]->() DELETE e RETURN e.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set[0][0], 1)
    
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

        self.env.assertEquals(e.relation, deleted_edge.relation)
        self.env.assertEquals(e.properties, deleted_edge.properties)

    def test03_deleted_edge_as_argument(self):
        # try to invoke a function on a deleted edge

        # create an edge
        q = "CREATE ()-[:R{v:1}]->()"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(res.relationships_created, 1)

        # invoke function on a deleted edge
        q = "MATCH ()-[e]->() DELETE e RETURN type(e)"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set[0][0], "R")

    def test04_update_deleted_edge(self):
        # try to update a deleted edge

        q = "CREATE ()-[:R{v:1}]->()"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(res.relationships_created, 1)

        # update a deleted edge
        q = "MATCH ()-[e]->() DELETE e SET e.v = 2 RETURN e.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_set, 0)
        self.env.assertEquals(res.result_set[0][0], 1)

        #-----------------------------------------------------------------------

        q = "CREATE ()-[:A{v:'value'}]->(), ()-[:B{v:1}]->()"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 4)
        self.env.assertEquals(res.relationships_created, 2)

        # set existing edge to a deleted edge
        q = "MATCH ()-[a:A]->(), ()-[b:B]->() DELETE a SET b = a RETURN b.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_set, 1)
        self.env.assertEquals(res.result_set[0][0], "value")

        # clear graph
        self.graph.delete()

        #-----------------------------------------------------------------------

        q = "CREATE ()-[:A{a:'value'}]->(), ()-[:B{b:1}]->()"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 4)
        self.env.assertEquals(res.relationships_created, 2)

        # set existing edge to a deleted edge
        q = "MATCH ()-[a:A]->(), ()-[b:B]->() DELETE a SET b += a RETURN b.a, b.b"
        res = self.graph.query(q)
        self.env.assertEquals(res.properties_set, 1)
        self.env.assertEquals(res.result_set[0][0], 'value')
        self.env.assertEquals(res.result_set[0][1], 1)

        # clear graph
        self.graph.delete()

    def test05_merge_using_deleted_edge_attr(self):
        # try to merge an edge based on a deleted edge attribute

        q = "CREATE ()-[:R{v:1}]->()"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(res.relationships_created, 1)

        q = "MATCH ()-[e]->() DELETE e MERGE (m {v:e.v+2}) RETURN m.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set[0][0], 3)

    def test06_merge_using_deleted_edge_attr(self):
        # try to merge an edge based on a deleted edge attribute
        
        q = "CREATE ()-[:R{v:1}]->()"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(res.relationships_created, 1)

        q = "MATCH (a)-[e]->(b) DELETE e MERGE (a)-[:R {v:e.v+2}]->(b) RETURN e.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set[0][0], 3)
        self.env.assertEquals(res.relationships_deleted, 1)
        self.env.assertEquals(res.relationships_created, 1)

    def test07_dobule_edge_delete(self):
        # clear graph
        self.graph.delete()

        # try to delete a deleted edge
        # expecting a single delete to be performed
        q = "CREATE ()-[:R]->()"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(res.relationships_created, 1)

        # delete a deleted edge
        q = "MATCH ()-[e]->() DELETE e WITH e DELETE e"
        res = self.graph.query(q)
        self.env.assertEquals(res.relationships_deleted, 1)

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
        self.env.assertEquals(res.relationships_deleted, 1)
        edges = res.result_set[0][0]
        self.env.assertEquals(len(edges), 2)

        # assert individual edges
        self.env.assertEquals(edges[0].properties['v'], 1)
        self.env.assertEquals(edges[0].relation, 'R1')

        self.env.assertEquals(edges[1].properties['v'], 2)
        self.env.assertEquals(edges[1].relation, 'R2')

    def test09_uaf_via_dangling_list_reference(self):
        # regression test for issue #1591
        # collecting a node into a list, deleting it, then accessing
        # the collected copy must not cause use-after-free
        q = "CREATE (:A {payload: 123})"
        self.graph.query(q)

        q = """MATCH (n:A)
               WITH collect(n) AS list
               DETACH DELETE list[0]
               RETURN list[0].payload"""
        res = self.graph.query(q)
        # the collected copy should still be accessible
        self.env.assertEquals(res.result_set[0][0], 123)
