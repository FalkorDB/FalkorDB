from common import *
from neo4j import GraphDatabase
from neo4j.spatial import WGS84Point
import neo4j.graph
# from neo4j.debug import watch

# watch("neo4j")

bolt_con = None

CACHE_SIZE = 16

class testBolt():
    def __init__(self):
        self.env = Env(decodeResponses=True)
        global bolt_con
        bolt_con = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    def test01_simple_values(self):
        with bolt_con.session() as session:
            result = session.run("""RETURN null, true, false, -1, 0, 1, 2, 255, 256, 257, 65535, 65536, 65537, 4294967295, 4294967296, 4294967297, 9223372036854775807, 1.23, 'Hello, World!', [1,2,3], {foo:'bar'}, POINT({longitude:1, latitude:2})""")
            record = result.single()
            self.env.assertEquals(record[0], None)
            self.env.assertEquals(record[1], True)
            self.env.assertEquals(record[2], False)
            self.env.assertEquals(record[3], -1)
            self.env.assertEquals(record[4], 0)
            self.env.assertEquals(record[5], 1)
            self.env.assertEquals(record[6], 2)
            self.env.assertEquals(record[7], 255)
            self.env.assertEquals(record[8], 256)
            self.env.assertEquals(record[9], 257)
            self.env.assertEquals(record[10], 65535)
            self.env.assertEquals(record[11], 65536)
            self.env.assertEquals(record[12], 65537)
            self.env.assertEquals(record[13], 4294967295)
            self.env.assertEquals(record[14], 4294967296)
            self.env.assertEquals(record[15], 4294967297)
            self.env.assertEquals(record[16], 9223372036854775807)
            self.env.assertEquals(record[17], 1.23)
            self.env.assertEquals(record[18], 'Hello, World!')
            self.env.assertEquals(record[19], [1,2,3])
            self.env.assertEquals(record[20], {'foo':'bar'})
            self.env.assertEquals(record[21], WGS84Point((1, 2)))
    
    def test02_graph_entities_values(self):
        with bolt_con.session() as session:
            result = session.run("""CREATE (a:A {v: 1})-[r1:R1]->(b:B)<-[r2:R2]-(c:C) RETURN a, r1, b, r2, c""")
            record = result.single()
            a:neo4j.graph.Node = record[0]
            r1:neo4j.graph.Relationship = record[1]
            b:neo4j.graph.Node = record[2]
            r2:neo4j.graph.Relationship = record[3]
            c:neo4j.graph.Node = record[4]

            self.env.assertEquals(a.id, 0)
            self.env.assertEquals(a.labels, set(['A']))

            self.env.assertEquals(r1.id, 0)
            self.env.assertEquals(r1.type, 'R1')
            self.env.assertEquals(r1.start_node, a)
            self.env.assertEquals(r1.end_node, b)

            self.env.assertEquals(b.id, 1)
            self.env.assertEquals(b.labels, set(['B']))

            self.env.assertEquals(r2.id, 1)
            self.env.assertEquals(r2.type, 'R2')
            self.env.assertEquals(r2.start_node, c)
            self.env.assertEquals(r2.end_node, b)

            self.env.assertEquals(c.id, 2)
            self.env.assertEquals(c.labels, set(['C']))

            result = session.run("""MATCH p=(:A)-[:R1]->(:B)<-[:R2]-(:C) RETURN p""")
            record = result.single()
            p:neo4j.graph.Path = record[0]
            self.env.assertEquals(p.start_node.labels, set(['A']))
            self.env.assertEquals(p.end_node.labels, set(['C']))
            self.env.assertEquals(p.nodes[0].labels, set(['A']))
            self.env.assertEquals(p.nodes[1].labels, set(['B']))
            self.env.assertEquals(p.nodes[2].labels, set(['C']))
            self.env.assertEquals(p.relationships[0].type, 'R1')
            self.env.assertEquals(p.relationships[1].type, 'R2')
