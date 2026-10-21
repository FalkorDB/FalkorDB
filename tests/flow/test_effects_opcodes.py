"""Effects -- does every opcode replicate at all?

One test per kind of mutation a query can produce: schema, attribute, node,
edge, labels, delete, merge. `test_effects_shapes.py` asks what a *buffer* has
to survive; this file asks whether each opcode reaches the replica in the first
place.

Effects are the only way a write reaches a replica, so nothing here turns them
on or off -- there is no query-replay alternative left to compare against.

See `effects_common.py` for the shared fixture.
"""

import time

from common import *
from index_utils import *

from effects_common import _EffectsBase


class testEffects_01_OpcodeWalk(_EffectsBase):
    """Every opcode in turn, against one graph that accumulates as it goes.

    The tests are a chain and that is the method, not debt: `test07` relabels
    the `:A:B` node `test03` created carrying eight properties, so SET_LABELS
    is exercised against a real attribute set rather than a bare one, and
    `test08` then removes a label `test07` added. `test02` and `test05` both
    update the `:L` node `test01` introduced. RLTest runs a class's tests in
    sorted order (`RLTest/loader.py:124` iterates `dir(module)`, which sorts),
    so the order holds -- but a test inserted between them has to leave those
    nodes alone.

    `test10` ends with `MATCH (n) DELETE n`; nothing after this class depends
    on anything it leaves behind.
    """

    GRAPH_ID = "effects_opcodes"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT')

        # create indices
        create_node_range_index(self.master_graph, "L", "a", "b", "c")
        create_edge_range_index(self.master_graph, "R", "a", "b", "c")
        self.wait_for_replica_offset()

        # Index DDL ships as effects of its own. Discard them so the first
        # test's window starts empty and `assert_effect_emitted` counts only
        # what that test wrote.
        self.monitor_mark()

    def test01_add_schema_effect(self):
        # test the introduction of a schema by an effect

        # introduce a new label which in turn creates a new schema
        q = "CREATE (:L)"
        res = self.query_and_sync(q)
        self.env.assertEqual(res.nodes_created, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # introduce multiple labels
        q = "CREATE (:X:Y)"
        res = self.query_and_sync(q)
        self.env.assertEqual(res.labels_added, 2)
        self.env.assertEqual(res.nodes_created, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # introduce a new relationship-type which in turn creates a new schema
        q = "CREATE ()-[:R]->()"
        res = self.query_and_sync(q)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test02_add_attribute_effect(self):
        # test the introduction of an attribute by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        # set a new attribute for each supported attribute type
        q = """MATCH (n:L) WITH n
                LIMIT 1
                SET
                n.a = 1,
                n.b = 'str',
                n.c = True,
                n.d = [1, [2], '3'],
                n.v = vecf32([1.0, 2.0, 3.0])
            """

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 5)

        self.assert_effect_emitted()

        q = """MATCH ()-[e]->()
                WITH e
                LIMIT 1
                SET
                e.e = point({latitude: 51, longitude: 0}),
                e.f=3.14,
                e.empty_string = '',
                e.v = vecf32([1.0, 2.0, 3.0])
            """

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 4)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test03_create_node_effect(self):
        # test the introduction of a new node by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        # empty node
        q0 = """CREATE ()"""

        # label-less node with attributes
        q1 = """CREATE ({
                            i:1,
                            s:'str',
                            b:True,
                            a:[1, [2], '3'],
                            p:point({latitude: 51, longitude: 0}),
                            f:3.14,
                            empty_string: '',
                            v: vecf32([1.0, 2.0, 3.0])
                        })"""

        # labeled node without attributes
        q2 = """CREATE (:L)"""

        # node with multiple labels and attributes
        q3 = """CREATE (:A:B {
                            i:1,
                            s:'str',
                            b:True,
                            a:[1, [2], '3'],
                            p:point({latitude: 51, longitude: 0}),
                            f:3.14,
                            empty_string: '',
                            v: vecf32([1.0, 2.0, 3.0])
                        })"""

        queries = [q0, q1, q2, q3]
        for q in queries:
            res = self.query_and_sync(q)
            self.env.assertEqual(res.nodes_created, 1)

            self.assert_effect_emitted()

        self.assert_graph_eq()

    def test04_create_edge_effect(self):
        # tests the introduction of a new edge by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        # edge without attributes
        q1 = """CREATE ()-[:R]->()"""

        # edge with attributes
        q2 = """CREATE ()-[:CONNECT {
                                      ei:1,
                                      s:'str',
                                      eb:True,
                                      a:[1, [2], '3'],
                                      ep:point({latitude: 51, longitude: 0}),
                                      f:3.14,
                                      empty_string: '',
                                      v: vecf32([1.0, 2.0, 3.0])}
                            ]->()"""

        # edge between an existing node and a new node
        q3 = """MATCH (a) WITH a LIMIT 1 CREATE (a)-[:R]->()"""

        # edge between two existing nodes
        q4 = """MATCH (a), (b) WITH a, b LIMIT 1 CREATE (a)-[:R]->(b)"""

        queries = [q1, q2, q3, q4]
        for q in queries:
            res = self.query_and_sync(q)
            self.env.assertEqual(res.relationships_created, 1)

            self.assert_effect_emitted()

        self.assert_graph_eq()

    def test05_update_node_effect(self):
        # test an entity attribute set update by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               SET
                    n.xa = 2,
                    n.b = 'string',
                    n.xc = False,
                    n.d = [[2], 1, '3'],
                    n.xe = point({latitude: 41, longitude: 2}),
                    n.f=6.28,
                    n.xempty_string = '',
                    n.v = vecf32([-1.0, -2.0, -3.0])"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update the same attribute multiple times
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               UNWIND range(0, 10) AS i
               SET
                    n.xa = n.xa + 1"""

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update using map overwrite
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               SET n = {
                a:3,
                b:'_string_',
                c:True,
                d:[['3'], 2, 1],
                e:point({latitude: 2, longitude: 41}),
                f:2.68,
                empty_string:'',
                v: vecf32([-1.1, 2.2, -3.3])}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update using map addition
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               SET n += {
                a:4,
                b:'string_',
                c:False,
                d:[['1'], 3, 2.0],
                e:point({latitude: 3, longitude: 40}),
                f:8.26,
                empty_string:'',
                v: vecf32([-1.2, 2.4, -3.6])}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove attribute

        q = "MATCH (n:L) WITH n LIMIT 1 SET n.b = NULL"

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove all attributes

        q = "MATCH (n:L) WITH n LIMIT 1 SET n = {}"

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # add attribute, remove all attributes and add again
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               SET n.v = 'value'
               WITH n
               SET n = {}
               WITH n
               SET n.v = 'value2'"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove attribute via map addition
        q = """MATCH (n:L)
               WITH n
               LIMIT 1
               SET n += {x:1, v:NULL, y:2}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test06_update_edge_effect(self):

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        # test an edge attribute set update by an effect
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               SET
                    e.a = 2,
                    e.b = 'string',
                    e.c = False,
                    e.d = [[2], 1, '3'],
                    e.e = point({latitude: 41, longitude: 2}),
                    e.f=6.28,
                    e.empty_string = '',
                    e.v = vecf32([-1.0, -2.0, -3.0])"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update the same attribute multiple times
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               UNWIND range(0, 10) AS i
               SET
                    e.a = e.a + 1"""

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update using map overwrite
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               SET e = {
                a:3,
                b:'_string_',
                c:True,
                d:[['3'], 2, 1],
                e:point({latitude: 2, longitude: 41}),
                f:2.68,
                empty_string:'',
                v: vecf32([-1.1, 2.2, -3.3])}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # update using map addition
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               SET e += {
                a:4,
                b:'string_',
                c:False,
                d:[['1'], 3, 2.0],
                e:point({latitude: 3, longitude: 40}),
                f:8.26,
                empty_string:'',
                v: vecf32([-1.2, 2.4, -3.6])}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove attribute

        q = "MATCH ()-[e]->() WITH e LIMIT 1 SET e.b = NULL"

        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove all attributes

        q = "MATCH ()-[e]->() WITH e LIMIT 1 SET e = {}"

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # add attribute, remove all attributes and add again
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               SET e.v = 'value'
               WITH e
               SET e = {}
               WITH e
               SET e.v = 'value2'"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # remove attribute via map addition
        q = """MATCH ()-[e]->()
               WITH e
               LIMIT 1
               SET e += {x:1, v:NULL, y:2}"""

        res = self.query_and_sync(q)
        self.env.assertGreater(res.properties_set, 0)
        self.env.assertGreater(res.properties_removed, 0)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test07_set_labels_effect(self):
        # test the addition of a new node label by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MATCH (n:A:B) SET n:C"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.labels_added, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # test the addition of an existing and anew node label by an effect
        q = """MATCH (n:A:B:C) SET n:C:D"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.labels_added, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test08_remove_labels_effect(self):
        # test the removal of a node label by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MATCH (n:C) REMOVE n:C RETURN n"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.labels_removed, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test09_delete_edge_effect(self):
        # test the deletion of an edge by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MATCH ()-[e]->() WITH e LIMIT 1 DELETE e"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.relationships_deleted, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test10_delete_node_effect(self):
        # test the deletion of a node by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        # using 'n' and 'x' to try and introduce "duplicated" deletions
        q = "MATCH (n) WITH n as n, n as x DELETE n, x"
        res = self.query_and_sync(q)
        self.env.assertGreater(res.nodes_deleted, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()


class testEffects_02_Merge(_EffectsBase):
    """MERGE, which commits either a create or an update and has to replicate
    whichever one it chose.

    Split from the walk above because its two branches (`ON CREATE` /
    `ON MATCH`) are the property under test, not the opcode: the same statement
    must produce a CREATE_NODE on one run and an UPDATE_NODE on the next.
    """

    GRAPH_ID = "effects_merge"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT')
        # Discard anything setup put on the wire so the first test's window
        # starts empty and `assert_effect_emitted` counts only what it wrote.
        self.monitor_mark()

    def test01_merge_node(self):
        # test create and update of a node by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MERGE (n:A {v:'red'})
               ON MATCH SET n.v = 'green'
               ON CREATE SET n.v = 'blue'"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.nodes_created, 1)
        self.env.assertEqual(res.properties_set, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # this time MERGE will match
        q = """MERGE (n:A {v:'blue'})
               ON MATCH SET n.v = 'green'
               ON CREATE SET n.v = 'red'"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 1)
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

    def test02_merge_edge(self):
        # test create and update of an edge by an effect

        # no leftovers from previous test
        self.assert_effect_emitted(0)

        q = """MERGE (n:A {v:'red'})
               MERGE (n)-[e:R{v:'red'}]->(n)
               ON MATCH SET e.v = 'green'
               ON CREATE SET e.v = 'blue'"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 2)
        self.env.assertEqual(res.relationships_created, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()

        # this time MERGE will match
        q = """MERGE (n:A {v:'red'})
               MERGE (n)-[e:R{v:'blue'}]->(n)
               ON MATCH SET e.v = 'green'
               ON CREATE SET e.v = 'red'"""
        res = self.query_and_sync(q)
        self.env.assertEqual(res.properties_set, 1)
        self.env.assertEqual(res.properties_removed, 1)

        self.assert_effect_emitted()

        self.assert_graph_eq()
