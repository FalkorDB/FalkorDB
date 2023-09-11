import asyncio
from common import *
from index_utils import *
from time import sleep, time
from collections import OrderedDict
from execution_plan_util import locate_operation

GRAPH_ID = "index_create"
con = None
graph = None

class testIndexCreationFlow():
    def __init__(self):
        self.env = Env(decodeResponses=True)
        global con
        global graph
        con = self.env.getConnection()
        graph = Graph(con, GRAPH_ID)

    # full-text index creation
    def test01_fulltext_index_creation(self):
        # create an index over L:v0
        result = create_node_fulltext_index(graph, 'L', 'v0')
        self.env.assertEquals(result.indices_created, 1)

        # create an index over L:v1 and L:v2
        result = create_node_fulltext_index(graph, 'L', 'v1', 'v2')
        self.env.assertEquals(result.indices_created, 2)

        # create an index over L:v3, L:v4, L:v5 and L:v6
        result = create_node_fulltext_index(graph, 'L', 'v3', 'v4', 'v5', 'v6', sync=True)
        self.env.assertEquals(result.indices_created, 4)

    def test02_fulltext_index_creation_label_config(self):
        # create an index over L1:v1
        result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L1' }, 'v1')")
        self.env.assertEquals(result.indices_created, 1)

        # create an index over L1:v2, v3
        result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L1' }, 'v2', 'v3')")
        self.env.assertEquals(result.indices_created, 2)

        # create an index over L2:v1 with stopwords
        result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L2', stopwords: ['The'] }, 'v1')")
        self.env.assertEquals(result.indices_created, 1)

        # create an index over L2:v2
        result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L2' }, 'v2')")
        self.env.assertEquals(result.indices_created, 1)

        try:
            # try to create an index, without specifying the label
            result = graph.query("CALL db.idx.fulltext.createNodeIndex({ stopwords: ['The'] }, 'v4')")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Label is missing", str(e))

        # create an index over L1:v4 with stopwords
        result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L1', stopwords: ['The'] }, 'v4')")
        self.env.assertEquals(result.indices_created, 1)

        # try to update L1 index stopwords should failed
        try:
            result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L1', stopwords: ['The'] }, 'v5')")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Can not override index configuration", str(e))

        # create an index over L1:v5 with language
        result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L1', language: 'english' }, 'v5')")
        self.env.assertEquals(result.indices_created, 1)

        # try to update L1 index language should failed
        try:
            result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L1', language: 'english' }, 'v6')")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Can not override index configuration", str(e))

        # drop L1 index
        result = graph.query("CALL db.idx.fulltext.drop('L1')")
        self.env.assertEquals(result.indices_deleted, 5)

        try:
            # create an index over L1:v4 with an unsupported language, expecting to failed
            result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L1', language: 'x' }, 'v4')")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Language is not supported", str(e))

        # create an index over L1:v4 with language
        result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L1', language: 'english' }, 'v4')")
        self.env.assertEquals(result.indices_created, 1)

        try:
            # create an index over L3:v1 with stopwords should failed
            # stopwords must be provided as an array of strings
            result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L3', stopwords: 'The' }, 'v1')")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Stopwords must be array", str(e))

        try:
            # create an index over L3:v1 with language should failed
            # language must be provided as a string
            result = graph.query("CALL db.idx.fulltext.createNodeIndex({ label: 'L3', language: ['english'] }, 'v1')")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Language must be string", str(e))

        try:
            # create an index over L3 should failed, missing field(s)
            result = graph.query("CALL db.idx.fulltext.createNodeIndex('L3', { })")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Field is missing", str(e))

        try:
            # create an index over L3:v1 with weight of type string should failed
            # weight must be provided as numeric
            result = graph.query("CALL db.idx.fulltext.createNodeIndex('L3', { field: 'v1', weight: '1' })")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Weight must be numeric", str(e))

        try:
            # create an index over L3:v1 with nostem of type string should failed
            # nostem must be boolean
            result = graph.query("CALL db.idx.fulltext.createNodeIndex('L3', { field: 'v1', nostem: 'true' })")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Nostem must be bool", str(e))
        
        try:
            # create an index over L3:v1 with phonetic of type bool should failed
            # phonetic must be a string
            result = graph.query("CALL db.idx.fulltext.createNodeIndex('L3', { field: 'v1', phonetic: true })")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Phonetic must be string", str(e))

    def test03_multi_prop_index_creation(self):
        # create an index over person:age and person:name
        result = graph.query("CREATE INDEX ON :person(age, name)")
        self.env.assertEquals(result.indices_created, 2)

        # try to create an index over person:age and person:name, index shouldn't be created as it already exist
        try:
            result = graph.query("CREATE INDEX ON :person(age, name)")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Attribute 'age' is already indexed", str(e))

        # try to create an index over person:name and person:age, index shouldn't be created as it already exist
        try:
            result = graph.query("CREATE INDEX ON :person(name, age)")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Attribute 'name' is already indexed", str(e))

        # try to create an index over person: age, name height,
        # operation should fail as 'age' and 'name' are already indexed
        try:
            result = graph.query("CREATE INDEX ON :person(age, name, height)")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Attribute 'age' is already indexed", str(e))

        # try to create an index over person: gender, name and height
        # operation should fail as 'name' is already indexed
        try:
            result = graph.query("CREATE INDEX ON :person(gender, name, height)")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Attribute 'name' is already indexed", str(e))

        # try to create an index with a duplicated field
        try:
            result = graph.query("CREATE INDEX ON :person(height, height)")
            assert(False)
        except ResponseError as e:
            self.env.assertIn("Attribute 'height' is already indexed", str(e))

    def test04_index_creation_pattern_syntax(self):
        # create an index over user:age and user:name
        result = graph.query("CREATE INDEX FOR (p:user) ON (p.age, p.name)")
        self.env.assertEquals(result.indices_created, 2)

        # create an index over follow:prop1 and follow:prop2
        result = graph.query("CREATE INDEX FOR ()-[r:follow]-() ON (r.prop1, r.prop2)")
        self.env.assertEquals(result.indices_created, 2)

    def test05_index_delete(self):
        if SANITIZER != "":
            self.env.skip()

        def create_drop_index(graph_id):
            env = Env(decodeResponses=True)
            con = env.getConnection()
            for _ in range(1, 100):
                pipe = con.pipeline()
                pipe.execute_command("GRAPH.QUERY", graph_id, "CREATE (a:L), (n:L), (n)-[:T]->(a)")
                pipe.execute_command("GRAPH.QUERY", graph_id, "CREATE INDEX FOR ()-[n:T]-() ON (n.p)")
                pipe.execute()
                wait_for_indices_to_sync(Graph(con, graph_id))
                con.execute_command("GRAPH.DELETE", graph_id)

        if "to_thread" not in dir(asyncio):
            create_drop_index(1)
        else:
            loop = asyncio.get_event_loop()
            tasks = []
            for i in range(1, 20):
                tasks.append(loop.create_task(asyncio.to_thread(create_drop_index, i)))

            loop.run_until_complete(asyncio.wait(tasks))

    def test06_syntax_error_index_creation(self):
        # create index on invalid property name
        try:
            graph.query("CREATE INDEX FOR (p:Person) ON (p.m.n, p.p.q)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid input '.': expected ',' or ')'", str(e))

        # create index on invalid identifier
        try:
            graph.query("CREATE INDEX FOR (p:Person) ON (1.b)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid input '1': expected an identifier", str(e))

        # create index on invalid property name: number
        try:
            graph.query("CREATE INDEX FOR (p:Person) ON (b.1)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid input '1': expected a property name", str(e))

        # create index without label
        try:
            graph.query("CREATE INDEX FOR (Person) ON (surname)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid input ')': expected a label", str(e))

        # create index without property name
        try:
            graph.query("CREATE INDEX FOR (p:Person) ON (surname)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid input ')': expected '.'", str(e))

        # create index without identifier
        try:
            graph.query("CREATE INDEX FOR (p:Person) ON ()")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid input ')': expected an identifier", str(e))

        # create index for relationship on invalid property name
        try:
            graph.query("CREATE INDEX FOR ()-[n:T]-() ON (n.p.q)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid input '.': expected ',' or ')'", str(e))

        # create index for relationship on invalid identifier
        try:
            graph.query("CREATE INDEX FOR ()-[n:T]-() ON (1.b)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid input '1': expected an identifier", str(e))

    def test07_index_creation_undefined_identifier(self):   
        # create index on undefined identifier
        try:
            graph.query("CREATE INDEX FOR (p:Person) ON (a.b)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("'a' not defined", str(e))

        # create index on undefined identifier after defined identifier
        try:
            graph.query("CREATE INDEX FOR (p:Person) ON (p.x, a.b)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("'a' not defined", str(e))
        
        # create index for relationship on undefined identifier
        try:
            graph.query("CREATE INDEX FOR ()-[n:T]-() ON (a.b)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("'a' not defined", str(e))

        # create index for relationship on undefined identifier after defined identifier
        try:
            graph.query("CREATE INDEX FOR ()-[n:T]-() ON (n.x, a.b)")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("'a' not defined", str(e))

    def test08_async_index_creation(self):
        # skip test if we're running under Valgrind
        if VALGRIND:
            self.env.skip()

        # 1. create a large graph
        # 2. create an index
        # 3. while the index is being constructed make sure:
        # 3.a. we're able to write to the graph
        # 3.b. we're able to read
        # 3.c. queries aren't utilizing the index while it is being constructed

        min_node_v = 0
        max_node_v = 1000000

        g = Graph(self.env.getConnection(), "async-index")

        #-----------------------------------------------------------------------
        # create a large graph
        #-----------------------------------------------------------------------

        q = "UNWIND range($min_v, $max_v) AS x CREATE (:L {v:x})"
        g.query(q, {'min_v': min_node_v, 'max_v': max_node_v})

        #-----------------------------------------------------------------------
        # create an index
        #-----------------------------------------------------------------------

        res = create_node_range_index(g, 'L', 'v', sync=False)
        self.env.assertEquals(res.indices_created, 1)

        #-----------------------------------------------------------------------
        # validate index is being populated
        #-----------------------------------------------------------------------

        self.env.assertTrue(index_under_construction(g, 'L'))

        # while the index is being constructed
        # perform CRUD operations

        #-----------------------------------------------------------------------
        # read while index is being constructed
        #-----------------------------------------------------------------------

        q = "MATCH (n:L) WHERE n.v = 41 RETURN n.v LIMIT 1"
        res = g.query(q)
        self.env.assertEquals(res.result_set[0][0], 41)

        plan = g.explain(q)
        self.env.assertIsNone(locate_operation(plan.structured_plan, "Node By Index Scan"))

        #-----------------------------------------------------------------------
        # write while index is being constructed
        #-----------------------------------------------------------------------
    
        # create a new node
        q = "CREATE (:L {v:$v})"
        g.query(q, {'v': max_node_v + 10})

        # update a node which had yet to be indexed
        q = "MATCH (n:L) WHERE ID(n) = $id WITH n LIMIT 1 SET n.v = $new_v"
        g.query(q, {'id': max_node_v - 10, 'new_v': -max_node_v})

        # update a node which is already indexed
        g.query(q, {'id': 1, 'new_v': -1})

        # delete a node which had yet to be indexed
        q = "MATCH (n:L) WHERE ID(n) = $id WITH n LIMIT 1 DELETE n"
        g.query(q, {'id': max_node_v - 9})

        # delete an indexed node
        q = "MATCH (n:L) WHERE ID(n) = $id WITH n LIMIT 1 DELETE n"
        g.query(q, {'id': 2})

        #-----------------------------------------------------------------------
        # validate index is being populated
        #-----------------------------------------------------------------------

        self.env.assertTrue(index_under_construction(g, 'L'))

        # wait for index to become operational
        wait_for_indices_to_sync(g)

        # index should be operational
        self.env.assertFalse(index_under_construction(g, 'L'))

        #-----------------------------------------------------------------------
        # validate index results
        #-----------------------------------------------------------------------

        # index should be utilized
        q = "MATCH (n:L) WHERE n.v = 41 RETURN n.v LIMIT 1"
        plan = g.explain(q)
        self.env.assertIsNotNone(locate_operation(plan.structured_plan, "Node By Index Scan"))

        # find newly created node
        q = "MATCH (n:L {v:$v}) RETURN count(n)"
        res = g.query(q, {'v': max_node_v + 10}).result_set
        self.env.assertEquals(res[0][0], 1)

        # find updated nodes
        q = "MATCH (n:L) WHERE n.v = $new_v RETURN count(n)"
        res = g.query(q, {'new_v': -max_node_v}).result_set
        self.env.assertEquals(res[0][0], 1)

        # find updated node
        res = g.query(q, {'new_v': -1}).result_set
        self.env.assertEquals(len(res), 1)

        # make sure deleted nodes aren't found 
        q = "MATCH (n:L) WHERE n.v = $id RETURN count(n)"
        res = g.query(q, {'id': max_node_v - 9}).result_set
        self.env.assertEquals(res[0][0], 0)
        res = g.query(q, {'id': 2}).result_set
        self.env.assertEquals(res[0][0], 0)

    def test09_async_fulltext_index_creation(self):
        # skip test if we're running under Valgrind
        if VALGRIND:
            self.env.skip()

        # 1. create a large graph
        # 2. create an index
        # 3. while the index is being constructed make sure:
        # 3.a. we're able to write to the graph
        # 3.b. we're able to read
        # 3.c. queries aren't utilizing the index while it is being constructed

        min_node_v = 0
        max_node_v = 1000000
        g = Graph(self.env.getConnection(), "async-fulltext-index")

        #-----------------------------------------------------------------------
        # create a large graph
        #-----------------------------------------------------------------------

        q = "UNWIND range($min_v, $max_v) AS x CREATE (:L {h:toString(x)})"
        g.query(q, {'min_v': min_node_v, 'max_v': max_node_v})

        #-----------------------------------------------------------------------
        # create a fulltext index
        #-----------------------------------------------------------------------

        res = create_node_fulltext_index(g, 'L', 'h', sync=False)
        self.env.assertEquals(res.indices_created, 1)

        #-----------------------------------------------------------------------
        # validate index is being populated
        #-----------------------------------------------------------------------

        self.env.assertTrue(index_under_construction(g, 'L'))

        # while the index is being constructed
        # perform CRUD operations

        #-----------------------------------------------------------------------
        # read while index is being constructed
        #-----------------------------------------------------------------------

        q = "RETURN 1"
        res = g.query(q)
        self.env.assertEquals(res.result_set[0][0], 1)

        #-----------------------------------------------------------------------
        # write while index is being constructed
        #-----------------------------------------------------------------------
    
        uids_to_match = []
        uids_to_unmatch = []

        # create a new node
        q = "CREATE (n:L {h:toString($v)}) RETURN n.h"
        res = g.query(q, {'v': max_node_v + 10})
        uids_to_match.append(res.result_set[0][0])

        # update a node which had yet to be indexed
        q = "MATCH (n:L) WHERE ID(n) = $id WITH n LIMIT 1 SET n.h = toString($new_v) RETURN n.h"
        res = g.query(q, {'id': max_node_v - 10, 'new_v': max_node_v + 15})
        uids_to_match.append(res.result_set[0][0])

        # update a node which is already indexed
        res = g.query(q, {'id': 1, 'new_v': max_node_v + 17})
        uids_to_match.append(res.result_set[0][0])

        # delete a node which had yet to be indexed
        q = "MATCH (n:L) WHERE ID(n) = $id RETURN n.h"
        res = g.query(q, {'id': max_node_v - 9})
        uids_to_unmatch.append(res.result_set[0][0])

        q = "MATCH (n:L) WHERE ID(n) = $id WITH n LIMIT 1 DELETE n"
        g.query(q, {'id': max_node_v - 9})

        # delete an indexed node
        q = "MATCH (n:L) WHERE ID(n) = $id RETURN n.h"
        res = g.query(q, {'id': 2})
        uids_to_unmatch.append(res.result_set[0][0])

        q = "MATCH (n:L) WHERE ID(n) = $id WITH n LIMIT 1 DELETE n"
        g.query(q, {'id': 2})

        #-----------------------------------------------------------------------
        # validate index is being populated
        #-----------------------------------------------------------------------

        self.env.assertTrue(index_under_construction(g, 'L'))

        # wait for index to become operational
        wait_for_indices_to_sync(g)

        # index should be operational
        self.env.assertFalse(index_under_construction(g, 'L'))

        #-----------------------------------------------------------------------
        # validate index results
        #-----------------------------------------------------------------------

        for uid in uids_to_match:
            q = "CALL db.idx.fulltext.queryNodes('L', $uid) YIELD node RETURN count(node)"
            res = g.query(q, {'uid': uid}).result_set
            self.env.assertEquals(res[0][0], 1)

        for uid in uids_to_unmatch:
            q = "CALL db.idx.fulltext.queryNodes('L', $uid) YIELD node RETURN count(node)"
            res = g.query(q, {'uid': uid}).result_set
            self.env.assertEquals(res[0][0], 0)

    def test10_delete_interrupt_async_index_creation(self):
        # skip test if we're running under Valgrind
        if VALGRIND:
            self.env.skip()

        # 1. create a large graph
        # 2. create an index
        # 3. delete the graph while the index is being constructed

        key = "async-index"
        min_node_v = 0
        max_node_v = 1000000
        conn = self.env.getConnection()

        # clear DB
        conn.flushall()

        g = Graph(self.env.getConnection(), key)

        #-----------------------------------------------------------------------
        # create a large graph
        #-----------------------------------------------------------------------

        q = "UNWIND range($min_v, $max_v) AS x CREATE (:L {v:x})"
        g.query(q, {'min_v': min_node_v, 'max_v': max_node_v})

        #-----------------------------------------------------------------------
        # create an index
        #-----------------------------------------------------------------------

        res = create_node_range_index(g, 'L', 'v', sync=False)
        self.env.assertEquals(res.indices_created, 1)

        #-----------------------------------------------------------------------
        # validate index is being populated
        #-----------------------------------------------------------------------

        self.env.assertTrue(index_under_construction(g, 'L'))

        #-----------------------------------------------------------------------
        # delete graph while the index is being constructed 
        #-----------------------------------------------------------------------

        conn.delete(key)

        # graph key should be removed, index creation should run to completion
        self.env.assertFalse(conn.exists(key))

        # at the moment there's no way of checking index status once its graph
        # key had been removed

    def test11_delete_interrupt_async_fulltext_index_creation(self):
        # skip test if we're running under Valgrind
        if VALGRIND:
            self.env.skip()

        # 1. create a large graph
        # 2. create an index
        # 3. delete the graph while the index is being constructed

        key = "async-fulltext-index"
        min_node_v = 0
        max_node_v = 1000000
        conn = self.env.getConnection()

        # clear DB
        conn.flushall()

        g = Graph(self.env.getConnection(), key)

        #-----------------------------------------------------------------------
        # create a large graph
        #-----------------------------------------------------------------------

        q = "UNWIND range($min_v, $max_v) AS x CREATE (:L {v:toString(x)})"
        g.query(q, {'min_v': min_node_v, 'max_v': max_node_v})

        #-----------------------------------------------------------------------
        # create an index
        #-----------------------------------------------------------------------

        res = create_node_fulltext_index(g, 'L', 'v', sync=False)
        self.env.assertEquals(res.indices_created, 1)

        #-----------------------------------------------------------------------
        # validate index is being populated
        #-----------------------------------------------------------------------

        self.env.assertTrue(index_under_construction(g, 'L'))

        #-----------------------------------------------------------------------
        # delete graph while the index is being constructed 
        #-----------------------------------------------------------------------

        conn.delete(key)

        # graph key should be removed, index creation should run to completion
        self.env.assertFalse(conn.exists(key))

        # at the moment there's no way of checking index status once its graph
        # key had been removed

    def test12_multi_index_creation(self):
        # skip test if we're running under Valgrind
        if VALGRIND:
            self.env.skip()

        # interrupt index creation by adding/removing fields
        #
        # 1. create a large graph
        # 2. create an index
        # 3. modify the index while it is being populated

        key = "async-index"
        min_node_v = 0
        max_node_v = 500000
        conn = self.env.getConnection()

        # clear DB
        conn.flushall()

        g = Graph(self.env.getConnection(), key)

        #-----------------------------------------------------------------------
        # create a large graph
        #-----------------------------------------------------------------------

        q = "UNWIND range($min_v, $max_v) AS x CREATE (:L {v:x, a:x, b:x})"
        g.query(q, {'min_v': min_node_v, 'max_v': max_node_v})

        #-----------------------------------------------------------------------
        # create an index
        #-----------------------------------------------------------------------

        # determine how much time does it take to construct our index
        start = time()

        res = create_node_range_index(g, 'L', 'v', sync=True)
        self.env.assertEquals(res.indices_created, 1)

        # total index creation time
        elapsed = time() - start

        #-----------------------------------------------------------------------
        # drop the index
        #-----------------------------------------------------------------------

        q = "DROP INDEX ON :L(v)"
        res = g.query(q)
        self.env.assertEquals(res.indices_deleted, 1)

        # recreate the index, but this time introduce additionl fields
        # while the index is being populated

        start = time()

        # introduce a new field
        res = create_node_range_index(g, 'L', 'a', sync=False)
        self.env.assertEquals(res.indices_created, 1)

        # introduce a new field
        res = create_node_range_index(g, 'L', 'b', sync=False)
        self.env.assertEquals(res.indices_created, 1)

        # remove field
        q = "DROP INDEX ON :L(a)"
        res = g.query(q)
        self.env.assertEquals(res.indices_deleted, 1)

        # introduce a new field
        res = create_node_range_index(g, 'L', 'v', sync=False)
        self.env.assertEquals(res.indices_created, 1)

        # wait for index to become operational
        wait_for_indices_to_sync(g)

        elapsed_2 = time() - start

        # although we've constructed a larger index
        # new index includes 2 fields (b,v) while the former index included just
        # one (v) we're expecting thier overall construction time to be similar
        self.env.assertTrue(elapsed_2 < elapsed * 2)

    def test13_multi_fulltext_index_creation(self):
        # skip test if we're running under Valgrind
        if VALGRIND:
            self.env.skip()

        # interrupt index creation by adding/removing fields
        #
        # 1. create a large graph
        # 2. create an index
        # 3. modify the index while it is being populated

        key = "async-fulltext-index"
        min_node_v = 0
        max_node_v = 500000
        conn = self.env.getConnection()

        # clear DB
        conn.flushall()

        g = Graph(self.env.getConnection(), key)

        #-----------------------------------------------------------------------
        # create a large graph
        #-----------------------------------------------------------------------

        q = "UNWIND range($min_v, $max_v) AS x CREATE (:L {v:toString(x), a:toString(x), b:toString(x)})"
        g.query(q, {'min_v': min_node_v, 'max_v': max_node_v})

        #-----------------------------------------------------------------------
        # create an index
        #-----------------------------------------------------------------------

        # determine how much time does it take to construct our index
        start = time()

        res = create_node_fulltext_index(g, 'L', 'v', sync=True)
        self.env.assertEquals(res.indices_created, 1)

        # total index creation time
        elapsed = time() - start

        #-----------------------------------------------------------------------
        # drop the index
        #-----------------------------------------------------------------------

        q = "CALL db.idx.fulltext.drop('L')"
        res = g.query(q)
        self.env.assertEquals(res.indices_deleted, 1)

        # recreate the index, but this time introduce additionl fields
        # while the index is being populated

        start = time()

        # introduce a new field
        res = create_node_fulltext_index(g, 'L', 'a', sync=False)
        self.env.assertEquals(res.indices_created, 1)

        # introduce a new field
        res = create_node_fulltext_index(g, 'L', 'b', sync=False)
        self.env.assertEquals(res.indices_created, 1)

        # remove index
        q = "CALL db.idx.fulltext.drop('L')"
        res = g.query(q)
        self.env.assertEquals(res.indices_deleted, 2)

        # introduce a new field
        res = create_node_fulltext_index(g, 'L', 'v', sync=False)
        self.env.assertEquals(res.indices_created, 1)

        # wait for index to become operational
        wait_for_indices_to_sync(g)

        elapsed_2 = time() - start

        # although we've constructed a larger index
        # new index includes 2 fields (b,v) while the former index included just
        # one (v) we're expecting thier overall construction time to be similar
        self.env.assertTrue(elapsed_2 < elapsed * 2)

    def test14_multi_type_index_listing(self):
        # clear DB
        con.flushall()

        # create index of multiple types
        # Label | Attributes | Types
        # --------------------------------------------
        # L     | a          | range
        # L     | b          | vector
        # L     | c          | fulltext
        # L     | d          | range, vector
        # L     | e          | range, fulltext
        # L     | f          | fulltext, vector
        # L     | g          | vector, range, fulltext

        create_node_range_index(graph, 'L', 'a', 'd', 'e', 'g')
        create_node_fulltext_index(graph, 'L', 'c', 'e', 'f', 'g')
        create_node_vector_index(graph, 'L', 'b', 'd', 'f', 'g')

        # list all indices
        res = list_indicies(graph).result_set

        label      = res[0][0]
        properties = res[0][1]
        types      = res[0][2]
        language   = res[0][3]
        stopwords  = res[0][4]
        entitytype = res[0][5]

        # sort
        properties.sort()

        # sort by key
        types = OrderedDict(sorted(types.items(), key=lambda t: t[0]))

        # sort values
        for k, v in types.items():
            types[k].sort()

        #-----------------------------------------------------------------------
        # validate
        #-----------------------------------------------------------------------

        self.env.assertEquals(label, 'L')

        self.env.assertEquals(properties, ['a', 'b', 'c', 'd', 'e', 'f', 'g'])

        expected_types = OrderedDict(
                [
                    ('a', ['RANGE']),
                    ('b', ['VECTOR']),
                    ('c', ['FULLTEXT']),
                    ('d', ['RANGE', 'VECTOR']),
                    ('e', ['FULLTEXT', 'RANGE']),
                    ('f', ['FULLTEXT', 'VECTOR']),
                    ('g', ['FULLTEXT', 'RANGE', 'VECTOR'])
                ]
            )

        self.env.assertEquals(types, expected_types)
        self.env.assertEquals(language, 'english')
        self.env.assertEquals(entitytype, 'NODE')

