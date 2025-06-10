from common import *
from index_utils import *
from collections import OrderedDict

GRAPH_ID = "index_delete"

class testNodeIndexDeletionFlow():
    def __init__(self):
        # skip test if we're running under Valgrind
        # drop index is an async operation which can cause Valgraind
        # to wrongfully report as a leak
        if VALGRIND:
            Environment.skip(None)

        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.g = Graph(self.redis_con, GRAPH_ID)

    def test_01_drop_missing_index(self):
        # drop range, fulltext and vector index
        funcs = [drop_node_range_index,
                 drop_node_fulltext_index,
                 drop_node_vector_index]

        for f in funcs:
            # drop none existing index
            try:
                result = f(self.g, 'L', 'age')
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Unable to drop index on :L(age): no such index.", str(e))

    def test_02_drop_unknown_label(self):
        # try to delete an index providing an unknown label/relationship
        result = create_node_range_index(self.g, 'L', 'age')
        self.env.assertEquals(result.indices_created, 1)

        # try to delete index providing an unknown label
        try:
            result = drop_node_range_index(self.g, 'missing', 'age')
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Unable to drop index on :missing(age): no such index.", str(e))

        # remove actual index
        result = drop_node_range_index(self.g, 'L', 'age')
        self.env.assertEquals(result.indices_deleted, 1)

    def test_03_drop_unknown_attribute(self):
        # try to delete an index providing an unknown attribute
        result = create_node_range_index(self.g, 'L', 'age')
        self.env.assertEquals(result.indices_created, 1)

        # try to delete index providing an unknown attribute
        try:
            result = drop_node_range_index(self.g, 'L', 'missing')
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Unable to drop index on :L(missing): no such index.", str(e))

        # remove actual index
        result = drop_node_range_index(self.g, 'L', 'age')

    def test_04_drop_wrong_index_type(self):
        def create_node_vector_index_default():
            return create_node_vector_index(self.g, 'L', 'age', dim=2)

        def create_node_fulltext_index_default():
            return create_node_fulltext_index(self.g, 'L', 'age')

        def create_node_range_index_default():
            return create_node_range_index(self.g, 'L', 'age')

        def drop_node_vector_index_default():
            return drop_node_vector_index(self.g, 'L', 'age')

        def drop_node_fulltext_index_default():
            return drop_node_fulltext_index(self.g, 'L', 'age')

        def drop_node_range_index_default():
            return drop_node_range_index(self.g, 'L', 'age')

        create_funcs = [create_node_vector_index_default,
                        create_node_fulltext_index_default,
                        create_node_range_index_default]

        drop_funcs = [
                (
                    drop_node_fulltext_index_default, # wrong index type
                    drop_node_range_index_default,    # wrong index type
                    drop_node_vector_index_default    # correct index type
                ),
                (
                    drop_node_vector_index_default,   # wrong index type
                    drop_node_range_index_default,    # wrong index type
                    drop_node_fulltext_index_default  # correct index type
                ),
                (
                    drop_node_fulltext_index_default, # wrong index type
                    drop_node_vector_index_default,   # wrong index type
                    drop_node_range_index_default     # correct index type
                )
            ]

        for create_func in create_funcs:
            # create index
            result = create_func()
            self.env.assertEquals(result.indices_created, 1)

            # try to delete index providing an unknown attribute
            drop_func = drop_funcs.pop(0)
            try:
                result = drop_func[0]()
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Unable to drop index on :L(age): no such index.", str(e))

            try:
                result = drop_func[1]()
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Unable to drop index on :L(age): no such index.", str(e))

            # remove actual index
            result = drop_func[2]()
            self.env.assertEquals(result.indices_deleted, 1)

    def test_05_drop_multi_type_node_indices(self):
        # create indices
        label = "person"
        attributes = ["a", "b", "c"]

        # create indices
        for attribute in attributes:
            result = create_node_range_index(self.g, label, attribute)
            self.env.assertEquals(result.indices_created, 1)

            result = create_node_fulltext_index(self.g, label, attribute)
            self.env.assertEquals(result.indices_created, 1)

            result = create_node_vector_index(self.g, label, attribute, dim=2, sync=True)
            self.env.assertEquals(result.indices_created, 1)

        # validate indices
        result = list_indicies(self.g)
        index_label = result.result_set[0][0]
        index_fields = result.result_set[0][1]
        index_fields_types = result.result_set[0][2]

        self.env.assertEquals(index_label, label)
        self.env.assertEquals(index_fields, attributes)
        all_types = ['RANGE', 'VECTOR', 'FULLTEXT']
        self.env.assertEquals(index_fields_types,
                              OrderedDict([('a', all_types),
                                           ('b', all_types),
                                           ('c', all_types)]))

        # drop indices
        for attribute in attributes:
            # remove range index
            result = drop_node_range_index(self.g, label, attribute)
            self.env.assertEquals(result.indices_deleted, 1)

            # validate index does not contains RANGE index for current attribute
            result = list_indicies(self.g)
            index_fields_types = result.result_set[0][2]
            self.env.assertEquals(index_fields_types[attribute], ['VECTOR', 'FULLTEXT'])

            # remove fulltext index
            result = drop_node_fulltext_index(self.g, label, attribute)
            self.env.assertEquals(result.indices_deleted, 1)

            # validate index does not contains FULLTEXT index for current attribute
            result = list_indicies(self.g)
            index_fields_types = result.result_set[0][2]
            self.env.assertEquals(index_fields_types[attribute], ['VECTOR'])

            # remove vector index
            result = drop_node_vector_index(self.g, label, attribute)
            self.env.assertEquals(result.indices_deleted, 1)

        # validate no indexes in graph
        result = list_indicies(self.g)
        self.env.assertEquals(len(result.result_set), 0)

    def test_06_drop_index_during_population(self):
        # 1. populate a graph
        # 2. create an index and wait for it to be sync
        # 3. constantly update indexed entities
        # 4. drop index
        # 5. validate execution-plan + indexes report

        #-----------------------------------------------------------------------
        # populate a graph
        #-----------------------------------------------------------------------
        q = "UNWIND range(0, 1000) AS x CREATE (:N{v:x})"
        self.g.query(q)

        #-----------------------------------------------------------------------
        # create an index and wait for it to be sync
        #-----------------------------------------------------------------------
        create_node_range_index(self.g, 'N', 'v', sync=True)

        #-----------------------------------------------------------------------
        # constantly update indexed entities
        # drop index
        # validate execution-plan + indexes report
        #-----------------------------------------------------------------------
        start_idx = 0
        end_idx = 100
        for i in range(start_idx, end_idx):
            # update indexed entities
            q = f"MATCH (n:N) WHERE n.v = {i} SET n.v = -n.v"
            self.g.query(q)

            if i < end_idx / 2:
                # validate execution-plan + indexes report
                plan = str(self.g.explain(q))
                self.env.assertIn("Index", plan)

                indicies = list_indicies(self.g).result_set
                self.env.assertEquals(len(indicies), 1)

            elif i == end_idx / 2:
                # drop index
                drop_node_range_index(self.g, 'N', 'v')

            else:
                # validate execution-plan + indexes report
                plan = str(self.g.explain(q))
                self.env.assertNotIn("Index", plan)

                indicies = list_indicies(self.g).result_set
                self.env.assertEquals(len(indicies), 0)

    def test_07_reset_order(self):
        """Tests that the reset order is correct, i.e., that the reading ops are
        reset before the writing ops (otherwise we write while a read-lock is
        held)."""

        # create data
        self.g.query(
            """
            WITH 1 AS x
            CREATE (:X {uid: toString(x)})-[:R]->(y:Y {v: x})
            """
        )

        # create an index
        create_node_range_index(self.g, 'X', 'uid', sync=True)
        create_node_range_index(self.g, 'Y', 'v', sync=True)

        # utilize the index for a scan, followed by a deletion of the indexed
        # entity and setting of a property on the other entity
        res = self.g.query(
            """
            MATCH (x:X {uid: '1'})-[:R]->(y:Y)
            DELETE y
            SET x.uid = '10'
            RETURN x
            """
        )

        # validate results
        self.env.assertEquals(res.nodes_deleted, 1)
        self.env.assertEquals(res.relationships_deleted, 1)
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0],
            Node(labels='X', properties={'uid': '10'}))

    def test_08_remove_range_field(self):
        # a single range field F is composed of 3 distinct fields:
        # 1. range:F (scalar exact matching) e.g. n.v = 3 or n.v > 4
        # 2. range:F:numeric:arr (array numeric element lookup) e.g. 3 in n.v
        # 3. range:F:string:arr  (array string element lookup) e.g. 'a' in n.v
        # we need to make sure that upon the removal of a field, all 3 fields
        # are deleted

        # create an index on N:a and N:b
        create_node_range_index(self.g, 'N', 'a')
        create_node_range_index(self.g, 'N', 'b', sync=True)

        # make sure we can see all six field
        res = self.g.query("""CALL db.indexes()
                              YIELD label, info
                              WHERE label = 'N'
                              RETURN [f in info['fields'] | f.name]""")

        fields = res.result_set[0][0]
        self.env.assertEquals(len(fields), 7)
        self.env.assertIn("range:a", fields)
        self.env.assertIn("range:b", fields)
        self.env.assertIn("range:a:string:arr", fields)
        self.env.assertIn("range:b:string:arr", fields)
        self.env.assertIn("range:a:numeric:arr", fields)
        self.env.assertIn("range:b:numeric:arr", fields)
        self.env.assertIn("NONE_INDEXABLE_FIELDS", fields)

        # drop the 'a' field
        drop_node_range_index(self.g, 'N', 'a')

        # make sure all 3 fields associated with 'a' been removed
        res = self.g.query("""CALL db.indexes()
                              YIELD label, info
                              WHERE label = 'N'
                              RETURN [f in info['fields'] | f.name]""")

        fields = res.result_set[0][0]
        self.env.assertEquals(len(fields), 4)
        self.env.assertIn("range:b", fields)
        self.env.assertIn("range:b:string:arr", fields)
        self.env.assertIn("range:b:numeric:arr", fields)
        self.env.assertIn("NONE_INDEXABLE_FIELDS", fields)

        # drop the last field in the index
        # drop the 'b' field
        drop_node_range_index(self.g, 'N', 'b')

        # make sure the N index doesn't exists
        res = self.g.query("""CALL db.indexes()
                              YIELD label, info
                              WHERE label = 'N'
                              RETURN [f in info['fields'] | f.name]""")

        self.env.assertEquals(len(res.result_set), 0)

class testEdgeIndexDeletionFlow():
    def __init__(self):
        # skip test if we're running under Valgrind
        # drop index is an async operation which can cause Valgraind
        # to wrongfully report as a leak
        if VALGRIND:
            Environment.skip(None)

        self.env, self.db = Env()
        self.g = self.db.select_graph(GRAPH_ID)

    def test_01_drop_missing_index(self):
        # drop range, fulltext and vector index
        funcs = [drop_edge_range_index,
                 drop_edge_vector_index,
                 drop_edge_fulltext_index]

        for f in funcs:
            # drop none existing index
            try:
                result = f(self.g, 'L', 'age')
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Unable to drop index on :L(age): no such index.", str(e))

    def test_02_drop_unknown_label(self):
        # try to delete an index providing an unknown label/relationship
        result = create_edge_range_index(self.g, 'L', 'age')
        self.env.assertEquals(result.indices_created, 1)

        # try to delete index providing an unknown relation
        try:
            result = drop_edge_range_index(self.g, 'missing', 'age')
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Unable to drop index on :missing(age): no such index.", str(e))

        result = drop_edge_range_index(self.g, 'L', 'age')
        self.env.assertEquals(result.indices_deleted, 1)

    def test_03_drop_unknown_attribute(self):
        # try to delete an index providing an unknown attribute
        result = create_edge_range_index(self.g, 'L', 'age')
        self.env.assertEquals(result.indices_created, 1)

        # try to delete index providing an unknown attribute
        try:
            result = drop_edge_range_index(self.g, 'L', 'missing')
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Unable to drop index on :L(missing): no such index.", str(e))

        # drop index
        result = drop_edge_range_index(self.g, 'L', 'age')

    def test_04_drop_wrong_index_type(self):
        def create_edge_vector_index_default():
            return create_edge_vector_index(self.g, 'L', 'age', dim=2)

        def create_edge_fulltext_index_default():
            return create_edge_fulltext_index(self.g, 'L', 'age')

        def create_edge_range_index_default():
            return create_edge_range_index(self.g, 'L', 'age')

        def drop_edge_vector_index_default():
            return drop_edge_vector_index(self.g, 'L', 'age')

        def drop_edge_fulltext_index_default():
            return drop_edge_fulltext_index(self.g, 'L', 'age')

        def drop_edge_range_index_default():
            return drop_edge_range_index(self.g, 'L', 'age')

        create_funcs = [create_edge_vector_index_default,
                        create_edge_fulltext_index_default,
                        create_edge_range_index_default]

        drop_funcs = [
                (
                    drop_edge_fulltext_index_default, # wrong index type
                    drop_edge_range_index_default,    # wrong index type
                    drop_edge_vector_index_default    # correct index type
                ),
                (
                    drop_edge_vector_index_default,   # wrong index type
                    drop_edge_range_index_default,    # wrong index type
                    drop_edge_fulltext_index_default  # correct index type
                ),
                (
                    drop_edge_fulltext_index_default, # wrong index type
                    drop_edge_vector_index_default,   # wrong index type
                    drop_edge_range_index_default     # correct index type
                )
            ]

        for create_func in create_funcs:
            # create index
            result = create_func()
            self.env.assertEquals(result.indices_created, 1)

            # try to delete index providing an unknown attribute
            drop_func = drop_funcs.pop(0)
            try:
                result = drop_func[0]()
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Unable to drop index on :L(age): no such index.", str(e))

            try:
                result = drop_func[1]()
                self.env.assertTrue(False)
            except ResponseError as e:
                self.env.assertContains("Unable to drop index on :L(age): no such index.", str(e))

            # remove actual index
            result = drop_func[2]()
            self.env.assertEquals(result.indices_deleted, 1)

    def test_05_drop_multi_type_edge_indices(self):
        # create indices
        label = "person"
        attributes = ["a", "b", "c"]

        # create indices
        for attribute in attributes:
            result = create_edge_range_index(self.g, label, attribute)
            self.env.assertEquals(result.indices_created, 1)

            result = create_edge_fulltext_index(self.g, label, attribute)
            self.env.assertEquals(result.indices_created, 1)

            result = create_edge_vector_index(self.g, label, attribute, dim=2, sync=True)
            self.env.assertEquals(result.indices_created, 1)

        # validate indices
        result = list_indicies(self.g)
        index_label = result.result_set[0][0]
        index_fields = result.result_set[0][1]
        index_fields_types = result.result_set[0][2]

        self.env.assertEquals(index_label, label)
        self.env.assertEquals(index_fields, attributes)
        all_types = ['RANGE', 'VECTOR', 'FULLTEXT']
        self.env.assertEquals(index_fields_types, OrderedDict([('a', all_types), ('b', all_types), ('c', all_types)]))

        # drop indices
        for attribute in attributes:
            # remove range index
            result = drop_edge_range_index(self.g, label, attribute)
            self.env.assertEquals(result.indices_deleted, 1)

            # validate index does not contains RANGE index for current attribute
            result = list_indicies(self.g)
            index_fields_types = result.result_set[0][2]
            self.env.assertEquals(index_fields_types[attribute], ['VECTOR', 'FULLTEXT'])

            # remove fulltext index
            result = drop_edge_fulltext_index(self.g, label, attribute)
            self.env.assertEquals(result.indices_deleted, 1)

            # validate index does not contains FULLTEXT index for current attribute
            result = list_indicies(self.g)
            index_fields_types = result.result_set[0][2]
            self.env.assertEquals(index_fields_types[attribute], ['VECTOR'])

            # remove vector index
            result = drop_edge_vector_index(self.g, label, attribute)
            self.env.assertEquals(result.indices_deleted, 1)

        # validate no indexes in graph
        result = list_indicies(self.g)
        self.env.assertEquals(len(result.result_set), 0)
        # create indices
        relation = "person"
        attributes = ["a", "b", "c"]

    def test_06_index_rollback(self):
        # make sure graph rollsback to its previous state if index creation fails

        # create an index over 'L', 'age'
        result = create_edge_range_index(self.g, 'L', 'age')
        self.env.assertEquals(result.indices_created, 1)

        # try to create index over multiple fields: some new to the graph
        # some already indexed
        try:
            result = create_edge_range_index(self.g, 'L', 'x', 'y', 'z', 'age')
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Attribute 'age' is already indexed", str(e))

        # make sure attributes 'x', 'y', 'z' are not part of the graph
        result = self.g.query("CALL db.propertyKeys()")
        self.env.assertFalse('x' in result.result_set[0])
        self.env.assertFalse('y' in result.result_set[0])
        self.env.assertFalse('z' in result.result_set[0])

        # try to create a vector index with wrong configuration
        try:
            result = create_edge_vector_index(self.g, 'NewRelation', 'NewAttr', dim=-1, similarity_function='tarnegol')
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Invalid vector index configuration", str(e))

        # make sure 'NewLabel' is not part of the graph
        result = self.g.query("CALL db.relationshipTypes()")
        self.env.assertFalse('NewRelation' in result.result_set[0])

        # make sure 'NewAttr' is not part of the graph
        result = self.g.query("CALL db.propertyKeys()")
        self.env.assertFalse('NewAttr' in result.result_set[0])

        # drop index over 'L', 'age'
        result = drop_edge_range_index(self.g, 'L', 'age')

