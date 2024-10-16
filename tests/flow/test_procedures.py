from common import *
from index_utils import *

GRAPH_ID = "procedures"

node1 = Node(alias="n1", labels="fruit", properties={"name": "Orange1", "value": 1})
node2 = Node(alias="n2", labels="fruit", properties={"name": "Orange2", "value": 2})
node3 = Node(alias="n3", labels="fruit", properties={"name": "Orange3", "value": 3})
node4 = Node(alias="n4", labels="fruit", properties={"name": "Orange4", "value": 4})
node5 = Node(alias="n5", labels="fruit", properties={"name": "Banana", "value": 5})


# Tests built in procedures,
# e.g. db.idx.fulltext.queryNodes
# Test over all procedure behavior in addition to procedure specifics.
class testProcedures(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        if GRAPH_ID in self.db.list_graphs():
            return

        edge = Edge(node1, 'goWellWith', node5)
        self.graph.query(f"CREATE {node1}, {node2}, {node3}, {node4}, {node5}, {edge}")

        # Create full-text index.
        create_node_fulltext_index(self.graph, 'fruit', 'name', sync=True)

    # Compares two nodes based on their properties.
    def _compareNodes(self, a, b):
        return a.properties == b.properties

    # Make sure given item is found within resultset.
    def _inResultSet(self, item, resultset):
        for i in range(len(resultset)):
            result = resultset[i][0]
            if self._compareNodes(item, result):
                return True
        return False

    # Issue query and validates resultset.
    def queryAndValidate(self, query, expected_results, query_params={}):
        actual_resultset = self.graph.query(query, query_params).result_set
        self.env.assertEquals(len(actual_resultset), len(expected_results))
        for i in range(len(actual_resultset)):
            self.env.assertTrue(self._inResultSet(expected_results[i], actual_resultset))

    # Call procedure, omit yield, expecting all procedure outputs to
    # be included in result-set.
    def test01_no_yield(self):
        actual_result = self.graph.call_procedure("db.idx.fulltext.queryNodes",
                                                  args=["fruit", "Orange1"])
        assert(len(actual_result.result_set) == 1)

        header = actual_result.header
        data = actual_result.result_set[0]
        assert(header[0][1] == 'node')
        assert(data[0] is not None)

    # Call procedure specify different outputs.
    def test02_yield(self):
        actual_result = self.graph.call_procedure("db.idx.fulltext.queryNodes",
                                                  args=['fruit', 'Orange1'],
                                                  emit=['node'])
        assert(len(actual_result.result_set) == 1)

        header = actual_result.header
        data = actual_result.result_set[0]
        assert(header[0][1] == 'node')
        assert(data[0] is not None)

        # Yield an unknown output.
        # Expect an error when trying to use an unknown procedure output.
        try:
            self.graph.call_procedure("db.idx.fulltext.queryNodes",
                                      args=["fruit", "Orange1"],
                                      emit=["unknown"])
            self.env.assertFalse(1)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

        # Yield the same output multiple times.
        # Expect an error when trying to use the same output multiple times.
        try:
            self.graph.call_procedure("db.idx.fulltext.queryNodes",
                                      args=["fruit", "Orange1"],
                                      emit=["node", "node"])
            self.env.assertFalse(1)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test03_arguments(self):
        # Omit arguments.
        # Expect an error when trying to omit arguments.
        try:
            self.graph.call_procedure("db.idx.fulltext.queryNodes")
            self.env.assertFalse(1)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

        # Omit arguments, queryNodes expecting 2 argument, provide 1.
        # Expect an error when trying to omit arguments.
        try:
            self.graph.call_procedure("db.idx.fulltext.queryNodes", args=["arg1"])
            self.env.assertFalse(1)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

        # Overload arguments.
        # Expect an error when trying to send too many arguments.
        try:
            self.graph.call_procedure("db.idx.fulltext.queryNodes",
                                      args=["fruit", "query", "fruit", "query"],
                                      emit=["node"])
            self.env.assertFalse(1)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    # Test procedure call while mixing a number of addition clauses.
    def test04_mix_clauses(self):
        query_params = {'prefix': 'Orange*'}
        # CALL + RETURN.

        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    RETURN node"""
        expected_results = [node4, node2, node3, node1]
        self.queryAndValidate(query, expected_results, query_params=query_params)


        # The combination of CALL and WHERE currently creates a syntax error in libcypher-parser.
        # CALL + WHERE + RETURN + ORDER.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    WHERE node.value > 2
                    RETURN node
                    """
        expected_results = [node3, node4]
        self.queryAndValidate(query, expected_results, query_params=query_params)


        # CALL + WHERE + RETURN + ORDER + SKIP.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    WHERE node.value > 2
                    RETURN node
                    ORDER BY node.value
                    SKIP 1"""
        expected_results = [node4]
        self.queryAndValidate(query, expected_results, query_params=query_params)


        # CALL + WHERE + RETURN + LIMIT.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    WHERE node.value > 2
                    RETURN node
                    LIMIT 2"""
        expected_results = [node3, node4]
        self.queryAndValidate(query, expected_results, query_params=query_params)


        # CALL + WHERE + RETURN + ORDER + SKIP + LIMIT.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    WHERE node.value > 2
                    RETURN node
                    ORDER BY node.value
                    SKIP 1
                    LIMIT 1"""
        expected_results = [node4]
        self.queryAndValidate(query, expected_results, query_params=query_params)

        # CALL + RETURN + ORDER.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    RETURN node
                    ORDER BY node.value
                    """
        expected_results = [node1, node2, node3, node4]
        self.queryAndValidate(query, expected_results, query_params=query_params)


        # CALL + RETURN + ORDER + SKIP.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    RETURN node
                    ORDER BY node.value
                    SKIP 1
                    """
        expected_results = [node2, node3, node4]
        self.queryAndValidate(query, expected_results, query_params=query_params)


        # CALL + RETURN + ORDER + LIMIT.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    RETURN node
                    ORDER BY node.value
                    LIMIT 2
                    """
        expected_results = [node1, node2]
        self.queryAndValidate(query, expected_results, query_params=query_params)


        # CALL + RETURN + ORDER + SKIP + LIMIT.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    RETURN node
                    ORDER BY node.value
                    SKIP 1
                    LIMIT 1
                    """
        expected_results = [node2]
        self.queryAndValidate(query, expected_results, query_params=query_params)


        # CALL + WHERE + RETURN + ORDER.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    WHERE node.value > 2
                    RETURN node
                    ORDER BY node.value"""
        expected_results = [node3, node4]
        self.queryAndValidate(query, expected_results, query_params=query_params)


        # CALL + WHERE + RETURN + ORDER + SKIP.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    WHERE node.value > 2
                    RETURN node
                    ORDER BY node.value
                    SKIP 1"""
        expected_results = [node4]
        self.queryAndValidate(query, expected_results, query_params=query_params)


        # CALL + WHERE + RETURN + ORDER + LIMIT.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    WHERE node.value > 2
                    RETURN node
                    ORDER BY node.value
                    LIMIT 1"""
        expected_results = [node3]
        self.queryAndValidate(query, expected_results, query_params=query_params)


        # CALL + WHERE + RETURN + ORDER + SKIP + LIMIT.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
                    WHERE node.value > 2
                    RETURN node
                    ORDER BY node.value
                    SKIP 1
                    LIMIT 1"""
        expected_results = [node4]
        self.queryAndValidate(query, expected_results, query_params=query_params)

        # CALL + MATCH + RETURN.
        query = """CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node
            MATCH (node)-[]->(z)
            RETURN z"""
        expected_results = [node5]
        self.queryAndValidate(query, expected_results, query_params=query_params)

        # UNWIND + CALL + RETURN.
        query = """UNWIND([1,2]) AS x CALL db.idx.fulltext.queryNodes('fruit', $prefix) YIELD node RETURN node"""
        expected_results = [node4, node2, node3, node1, node4, node2, node3, node1]
        self.queryAndValidate(query, expected_results, query_params=query_params)

    def test05_procedure_labels(self):
        actual_resultset = self.graph.call_procedure("db.labels").result_set
        expected_results = [["fruit"]]
        self.env.assertEquals(actual_resultset, expected_results)

    def test06_procedure_relationshipTypes(self):
        actual_resultset = self.graph.call_procedure("db.relationshipTypes").result_set
        expected_results = [["goWellWith"]]
        self.env.assertEquals(actual_resultset, expected_results)

    def test07_procedure_propertyKeys(self):
        actual_resultset = self.graph.call_procedure("db.propertyKeys").result_set
        expected_results = [["name"], ["value"]]
        self.env.assertEquals(actual_resultset, expected_results)

    def test08_procedure_fulltext_syntax_error(self):
        try:
            query = """CALL db.idx.fulltext.queryNodes('fruit', 'Orange || Apple') YIELD node RETURN node"""
            self.graph.query(query)
            self.env.assertFalse(1)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

    def test09_procedure_lookup(self):
        try:
            self.graph.call_procedure("dB.LaBeLS")
        except redis.exceptions.ResponseError:
            # This should not cause an error
            self.env.assertFalse(1)
            pass

        try:
            # looking for a non existing procedure
            self.graph.call_procedure("db.nonExistingProc")
            self.env.assertFalse(1)
        except redis.exceptions.ResponseError:
            # Expecting an error.
            pass

        try:
            self.graph.call_procedure("db.IDX.FulLText.QueRyNoDes", args=["fruit", "or"])
        except redis.exceptions.ResponseError:
            # This should not cause an error
            self.env.assertFalse(1)
            pass

    def test10_procedure_get_all_procedures(self):
        actual_resultset = self.graph.call_procedure("dbms.procedures").result_set

        # The following two procedure are a part of the expected results
        expected_result = [["db.labels", "READ"], ["db.idx.fulltext.createNodeIndex", "WRITE"],
                           ["db.propertyKeys", "READ"], ["dbms.procedures", "READ"], ["db.relationshipTypes", "READ"],
                           ["algo.BFS", "READ"], ["algo.pageRank", "READ"], ["db.idx.fulltext.queryNodes", "READ"],
                           ["db.idx.fulltext.drop", "WRITE"]]
        for res in expected_result:
            self.env.assertContains(res, actual_resultset)

    def test11_procedure_indexes(self):
        # Verify that the full-text index is reported properly.
        actual_resultset = self.graph.query("CALL db.indexes() YIELD label, properties").result_set
        expected_results = [["fruit", ["name"]]]
        self.env.assertEquals(actual_resultset, expected_results)

        # Add an exact-match index to a different property on the same label..
        result = create_node_range_index(self.graph, 'fruit', 'other_property')
        self.env.assertEquals(result.indices_created, 1)

        # Verify that all indexes are reported.
        actual_resultset = self.graph.query("CALL db.indexes() YIELD label, properties RETURN * ORDER BY properties").result_set
        expected_results = [["fruit", ["name", "other_property"]]]
        self.env.assertEquals(actual_resultset, expected_results)

        # Add an exact-match index to the full-text indexed property on the same label..
        result = create_node_range_index(self.graph, 'fruit', 'name', sync=True)
        self.env.assertEquals(result.indices_created, 1)

        # Verify that all indexes are reported.
        actual_resultset = self.graph.query("CALL db.indexes() YIELD label, properties RETURN * ORDER BY properties").result_set
        expected_results = [["fruit", ["name", "other_property"]]]
        self.env.assertEquals(actual_resultset, expected_results)

        # Validate the results when yielding only one element.
        actual_resultset = self.graph.query("CALL db.indexes() YIELD label").result_set
        expected_results = [["fruit"]]
        self.env.assertEquals(actual_resultset, expected_results)

    def test12_procedure_reordered_yields(self):
        # Yield results of procedure in a non-default sequence
        actual_resultset = self.graph.query("CALL dbms.procedures() YIELD mode, name RETURN mode, name ORDER BY name").result_set

        expected_result = [["READ",  "algo.BFS"],
                           ['READ',  'algo.SPpaths'],
                           ['READ',  'algo.SSpaths'],
                           ["READ",  "algo.pageRank"],
                           ['READ',  'db.constraints'],
                           ["WRITE", "db.idx.fulltext.createNodeIndex"],
                           ["WRITE", "db.idx.fulltext.drop"],
                           ["READ",  "db.idx.fulltext.queryNodes"],
                           ["READ",  "db.idx.vector.queryNodes"],
                           ["READ",  "db.idx.vector.queryRelationships"],
                           ["READ",  "db.indexes"],
                           ["READ",  "db.labels"],
                           ["READ",  "db.propertyKeys"],
                           ["READ",  "db.relationshipTypes"],
                           ["READ",  "dbms.functions"],
                           ["READ",  "dbms.procedures"]]
        self.env.assertEquals(actual_resultset, expected_result)

    def test13_functions(self):
        # Yield results of procedure in a non-default sequence
        actual_resultset = self.graph.query("CALL dbms.functions() YIELD name RETURN name ORDER BY name").result_set
        actual_len = len(actual_resultset)
        self.env.assertEquals(actual_len, 95)

    def test14_procedure_get_all_functions(self):
        actual_resultset = self.graph.query("CALL dbms.functions() YIELD name, signature, description, return_description, argument_description RETURN name, signature, description, return_description, argument_description ORDER BY name").result_set
        actual_len = len(actual_resultset)
        self.env.assertEquals(actual_len, 95)
    
        expected_first_result = ["abs", 
                                 "abs: Integer|Float -> Integer|Float",
                                 "Returns the absolute value of a numeric value\nReturns null when expr evaluates to null\n\nParameters:\n- expr: Numeric expression\n\nReturns the absolute value",
                                 "List|Null", 
                                 "[Integer|Float]"]
        self.env.assertEquals(actual_resultset[0][0], "abs")            
