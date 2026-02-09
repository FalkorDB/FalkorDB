from common import *

people = ["Roi", "Alon", "Ailon", "Boaz"]
GRAPH_ID = "G"


class testResultSetFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        nodes = {}
        # Create entities
        for idx, p in enumerate(people):
            node = Node(alias=f"n{idx}", labels="person", properties={"name": p, "val": idx})
            nodes[p] = node

        # Fully connected graph
        edges = []
        for src in nodes:
            for dest in nodes:
                if src != dest:
                    edges.append(Edge(nodes[src], "know", nodes[dest]))

        nodes_str = [str(node) for node in nodes.values()]
        edges_str = [str(edge) for edge in edges]
        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")


    # Verify that scalar returns function properly
    def test01_return_scalars(self):
        query = """MATCH (a) RETURN a.name, a.val ORDER BY a.val"""
        result = self.graph.query(query)

        expected_result = [['Roi', 0],
                           ['Alon', 1],
                           ['Ailon', 2],
                           ['Boaz', 3]]

        self.env.assertEquals(len(result.result_set), 4)
        self.env.assertEquals(len(result.header), 2) # 2 columns in result set
        self.env.assertEquals(result.result_set, expected_result)

    # Verify that full node returns function properly
    def test02_return_nodes(self):
        query = """MATCH (a) RETURN a"""
        result = self.graph.query(query)

        # TODO add more assertions after updated client format is defined
        self.env.assertEquals(len(result.result_set), 4)
        self.env.assertEquals(len(result.header), 1) # 1 column in result set

    # Verify that full edge returns function properly
    def test03_return_edges(self):
        query = """MATCH ()-[e]->() RETURN e"""
        result = self.graph.query(query)

        # TODO add more assertions after updated client format is defined
        self.env.assertEquals(len(result.result_set), 12) # 12 relations (fully connected graph)
        self.env.assertEquals(len(result.header), 1) # 1 column in result set

    def test04_mixed_returns(self):
        query = """MATCH (a)-[e]->() RETURN a.name, a, e ORDER BY a.val"""
        result = self.graph.query(query)

        # TODO add more assertions after updated client format is defined
        self.env.assertEquals(len(result.result_set), 12) # 12 relations (fully connected graph)
        self.env.assertEquals(len(result.header), 3) # 3 columns in result set

    # Verify that the DISTINCT operator works with full entity returns
    def test05_distinct_full_entities(self):
        graph2 = self.db.select_graph("H")
        query = """CREATE (a)-[:e]->(), (a)-[:e]->()"""
        result = graph2.query(query)
        self.env.assertEquals(result.nodes_created, 3)
        self.env.assertEquals(result.relationships_created, 2)

        query = """MATCH (a)-[]->() RETURN a"""
        non_distinct = graph2.query(query)
        query = """MATCH (a)-[]->() RETURN DISTINCT a"""
        distinct = graph2.query(query)

        self.env.assertEquals(len(non_distinct.result_set), 2)
        self.env.assertEquals(len(distinct.result_set), 1)

    # Verify that RETURN * projections include all user-defined aliases.
    def test06_return_all(self):
        query = """MATCH (a)-[e]->(b) RETURN *"""
        result = self.graph.query(query)
        # Validate the header strings of the 3 columns.
        # NOTE - currently, RETURN * populates values in alphabetical order, but that is subject to later change.
        self.env.assertEqual(result.header[0][1], 'a')
        self.env.assertEqual(result.header[1][1], 'b')
        self.env.assertEqual(result.header[2][1], 'e')
        # Verify that 3 columns are returned
        self.env.assertEqual(len(result.result_set[0]), 3)

    # Tests for aggregation functions default values. Fix for issue 767.
    def test07_agg_func_default_values(self):
        # Test for aggregation over non existing node properties.
        # Max default value is null.
        query = """MATCH (a) return max(a.missing_field)"""
        result = self.graph.query(query)
        self.env.assertEqual(None, result.result_set[0][0])

        # Min default value is null.
        query = """MATCH (a) return min(a.missing_field)"""
        result = self.graph.query(query)
        self.env.assertEqual(None, result.result_set[0][0])

        # Count default value is 0.
        query = """MATCH (a) return count(a.missing_field)"""
        result = self.graph.query(query)
        self.env.assertEqual(0, result.result_set[0][0])

        # Avarage default value is 0.
        query = """MATCH (a) return avg(a.missing_field)"""
        result = self.graph.query(query)
        self.env.assertEqual(None, result.result_set[0][0])

        # Collect default value is an empty array.
        query = """MATCH (a) return collect(a.missing_field)"""
        result = self.graph.query(query)
        self.env.assertEqual([], result.result_set[0][0])

        # StdDev default value is 0.
        query = """MATCH (a) return stdev(a.missing_field)"""
        result = self.graph.query(query)
        self.env.assertEqual(0, result.result_set[0][0])

        # percentileCont default value is null.
        query = """MATCH (a) return percentileCont(a.missing_field, 0.1)"""
        result = self.graph.query(query)
        self.env.assertEqual(None, result.result_set[0][0])

        # percentileDisc default value is null.
        query = """MATCH (a) return percentileDisc(a.missing_field, 0.1)"""
        result = self.graph.query(query)
        self.env.assertEqual(None, result.result_set[0][0])

    # Test returning multiple occurrence of an expression.
    def test08_return_duplicate_expression(self):
        query = """MATCH (a) return max(a.val) as x, max(a.val) as y"""
        result = self.graph.query(query)
        self.env.assertEqual(result.result_set[0][0], result.result_set[0][1])

        query = """MATCH (a) return a.val as x, a.val as y LIMIT 1"""
        result = self.graph.query(query)
        self.env.assertEqual(result.result_set[0][0], result.result_set[0][1])

        query = """return 1 as x, 1 as y"""
        result = self.graph.query(query)
        self.env.assertEqual(result.result_set[0][0], result.result_set[0][1])

    # Test implicit result-set size limit
    def test09_implicit_resultset_limit(self):
        query = "MATCH (a) RETURN a"

        result = self.graph.query(query)
        record_count = len(result.result_set)

        # make sure limit is greater than 0
        assert(record_count > 1)
        limit = record_count -1

        # enforce implicit limit
        self.db.config_set("RESULTSET_SIZE", limit)

        result = self.graph.query(query)
        limited_record_count = len(result.result_set)
        assert(limited_record_count == limit)

        # lift limit, -1 stands for unlimited
        self.db.config_set("RESULTSET_SIZE", -1)

        # re-issue query
        result = self.graph.query(query)
        unlimited_record_count = len(result.result_set)
        assert(unlimited_record_count == record_count)

    def test10_carriage_return_in_result(self):
        query = """RETURN 'Foo\r\nBar'"""
        result = self.graph.query(query)
        self.env.assertEqual(result.result_set[0][0], 'Foo\r\nBar')

    # Test entity functions on a directly deleted node
    # the node is still in scope after DELETE, attributes are valid
    def test11_entity_functions_on_deleted_node(self):
        # id() on deleted node should return a valid integer
        query = """CREATE (n:Person {name: 'Alice', age: 30}) DELETE n RETURN id(n)"""
        result = self.graph.query(query)
        self.env.assertNotEqual(result.result_set[0][0], None)

        # labels() on deleted node should return empty array
        query = """CREATE (n:Person {name: 'Alice', age: 30}) DELETE n RETURN labels(n)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], [])

        # hasLabels() on deleted node should return false
        query = """CREATE (n:Person {name: 'Alice', age: 30}) DELETE n RETURN hasLabels(n, ['Person'])"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], False)

        # properties() on deleted node should return the node's properties
        query = """CREATE (n:Person {name: 'Alice', age: 30}) DELETE n RETURN properties(n)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], {'name': 'Alice', 'age': 30})

        # keys() on deleted node should return the node's keys
        query = """CREATE (n:Person {name: 'Alice', age: 30}) DELETE n RETURN keys(n)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], ['name', 'age'])

        # property access on deleted node should return the property value
        query = """CREATE (n:Person {name: 'Alice', age: 30}) DELETE n RETURN n.name"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], 'Alice')

        # typeof on deleted node should return "Node"
        query = """CREATE (n:Person {name: 'Alice', age: 30}) DELETE n RETURN typeof(n)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], "Node")

        # indegree on deleted node should return 0
        query = """CREATE (n:Person {name: 'Alice', age: 30}) DELETE n RETURN indegree(n)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], 0)

        # outdegree on deleted node should return 0
        query = """CREATE (n:Person {name: 'Alice', age: 30}) DELETE n RETURN outdegree(n)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], 0)

    # Test entity functions on a directly deleted edge
    # the edge is still in scope after DELETE, attributes are valid
    def test12_entity_functions_on_deleted_edge(self):
        # id() on deleted edge should return a valid integer
        query = """CREATE (a)-[r:KNOWS {since: 2020}]->(b) DELETE r RETURN id(r)"""
        result = self.graph.query(query)
        self.env.assertNotEqual(result.result_set[0][0], None)

        # type() on deleted edge should return the relationship type
        query = """CREATE (a)-[r:KNOWS {since: 2020}]->(b) DELETE r RETURN type(r)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], "KNOWS")

        # properties() on deleted edge should return the edge's properties
        query = """CREATE (a)-[r:KNOWS {since: 2020}]->(b) DELETE r RETURN properties(r)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], {'since': 2020})

        # keys() on deleted edge should return the edge's keys
        query = """CREATE (a)-[r:KNOWS {since: 2020}]->(b) DELETE r RETURN keys(r)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], ['since'])

        # property access on deleted edge should return the property value
        query = """CREATE (a)-[r:KNOWS {since: 2020}]->(b) DELETE r RETURN r.since"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], 2020)

        # typeof on deleted edge should return "Edge"
        query = """CREATE (a)-[r:KNOWS {since: 2020}]->(b) DELETE r RETURN typeof(r)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], "Edge")

    # Test returning deleted node via startNode/endNode
    # when endpoints are deleted and accessed via startNode/endNode,
    # the node has a NULL attribute-set so labels and properties are empty
    def test13_deleted_node_via_startNode_endNode(self):
        # startNode returns the deleted node with empty labels and properties
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(a) WITH r, a DELETE a RETURN startNode(r)"""
        result = self.graph.query(query)
        self.env.assertEquals(len(result.result_set), 1)
        node = result.result_set[0][0]
        self.env.assertEquals(node.labels, None)
        self.env.assertEquals(node.properties, {})

        # endNode returns the deleted node with empty labels and properties
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(a) WITH r, a DELETE a RETURN endNode(r)"""
        result = self.graph.query(query)
        self.env.assertEquals(len(result.result_set), 1)
        node = result.result_set[0][0]
        self.env.assertEquals(node.labels, None)
        self.env.assertEquals(node.properties, {})

        # both startNode and endNode are deleted
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(b:Person {name: 'Bob', age: 25}) WITH r, a, b DELETE a, b RETURN startNode(r), endNode(r)"""
        result = self.graph.query(query)
        self.env.assertEquals(len(result.result_set), 1)
        start_node = result.result_set[0][0]
        end_node = result.result_set[0][1]
        self.env.assertEquals(start_node.labels, None)
        self.env.assertEquals(start_node.properties, {})
        self.env.assertEquals(end_node.labels, None)
        self.env.assertEquals(end_node.properties, {})

    # Test returning a deleted edge with properties
    def test14_deleted_edge_reply(self):
        # returning a deleted edge should include its relation type and properties
        query = """CREATE (a:Person {name: 'Alice'})-[r:KNOWS {since: 2020, weight: 0.5}]->(a) WITH r, a DELETE a RETURN r"""
        result = self.graph.query(query)
        self.env.assertEquals(len(result.result_set), 1)
        edge = result.result_set[0][0]
        self.env.assertEquals(edge.relation, "KNOWS")
        self.env.assertEquals(edge.properties, {'since': 2020, 'weight': 0.5})

    # Test entity functions on deleted node accessed via startNode
    # the node has a NULL attribute-set
    def test15_entity_functions_on_deleted_start_node(self):
        # properties() on deleted node via startNode should return empty map
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(a) WITH r, a DELETE a RETURN properties(startNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], {})

        # keys() on deleted node via startNode should return empty array
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(a) WITH r, a DELETE a RETURN keys(startNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], [])

        # labels() on deleted node via startNode should return empty array
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(a) WITH r, a DELETE a RETURN labels(startNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], [])

        # id() on deleted node via startNode should return a valid integer
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(a) WITH r, a DELETE a RETURN id(startNode(r))"""
        result = self.graph.query(query)
        self.env.assertNotEqual(result.result_set[0][0], None)

        # hasLabels on deleted node via startNode should return false
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(a) WITH r, a DELETE a RETURN hasLabels(startNode(r), ['Person'])"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], False)

        # typeof on deleted node via startNode should return "Node"
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(a) WITH r, a DELETE a RETURN typeof(startNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], "Node")

        # indegree on deleted node via startNode should return 0
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(a) WITH r, a DELETE a RETURN indegree(startNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], 0)

        # outdegree on deleted node via startNode should return 0
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(a) WITH r, a DELETE a RETURN outdegree(startNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], 0)

    # Test entity functions on deleted node accessed via endNode
    # the node has a NULL attribute-set
    def test16_entity_functions_on_deleted_end_node(self):
        # properties() on deleted node via endNode should return empty map
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(b:Person {name: 'Bob', age: 25}) WITH r, a, b DELETE a, b RETURN properties(endNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], {})

        # keys() on deleted node via endNode should return empty array
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(b:Person {name: 'Bob', age: 25}) WITH r, a, b DELETE a, b RETURN keys(endNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], [])

        # labels() on deleted node via endNode should return empty array
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(b:Person {name: 'Bob', age: 25}) WITH r, a, b DELETE a, b RETURN labels(endNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], [])

        # id() on deleted node via endNode should return a valid integer
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(b:Person {name: 'Bob', age: 25}) WITH r, a, b DELETE a, b RETURN id(endNode(r))"""
        result = self.graph.query(query)
        self.env.assertNotEqual(result.result_set[0][0], None)

        # hasLabels on deleted node via endNode should return false
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(b:Person {name: 'Bob', age: 25}) WITH r, a, b DELETE a, b RETURN hasLabels(endNode(r), ['Person'])"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], False)

        # typeof on deleted node via endNode should return "Node"
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(b:Person {name: 'Bob', age: 25}) WITH r, a, b DELETE a, b RETURN typeof(endNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], "Node")

        # indegree on deleted node via endNode should return 0
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(b:Person {name: 'Bob', age: 25}) WITH r, a, b DELETE a, b RETURN indegree(endNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], 0)

        # outdegree on deleted node via endNode should return 0
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(b:Person {name: 'Bob', age: 25}) WITH r, a, b DELETE a, b RETURN outdegree(endNode(r))"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], 0)

    # Test property access on deleted node via startNode/endNode
    # accessing a property on a node with NULL attribute-set should raise RuntimeError
    def test17_property_access_on_deleted_node(self):
        # accessing a property of a deleted node via startNode should raise RuntimeError
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(a) WITH r, a DELETE a RETURN startNode(r).name"""
        self.env.assertRaises(RuntimeError, lambda: self.graph.query(query))

        # accessing a property of a deleted node via endNode should raise RuntimeError
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS]->(b:Person {name: 'Bob', age: 25}) WITH r, a, b DELETE a, b RETURN endNode(r).name"""
        self.env.assertRaises(RuntimeError, lambda: self.graph.query(query))

    # Test edge entity functions when endpoints are deleted
    # edge is still in scope but implicitly deleted via endpoint deletion
    def test18_edge_functions_with_deleted_endpoints(self):
        # id() on edge with deleted endpoints
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS {since: 2020}]->(a) WITH r, a DELETE a RETURN id(r)"""
        result = self.graph.query(query)
        self.env.assertNotEqual(result.result_set[0][0], None)

        # type() on edge with deleted endpoints
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS {since: 2020}]->(a) WITH r, a DELETE a RETURN type(r)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], "KNOWS")

        # properties() on edge with deleted endpoints
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS {since: 2020}]->(a) WITH r, a DELETE a RETURN properties(r)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], {'since': 2020})

        # keys() on edge with deleted endpoints
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS {since: 2020}]->(a) WITH r, a DELETE a RETURN keys(r)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], ['since'])

        # property access on edge with deleted endpoints
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS {since: 2020}]->(a) WITH r, a DELETE a RETURN r.since"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], 2020)

        # typeof on edge with deleted endpoints should return "Edge"
        query = """CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS {since: 2020}]->(a) WITH r, a DELETE a RETURN typeof(r)"""
        result = self.graph.query(query)
        self.env.assertEquals(result.result_set[0][0], "Edge")
