# -*- coding: utf-8 -*-
import csv
import time
import random
import threading
from common import *
from click.testing import CliRunner
from falkordb_bulk_loader.bulk_insert import bulk_insert

GRAPH_ID = "bulk_insert"

def ping_server(stop_event, res, self, interval = 0.1, delay = 2):
    ping_count = 0
    while not stop_event.is_set():
        t0 = time.time()
        self.db.connection.ping()
        t1 = time.time() - t0
        # Verify that pinging the server takes less than 1 second during bulk insertion
        self.env.assertLess(t1, delay)
        ping_count += 1
        time.sleep(interval)

    res[0] = ping_count

class testGraphBulkInsertFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

        # skip test if we're running under Valgrind
        if VALGRIND:
            self.env.skip() # valgrind is not working correctly with replication

        self.port = self.env.envRunner.port
        self.graph = self.db.select_graph(GRAPH_ID)

    # Run bulk loader script and validate terminal output
    def test01_run_script(self):
        runner = CliRunner()

        csv_path = os.path.dirname(os.path.abspath(__file__)) + '/../../demo/social/resources/bulk_formatted/'
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          '--nodes', csv_path + 'Person.csv',
                                          '--nodes', csv_path + 'Country.csv',
                                          '--relations', csv_path + 'KNOWS.csv',
                                          '--relations', csv_path + 'VISITED.csv',
                                          GRAPH_ID])

        # The script should report 27 node creations and 56 edge creations
        self.env.assertEquals(res.exit_code, 0)
        self.env.assertIn('27 nodes created', res.output)
        self.env.assertIn('56 relations created', res.output)

    # Validate that the expected nodes and properties have been constructed
    def test02_validate_nodes(self):
        # Query the newly-created graph
        query_result = self.graph.query('MATCH (p:Person) RETURN p.name, p.age, p.gender, p.status, ID(p) ORDER BY p.name')
        # Verify that the Person label exists, has the correct attributes, and is properly populated
        expected_result = [['Ailon Velger', 32, 'male', 'married', 2],
                           ['Alon Fital', 32, 'male', 'married', 1],
                           ['Boaz Arad', 31, 'male', 'married', 4],
                           ['Gal Derriere', 26, 'male', 'single', 11],
                           ['Jane Chernomorin', 31, 'female', 'married', 8],
                           ['Lucy Yanfital', 30, 'female', 'married', 7],
                           ['Mor Yesharim', 31, 'female', 'married', 12],
                           ['Noam Nativ', 34, 'male', 'single', 13],
                           ['Omri Traub', 33, 'male', 'single', 5],
                           ['Ori Laslo', 32, 'male', 'married', 3],
                           ['Roi Lipman', 32, 'male', 'married', 0],
                           ['Shelly Laslo Rooz', 31, 'female', 'married', 9],
                           ['Tal Doron', 32, 'male', 'single', 6],
                           ['Valerie Abigail Arad', 31, 'female', 'married', 10]]
        self.env.assertEquals(query_result.result_set, expected_result)

        # Verify that the Country label exists, has the correct attributes, and is properly populated
        query_result = self.graph.query('MATCH (c:Country) RETURN c.name, ID(c) ORDER BY c.name')
        expected_result = [['Andora', 21],
                           ['Canada', 18],
                           ['China', 19],
                           ['Germany', 24],
                           ['Greece', 17],
                           ['Italy', 25],
                           ['Japan', 16],
                           ['Kazakhstan', 22],
                           ['Netherlands', 20],
                           ['Prague', 15],
                           ['Russia', 23],
                           ['Thailand', 26],
                           ['USA', 14]]
        self.env.assertEquals(query_result.result_set, expected_result)

    # Validate that the expected relations and properties have been constructed
    def test03_validate_relations(self):
        # Query the newly-created graph
        query_result = self.graph.query('MATCH (a)-[e:KNOWS]->(b) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name')

        expected_result = [['Ailon Velger', 'friend', 'Noam Nativ'],
                           ['Alon Fital', 'friend', 'Gal Derriere'],
                           ['Alon Fital', 'friend', 'Mor Yesharim'],
                           ['Boaz Arad', 'friend', 'Valerie Abigail Arad'],
                           ['Roi Lipman', 'friend', 'Ailon Velger'],
                           ['Roi Lipman', 'friend', 'Alon Fital'],
                           ['Roi Lipman', 'friend', 'Boaz Arad'],
                           ['Roi Lipman', 'friend', 'Omri Traub'],
                           ['Roi Lipman', 'friend', 'Ori Laslo'],
                           ['Roi Lipman', 'friend', 'Tal Doron'],
                           ['Ailon Velger', 'married', 'Jane Chernomorin'],
                           ['Alon Fital', 'married', 'Lucy Yanfital'],
                           ['Ori Laslo', 'married', 'Shelly Laslo Rooz']]
        self.env.assertEquals(query_result.result_set, expected_result)

        query_result = self.graph.query('MATCH (a)-[e:VISITED]->(b) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name')

        expected_result = [['Alon Fital', 'business', 'Prague'],
                           ['Alon Fital', 'business', 'USA'],
                           ['Boaz Arad', 'business', 'Netherlands'],
                           ['Boaz Arad', 'business', 'USA'],
                           ['Gal Derriere', 'business', 'Netherlands'],
                           ['Jane Chernomorin', 'business', 'USA'],
                           ['Lucy Yanfital', 'business', 'USA'],
                           ['Mor Yesharim', 'business', 'Germany'],
                           ['Ori Laslo', 'business', 'China'],
                           ['Ori Laslo', 'business', 'USA'],
                           ['Roi Lipman', 'business', 'Prague'],
                           ['Roi Lipman', 'business', 'USA'],
                           ['Tal Doron', 'business', 'Japan'],
                           ['Tal Doron', 'business', 'USA'],
                           ['Alon Fital', 'pleasure', 'Greece'],
                           ['Alon Fital', 'pleasure', 'Prague'],
                           ['Alon Fital', 'pleasure', 'USA'],
                           ['Boaz Arad', 'pleasure', 'Netherlands'],
                           ['Boaz Arad', 'pleasure', 'USA'],
                           ['Jane Chernomorin', 'pleasure', 'Greece'],
                           ['Jane Chernomorin', 'pleasure', 'Netherlands'],
                           ['Jane Chernomorin', 'pleasure', 'USA'],
                           ['Lucy Yanfital', 'pleasure', 'Kazakhstan'],
                           ['Lucy Yanfital', 'pleasure', 'Prague'],
                           ['Lucy Yanfital', 'pleasure', 'USA'],
                           ['Mor Yesharim', 'pleasure', 'Greece'],
                           ['Mor Yesharim', 'pleasure', 'Italy'],
                           ['Noam Nativ', 'pleasure', 'Germany'],
                           ['Noam Nativ', 'pleasure', 'Netherlands'],
                           ['Noam Nativ', 'pleasure', 'Thailand'],
                           ['Omri Traub', 'pleasure', 'Andora'],
                           ['Omri Traub', 'pleasure', 'Greece'],
                           ['Omri Traub', 'pleasure', 'USA'],
                           ['Ori Laslo', 'pleasure', 'Canada'],
                           ['Roi Lipman', 'pleasure', 'Japan'],
                           ['Roi Lipman', 'pleasure', 'Prague'],
                           ['Shelly Laslo Rooz', 'pleasure', 'Canada'],
                           ['Shelly Laslo Rooz', 'pleasure', 'China'],
                           ['Shelly Laslo Rooz', 'pleasure', 'USA'],
                           ['Tal Doron', 'pleasure', 'Andora'],
                           ['Tal Doron', 'pleasure', 'USA'],
                           ['Valerie Abigail Arad', 'pleasure', 'Netherlands'],
                           ['Valerie Abigail Arad', 'pleasure', 'Russia']]

        self.env.assertEquals(query_result.result_set, expected_result)

    def test04_private_identifiers(self):
        graphname = "tmpgraph1"
        # Write temporary files
        with open('/tmp/nodes.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["_identifier", "nodename"])
            out.writerow([0, "a"])
            out.writerow([5, "b"])
            out.writerow([3, "c"])
        with open('/tmp/relations.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["src", "dest"])
            out.writerow([0, 3])
            out.writerow([5, 3])

        runner = CliRunner()
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          '--nodes', '/tmp/nodes.tmp',
                                          '--relations', '/tmp/relations.tmp',
                                          graphname])

        # The script should report 3 node creations and 2 edge creations
        self.env.assertEquals(res.exit_code, 0)
        self.env.assertIn('3 nodes created', res.output)
        self.env.assertIn('2 relations created', res.output)

        # Delete temporary files
        os.remove('/tmp/nodes.tmp')
        os.remove('/tmp/relations.tmp')

        tmp_graph = self.db.select_graph(graphname)
        # The field "_identifier" should not be a property in the graph
        query_result = tmp_graph.query('MATCH (a) RETURN a')

        for propname in query_result.header:
            self.env.assertNotIn('_identifier', propname)

    def test05_reused_identifier(self):
        graphname = "tmpgraph2"
        # Write temporary files
        with open('/tmp/nodes.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["_identifier", "nodename"])
            out.writerow([0, "a"])
            out.writerow([5, "b"])
            out.writerow([0, "c"]) # reused identifier
        with open('/tmp/relations.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["src", "dest"])
            out.writerow([0, 3])

        runner = CliRunner()
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          '--nodes', '/tmp/nodes.tmp',
                                          '--relations', '/tmp/relations.tmp',
                                          graphname])

        # The script should fail because a node identifier is reused
        self.env.assertNotEqual(res.exit_code, 0)
        self.env.assertIn('used multiple times', res.output)

        # Run the script again without creating relations
        runner = CliRunner()
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          '--nodes', '/tmp/nodes.tmp',
                                          graphname])

        # The script should succeed and create 3 nodes
        self.env.assertEquals(res.exit_code, 0)
        self.env.assertIn('3 nodes created', res.output)

        # Delete temporary files
        os.remove('/tmp/nodes.tmp')
        os.remove('/tmp/relations.tmp')

    def test06_batched_build(self):
        # Create demo graph wth one query per input file
        graphname = "batched_graph"
        runner = CliRunner()

        csv_path = os.path.dirname(os.path.abspath(__file__)) + '/../../demo/social/resources/bulk_formatted/'
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          '--nodes', csv_path + 'Person.csv',
                                          '--nodes', csv_path + 'Country.csv',
                                          '--relations', csv_path + 'KNOWS.csv',
                                          '--relations', csv_path + 'VISITED.csv',
                                          '--max-token-count', 1,
                                          graphname])

        self.env.assertEquals(res.exit_code, 0)
        # The script should report statistics multiple times
        self.env.assertGreater(res.output.count('nodes created'), 1)

        new_graph = self.db.select_graph(graphname)

        # Newly-created graph should be identical to graph created in single query
        original_result = self.graph.query('MATCH (p:Person) RETURN p, ID(p) ORDER BY p.name')
        new_result = new_graph.query('MATCH (p:Person) RETURN p, ID(p) ORDER BY p.name')
        self.env.assertEquals(original_result.result_set, new_result.result_set)

        original_result = self.graph.query('MATCH (a)-[e:KNOWS]->(b) RETURN a.name, e, b.name ORDER BY e.relation, a.name')
        new_result = new_graph.query('MATCH (a)-[e:KNOWS]->(b) RETURN a.name, e, b.name ORDER BY e.relation, a.name')
        self.env.assertEquals(original_result.result_set, new_result.result_set)

    def test07_script_failures(self):
        graphname = "tmpgraph3"
        # Write temporary files
        with open('/tmp/nodes.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["id", "nodename"])
            out.writerow([0]) # Wrong number of properites

        runner = CliRunner()
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          '--nodes', '/tmp/nodes.tmp',
                                          graphname])

        # The script should fail because a row has the wrong number of fields
        self.env.assertNotEqual(res.exit_code, 0)
        self.env.assertIn('Expected 2 columns', str(res.exception))

        # Write temporary files
        with open('/tmp/nodes.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["id", "nodename"])
            out.writerow([0, "a"])

        with open('/tmp/relations.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["src"]) # Incomplete relation description
            out.writerow([0])

        runner = CliRunner()
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          '--nodes', '/tmp/nodes.tmp',
                                          '--relations', '/tmp/relations.tmp',
                                          graphname])

        # The script should fail because a row has the wrong number of fields
        self.env.assertNotEqual(res.exit_code, 0)
        self.env.assertIn('should have at least 2 elements', str(res.exception))

        with open('/tmp/relations.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["src", "dest"])
            out.writerow([0, "fakeidentifier"])

        runner = CliRunner()
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          '--nodes', '/tmp/nodes.tmp',
                                          '--relations', '/tmp/relations.tmp',
                                          graphname])

        # The script should fail because an invalid node identifier was used
        self.env.assertNotEqual(res.exit_code, 0)
        self.env.assertIn('fakeidentifier', str(res.exception))
        os.remove('/tmp/nodes.tmp')
        os.remove('/tmp/relations.tmp')

        # Test passing invalid arguments directly to the GRAPH.BULK endpoint
        try:
            self.db.execute_command("GRAPH.BULK", "a", "a", "a")
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn("Invalid graph operation on empty key", str(e))

    # Verify that numeric, boolean, and null types are properly handled
    def test08_property_types(self):
        graphname = "tmpgraph4"
        # Write temporary files
        with open('/tmp/nodes.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["numeric", "mixed", "bool"])
            out.writerow([0, '', True])
            out.writerow([5, "notnull", False])
            out.writerow([7, '', False]) # reused identifier
        with open('/tmp/relations.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["src", "dest", "prop"])
            out.writerow([0, 5, True])
            out.writerow([5, 7, 3.5])
            out.writerow([7, 0, ''])

        runner = CliRunner()
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          '--nodes', '/tmp/nodes.tmp',
                                          '--relations', '/tmp/relations.tmp',
                                          graphname])

        self.env.assertEquals(res.exit_code, 0)
        self.env.assertIn('3 nodes created', res.output)
        self.env.assertIn('3 relations created', res.output)

        graph = self.db.select_graph(graphname)
        query_result = graph.query('MATCH (a)-[e]->() RETURN a.numeric, a.mixed, a.bool, e.prop ORDER BY a.numeric, e.prop')
        expected_result = [[0, None, True, True],
                           [5, 'notnull', False, 3.5],
                           [7, None, False, None]]

        # The graph should have the correct types for all properties
        self.env.assertEquals(query_result.result_set, expected_result)

    # Verify that the bulk loader does not block the server
    def test09_large_bulk_insert(self):
        graphname = "tmpgraph5"
        prop_str = "Property value to be repeated 1 million generating a multi-megabyte CSV"

        # Write temporary files
        filename = '/tmp/nodes.tmp'
        with open(filename, mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["long_property_string"])
            for i in range(100_000):
                out.writerow([prop_str])

        runner = CliRunner()

        # Instantiate a thread to run the bulk loader
        res = [None]
        stop_event = threading.Event()
        thread = threading.Thread(target=ping_server, args=(stop_event, res, self))
        thread.start()

        # Run bulk insert
        runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}", '--nodes', filename, graphname])

        # Signal the thread to stop
        stop_event.set()

        thread.join()
        ping_count = res[0]
        # Verify that at least one ping was issued
        self.env.assertGreaterEqual(ping_count, 1)

    # Verify that nodes with multiple labels are created correctly
    def test10_multiple_labels(self):
        graphname = "tmpgraph6"
        # Write temporary files
        with open('/tmp/City:Place.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["name"])
            out.writerow(["Binghamton"])
            out.writerow(["Geneseo"])
            out.writerow(["Stamford"])

        with open('/tmp/Place:State.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["name"])
            out.writerow(["New York"])
            out.writerow(["Connecticut"])

        with open('/tmp/Place:Country.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["name"])
            out.writerow(["US"])

        with open('/tmp/PART_OF.tmp', mode='w') as csv_file:
            out = csv.writer(csv_file)
            out.writerow(["src", "dest"])
            out.writerow(["Binghamton", "New York"])
            out.writerow(["Geneseo", "New York"])
            out.writerow(["Stamford", "Connecticut"])
            out.writerow(["New York", "US"])
            out.writerow(["Connecticut", "US"])

        runner = CliRunner()
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          '--nodes', '/tmp/City:Place.tmp',
                                          '--nodes', '/tmp/Place:State.tmp',
                                          '--nodes', '/tmp/Place:Country.tmp',
                                          '--relations', '/tmp/PART_OF.tmp',
                                          graphname])

        self.env.assertEquals(res.exit_code, 0)
        self.env.assertIn('6 nodes created', res.output)
        self.env.assertIn('5 relations created', res.output)

        graph = self.db.select_graph(graphname)

        #-----------------------------------------------------------------------
        expected_result = [["Binghamton"],
                           ["Connecticut"],
                           ["Geneseo"],
                           ["New York"],
                           ["Stamford"],
                           ["US"]]
        queries = [
                'MATCH (a) RETURN a.name ORDER BY a.name',
                'MATCH (a:Place) RETURN a.name ORDER BY a.name']
        for q in queries:
            query_result = graph.query(q)
            self.env.assertEquals(query_result.result_set, expected_result)

        #-----------------------------------------------------------------------
        expected_result = [["Binghamton"],
                           ["Geneseo"],
                           ["Stamford"]]
        queries = [
                'MATCH (a:City) RETURN a.name ORDER BY a.name',
                'MATCH (a:Place:City) RETURN a.name ORDER BY a.name',
                'MATCH (a:City:Place) RETURN a.name ORDER BY a.name']

        for q in queries:
            query_result = graph.query(q)
            self.env.assertEquals(query_result.result_set, expected_result)


        expected_result = [["Connecticut"],
                           ["New York"]]
        queries = [
                'MATCH (a:State) RETURN a.name ORDER BY a.name',
                'MATCH (a:Place:State) RETURN a.name ORDER BY a.name',
                'MATCH (a:State:Place) RETURN a.name ORDER BY a.name']

        for q in queries:
            query_result = graph.query(q)
            self.env.assertEquals(query_result.result_set, expected_result)

        #-----------------------------------------------------------------------
        expected_result = [["Stamford", "Connecticut"],
                           ["Binghamton", "New York"],
                           ["Geneseo", "New York"],
                           ["Connecticut", "US"],
                           ["New York", "US"]]
        queries = [
                'MATCH (a)-[{rel}]->(b) RETURN a.name, b.name ORDER BY b.name, a.name',
                'MATCH (a)-[{rel}]->(b:Place) RETURN a.name, b.name ORDER BY b.name, a.name',
                'MATCH (a:Place)-[{rel}]->(b) RETURN a.name, b.name ORDER BY b.name, a.name',
                'MATCH (a:Place)-[{rel}]->(b:Place) RETURN a.name, b.name ORDER BY b.name, a.name']

        relations = ['', ':PART_OF']
        for r in relations:
            for q in queries:
                query_result = graph.query(q.format(rel=r))
                self.env.assertEquals(query_result.result_set, expected_result)

        #-----------------------------------------------------------------------
        expected_result = [["Connecticut", "US"],
                           ["New York", "US"]]
        queries = [
            'MATCH (a)-[{rel}]->(b:Country) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a)-[{rel}]->(b:Place:Country) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a)-[{rel}]->(b:Country:Place) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:Place)-[{rel}]->(b:Country) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:Place)-[{rel}]->(b:Place:Country) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:Place)-[{rel}]->(b:Country:Place) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:State)-[{rel}]->(b:Place) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:State)-[{rel}]->(b:Country) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:State)-[{rel}]->(b:Place:Country) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:State)-[{rel}]->(b:Country:Place) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:Place:State)-[{rel}]->(b) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:Place:State)-[{rel}]->(b:Place) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:Place:State)-[{rel}]->(b:Place:Country) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:Place:State)-[{rel}]->(b:Country:Place) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:State:Place)-[{rel}]->(b) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:State:Place)-[{rel}]->(b:Place) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:State:Place)-[{rel}]->(b:Place:Country) RETURN a.name, b.name ORDER BY b.name, a.name',
            'MATCH (a:State:Place)-[{rel}]->(b:Country:Place) RETURN a.name, b.name ORDER BY b.name, a.name']

        relations = ['', ':PART_OF']
        for r in relations:
            for q in queries:
                query_result = graph.query(q.format(rel=r))
                self.env.assertEquals(query_result.result_set, expected_result)

    # Verify that nodes with multiple labels are created correctly
    def test11_social_multiple_labels(self):
        # Create the social graph with multi-labeled nodes
        graphname = "multilabel_social"
        graph = self.db.select_graph(graphname)
        csv_path = os.path.dirname(os.path.abspath(__file__)) + '/../../demo/social/resources/bulk_formatted/'

        runner = CliRunner()
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          '--nodes-with-label', "Person:Visitor", csv_path + 'Person.csv',
                                          '--nodes-with-label', "Country:Place", csv_path + 'Country.csv',
                                          '--relations', csv_path + 'KNOWS.csv',
                                          '--relations', csv_path + 'VISITED.csv',
                                          graphname])

        # The script should report 27 node creations and 48 edge creations
        self.env.assertEquals(res.exit_code, 0)
        self.env.assertIn('27 nodes created', res.output)
        self.env.assertIn('56 relations created', res.output)

        #-----------------------------------------------------------------------
        # Verify that the Person and Visitor labels both exist and produce the same results when queried
        expected_result = [['Ailon Velger', 32, 'male', 'married', 2],
                           ['Alon Fital', 32, 'male', 'married', 1],
                           ['Boaz Arad', 31, 'male', 'married', 4],
                           ['Gal Derriere', 26, 'male', 'single', 11],
                           ['Jane Chernomorin', 31, 'female', 'married', 8],
                           ['Lucy Yanfital', 30, 'female', 'married', 7],
                           ['Mor Yesharim', 31, 'female', 'married', 12],
                           ['Noam Nativ', 34, 'male', 'single', 13],
                           ['Omri Traub', 33, 'male', 'single', 5],
                           ['Ori Laslo', 32, 'male', 'married', 3],
                           ['Roi Lipman', 32, 'male', 'married', 0],
                           ['Shelly Laslo Rooz', 31, 'female', 'married', 9],
                           ['Tal Doron', 32, 'male', 'single', 6],
                           ['Valerie Abigail Arad', 31, 'female', 'married', 10]]

        queries = [
                'MATCH (p:Person) RETURN p.name, p.age, p.gender, p.status, ID(p) ORDER BY p.name',
                'MATCH (p:Visitor) RETURN p.name, p.age, p.gender, p.status, ID(p) ORDER BY p.name',
                'MATCH (p:Person:Visitor) RETURN p.name, p.age, p.gender, p.status, ID(p) ORDER BY p.name']

        for q in queries:
            query_result = graph.query(q)
            self.env.assertEquals(query_result.result_set, expected_result)

        #-----------------------------------------------------------------------
        # Verify that the Country and Place labels both exist and produce the same results when queried
        expected_result = [['Andora', 21],
                           ['Canada', 18],
                           ['China', 19],
                           ['Germany', 24],
                           ['Greece', 17],
                           ['Italy', 25],
                           ['Japan', 16],
                           ['Kazakhstan', 22],
                           ['Netherlands', 20],
                           ['Prague', 15],
                           ['Russia', 23],
                           ['Thailand', 26],
                           ['USA', 14]]

        queries = [
                'MATCH (c:Country) RETURN c.name, ID(c) ORDER BY c.name',
                'MATCH (c:Place) RETURN c.name, ID(c) ORDER BY c.name',
                'MATCH (c:Country:Place) RETURN c.name, ID(c) ORDER BY c.name']

        for q in queries:
            query_result = graph.query(q)
            self.env.assertEquals(query_result.result_set, expected_result)

        #-----------------------------------------------------------------------
        # Validate results when performing traversals using all combinations of labels
        expected_result = [['Ailon Velger', 'friend', 'Noam Nativ'],
                           ['Alon Fital', 'friend', 'Gal Derriere'],
                           ['Alon Fital', 'friend', 'Mor Yesharim'],
                           ['Boaz Arad', 'friend', 'Valerie Abigail Arad'],
                           ['Roi Lipman', 'friend', 'Ailon Velger'],
                           ['Roi Lipman', 'friend', 'Alon Fital'],
                           ['Roi Lipman', 'friend', 'Boaz Arad'],
                           ['Roi Lipman', 'friend', 'Omri Traub'],
                           ['Roi Lipman', 'friend', 'Ori Laslo'],
                           ['Roi Lipman', 'friend', 'Tal Doron'],
                           ['Ailon Velger', 'married', 'Jane Chernomorin'],
                           ['Alon Fital', 'married', 'Lucy Yanfital'],
                           ['Ori Laslo', 'married', 'Shelly Laslo Rooz']]
        queries = [
                'MATCH (a)-[e:KNOWS]->(b) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a)-[e:KNOWS]->(b:Visitor) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a)-[e:KNOWS]->(b:Visitor:Person) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a)-[e:KNOWS]->(b:Person:Visitor) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Person)-[e:KNOWS]->(b) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Person)-[e:KNOWS]->(b:Person) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Person)-[e:KNOWS]->(b:Visitor) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Person)-[e:KNOWS]->(b:Person:Visitor) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Person)-[e:KNOWS]->(b:Visitor:Person) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Visitor)-[e:KNOWS]->(b) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Visitor)-[e:KNOWS]->(b:Person) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Visitor)-[e:KNOWS]->(b:Visitor) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Visitor)-[e:KNOWS]->(b:Person:Visitor) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Visitor)-[e:KNOWS]->(b:Visitor:Person) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e:KNOWS]->(b) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e:KNOWS]->(b:Person) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e:KNOWS]->(b:Visitor) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e:KNOWS]->(b:Person:Visitor) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e:KNOWS]->(b:Visitor:Person) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e:KNOWS]->(b) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e:KNOWS]->(b:Person) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e:KNOWS]->(b:Visitor) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e:KNOWS]->(b:Person:Visitor) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e:KNOWS]->(b:Visitor:Person) RETURN a.name, e.relation, b.name ORDER BY e.relation, a.name, b.name']

        for q in queries:
            query_result = graph.query(q)
            self.env.assertEquals(query_result.result_set, expected_result)

        #-----------------------------------------------------------------------
        expected_result = [['Alon Fital', 'business', 'Prague'],
                           ['Alon Fital', 'business', 'USA'],
                           ['Boaz Arad', 'business', 'Netherlands'],
                           ['Boaz Arad', 'business', 'USA'],
                           ['Gal Derriere', 'business', 'Netherlands'],
                           ['Jane Chernomorin', 'business', 'USA'],
                           ['Lucy Yanfital', 'business', 'USA'],
                           ['Mor Yesharim', 'business', 'Germany'],
                           ['Ori Laslo', 'business', 'China'],
                           ['Ori Laslo', 'business', 'USA'],
                           ['Roi Lipman', 'business', 'Prague'],
                           ['Roi Lipman', 'business', 'USA'],
                           ['Tal Doron', 'business', 'Japan'],
                           ['Tal Doron', 'business', 'USA'],
                           ['Alon Fital', 'pleasure', 'Greece'],
                           ['Alon Fital', 'pleasure', 'Prague'],
                           ['Alon Fital', 'pleasure', 'USA'],
                           ['Boaz Arad', 'pleasure', 'Netherlands'],
                           ['Boaz Arad', 'pleasure', 'USA'],
                           ['Jane Chernomorin', 'pleasure', 'Greece'],
                           ['Jane Chernomorin', 'pleasure', 'Netherlands'],
                           ['Jane Chernomorin', 'pleasure', 'USA'],
                           ['Lucy Yanfital', 'pleasure', 'Kazakhstan'],
                           ['Lucy Yanfital', 'pleasure', 'Prague'],
                           ['Lucy Yanfital', 'pleasure', 'USA'],
                           ['Mor Yesharim', 'pleasure', 'Greece'],
                           ['Mor Yesharim', 'pleasure', 'Italy'],
                           ['Noam Nativ', 'pleasure', 'Germany'],
                           ['Noam Nativ', 'pleasure', 'Netherlands'],
                           ['Noam Nativ', 'pleasure', 'Thailand'],
                           ['Omri Traub', 'pleasure', 'Andora'],
                           ['Omri Traub', 'pleasure', 'Greece'],
                           ['Omri Traub', 'pleasure', 'USA'],
                           ['Ori Laslo', 'pleasure', 'Canada'],
                           ['Roi Lipman', 'pleasure', 'Japan'],
                           ['Roi Lipman', 'pleasure', 'Prague'],
                           ['Shelly Laslo Rooz', 'pleasure', 'Canada'],
                           ['Shelly Laslo Rooz', 'pleasure', 'China'],
                           ['Shelly Laslo Rooz', 'pleasure', 'USA'],
                           ['Tal Doron', 'pleasure', 'Andora'],
                           ['Tal Doron', 'pleasure', 'USA'],
                           ['Valerie Abigail Arad', 'pleasure', 'Netherlands'],
                           ['Valerie Abigail Arad', 'pleasure', 'Russia']]

        queries = [
                'MATCH (a)-[e]->(b:Place) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a)-[e]->(b:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a)-[e]->(b:Place:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a)-[e]->(b:Country:Place) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person)-[e:VISITED]->(b) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person)-[e:VISITED]->(b:Place) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person)-[e:VISITED]->(b:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person)-[e:VISITED]->(b:Place:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person)-[e:VISITED]->(b:Country:Place) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor)-[e:VISITED]->(b) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor)-[e:VISITED]->(b:Place) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor)-[e:VISITED]->(b:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor)-[e:VISITED]->(b:Place:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor)-[e:VISITED]->(b:Country:Place) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e]->(b:Place) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e]->(b:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e]->(b:Place:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e:VISITED]->(b) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e:VISITED]->(b:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e:VISITED]->(b:Place:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Person:Visitor)-[e:VISITED]->(b:Country:Place) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e]->(b:Place) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e]->(b:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e]->(b:Place:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e:VISITED]->(b) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e:VISITED]->(b:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e:VISITED]->(b:Place:Country) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name',
                'MATCH (a:Visitor:Person)-[e:VISITED]->(b:Country:Place) RETURN a.name, e.purpose, b.name ORDER BY e.purpose, a.name, b.name']

        for q in queries:
            query_result = graph.query(q)
            self.env.assertEquals(query_result.result_set, expected_result)

    def test12_load_large_graph(self):
        """make sure bulk-loader is able to load a large graph quickly"""

        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        graphname = "bulk-loader-large-graph"
        n_lbls = 3
        n_rels = 2
        edge_rel_min = 500000
        edge_rel_max = 10000000
        node_lbl_count = 2000000
        total_node_count = node_lbl_count * n_lbls

        node_csvs = [f"./node_{i}.csv" for i in range(0, n_lbls)]
        edge_csvs = [f"./edge_{i}.csv" for i in range(0, n_rels)]

        # delete old csv files
        for node_csv in node_csvs:
            if os.path.exists(node_csv):
                os.remove(node_csv)

        for edge_csv in edge_csvs:
            if os.path.exists(edge_csv):
                os.remove(edge_csv)

        #-----------------------------------------------------------------------
        # create node csv files
        #-----------------------------------------------------------------------

        node_count = 0
        for node_csv in node_csvs:
            with open(node_csv, 'w') as f:
                writer = csv.writer(f)

                # header row
                writer.writerow(["id:ID"])

                for i in range(0, node_lbl_count):
                    writer.writerow([node_count])
                    node_count = node_count + 1

        #-----------------------------------------------------------------------
        # create edge csv files
        #-----------------------------------------------------------------------

        edge_count = 0
        for edge_csv in edge_csvs:
            with open(edge_csv, 'w') as f:
                writer = csv.writer(f)

                # Header row
                writer.writerow(['src', 'dest'])

                count = random.randint(edge_rel_min, edge_rel_max)
                for i in range(0, count):
                    src  = edge_count % total_node_count
                    dest = min(max(0, src + random.randint(-1, 1)), total_node_count-1) # 1/3 self pointing edge
                    writer.writerow([src, dest])
                    edge_count = edge_count + 1

        #-----------------------------------------------------------------------
        # load graph
        #-----------------------------------------------------------------------

        nodes_args = [arg for node_csv in node_csvs for arg in ('--nodes', node_csv)]
        edges_args = [arg for edge_csv in edge_csvs for arg in ('--relations', edge_csv)]

        runner = CliRunner()

        # ping server during bulk-load
        pings = [None]
        stop_event = threading.Event()
        thread = threading.Thread(target=ping_server, args=(stop_event, pings, self, 1, 4))
        thread.start()

        # start bulk-insert
        start_time = time.perf_counter()
        res = runner.invoke(bulk_insert, ['--server-url', f"redis://localhost:{self.port}",
                                          *nodes_args, *edges_args, graphname])
        # calculate the execution time
        execution_time = time.perf_counter() - start_time

        # Signal the thread to stop
        stop_event.set()
        thread.join()
        ping_count = pings[0]
        # Expecting at minimum ping every 2 seconds
        self.env.assertGreaterEqual(ping_count, execution_time / 2)

        # validate script results
        self.env.assertEquals(res.exit_code, 0)
        self.env.assertIn(f'{node_lbl_count} nodes created', res.output)
        self.env.assertIn(f'{edge_count} relations created', res.output)

        # make sure load time did not exceeds 80 seconds
        self.env.assertLess(execution_time, 80)

        # clean up
        for node_csv in node_csvs:
            os.remove(node_csv)

        for edge_csv in edge_csvs:
            os.remove(edge_csv)

