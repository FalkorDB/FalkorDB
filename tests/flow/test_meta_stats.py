import random
from common import *

GRAPH_ID = "meta_stats"

class testMetaStats():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def meta_stats(self, outputs=None):
        # Default outputs
        if outputs is None:
            outputs = [
                "labels",
                "relTypes",
                "relCount",
                "nodeCount",
                "labelCount",
                "relTypeCount",
                "propertyKeyCount"
            ]

        # Build column name → index map
        col_map = {name: idx for idx, name in enumerate(outputs)}

        proc_yield = ", ".join(outputs)
        query = f"CALL db.meta.stats() YIELD {proc_yield}"

        # Execute query
        result = self.graph.query(query)

        # Expect exactly one row
        self.env.assertEquals(len(result.result_set), 1)
        record = result.result_set[0]

        # Expect record length matches outputs
        self.env.assertEquals(len(record), len(outputs))

        # Return dict of column_name → value
        return {k: record[col_map[k]] for k in outputs}

    def populate_graph(self):
        """Create a small graph with labels, rel types, and properties."""
        q = """CREATE
                (john :Person {name: 'John', age: 33}),
                (mary :Person {name: 'Mary', age: 29}),
                (telaviv :City {name: 'Tel Aviv'}),
                (john)-[:KNOWS {since: 2010}]->(mary),
                (john)-[:VISITED]->(telaviv),
                (mary)-[:VISITED]->(telaviv)"""

        self.graph.query(q)

    def test_01_empty_graph(self):
        """CALL db.meta.stats() on an empty graph."""
        record = self.meta_stats()

        # All counts are zero
        labels = record['labels']
        relTypes = record['relTypes']   # relTypes

        self.env.assertEquals(len(labels), 0)
        self.env.assertEquals(len(relTypes), 0)

        self.env.assertEquals(record["nodeCount"], 0)
        self.env.assertEquals(record["relCount"], 0)
        self.env.assertEquals(record["labelCount"], 0)
        self.env.assertEquals(record["relTypeCount"], 0)
        self.env.assertEquals(record["propertyKeyCount"], 0)

    def test_02_populated_graph(self):
        """CALL db.meta.stats() on a graph with data."""
        self.populate_graph()

        record = self.meta_stats()

        labels = record["labels"]
        relTypes = record["relTypes"]

        # Verify expected label counts
        self.env.assertIn("Person", labels)
        self.env.assertIn("City", labels)
        self.env.assertEquals(labels["Person"], 2)
        self.env.assertEquals(labels["City"], 1)

        # Verify expected relationship types
        self.env.assertIn("KNOWS", relTypes)
        self.env.assertIn("VISITED", relTypes)
        self.env.assertEquals(relTypes["KNOWS"], 1)
        self.env.assertEquals(relTypes["VISITED"], 2)

        # Verify global counts
        self.env.assertEquals(record["nodeCount"], 3)
        self.env.assertEquals(record["relCount"], 3)
        self.env.assertEquals(record["labelCount"], 2)
        self.env.assertEquals(record["relTypeCount"], 2)
        self.env.assertGreaterEqual(record["propertyKeyCount"], 3)

    def test_03_partial_yield(self):
        """CALL db.meta.stats() YIELD nodeCount, relCount"""
        record = self.meta_stats(["nodeCount", "relCount"])

        self.env.assertEquals(record["nodeCount"], 3)
        self.env.assertEquals(record["relCount"], 3)

    def test_04_invalid_args(self):
        """CALL db.meta.stats(123) should error"""
        query = "CALL db.meta.stats(123)"
        try:
            self.graph.query(query)
            self.env.assertTrue(False, "Expected error for invalid arguments")
        except ResponseError as e:
            self.env.assertContains("procedure", str(e).lower())

    def test_05_performance_with_large_graph(self):
        """CALL db.meta.stats() on a larger graph to ensure no timeouts."""
        self.graph = self.db.select_graph("large_meta_stats")
        labels = ["Person", "Company", "Event"]
        rels = ["KNOWS", "WORKS_AT", "ATTENDED"]

        q = """UNWIND range (0, 500) AS x
               CREATE (:Person {id:x})
               WITH x
               WHERE x % 2 = 0
               CREATE (:Company {id:x})
               WITH x
               WHERE x % 4 = 0
               CREATE (:Event {id:x})
            """

        self.graph.query(q)

        q = """UNWIND range (0, 1000) AS x
               MATCH (a), (b)
               WHERE ID(a) = toInteger(rand() * 200) AND
                     ID(b) = toInteger(rand() * 200)
               CREATE (a)-[:KNOWS]->(b)
               WITH a, b, x
               WHERE x % 2 = 0
               CREATE (a)-[:WORKS_AT]->(b)
               WITH a, b, x
               WHERE x % 4 = 0
               CREATE (a)-[:ATTENDED]->(b)"""

        self.graph.query(q)

        record = self.meta_stats()

        # Sanity checks
        self.env.assertGreater(record["nodeCount"], 0)
        self.env.assertGreater(record["relCount"], 0)
        self.env.assertGreater(record["labelCount"], 0)
        self.env.assertGreater(record["relTypeCount"], 0)
        self.env.assertGreater(record["propertyKeyCount"], 0)

