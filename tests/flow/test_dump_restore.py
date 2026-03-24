from graph_utils import graph_eq
from common import Env, SANITIZER


GRAPH_ID = "dump_restore"


# tests Redis DUMP / RESTORE with FalkorDB graph keys
class testDumpRestore():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

    def test_01_dump_restore_simple(self):
        """DUMP/RESTORE a graph to a different key produces an independent copy"""

        src_id  = "src_simple"
        dest_id = "dest_simple"

        src_graph = self.db.select_graph(src_id)
        src_graph.query("CREATE (:Person {name: 'Alice'})")

        # DUMP src and RESTORE to dest
        payload = self.conn.dump(src_id)
        self.env.assertIsNotNone(payload)

        self.conn.restore(dest_id, 0, payload)

        dest_graph = self.db.select_graph(dest_id)

        # both graphs should be queryable and equal
        self.env.assertTrue(graph_eq(src_graph, dest_graph))

        # src should still have exactly 1 node (not 2)
        src_count = src_graph.query(
            "MATCH (n) RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEqual(src_count, 1)

        # dest should have exactly 1 node
        dest_count = dest_graph.query(
            "MATCH (n) RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEqual(dest_count, 1)

        # clean up
        src_graph.delete()
        dest_graph.delete()

    def test_02_dump_restore_graph_list(self):
        """Restored graph appears in GRAPH.LIST"""

        src_id  = "src_list"
        dest_id = "dest_list"

        src_graph = self.db.select_graph(src_id)
        src_graph.query("CREATE (:X)")

        # DUMP and RESTORE
        payload = self.conn.dump(src_id)
        self.conn.restore(dest_id, 0, payload)

        # GRAPH.LIST should contain both graphs
        graphs = self.conn.execute_command("GRAPH.LIST")
        self.env.assertContains(src_id, graphs)
        self.env.assertContains(dest_id, graphs)

        # clean up
        src_graph.delete()
        dest_graph = self.db.select_graph(dest_id)
        dest_graph.delete()

    def test_03_dump_restore_independence(self):
        """Mutating the restored graph does not affect the source"""

        src_id  = "src_indep"
        dest_id = "dest_indep"

        src_graph = self.db.select_graph(src_id)
        src_graph.query("CREATE (:Person {name: 'Alice'})")

        # DUMP and RESTORE
        payload = self.conn.dump(src_id)
        self.conn.restore(dest_id, 0, payload)

        dest_graph = self.db.select_graph(dest_id)

        # mutate dest only
        dest_graph.query("CREATE (:Person {name: 'Bob'})")

        # src should still have 1 node
        src_count = src_graph.query(
            "MATCH (n) RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEqual(src_count, 1)

        # dest should have 2 nodes
        dest_count = dest_graph.query(
            "MATCH (n) RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEqual(dest_count, 2)

        # clean up
        src_graph.delete()
        dest_graph.delete()

    def test_04_dump_restore_with_edges(self):
        """DUMP/RESTORE preserves nodes, edges, and properties"""

        src_id  = "src_edges"
        dest_id = "dest_edges"

        src_graph = self.db.select_graph(src_id)
        src_graph.query(
            "CREATE (:A {v:1})-[:R {w:2}]->(:B {v:3})"
        )

        payload = self.conn.dump(src_id)
        self.conn.restore(dest_id, 0, payload)

        dest_graph = self.db.select_graph(dest_id)
        self.env.assertTrue(graph_eq(src_graph, dest_graph))

        # clean up
        src_graph.delete()
        dest_graph.delete()

    def test_05_dump_restore_with_index(self):
        """DUMP/RESTORE preserves indices"""

        src_id  = "src_idx"
        dest_id = "dest_idx"

        src_graph = self.db.select_graph(src_id)
        src_graph.query("CREATE (:Person {name: 'Alice', age: 30})")
        src_graph.query("CREATE INDEX FOR (n:Person) ON (n.name)")

        # wait for index to be populated
        src_graph.query("MATCH (n:Person) WHERE n.name = 'Alice' RETURN n")

        payload = self.conn.dump(src_id)
        self.conn.restore(dest_id, 0, payload)

        dest_graph = self.db.select_graph(dest_id)
        self.env.assertTrue(graph_eq(src_graph, dest_graph))

        # clean up
        src_graph.delete()
        dest_graph.delete()

    def test_06_dump_restore_multiple_times(self):
        """DUMP the same graph and RESTORE it to multiple keys"""

        src_id = "src_multi"

        src_graph = self.db.select_graph(src_id)
        src_graph.query("CREATE (:N {v:1})-[:E]->(:N {v:2})")

        payload = self.conn.dump(src_id)

        copies = []
        for i in range(3):
            dest_id = f"dest_multi_{i}"
            self.conn.restore(dest_id, 0, payload)
            copies.append(dest_id)

        # all copies should match the source
        for dest_id in copies:
            dest_graph = self.db.select_graph(dest_id)
            self.env.assertTrue(graph_eq(src_graph, dest_graph))

        # src should still have exactly 2 nodes
        src_count = src_graph.query(
            "MATCH (n) RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEqual(src_count, 2)

        # GRAPH.LIST should contain all graphs
        graphs = self.conn.execute_command("GRAPH.LIST")
        self.env.assertContains(src_id, graphs)
        for dest_id in copies:
            self.env.assertContains(dest_id, graphs)

        # clean up
        src_graph.delete()
        for dest_id in copies:
            g = self.db.select_graph(dest_id)
            g.delete()

    def test_07_dump_restore_with_existing_graphs(self):
        """DUMP/RESTORE works correctly when other graphs already exist"""

        other_id = "other_graph"
        src_id   = "src_existing"
        dest_id  = "dest_existing"

        # create another graph first
        other_graph = self.db.select_graph(other_id)
        other_graph.query("CREATE (:Dummy {v: 42})")

        # create src graph
        src_graph = self.db.select_graph(src_id)
        src_graph.query("CREATE (:Person {name: 'Alice'})")

        # DUMP and RESTORE
        payload = self.conn.dump(src_id)
        self.conn.restore(dest_id, 0, payload)

        dest_graph = self.db.select_graph(dest_id)

        # dest should match src
        self.env.assertTrue(graph_eq(src_graph, dest_graph))

        # other graph should be unaffected
        other_count = other_graph.query(
            "MATCH (n) RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEqual(other_count, 1)

        # clean up
        other_graph.delete()
        src_graph.delete()
        dest_graph.delete()

    def test_08_dump_restore_reload(self):
        """Restored graph survives DEBUG RELOAD"""

        src_id  = "src_reload"
        dest_id = "dest_reload"

        src_graph = self.db.select_graph(src_id)
        src_graph.query("CREATE (:A {v:1})-[:R {w:2}]->(:B {v:3})")

        payload = self.conn.dump(src_id)
        self.conn.restore(dest_id, 0, payload)

        # reload keyspace
        self.conn.execute_command("DEBUG", "RELOAD")

        # both graphs should survive reload and be equal
        self.env.assertTrue(graph_eq(src_graph, self.db.select_graph(dest_id)))

        # clean up
        src_graph.delete()
        dest_graph = self.db.select_graph(dest_id)
        dest_graph.delete()
