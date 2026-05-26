from common import Env

# Regression test for issue #2048.
#
# DUMP-ing a graph and RESTORE-ing the payload under a *different* key while the
# original graph still exists used to crash the server with a NULL dereference:
# the dumped payload embeds the original graph name, so on RESTORE the decoder
# found the still-live graph, treated the payload as that graph's first virtual
# key, and re-loaded the schema on top of the populated graph - doubling the
# relation-matrix count and driving the matrix decode loop past the end of the
# matrices array.
#
# The decoder now threads the destination key name through and, for a single-key
# restore to a different name, creates an independent new graph under the
# destination key instead of corrupting the live one. So the restore succeeds and
# produces an independent copy. (Aligned with the approach in PR #1737.)
class testDumpRestoreRename():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test01_restore_to_different_key_while_original_exists(self):
        # the exact #2048 crash scenario: original key is NOT deleted
        src = self.db.select_graph("g")
        src.query("CREATE (:A)-[:R]->(:B)")

        payload = self.conn.dump("g")
        self.env.assertIsNotNone(payload)

        # RESTORE under a different key while 'g' still exists -> must not crash
        self.conn.restore("g2", 0, payload)

        # server is alive
        self.env.assertTrue(self.conn.ping())

        # the restored graph is an independent copy with the same content
        dst = self.db.select_graph("g2")
        self.env.assertEqual(
            dst.query("MATCH (:A)-[r:R]->(:B) RETURN count(r)").result_set[0][0], 1)

        # the original graph is untouched
        self.env.assertEqual(
            src.query("MATCH (:A)-[r:R]->(:B) RETURN count(r)").result_set[0][0], 1)

        # both graphs are listed
        graphs = self.conn.execute_command("GRAPH.LIST")
        self.env.assertContains("g", graphs)
        self.env.assertContains("g2", graphs)

    def test02_restored_copy_is_independent(self):
        src = self.db.select_graph("src")
        src.query("CREATE (:Person {name: 'Alice'})")

        payload = self.conn.dump("src")
        self.conn.restore("dst", 0, payload)
        dst = self.db.select_graph("dst")

        # mutate the copy only
        dst.query("CREATE (:Person {name: 'Bob'})")

        self.env.assertEqual(
            src.query("MATCH (n) RETURN count(n)").result_set[0][0], 1)
        self.env.assertEqual(
            dst.query("MATCH (n) RETURN count(n)").result_set[0][0], 2)
