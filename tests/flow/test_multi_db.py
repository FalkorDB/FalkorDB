import re
import redis
from common import Env, SANITIZER

# Graphs living in different Redis logical databases (#2782).
#
# The module tracks every graph it holds in one process-wide registry. When
# that registry was keyed by key name alone, a graph created under a name
# another database already used displaced the first graph from it: the
# displaced graph then failed the writer-mode "is this graph still in the
# keyspace" check, and every later write to it aborted with "graph was deleted
# or replaced while the query was running" (a debug build died on an assert
# instead). MOVE and SWAPDB reached the same state without reusing a name in
# the same command, and a multi-key SAVE mixed up two graphs sharing a name.


class testMultiDb():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()
        self.port = self.env.envRunner.port
        self.host = self.env.host if hasattr(self.env, 'host') else 'localhost'

    def tearDown(self):
        self.conn.flushall()

    def con(self, db):
        return redis.Redis(host=self.host, port=self.port, db=db)

    def query(self, db, key, q):
        return self.con(db).execute_command("GRAPH.QUERY", key, q)

    def scalar(self, db, key, q):
        v = self.query(db, key, q)[1][0][0]
        # verbose replies carry some numbers as strings
        if isinstance(v, bytes) and re.fullmatch(rb'-?\d+', v):
            return int(v)
        return v

    def assert_writable(self, db, key, label):
        # a write that must not abort, then read it back
        before = self.scalar(db, key, f"MATCH (n:{label}) RETURN count(n)")
        self.query(db, key, f"CREATE (:{label} {{w: true}})")
        after = self.scalar(db, key, f"MATCH (n:{label}) RETURN count(n)")
        self.env.assertEqual(after, before + 1)

    def assert_alive(self):
        self.env.assertTrue(self.conn.ping())

    def test01_same_name_two_dbs(self):
        self.query(0, "shared", "CREATE (:InDB0 {n: 'zero'})")
        self.query(1, "shared", "CREATE (:InDB1 {n: 'one'})")
        self.assert_alive()

        # each db sees its own graph
        self.env.assertEqual(self.scalar(0, "shared", "MATCH (n:InDB0) RETURN n.n"), b'zero')
        self.env.assertEqual(self.scalar(1, "shared", "MATCH (n:InDB1) RETURN n.n"), b'one')
        self.env.assertEqual(self.scalar(0, "shared", "MATCH (n:InDB1) RETURN count(n)"), 0)
        self.env.assertEqual(self.scalar(1, "shared", "MATCH (n:InDB0) RETURN count(n)"), 0)

        # and both are writable
        self.assert_writable(0, "shared", "InDB0")
        self.assert_writable(1, "shared", "InDB1")
        self.assert_alive()

    def test02_del_in_one_db_leaves_other_writable(self):
        self.query(0, "shared", "CREATE (:InDB0)")
        self.query(1, "shared", "CREATE (:InDB1)")
        self.env.assertEqual(self.con(1).delete("shared"), 1)

        self.assert_writable(0, "shared", "InDB0")
        # recreate in db1, both writable again
        self.query(1, "shared", "CREATE (:Again)")
        self.assert_writable(0, "shared", "InDB0")
        self.assert_writable(1, "shared", "Again")
        self.assert_alive()

    def test03_move(self):
        self.query(0, "m", "CREATE (:Moved)")
        self.env.assertEqual(self.con(0).move("m", 2), True)
        self.query(0, "m", "CREATE (:Fresh)")

        self.env.assertEqual(self.scalar(2, "m", "MATCH (n:Moved) RETURN count(n)"), 1)
        self.env.assertEqual(self.scalar(0, "m", "MATCH (n:Moved) RETURN count(n)"), 0)
        self.assert_writable(2, "m", "Moved")
        self.assert_writable(0, "m", "Fresh")

        # and back again, onto a now-empty key
        self.con(0).delete("m")
        self.env.assertEqual(self.con(2).move("m", 0), True)
        self.assert_writable(0, "m", "Moved")
        self.assert_alive()

    def test04_swapdb(self):
        self.query(0, "s", "CREATE (:Swapped)")
        self.conn.swapdb(0, 3)
        self.query(0, "s", "CREATE (:Fresh)")

        self.env.assertEqual(self.scalar(3, "s", "MATCH (n:Swapped) RETURN count(n)"), 1)
        self.assert_writable(3, "s", "Swapped")
        self.assert_writable(0, "s", "Fresh")

        # swap back: each graph follows its database
        self.conn.swapdb(0, 3)
        self.env.assertEqual(self.scalar(0, "s", "MATCH (n:Swapped) RETURN count(n)"), 2)
        self.env.assertEqual(self.scalar(3, "s", "MATCH (n:Fresh) RETURN count(n)"), 2)
        self.assert_writable(0, "s", "Swapped")
        self.assert_writable(3, "s", "Fresh")
        self.assert_alive()

    def test05_rename_while_other_db_has_name(self):
        self.query(0, "r", "CREATE (:InDB0)")
        self.query(1, "x", "CREATE (:InDB1)")
        self.con(1).rename("x", "r")

        self.assert_writable(0, "r", "InDB0")
        self.assert_writable(1, "r", "InDB1")
        self.env.assertEqual(self.scalar(1, "r", "MATCH (n:InDB0) RETURN count(n)"), 0)

        # and the name it left is free in db1, while db0 is untouched
        self.query(1, "x", "CREATE (:New)")
        self.assert_writable(1, "x", "New")
        self.assert_writable(0, "r", "InDB0")
        self.assert_alive()

    def test06_save_reload_multi_key_same_name(self):
        if SANITIZER:
            self.env.skip()
            return

        # small virtual-key threshold so both graphs are split across keys
        self.conn.execute_command("GRAPH.CONFIG", "SET", "VKEY_MAX_ENTITY_COUNT", 10)
        try:
            self.query(0, "g", "UNWIND range(1, 100) AS i CREATE (:A {v: i})-[:R {v: i}]->(:A {v: -i})")
            self.query(1, "g", "UNWIND range(1, 60) AS i CREATE (:B {v: i * 10})")
            # a second, uniquely-named split graph in db1 too
            self.query(1, "solo", "UNWIND range(1, 40) AS i CREATE (:C {v: i})")

            def check():
                self.env.assertEqual(self.scalar(0, "g", "MATCH (n:A) RETURN count(n)"), 200)
                self.env.assertEqual(self.scalar(0, "g", "MATCH (n:A) RETURN sum(n.v)"), 0)
                self.env.assertEqual(self.scalar(0, "g", "MATCH ()-[e:R]->() RETURN count(e)"), 100)
                self.env.assertEqual(self.scalar(0, "g", "MATCH ()-[e:R]->() RETURN sum(e.v)"), 5050)
                self.env.assertEqual(self.scalar(0, "g", "MATCH (n:B) RETURN count(n)"), 0)
                self.env.assertEqual(self.scalar(1, "g", "MATCH (n:B) RETURN count(n)"), 60)
                self.env.assertEqual(self.scalar(1, "g", "MATCH (n:B) RETURN sum(n.v)"), 18300)
                self.env.assertEqual(self.scalar(1, "g", "MATCH (n:A) RETURN count(n)"), 0)
                self.env.assertEqual(self.scalar(1, "solo", "MATCH (n:C) RETURN sum(n.v)"), 820)
                # no virtual keys left behind in any db (telemetry streams
                # live next to their graph, and are not what this checks)
                def keys(db):
                    return sorted(k for k in self.con(db).keys("*") if not k.startswith(b'telemetry{'))
                self.env.assertEqual(keys(0), [b'g'])
                self.env.assertEqual(keys(1), [b'g', b'solo'])
                for db in range(16):
                    self.env.assertEqual(self.con(db).execute_command("SCAN", 0, "TYPE", "graphmeta")[1], [])

            check()
            self.conn.execute_command("SAVE")
            check()
            self.conn.execute_command("DEBUG", "RELOAD")
            check()

            self.assert_writable(0, "g", "A")
            self.assert_writable(1, "g", "B")
            self.assert_writable(1, "solo", "C")
            self.assert_alive()
        finally:
            self.conn.execute_command("GRAPH.CONFIG", "SET", "VKEY_MAX_ENTITY_COUNT", 100000)
