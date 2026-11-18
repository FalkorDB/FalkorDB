from common import *
from redis import ResponseError

VERSION = 0
GRAPH_ID = "GraphVersion"


class testGraphVersioning(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True, enableDebugCommand=True)
        self.master = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        self.master_graph = Graph(self.master, GRAPH_ID)
        self.replica_graph = Graph(self.replica, GRAPH_ID)
        self.master.wait(1, 0)

    # Make sure graph version changes once a new label is created
    def test01_version_update_on_label_creation(self):
        global VERSION

        con = self.master

        # Adding a node without a label shouldn't update graph version.
        q = """CREATE ()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        # Adding a labeled node should update graph version.
        q = """CREATE (:L)"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertTrue(isinstance(res[0], ResponseError))

        # Update version
        VERSION = int(res[1])

        # Adding a node with an existing label shouldn't update graph version
        q = """CREATE (:L)"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

    # Make sure graph version changes once a new relationship type is created
    def test02_version_update_on_relation_creation(self):
        global VERSION
        con = self.master

        # Adding edge with a new relationship type should update graph version
        q = """CREATE ()-[:R]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertTrue(isinstance(res[0], ResponseError))

        # Update version
        VERSION = int(res[1])

        # Adding edge with existing relationship type shouldn't update graph version
        q = """CREATE ()-[:R]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

    # Make sure graph version changes once a new attribute is created
    def test03_version_update_on_attribute_creation(self):
        global VERSION
        con = self.master

        # Adding a new attribute should update graph version
        q = """CREATE ({v:1})"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertTrue(isinstance(res[0], ResponseError))

        # Update version
        VERSION = int(res[1])

        # Adding a new node with existing attribute shouldn't update graph version
        q = """CREATE ({v:1})"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        # Adding a new edge with a new attribute should update graph version
        q = """CREATE ()-[:R {q:1}]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertTrue(isinstance(res[0], ResponseError))

        # Update version
        VERSION = int(res[1])

        # Adding a new edge with existing attribute shouldn't update graph version
        q = """CREATE ()-[:R {v:1}]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

    # make sure the graph version on the replica matches the one on the master
    # after each schema-changing operation (new label, relation type, attribute).
    def test04_version_sync_master_replica(self):
        global VERSION

        def get_version(con):
            """
            Execute a no-op query and read the graph version
            """
            res = con.execute_command("GRAPH.RO_QUERY", GRAPH_ID, "RETURN 1", "version", VERSION)
            return int(res[1])

        # ------------------------------------------------------------------ #
        # 1. New label
        # ------------------------------------------------------------------ #
        q = """CREATE (:NewLabel)"""
        self.master_graph.query(q)
        # Wait for the replica to catch up before reading its version.
        self.master.wait(1, 0)

        master_version  = get_version(self.master)
        replica_version = get_version(self.replica)
        self.env.assertEqual(master_version, replica_version)
        VERSION = master_version

        # ------------------------------------------------------------------ #
        # 2. New relationship type
        # ------------------------------------------------------------------ #
        q = """CREATE ()-[:NewRelType]->()"""
        self.master_graph.query(q)
        self.master.wait(1, 0)

        master_version  = get_version(self.master)
        replica_version = get_version(self.replica)
        self.env.assertEqual(master_version, replica_version)
        VERSION = master_version

        # ------------------------------------------------------------------ #
        # 3. New attribute key
        # ------------------------------------------------------------------ #
        q = """CREATE ({newAttr: 42})"""
        self.master_graph.query(q)
        self.master.wait(1, 0)

        master_version  = get_version(self.master)
        replica_version = get_version(self.replica)
        self.env.assertEqual(master_version, replica_version)
        VERSION = master_version

        # ------------------------------------------------------------------ #
        # 4. Multiple schema changes
        # ------------------------------------------------------------------ #
        q = """CREATE (:A {a:1})-[:R {b:2, c:3}]->(:Z {z:4, y:5})"""
        self.master_graph.query(q)
        self.master.wait(1, 0)

        master_version  = get_version(self.master)
        replica_version = get_version(self.replica)
        self.env.assertEqual(master_version, replica_version)
        VERSION = master_version

    # ---------------------------------------------------------------------- #
    # SCHEMA_HASH h (#3757): the schema hash S, and the name lists when h != S.
    # ---------------------------------------------------------------------- #

    # labels [Person, City], types [LIVES_IN], keys [name, age]: the value
    # graph::schema_hash pins in its golden test, above i64::MAX, so the reply
    # carries it as the negative integer with the same bits.
    GOLDEN = 17300287264737141393
    GOLDEN_SIGNED = GOLDEN - 2**64
    GOLDEN_MAP = {"hash": GOLDEN_SIGNED, "labels": ["Person", "City"],
                  "relationshipTypes": ["LIVES_IN"], "propertyKeys": ["name", "age"]}

    def _client(self, con, protocol):
        """A client on `con`'s server that speaks `protocol`. The env's own
        connections speak RESP2 locally but RESP3 in CI, redis-py 8's default,
        and the schema element is a map, whose shape depends on the protocol."""
        kw = con.connection_pool.connection_kwargs
        return redis.Redis(host=kw["host"], port=kw["port"], password=kw.get("password"),
                           protocol=protocol, decode_responses=True,
                           socket_timeout=SOCKET_TIMEOUT, retry=NO_RETRY)

    def _as_map(self, element):
        """The schema element of a RESP2 reply as a dict: RESP2 has no map
        type, so a map arrives as a flat [key, value, ...] array."""
        self.env.assertTrue(isinstance(element, list) and len(element) % 2 == 0)
        return dict(zip(element[::2], element[1::2]))

    def _schema(self, con, key, h, q="RETURN 1", cmd="GRAPH.RO_QUERY"):
        """The schema element of a reply: always the last top-level element."""
        resp2 = self._client(con, 2)
        res = resp2.execute_command(cmd, key, q, "--compact", "SCHEMA_HASH", h)
        resp2.close()
        return self._as_map(res[-1])

    def _build_golden(self, key):
        # One name per query, so interning order (= id order) is fixed.
        for q in ["CREATE (:Person)", "CREATE (:City)", "CREATE ()-[:LIVES_IN]->()",
                  "CREATE ({name: 1})", "CREATE ({age: 1})"]:
            self.master.execute_command("GRAPH.QUERY", key, q)

    def test05_schema_reply_shape(self):
        key = "schema_shape"
        con = self._client(self.master, 2)

        # A query creates the graph: an empty schema hashes to 0, so h = 0 is current.
        res = con.execute_command("GRAPH.QUERY", key, "RETURN 1", "--compact",
                                  "SCHEMA_HASH", 0)
        self.env.assertEqual(len(res), 4)
        self.env.assertEqual(res[-1], ["hash", 0])
        # Any other h gets the hash and three empty lists.
        self.env.assertEqual(self._schema(con, key, 5),
                             {"hash": 0, "labels": [], "relationshipTypes": [],
                              "propertyKeys": []})

        # A write without RETURN: [stats, schema], with the names it added itself.
        res = con.execute_command("GRAPH.QUERY", key, "CREATE (:L {k: 1})-[:R]->()",
                                  "--compact", "SCHEMA_HASH", 0)
        self.env.assertEqual(len(res), 2)
        self.env.assertTrue(all(isinstance(line, str) for line in res[0]))
        schema = self._as_map(res[1])
        s = schema.pop("hash")
        self.env.assertNotEqual(s, 0)
        self.env.assertEqual(schema, {"labels": ["L"], "relationshipTypes": ["R"],
                                      "propertyKeys": ["k"]})

        # h = S: the hash alone, on both query commands, with --compact anywhere.
        for cmd in ["GRAPH.QUERY", "GRAPH.RO_QUERY"]:
            for flags in [("--compact", "SCHEMA_HASH", s),
                          ("--compact", "TIMEOUT", "1000", "SCHEMA_HASH", s),
                          ("SCHEMA_HASH", s, "TIMEOUT", "1000", "--compact")]:
                res = con.execute_command(cmd, key, "MATCH (n:L) RETURN n.k", *flags)
                self.env.assertEqual(len(res), 4)
                self.env.assertEqual(res[1], [[[3, 1]]])
                self.env.assertEqual(res[-1], ["hash", s])
        # (Without --compact the argument is refused: test_query_validation test17b.)

        # RESP3 has maps, and gets a real one.
        resp3 = self._client(self.master, 3)
        res = resp3.execute_command("GRAPH.RO_QUERY", key, "RETURN 1", "--compact",
                                    "SCHEMA_HASH", s)
        self.env.assertEqual(res[-1], {"hash": s})
        res = resp3.execute_command("GRAPH.QUERY", key, "CREATE ()", "--compact",
                                    "SCHEMA_HASH", 0)
        self.env.assertEqual(res[-1], {"hash": s, "labels": ["L"],
                                       "relationshipTypes": ["R"], "propertyKeys": ["k"]})
        resp3.close()

        # Without SCHEMA_HASH, the reply is what it was before the argument existed.
        res = con.execute_command("GRAPH.QUERY", key, "MATCH (n:L) RETURN n.k")
        self.env.assertEqual(len(res), 3)
        self.env.assertEqual(res[1], [[1]])
        self.env.assertTrue(all(isinstance(line, str) for line in res[2]))
        res = con.execute_command("GRAPH.QUERY", key, "CREATE ()")
        self.env.assertEqual(len(res), 1)
        self.env.assertTrue(all(isinstance(line, str) for line in res[0]))
        res = con.execute_command("GRAPH.QUERY", key, "MATCH (n:L) RETURN n.k", "--compact")
        self.env.assertEqual(len(res), 3)
        self.env.assertEqual(res[1], [[[3, 1]]])
        self.env.assertTrue(all(isinstance(line, str) for line in res[2]))
        res = con.execute_command("GRAPH.QUERY", key, "CREATE ()", "--compact")
        self.env.assertEqual(len(res), 1)
        self.env.assertTrue(all(isinstance(line, str) for line in res[0]))

        # EXPLAIN and PROFILE take the argument and ignore it.
        for cmd in ["GRAPH.EXPLAIN", "GRAPH.PROFILE"]:
            res = con.execute_command(cmd, key, "RETURN 1", "--compact", "SCHEMA_HASH", 0)
            self.env.assertTrue(all(isinstance(line, str) for line in res))

        # A write that returns rows, on both write paths to the reply: inline inside
        # MULTI, and queued for the write thread outside it. The rows stay compact.
        pipe = con.pipeline(transaction=True)
        pipe.execute_command("GRAPH.QUERY", key, "CREATE (:M) RETURN 1", "--compact",
                             "SCHEMA_HASH", s)
        res = pipe.execute()[0]
        self.env.assertEqual(len(res), 4)
        self.env.assertEqual(res[1], [[[3, 1]]])
        self.env.assertEqual(self._as_map(res[-1])["labels"], ["L", "M"])
        res = con.execute_command("GRAPH.QUERY", key, "CREATE (:N) RETURN 1", "--compact",
                                  "SCHEMA_HASH", s)
        self.env.assertEqual(len(res), 4)
        self.env.assertEqual(res[1], [[[3, 1]]])
        self.env.assertEqual(self._as_map(res[-1])["labels"], ["L", "M", "N"])
        con.close()

    def test06_schema_hash_value_round_trip(self):
        key = "schema_golden"
        self._build_golden(key)
        con = self.master

        self.env.assertEqual(self._schema(con, key, 0), self.GOLDEN_MAP)
        # Echoed back as the signed integer it arrived as, or as its unsigned bits.
        current = {"hash": self.GOLDEN_SIGNED}
        self.env.assertEqual(self._schema(con, key, self.GOLDEN_SIGNED), current)
        self.env.assertEqual(self._schema(con, key, self.GOLDEN), current)
        # Both ends of both ranges are valid values that don't match.
        for h in [2**64 - 1, -2**63, 2**63 - 1, 2**63]:
            self.env.assertEqual(self._schema(con, key, h), self.GOLDEN_MAP)

    def test07_schema_hash_drop_and_recreate(self):
        key = "schema_recreate"
        con = self.master

        con.execute_command("GRAPH.QUERY", key, "CREATE (:A {x: 1})")
        before = self._schema(con, key, 0)
        self.env.assertEqual(before, {"hash": before["hash"], "labels": ["A"],
                                      "relationshipTypes": [], "propertyKeys": ["x"]})

        # Same ids, other names: a client holding the old S gets the new names.
        con.delete(key)
        con.execute_command("GRAPH.QUERY", key, "CREATE (:B {y: 1})")
        after = self._schema(con, key, before["hash"])
        self.env.assertNotEqual(after["hash"], before["hash"])
        self.env.assertEqual(after, {"hash": after["hash"], "labels": ["B"],
                                     "relationshipTypes": [], "propertyKeys": ["y"]})

        # S depends on the names alone: recreating the first schema gives its S back.
        con.delete(key)
        con.execute_command("GRAPH.QUERY", key, "CREATE (:A {x: 2})")
        self.env.assertEqual(self._schema(con, key, before["hash"]),
                             {"hash": before["hash"]})

    def test08_schema_hash_replica_copy_and_reload(self):
        key = "schema_replica"
        self._build_golden(key)
        self.master.wait(1, 0)

        # Replayed from effects on the replica.
        self.env.assertEqual(self._schema(self.replica, key, 0), self.GOLDEN_MAP)

        # Loaded from a dump by GRAPH.COPY.
        self.master.execute_command("GRAPH.COPY", key, "schema_replica_copy")
        self.env.assertEqual(self._schema(self.master, "schema_replica_copy", 0),
                             self.GOLDEN_MAP)

        # Rebuilt by an RDB load.
        self.master.execute_command("DEBUG", "RELOAD")
        self.env.assertEqual(self._schema(self.master, key, 0), self.GOLDEN_MAP)
        self.env.assertEqual(self._schema(self.master, "schema_replica_copy", 0),
                             self.GOLDEN_MAP)
