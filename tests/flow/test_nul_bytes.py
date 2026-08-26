import time

from common import Env, FlowTestsBase, ResponseError

# A NUL byte is an ordinary byte in a Redis key and in a query string, and it used to
# be a way for any client to abort the server: it travelled into a `CString::new(..)`
# on the telemetry flusher, in GRAPH.DELETE, in a RENAME keyspace event, in GRAPH.LIST
# and in GRAPH.MEMORY (#2490).
#
# The engine now handles it the way C does — a name and a query both end at their
# first NUL, because C reads them with the length discarded and treats what it gets as
# a C string. These tests pin that behaviour against what the C engine actually
# answers, case by case, and assert the server is still there afterwards.
NUL  = "a\x00b"
CRLF = "a\r\nb"


class testNulBytesInQueries(FlowTestsBase):
    """A query ends at its first NUL, exactly as libcypher-parser sees it in C."""

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def tearDown(self):
        for key in self.conn.execute_command("GRAPH.LIST"):
            self.conn.delete(key)

    def test_trailing_bytes_are_dropped(self):
        # C: succeeds, returning 1 — the bytes after the NUL are not part of the query.
        # This is a truncation, not a rejection, so anything that parses still runs.
        res = self.conn.execute_command("GRAPH.QUERY", "g", "RETURN 1\x00 GARBAGE")
        self.env.assertEqual(res[1], [[1]])
        res = self.conn.execute_command("GRAPH.QUERY", "g", "RETURN 1\x00")
        self.env.assertEqual(res[1], [[1]])

    def test_nul_inside_a_literal_fails_to_parse(self):
        # C: `Invalid input at end of input: expected "` — it parsed `RETURN "a`.
        # The wording is our parser's, the outcome is C's: the string is unterminated.
        try:
            self.conn.execute_command("GRAPH.QUERY", "g", f'RETURN "{NUL}"')
            self.env.assertTrue(False)  # unreachable: the query must not parse
        except ResponseError as e:
            self.env.assertContains("Unterminated string", str(e))

    def test_nul_in_a_parameter_fails_to_parse(self):
        # The originally reported repro. C: `Failed to parse query parameter 'p' value`.
        try:
            self.conn.execute_command("GRAPH.QUERY", "g", f'CYPHER `p`="{NUL}" RETURN $p')
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("parameter 'p'", str(e))

    def test_nul_in_a_label_fails_to_parse(self):
        # Keeps a NUL out of a schema name, which is what GRAPH.MEMORY replies.
        try:
            self.conn.execute_command("GRAPH.QUERY", "g", f'CREATE (:`{NUL}`)')
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("expected an identifier", str(e))

    def test_leading_nul_is_the_empty_query(self):
        # C: `Error: empty query.` — nothing precedes the NUL.
        try:
            self.conn.execute_command("GRAPH.QUERY", "g", "\x00RETURN 1")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("empty query", str(e))

    def test_memory_frames_a_label_holding_cr_lf(self):
        # C replies each schema name with `RM_ReplyWithCString`, which is a *bulk*
        # reply. A label name can hold CR/LF even though it can no longer hold a NUL,
        # and replying it as a RESP status left those bytes unescaped — the GRAPH.LIST
        # problem again, one command over.
        injected = "L\r\n+PWNED"
        self.conn.execute_command("GRAPH.QUERY", "g", f"CREATE (:`{injected}` {{a: 1}})")
        flat = []
        def walk(v):
            for x in v:
                walk(x) if isinstance(x, list) else flat.append(x)
        walk(self.conn.execute_command("GRAPH.MEMORY", "USAGE", "g"))
        self.env.assertContains(injected, flat)
        self.env.assertTrue(self.conn.ping())

    def test_the_server_survives_all_of_it(self):
        # The crash was delayed and off-thread (the telemetry flusher), so liveness is
        # asserted after the queries above have been flushed, not just after they reply.
        self.env.assertTrue(self.conn.ping())


class testNulBytesInGraphKeys(FlowTestsBase):
    """A graph is named by its key up to the first NUL, and C stores a newly created
       graph under that *name* while looking one up by the full key it was given. That
       asymmetry is C's, reproduced here so both engines answer alike."""

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def tearDown(self):
        self.conn.flushall()

    def graph_keys(self):
        """Every key in the keyspace except the telemetry streams, which appear
           asynchronously once the flusher runs and would race these assertions."""
        return sorted(k for k in self.conn.execute_command("KEYS", "*")
                      if not k.startswith("telemetry{"))

    def test_created_under_the_truncated_key(self):
        # C: the node lands in graph `a`; the addressed key does not exist at all.
        self.conn.execute_command("GRAPH.QUERY", NUL, "CREATE (:L)")
        self.env.assertEqual(self.graph_keys(), ["a"])
        self.env.assertEqual(self.conn.type(NUL), "none")

        res = self.conn.execute_command("GRAPH.QUERY", "a", "MATCH (n) RETURN count(n)")
        self.env.assertEqual(res[1], [[1]])

    def test_a_second_write_replaces_the_first(self):
        # C: each write on the NUL key finds the addressed key empty, creates a fresh
        # graph, and stores it over the previous one — so the count stays 1, it does
        # not accumulate. Faithfully destructive.
        for _ in range(2):
            self.conn.execute_command("GRAPH.QUERY", NUL, "CREATE (:L)")
        res = self.conn.execute_command("GRAPH.QUERY", "a", "MATCH (n) RETURN count(n)")
        self.env.assertEqual(res[1], [[1]])
        self.env.assertEqual(self.graph_keys(), ["a"])

    def test_read_only_commands_see_an_empty_key(self):
        # C: the lookup uses the full key bytes, which hold nothing, so every command
        # that will not create a graph reports the empty key — including GRAPH.DELETE,
        # which is the crash from the issue.
        self.conn.execute_command("GRAPH.QUERY", "a", "CREATE (:L)")
        for cmd in [("GRAPH.RO_QUERY", NUL, "MATCH (n) RETURN n"),
                    ("GRAPH.EXPLAIN", NUL, "MATCH (n) RETURN n"),
                    ("GRAPH.SLOWLOG", NUL),
                    ("GRAPH.DELETE", NUL)]:
            try:
                self.conn.execute_command(*cmd)
                self.env.assertTrue(False)  # unreachable
            except ResponseError as e:
                self.env.assertContains("empty key", str(e))
        self.env.assertTrue(self.conn.ping())

    def test_bulk_insert_creates_under_the_truncated_key(self):
        self.conn.execute_command("GRAPH.BULK", NUL, "BEGIN", 0, 0, 0, 0)
        self.env.assertEqual(self.graph_keys(), ["a"])

    def test_empty_name_when_the_key_starts_with_nul(self):
        # C: a key of `\0abc` names the graph "", and that is the key it is stored at.
        self.conn.execute_command("GRAPH.QUERY", "\x00abc", "CREATE (:L)")
        self.env.assertEqual(self.graph_keys(), [""])

    def test_list_reports_names_never_raw_key_bytes(self):
        # C replies `gc->graph_name` per graph, so a NUL never reaches the reply. A
        # RENAME is the one way a graphdata key itself can still hold a NUL: Redis
        # renames the key by its full bytes while the graph keeps a truncated name.
        self.conn.execute_command("GRAPH.QUERY", "src", "CREATE (:L)")
        self.conn.rename("src", "c\x00d")
        self.env.assertEqual(self.conn.execute_command("GRAPH.LIST"), ["c"])
        self.env.assertTrue(self.conn.ping())

        # The graph is still reachable under the key Redis actually holds, and deleting
        # it — which builds the telemetry stream name from the graph's name — is the
        # path that used to abort the process.
        res = self.conn.execute_command("GRAPH.QUERY", "c\x00d", "MATCH (n) RETURN count(n)")
        self.env.assertEqual(res[1], [[1]])
        self.conn.execute_command("GRAPH.DELETE", "c\x00d")
        self.env.assertTrue(self.conn.ping())

    def test_cr_lf_in_a_key_is_left_alone(self):
        # Not a NUL: CR and LF end nothing, so the name keeps them, as C's does.
        self.conn.execute_command("GRAPH.QUERY", CRLF, "RETURN 1")
        self.env.assertEqual(self.graph_keys(), [CRLF])
        self.env.assertTrue(self.conn.ping())

    def test_list_frames_a_name_holding_cr_lf(self):
        # C replies each name with `RM_ReplyWithStringBuffer`, which is length-framed.
        # `up_to_nul` says nothing about CR/LF, so replying the name as a RESP *status*
        # instead left these bytes unescaped in the reply: a client could inject
        # `+PWNED` into the *next* reply on the connection, and the name itself came
        # back mangled on a Redis that sanitises a status instead.
        injected = "g\r\n+PWNED"
        self.conn.execute_command("GRAPH.QUERY", injected, "RETURN 1")
        self.env.assertEqual(self.conn.execute_command("GRAPH.LIST"), [injected])
        # A desync shows up here: `ping()` compares the reply against PONG, so it is
        # False if this connection read the tail of the previous reply instead.
        self.env.assertTrue(self.conn.ping())

    def test_bulk_insert_failure_on_a_nul_key(self):
        # The `BEGIN` cleanup path re-opens the key the batch created, and built that
        # key with `Context::create_string` — `CString::new(..).unwrap()`. One malformed
        # batch on a NUL key was enough to abort the server, the same class as #2490.
        try:
            self.conn.execute_command("GRAPH.BULK", NUL, "BEGIN", 0, 0, 1, 0,
                                      b"L\x00\x00\x00\x00\x00X")
            self.env.assertTrue(False)  # unreachable: the batch must be rejected
        except ResponseError as e:
            self.env.assertContains("bulk insert failed", str(e))
        self.env.assertTrue(self.conn.ping())
        # and a failed first batch leaves no key behind
        self.env.assertEqual(self.graph_keys(), [])


class testNulBytesInGraphCopy(FlowTestsBase):
    """`GRAPH.COPY` and `GRAPH.RESTORE` address their destination by its full key bytes,
       for the existence check as much as for the write. C opens `rm_dest` (`argv[1]`)
       both times and never rebuilds the key from the name, so the truncation C applies
       when a *query* creates a graph has no place here — and guarding one key while
       writing another is how `GRAPH.COPY src "victim\0bypass"` came to pass a check on
       an empty key and then overwrite the live `victim`."""

    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

    def tearDown(self):
        self.conn.flushall()

    def node_count(self, key):
        res = self.conn.execute_command("GRAPH.QUERY", key, "MATCH (n) RETURN count(n)")
        return res[1][0][0]

    def graph_copy(self, src, dest):
        # GRAPH.COPY forks; a fork failure is environmental, so retry it briefly the way
        # test_graph_copy does rather than reporting it as this behaviour breaking.
        deadline = time.time() + 60
        while True:
            try:
                self.conn.execute_command("GRAPH.COPY", src, dest)
                return
            except ResponseError as e:
                if "could not fork" in str(e).lower() and time.time() + 5 <= deadline:
                    time.sleep(5)
                else:
                    raise

    def test_copy_does_not_overwrite_the_truncated_destination(self):
        self.conn.execute_command("GRAPH.QUERY", "victim",
                                  "UNWIND range(1, 5) AS x CREATE (:V)")
        self.conn.execute_command("GRAPH.QUERY", "csrc",
                                  "UNWIND range(1, 2) AS x CREATE (:S)")
        self.graph_copy("csrc", "victim\x00bypass")

        # The copy landed at the key that was named, and `victim` is untouched.
        self.env.assertEqual(self.node_count("victim\x00bypass"), 2)
        self.env.assertEqual(self.node_count("victim"), 5)
        self.env.assertTrue(self.conn.ping())

    def test_copy_refuses_a_destination_holding_a_nul(self):
        # The guard is on the same full key bytes, so an occupied destination is still
        # refused. A RENAME is the only way to get a graph to sit at such a key.
        self.conn.execute_command("GRAPH.QUERY", "tmp", "CREATE (:T)")
        self.conn.rename("tmp", "dst\x00x")
        self.conn.execute_command("GRAPH.QUERY", "csrc", "CREATE (:S)")
        try:
            self.graph_copy("csrc", "dst\x00x")
            self.env.assertTrue(False)  # unreachable: the destination is occupied
        except ResponseError as e:
            self.env.assertContains("already exists", str(e))
        self.env.assertEqual(self.node_count("dst\x00x"), 1)

    def test_a_copied_graph_is_named_after_its_key_up_to_the_nul(self):
        # The graph's own name is truncated even though its key is not: C renames the
        # clone to a C string in the forked child (`GraphContext_Rename(gc, dest)`), so
        # the truncated name is what the dump header carries — and what GRAPH.LIST,
        # which replies names, reports.
        self.conn.execute_command("GRAPH.QUERY", "csrc", "CREATE (:S)")
        self.graph_copy("csrc", "named\x00tail")
        self.env.assertEqual(sorted(self.conn.execute_command("GRAPH.LIST")),
                             ["csrc", "named"])

    def test_a_nul_key_survives_debug_reload(self):
        self.conn.execute_command("GRAPH.QUERY", "src",
                                  "UNWIND range(1, 3) AS x CREATE (:L)")
        self.conn.rename("src", "c\x00d")
        self.conn.execute_command("DEBUG", "RELOAD")
        self.env.assertEqual(self.node_count("c\x00d"), 3)
        self.env.assertEqual(self.conn.execute_command("GRAPH.LIST"), ["c"])
        self.env.assertTrue(self.conn.ping())


class testNulBytesReplication(FlowTestsBase):
    """A write is replicated against the key it landed on, not against the graph's name.
       Those differ once a key holds a NUL, and `GRAPH.EFFECT` *creates* a graph when the
       key is missing — so replicating against the name grew a phantom graph on the
       replica beside the one the master has, and neither side ever reconverged."""

    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True)
        self.master = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        # Block until the replica has attached, or the WAIT in the test below has
        # nothing to wait for and reports a sync that never happened.
        self.master.wait(1, 0)

    def __del__(self):
        self.replica.shutdown()

    def node_count(self, conn, key):
        # RO_QUERY on both sides: a replica refuses GRAPH.QUERY as a write.
        res = conn.execute_command("GRAPH.RO_QUERY", key, "MATCH (n) RETURN count(n)")
        return res[1][0][0]

    def test_a_write_to_a_nul_key_replicates_to_that_key(self):
        self.master.execute_command("GRAPH.QUERY", "src", "CREATE (:L)")
        self.master.rename("src", "c\x00d")
        self.master.execute_command("GRAPH.QUERY", "c\x00d", "CREATE (:L)")
        self.env.assertEqual(self.master.wait(1, 5000), 1)

        # Same key, same graph, and no second graph invented alongside it.
        self.env.assertEqual(self.node_count(self.replica, "c\x00d"), 2)
        self.env.assertEqual(self.node_count(self.master, "c\x00d"), 2)
        self.env.assertEqual(sorted(k for k in self.replica.execute_command("KEYS", "*")
                                    if not k.startswith("telemetry{")),
                             ["c\x00d"])
