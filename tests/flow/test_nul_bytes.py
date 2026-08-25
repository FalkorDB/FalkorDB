from common import *

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
