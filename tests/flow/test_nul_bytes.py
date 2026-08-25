import time
from common import *

# A NUL byte is an ordinary byte in a Redis key, in a Cypher string literal and in
# a query parameter — nothing in the protocol or the language reserves it. It is
# also the one byte `CString::new` rejects, so every place the module hands a
# user-controlled string to a C API that wants a NUL-terminated string used to be a
# way for any client to abort the server with one well-formed request (#2490).
#
# These tests drive a NUL through those paths and assert the server is still there
# afterwards, and that the byte survived the round trip rather than truncating the
# value or mangling the reply.
NUL   = "a\x00b"
CRLF  = "a\r\nb"

def stream_name(graph):
    return f"telemetry{{{graph}}}"

def poll_until(f, description, timeout=30, interval=0.01):
    """Call `f` until it returns a truthy value; raise once `timeout` elapses.

       Bounded rather than `while True`: an unmet condition should fail the test
       with this description, not hang the run."""
    deadline = time.monotonic() + timeout
    while True:
        res = f()
        if res:
            return res
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {description}")
        time.sleep(interval)


class testNulBytes(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def tearDown(self):
        # Leave no graph behind: `test_list_graphs` asserts on the whole keyspace.
        for key in self.conn.execute_command("GRAPH.LIST"):
            self.conn.delete(key)

    def test_query_text_with_nul(self):
        """The reported repro: a NUL in a parameter, and in a plain string literal.

           Both used to return normally and kill the process about a second later,
           on the telemetry flusher thread, which made the crash unattributable to
           the query that caused it. The entry has to reach the stream intact — a
           truncated or dropped one would pass a liveness-only check."""
        graph = "nul_query"
        for query in [f'CYPHER `p`="{NUL}" RETURN $p', f'RETURN "{NUL}"']:
            res = self.conn.execute_command("GRAPH.QUERY", graph, query)
            self.env.assertEqual(res[1], [[NUL]])

        # The flusher writes within ~10ms; the crash it used to take was ~1s later.
        stream = stream_name(graph)
        entries = poll_until(lambda: self.conn.xrange(stream)
                             if self.conn.type(stream) == "stream"
                             and self.conn.xlen(stream) == 2 else None,
                             "both queries to reach the telemetry stream")
        # The parameter header is logged in the parameters field, not in the query
        # text, so the NUL shows up in a different field for each of the two.
        self.env.assertEqual(entries[0][1]["Query"], "RETURN $p")
        self.env.assertContains(NUL, entries[0][1]["Query parameters"])
        self.env.assertEqual(entries[1][1]["Query"], f'RETURN "{NUL}"')
        self.env.assertTrue(self.conn.ping())

    def test_delete_graph_keyed_with_nul(self):
        """GRAPH.DELETE builds its graph's telemetry stream name from the key."""
        self.conn.execute_command("GRAPH.QUERY", NUL, "RETURN 1")
        self.conn.execute_command("GRAPH.DELETE", NUL)
        self.env.assertTrue(self.conn.ping())
        self.env.assertEqual(self.conn.type(NUL), "none")
        self.env.assertEqual(self.conn.type(stream_name(NUL)), "none")

    def test_rename_graph_keyed_with_nul(self):
        """RENAME reaches the same stream-name path through the keyspace event, on
           the main thread, where the graph also has to survive re-keying."""
        after = "c\x00d"
        self.conn.execute_command("GRAPH.QUERY", NUL, "CREATE (:L {v: 1})")
        self.conn.rename(NUL, after)
        self.env.assertTrue(self.conn.ping())

        res = self.conn.execute_command("GRAPH.QUERY", after, "MATCH (n:L) RETURN n.v")
        self.env.assertEqual(res[1], [[1]])

    def test_list_graphs(self):
        """GRAPH.LIST replied with each key name as a RESP status, which aborted on
           a NUL and silently rewrote CR/LF to spaces. Bulk strings carry both,
           which is what the C engine replies with."""
        for name in [NUL, CRLF, "plain"]:
            self.conn.execute_command("GRAPH.QUERY", name, "RETURN 1")

        graphs = self.conn.execute_command("GRAPH.LIST")
        self.env.assertTrue(self.conn.ping())
        self.env.assertEqual(sorted(graphs), sorted([NUL, CRLF, "plain"]))

    def test_memory_usage_with_nul_in_schema(self):
        """GRAPH.MEMORY reports per-label and per-relationship-type sizes, keyed by
           the names the query supplied — which a query is free to write a NUL
           into, and which used to be replied as a RESP status."""
        graph = "nul_schema"
        self.conn.execute_command("GRAPH.QUERY", graph,
                                  f'CREATE (:`{NUL}` {{v: 1}})-[:`{NUL}` {{v: 1}}]->(:L)')

        res = self.conn.execute_command("GRAPH.MEMORY", "USAGE", graph)
        self.env.assertTrue(self.conn.ping())

        reply = dict(zip(res[::2], res[1::2]))
        by_label = reply["amortized_node_attributes_by_label_sz_mb"]
        by_type  = reply["amortized_edge_attributes_by_type_sz_mb"]
        self.env.assertContains(NUL, by_label[::2])
        self.env.assertContains(NUL, by_type[::2])
