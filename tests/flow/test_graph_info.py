import time
import queue
import string
import random
import datetime
import threading
import multiprocessing
from common import *

GRAPH_ID = "info"

class LoggedQuery:
    def __init__(self, event):
        # make sure event contains all expected fields
        fields = ["Received at", "Query", "Query parameters", "Total duration", "Wait duration",
                  "Execution duration", "Report duration", "Utilized cache",
                  "Write", "Timeout"]
        assert(all(field in event for field in fields))

        # cast and initialize
        self.received_at        = datetime.datetime.fromtimestamp(int(event['Received at']))
        self.query              = event['Query']
        self.params             = event['Query parameters']
        self.total_duration     = float(event['Total duration'])
        self.wait_duration      = float(event['Wait duration'])
        self.execution_duration = float(event['Execution duration'])
        self.report_duration    = float(event['Report duration'])
        self.utilized_cache     = False if event['Utilized cache'] == '0' else True

        assert (self.TotalDuration >= (self.ExecutionDuration + self.ReportDuration))

    def __str__(self):
        return f"""ReceivedAt: {self.ReceivedAt}
                 Query: {self.Query}
                 Parameters: {self.params}
                 TotalDuration: {self.TotalDuration}
                 WaitDuration: {self.WaitDuration}
                 ExecutionDuration: {self.ExecutionDuration}
                 ReportDuration: {self.ReportDuration}
                 UtilizedCache: {self.UtilizedCache}"""

    @property
    def ReceivedAt(self):
        return self.received_at

    @property
    def Query(self):
        return self.query

    @property
    def Parameters(self):
        return self.params

    @property
    def TotalDuration(self):
        return self.total_duration

    @property
    def WaitDuration(self):
        return self.wait_duration

    @property
    def ExecutionDuration(self):
        return self.execution_duration

    @property
    def ReportDuration(self):
        return self.report_duration

    @property
    def UtilizedCache(self):
        return self.utilized_cache

def StreamName(graph):
    return f"telemetry{{{graph.name}}}"

def pollUntil(f, description, timeout=30, interval=0.01):
    """Call `f` until it returns a truthy value, then return that value.
       Raises AssertionError once `timeout` seconds have elapsed.

       Bounded on purpose: an unbounded `while True` poll turns a condition that
       is never met into a hang — the run dies minutes later with RLTest's
       "Failed to get job result" instead of reporting a failed test. Callers
       must also keep env.assert* out of `f`: every assertion is recorded, and a
       poll running at thousands of iterations per second grows the result set
       without bound."""

    deadline = time.monotonic() + timeout
    while True:
        res = f()
        if res:
            return res
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {description}")
        time.sleep(interval)

def consumeStream(conn, env, stream, drop=True, n_items=1, timeout=30):
    # wait for telemetry stream to be created
    t = 'none' # type of stream_key
    deadline = time.time() + timeout

    while t == 'none' and time.time() < deadline:
        time.sleep(0.2)
        t = conn.type(stream)

    env.assertEqual(t, "stream")
    if t != "stream":
        return []

    # convert stream events to LoggedQueries
    logged_queries = []
    streams = {stream: '0-0'}

    # a finite block, bounded by the deadline, so a missing entry fails the
    # test promptly instead of hanging on XREAD BLOCK 0 forever
    while len(logged_queries) < n_items and time.time() < deadline:
        messages = conn.xread(streams, block=200)

        if len(messages) > 0:
            # process each message received
            stream_messages = messages[0][1]
            for message_id, message_payload in stream_messages:
                logged_queries.append(LoggedQuery(message_payload))

            # update stream last ID
            streams[stream] = stream_messages[-1][0]

    # drop stream
    if drop:
        conn.delete(stream)

    # reverse order to match expected order of events
    logged_queries.reverse()

    return logged_queries

def _wait_for(condition, timeout=10):
    """poll condition until it holds or timeout seconds pass"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return False

class testGraphInfo():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def assertLoggedQuery(self, logged_query, query, utilized_cache):
        # validate event values
        self.env.assertEqual(logged_query.Query, query)
        self.env.assertEqual(logged_query.UtilizedCache, utilized_cache)

    def test01_read_logged_queries(self):
        """issue a number of queries
           make sure they show up within the telemetry stream"""

        q0 = "RETURN 1"
        q1 = "CREATE ()"
        q2 = "MATCH (n) RETURN n"
        queries = [q0, q1, q2]

        # issue queries
        for q in queries:
            self.graph.query(q)

        # read stream
        logged_queries = consumeStream(self.conn, self.env, StreamName(self.graph), n_items=3)

        # validate events
        self.env.assertEqual(len(logged_queries), 3)
        utilized_cache = False # first time executing queies, no cache
        self.assertLoggedQuery(logged_queries[0], q2, utilized_cache)
        self.assertLoggedQuery(logged_queries[1], q1, utilized_cache)
        self.assertLoggedQuery(logged_queries[2], q0, utilized_cache)

        #-----------------------------------------------------------------------
        # re-issue queries
        #-----------------------------------------------------------------------

        for i in range(0, 2):
            for q in queries:
                self.graph.query(q)

        # read stream
        logged_queries = consumeStream(self.conn, self.env, StreamName(self.graph), n_items=6)

        # validate events
        self.env.assertEqual(len(logged_queries), 6)
        utilized_cache = True # second time executing queies
        self.assertLoggedQuery(logged_queries[0], q2, utilized_cache)
        self.assertLoggedQuery(logged_queries[1], q1, utilized_cache)
        self.assertLoggedQuery(logged_queries[2], q0, utilized_cache)
        self.assertLoggedQuery(logged_queries[3], q2, utilized_cache)
        self.assertLoggedQuery(logged_queries[4], q1, utilized_cache)
        self.assertLoggedQuery(logged_queries[5], q0, utilized_cache)

    def test02_capped_logged_queries(self):
        """make sure number of queries is capped"""

        q = "RETURN 1"

        # worker function, invoked by multiple threads
        def issue_query(g, q):
            for i in range(125):
                g.query(q)

        # create multiple connections
        connections = []
        for i in range(16):
            connections.append(self.env.getConnection())

        # create multiple threads
        threads = []
        for i in range(16):
            t = threading.Thread(target=issue_query, args=(Graph(connections[i], GRAPH_ID), q))
            threads.append(t)

        # issue threads
        for t in threads:
            t.start()

        # wait for all threads to complete
        for t in threads:
            t.join()

        # read stream
        logged_queries = consumeStream(self.conn, self.env, StreamName(self.graph))

        # make sure number of logged queries is capped
        self.env.assertLess(len(logged_queries), 1200)

    def test03_long_query(self):
        """long queries are truncated with: ..."""

        # create a long string
        length = 4000
        long_str = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

        long_query       = f"RETURN '{long_str}'"
        long_param       = "RETURN $i"
        long_param_query = f"RETURN $i, '{long_str}'"

        self.graph.query(long_query)
        self.graph.query(long_param, {'i': long_str})
        self.graph.query(long_param_query, {'i': long_str})

        # read stream
        logged_queries = consumeStream(self.conn, self.env, StreamName(self.graph), drop=False)

        # assert long query
        logged_query = logged_queries[2]
        self.env.assertLess(len(logged_query.Query), 4000)
        self.env.assertEqual(logged_query.Query[:2000], long_query[:2000])
        self.env.assertEqual(logged_query.Query[-3:], "...")

        # assert long param
        logged_query = logged_queries[1]
        self.env.assertLess(len(logged_query.Parameters), 4000)
        self.env.assertEqual(logged_query.Parameters[-3:], "...")

        # assert long param and query
        logged_query = logged_queries[0]
        self.env.assertLess(len(logged_query.Query), 4000)
        self.env.assertEqual(logged_query.Query[:2000], long_param_query[:2000])
        self.env.assertEqual(logged_query.Query[-3:], "...")
        self.env.assertLess(len(logged_query.Parameters), 4000)
        self.env.assertEqual(logged_query.Parameters[-3:], "...")

    def test04_delete_graph(self):
        """make sure reporting stream is deleted when graph is deleted"""

        # validate that stream exists
        stream_name = StreamName(self.graph)
        self.env.assertEqual(self.conn.type(stream_name), "stream")

        # make sure graph is deleted synchronously
        self.db.config_set("ASYNC_DELETE", "no")

        # delete graph
        self.graph.delete()

        # validate that stream was deleted
        self.env.assertEqual(self.conn.type(stream_name), "none")

        # restore ASYNC_DELETE
        self.db.config_set("ASYNC_DELETE", "yes")

    def test05_rename_graph(self):
        """make sure reporting stream is renamed when graph is renamed"""

        old_name = "old_graph"
        new_name = "new_graph"
        old_graph = Graph(self.conn, old_name)
        new_graph = Graph(self.conn, new_name)

        # issue query to create and populate stream
        old_graph.query("RETURN 1")

        # wait for stream to be created
        logged_queries = consumeStream(self.conn, self.env, StreamName(old_graph), drop=False)
        self.env.assertEqual(len(logged_queries), 1)

        # validate that stream exists
        self.env.assertEqual(self.conn.type(StreamName(old_graph)), "stream")

        # rename graph
        self.conn.rename(old_name, new_name)

        # issue query to create and populate stream
        new_graph.query("RETURN 1")

        # wait for stream to be created
        logged_queries = consumeStream(self.conn, self.env, StreamName(new_graph), drop=False)
        self.env.assertEqual(len(logged_queries), 1)

        # validate that stream was renamed
        self.env.assertEqual(self.conn.type(StreamName(old_graph)), "none")
        self.env.assertEqual(self.conn.type(StreamName(new_graph)), "stream")

    def test06_multiple_streams(self):
        """test a more realistic example for how logged-queries streams
        will be processed"""

        # shared variable, single consumer thread to exit
        alive = True

        # streams consumer thread
        def consume_streams(conn, queue):
            # continuously poll for new messages
            streams = {'telemetry{g}': '0-0', 'telemetry{x}': '0-0'}

            # as long as we're alive
            while alive:
                # read messages from the stream
                messages = conn.xread(streams, block=0)

                # process each message received
                for stream, stream_messages in messages:
                    for message_id, message_payload in stream_messages:
                        queue.put((stream, LoggedQuery(message_payload)))

                    if messages:
                        # update stream last ID
                        streams[stream] = stream_messages[-1][0]

        # create two graphs: 'g' and 'x'
        g = Graph(self.conn, "g")
        x = Graph(self.conn, "x")

        # create threads communication queue
        q = queue.Queue()

        # start streams consumer thread
        t = threading.Thread(target=consume_streams, args=(self.conn, q))
        t.start()

        # issue queries multiple times against graphs 'g' and 'x'
        for i in range (2):
            # issue queries
            g.query("RETURN 1")
            x.query("RETURN 1")

            # read logged queries
            logged_query = q.get()
            logged_query = q.get()

        # signal consumer thread to stop
        alive = False

        # issue another query to unblock consumer thread
        g.query("RETURN 1")

        # wait for stream consumer thread to exit
        t.join()

    def test07_current_queries(self):
        """test currently running queries"""

        # clear graph
        self.conn.delete(GRAPH_ID)

        # shared variable, single consumer thread to exit
        alive = True

        # issue a number of threads all running the same query
        def issue_query(g, q):
            while alive:
                g.query(q)

        def issue_2_query(g, q1, q2):
            while alive:
                g.query(q1)
                time.sleep(1)
                g.query(q2)

        num_threads = multiprocessing.cpu_count() * 2

        # create multiple connections
        connections = []
        for i in range(num_threads+1):
            connections.append(self.env.getConnection())

        read_query = "MATCH (n) WHERE n.v > 100 RETURN count(1)"
        write_query1 = "UNWIND range(1, 10000) AS x CREATE (v: x)"
        write_query2 = "MATCH (n) DELETE n"
        # create multiple threads
        threads = []
        for i in range(num_threads):
            # read queries
            t = threading.Thread(target=issue_query, args=(Graph(connections[i], GRAPH_ID), read_query))
            threads.append(t)

        # write query
        t = threading.Thread(target=issue_2_query, args=(Graph(connections[-1], GRAPH_ID), write_query1, write_query2))
        threads.append(t)

        # both sections are global: they report every graph's queries, so pick
        # out the entries belonging to this test's graph
        def queriesForGraph(section):
            return [q for q in section if q[3] == GRAPH_ID]

        # issue threads
        for t in threads:
            t.start()

        try:
            # wait for graph to be created
            pollUntil(lambda: self.conn.type(GRAPH_ID) == "graphdata",
                      "graph to be created")

            def readInfo():
                res = self.conn.execute_command("GRAPH.INFO", "RunningQueries",
                                                "WaitingQueries")
                # plain asserts, not env.assert*: this runs inside a poll loop
                assert len(res) == 4
                assert res[0] == "# Running queries"
                assert res[2] == "# Waiting queries"
                return res

            # record the response structure once, outside the poll loops
            res = readInfo()
            self.env.assertEqual(len(res), 4)
            self.env.assertEqual(res[0], "# Running queries")
            self.env.assertEqual(res[2], "# Waiting queries")

            #-------------------------------------------------------------------
            # validate running queries
            #-------------------------------------------------------------------

            running_queries = pollUntil(lambda: queriesForGraph(readInfo()[1]),
                                        f"a running query on '{GRAPH_ID}'")
            running_query = running_queries[0]
            self.env.assertEqual(running_query[0], "Received at")
            self.env.assertEqual(running_query[2], "Graph name")
            self.env.assertEqual(running_query[4], "Query")
            self.env.assertEqual(running_query[6], "Execution duration")
            self.env.assertEqual(running_query[8], "Replicated command")

            self.env.assertEqual(running_query[3], GRAPH_ID)
            self.env.assertTrue(running_query[5] == read_query or
                                running_query[5] == write_query1 or
                                running_query[5] == write_query2)
            self.env.assertEqual(running_query[9], False)

            #-------------------------------------------------------------------
            # validate waiting queries
            #
            # more client threads than thread-pool workers, so queries queue up
            # waiting for a worker to pick them up
            #-------------------------------------------------------------------

            waiting_queries = pollUntil(lambda: queriesForGraph(readInfo()[3]),
                                        f"a waiting query on '{GRAPH_ID}'")
            waiting_query = waiting_queries[0]
            self.env.assertEqual(waiting_query[0], "Received at")
            self.env.assertEqual(waiting_query[2], "Graph name")
            self.env.assertEqual(waiting_query[4], "Query")
            self.env.assertEqual(waiting_query[6], "Wait duration")

            self.env.assertEqual(waiting_query[3], GRAPH_ID)
            self.env.assertTrue(waiting_query[5] == read_query or
                                waiting_query[5] == write_query1 or
                                waiting_query[5] == write_query2)
        finally:
            # signal worker threads to stop, on every path out: a worker left
            # running would keep querying forever and hang the rest of the run
            alive = False

            # wait for all threads to complete
            for t in threads:
                t.join()

    def test08_no_sections_reports_everything(self):
        """a bare GRAPH.INFO reports every section, as the C engine does"""

        def shape(res):
            """The labels and the shape of each section, but not the counters.

               Comparing two replies field for field would compare the object-pool
               counters too, and those move on their own: a graph displaced by
               `register_graph` or freed by `graph_free` is dropped on a background
               thread, and the interned strings it releases change `Unique Objects in
               Pool` between one call and the next."""
            return [res[0], res[2], res[4],
                    len(res), type(res[1]), type(res[3]),
                    [entry[0] for entry in res[5]]]

        res = self.conn.execute_command("GRAPH.INFO")

        # same sections, in the same order, as asking for all of them
        self.env.assertEqual(shape(res),
                             shape(self.conn.execute_command("GRAPH.INFO",
                                                             "RunningQueries",
                                                             "WaitingQueries",
                                                             "ObjectPool")))

        self.env.assertEqual(len(res), 6)
        self.env.assertEqual(res[0], "# Running queries")
        self.env.assertEqual(res[2], "# Waiting queries")
        self.env.assertEqual(res[4], "Object Pool")

        # an unknown section is ignored, and the recognised one still answers —
        # C's `_handle_sections` matches what it knows and skips the rest
        res = self.conn.execute_command("GRAPH.INFO", "RunningQueries", "NoSuchSection")
        self.env.assertEqual(len(res), 2)
        self.env.assertEqual(res[0], "# Running queries")

        # nothing recognised at all: C replies with a string, not an error
        self.env.assertEqual(
            self.conn.execute_command("GRAPH.INFO", "NoSuchSection"), "no section found")

    def test09_cmd_info_runtime_toggle(self):
        """CMD_INFO is settable at run-time, and gates query logging"""

        stream = StreamName(self.graph)
        self.conn.delete(stream)

        # on by default: a query shows up in the telemetry stream
        self.graph.query("RETURN 1")
        pollUntil(lambda: self.conn.xlen(stream), "a logged query")

        #-----------------------------------------------------------------------
        # turn logging off
        #-----------------------------------------------------------------------

        self.env.assertEqual(
            self.conn.execute_command("GRAPH.CONFIG", "SET", "CMD_INFO", "no"), "OK")
        self.env.assertEqual(self.db.config_get("CMD_INFO"), 0)

        # Take the baseline only once the stream has stopped growing. With logging off
        # nothing new can be enqueued, so the stream reaches a fixed point — but an
        # entry enqueued while it was still on (this class runs queries from threads
        # two tests earlier) can still be in flight, and reading `n` before it lands
        # blames the arrival on the config that was already off. Observed as a rare
        # `2 == 1` here under a parallel run.
        def settled():
            before = self.conn.xlen(stream)
            time.sleep(0.1)
            return before == self.conn.xlen(stream)

        pollUntil(settled, "the telemetry stream to stop growing")

        n = self.conn.xlen(stream)
        self.graph.query("RETURN 2")
        # the flusher wakes every 5ms; half a second is well past the point
        # where an entry would have landed had logging still been on
        time.sleep(0.5)
        self.env.assertEqual(self.conn.xlen(stream), n)

        #-----------------------------------------------------------------------
        # and back on
        #-----------------------------------------------------------------------

        self.env.assertEqual(
            self.conn.execute_command("GRAPH.CONFIG", "SET", "CMD_INFO", "yes"), "OK")
        self.env.assertEqual(self.db.config_get("CMD_INFO"), 1)

        self.graph.query("RETURN 3")
        pollUntil(lambda: self.conn.xlen(stream) > n, "query logging to resume")

        # a non-boolean value is rejected, leaving the config untouched
        try:
            self.conn.execute_command("GRAPH.CONFIG", "SET", "CMD_INFO", "maybe")
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("Failed to set config value CMD_INFO to maybe", str(e))
        self.env.assertEqual(self.db.config_get("CMD_INFO"), 1)

    def test10_max_info_queries_zero_keeps_nothing(self):
        """MAX_INFO_QUERIES 0 bounds the stream, as C's unconditional trim does.

           0 is accepted by both `GRAPH.CONFIG SET` and the module argument, and C
           passes it straight to `RedisModule_StreamTrimByLength`. Skipping the trim
           for 0 turned the one value that means "keep nothing" into "keep
           everything": 2000 queries left all 2000 entries in the stream, with no
           setting left that could bound it."""

        stream = StreamName(self.graph)
        self.conn.delete(stream)

        # restored rather than reset to the default, so this test cannot quietly
        # change the cap for whatever runs after it
        previous = self.db.config_get("MAX_INFO_QUERIES")

        self.env.assertEqual(
            self.conn.execute_command("GRAPH.CONFIG", "SET", "MAX_INFO_QUERIES", "0"), "OK")
        self.env.assertEqual(self.db.config_get("MAX_INFO_QUERIES"), 0)

        try:
            for _ in range(500):
                self.graph.query("RETURN 1")
            time.sleep(0.5)   # well past the flusher's window

            # approximate trimming works in whole listpack nodes, so some of the last
            # entries can survive; what must not survive is all 500 of them
            xlen = self.conn.xlen(stream) if self.conn.exists(stream) else 0
            self.env.assertLess(xlen, 200)
        finally:
            self.conn.execute_command("GRAPH.CONFIG", "SET", "MAX_INFO_QUERIES", str(previous))
            self.conn.delete(stream)

    def test11_telemetry_stripped_on_reload(self):
        """telemetry is node-local and is kept out of snapshots: a synchronous
           save (DEBUG RELOAD) drops the stream, so the divergent consumer-group
           cursor Redis replicates for it can never corrupt an RDB"""

        stream = StreamName(self.graph)

        # produce a telemetry entry
        self.graph.query("RETURN 'reload'")
        logged_queries = consumeStream(self.conn, self.env, stream, drop=False,
                n_items=1)
        self.env.assertEqual([q.Query for q in logged_queries], ["RETURN 'reload'"])

        # DEBUG RELOAD saves and loads the RDB in the main process; the stream
        # is stripped at save, so it is gone afterwards and the reload succeeds
        self.env.assertEqual(self.conn.execute_command("DEBUG", "RELOAD"), "OK")
        self.env.assertTrue(self.conn.ping())
        self.env.assertFalse(self.conn.exists(stream))

    def test12_bgsave_keeps_live_telemetry_but_strips_rdb(self):
        """a background save strips telemetry in the fork child only: the live
           server keeps its stream, but the RDB the save produced holds none of
           it"""

        stream = StreamName(self.graph)

        # produce a telemetry entry
        self.graph.query("RETURN 'bgsave'")
        self.env.assertTrue(_wait_for(lambda: self.conn.exists(stream)))

        # BGSAVE forks a child that strips its copy of the stream; the live
        # server is a separate process and keeps the stream intact
        self.conn.execute_command("BGSAVE")
        self.env.assertTrue(_wait_for(lambda:
            self.conn.info("persistence")["rdb_bgsave_in_progress"] == 0))
        self.env.assertEqual(
            self.conn.info("persistence")["rdb_last_bgsave_status"], "ok")
        self.env.assertTrue(self.conn.exists(stream))

        # load the RDB the background save produced (NOSAVE, no re-save): it
        # holds no telemetry, so the stream is absent after loading it
        self.env.assertEqual(
            self.conn.execute_command("DEBUG", "RELOAD", "NOSAVE"), "OK")
        self.env.assertFalse(self.conn.exists(stream))

    def test13_removing_the_key_removes_the_stream(self):
        """a graph takes its telemetry stream with it however its key goes: the
           persistence strip finds streams through the graphs, so a stream that
           outlived its graph would be saved

           GRAPH.DELETE deletes the stream itself; the others free the graph from
           inside the server's own key removal, where a second key cannot be
           deleted, so the stream follows within a flush interval"""

        removals = {
            "GRAPH.DELETE": lambda key: self.conn.execute_command("GRAPH.DELETE", key),
            "DEL":          lambda key: self.conn.delete(key),
            "UNLINK":       lambda key: self.conn.unlink(key),
            "overwrite":    lambda key: self.conn.set(key, "not a graph"),
        }

        for how, remove in removals.items():
            key    = f"removed_by_{how}"
            g      = self.db.select_graph(key)
            stream = StreamName(g)

            g.query("CREATE ()")
            self.env.assertTrue(_wait_for(lambda: self.conn.exists(stream)),
                message=f"no telemetry stream to remove by {how}")

            remove(key)
            self.env.assertTrue(_wait_for(lambda: not self.conn.exists(stream)),
                message=f"the telemetry stream outlived its graph, removed by {how}")
            self.conn.delete(key)

class testGraphInfoStaleEntry():
    """A queued telemetry entry belongs to the graph that produced it, not to that
       graph's name.

       Entries are written by a background thread a few milliseconds later, so a
       name can be flushed and rebound to a different graph while one is still in
       flight. Writing it then attributes one graph's query to another's stream and
       recreates a stream key `FLUSHALL` removed — on the master only, since a
       key-API write does not replicate, which is what made
       `test_replication_states` fail: it compares master and replica keyspaces.
       The C engine cannot get this wrong because its queries log is owned by the
       GraphContext and is freed with it.

       The corollary is that a graph which merely *moved* keeps its entries: they are
       re-addressed to the key it answers to at write time, which is C's behaviour
       too."""

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test01_stale_entry_not_written_to_a_new_graph(self):
        g = self.db.select_graph("stale")
        g.query("CREATE (:N {v: 1})")          # queues an entry for this graph
        payload = self.conn.dump("stale")      # dump/restore keep the payload binary
        self.conn.flushall()                   # that graph, and its stream, are gone
        self.conn.restore("stale", 0, payload) # same name, new graph
        # well past the flusher's window: whatever was going to be written, was
        time.sleep(1)
        self.env.assertEqual(self.conn.type(StreamName(g)), "none")
        keys = sorted(k.decode() if isinstance(k, bytes) else k for k in self.conn.keys("*"))
        self.env.assertEqual(keys, ["stale"])

    def test02_entry_follows_a_renamed_graph(self):
        """The other half: same graph, different name.

           A `RENAME` re-keys the graph and deletes the stream of the old key, so a
           still-queued entry naming that key must not recreate it — it belongs to the
           graph, which now lives at the new key, so it is written there. That is what
           C does: its cron task takes the stream name from the `GraphContext`, and
           `GraphContext_Rename` deletes the old stream and rebuilds that name.
           Dropping the entry instead loses every query in flight across a blue/green
           key swap.

           The query and the `RENAME` are pipelined, because "in flight" is a race the
           test has to win: the entry is written a linger window (5ms) after the query
           finishes, so a client round trip in between can lose it — the entry then
           lands in telemetry{before} and the rename deletes it, leaving nothing to
           follow. Pipelining removes the round trip, leaving only the microseconds
           Redis takes to unblock the client, and the attempt is retried on the rare
           occasion that is not enough. The invariant that holds either way — that the
           stream the rename deleted stays deleted — is asserted on every attempt."""

        attempts = 20
        for attempt in range(attempts):
            self.conn.flushall()
            before, after = f"before{attempt}", f"after{attempt}"

            pipe = self.conn.pipeline(transaction=False)
            pipe.execute_command("GRAPH.QUERY", before, "CREATE (:N {v: 1})")
            pipe.rename(before, after)
            pipe.execute()

            # well past the flusher's window: whatever was going to be written, was
            time.sleep(0.5)

            # the stream the rename deleted stays deleted, whoever won the race
            self.env.assertEqual(self.conn.type(f"telemetry{{{before}}}"), "none")

            keys = sorted(k.decode() if isinstance(k, bytes) else k
                          for k in self.conn.keys("*"))
            if keys == [after, f"telemetry{{{after}}}"]:
                # the entry was still queued at the rename and followed the graph
                self.env.assertEqual(self.conn.xlen(f"telemetry{{{after}}}"), 1)
                return
            # the flusher got there first, so there was nothing in flight to carry
            self.env.assertEqual(keys, [after])

        raise AssertionError(
            f"in {attempts} attempts the flusher was never still holding an entry when "
            "the rename landed, so the behaviour under test was never exercised")

class testGraphInfoCmdInfoDisabled():
    """`CMD_INFO no` at *load time* must stop logging finished queries.

    `test09_cmd_info_runtime_toggle` covers `GRAPH.CONFIG SET`; this covers the
    module argument, which reaches the same atomic by a different route.

    The config was registered and reported by `GRAPH.CONFIG GET`, but nothing
    consulted it, so queries were logged regardless — and logging one is not free.
    On an M3 Pro, `RETURN 1` costs ~150k instructions with logging off; logging it
    cost a further ~83k, against ~11k on the C engine. Where the 7-8x went:

      ~49k  waking the flusher once per query — a blocking wait returned the
            moment an entry was queued, so nothing was batched but the flush
            itself, and the wakeup is most of what an entry costs. C's cron task
            runs on a timer instead and the query thread only appends to a buffer.
      ~26k  `RM_Call("XADD", ...)`: command lookup, dispatch, reply machinery.
      ~6k   a fresh `RedisModuleString` for all 25 argument tokens, including the
            ten constant field names, per entry.
      ~2k   replicating each XADD — which C never did.

    Fixed by writing the way C does: `StreamAdd` on an opened key, shared field-name
    strings, one key open and trim per graph per batch, and a batch window the
    flusher sleeps out. That leaves ~14k, or 1.28x of C. This test covers only the
    remaining escape hatch: with `CMD_INFO no` the cost is zero because nothing is
    enqueued at all.
    """

    def __init__(self):
        self.env, self.db = Env(moduleArgs="CMD_INFO no")
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_config_reports_disabled(self):
        self.env.assertEqual(
            self.conn.execute_command("GRAPH.CONFIG", "GET", "CMD_INFO"), ["CMD_INFO", 0]
        )

    def test02_no_queries_logged(self):
        stream = StreamName(self.graph)
        self.conn.delete(stream)
        for i in range(20):
            self.graph.query(f"RETURN {i}")
        # The flusher wakes on a 5ms timer; a full second is far longer than it
        # needs, so an empty stream here means nothing was ever enqueued.
        time.sleep(1)
        self.env.assertEqual(self.conn.type(stream), "none")
        self.env.assertEqual(self.conn.xlen(stream), 0)

class testGraphInfoShutdownUnderLoad():
    """`SHUTDOWN` must not hang because the telemetry flusher is behind.

    `shutdown_flusher_thread` drops the sender and then joins the flusher from the
    main thread, which holds the module GIL for the whole shutdown event callback.
    So the flusher must never park on the GIL after that point: it has to notice the
    disconnect and leave without writing.

    Noticing it means *asking*. `drain_queued` stops calling `try_recv` the moment the
    batch reaches `FLUSH_BATCH_MAX`, and crossfire reports `Disconnected` only when the
    channel is empty as well as closed — so with a backlog the flusher used to walk
    past its `if disconnected` guard straight into a blocking `hold_gil()` and park
    there forever, main thread waiting on it in `pthread_join`. The two halves of
    #2554 interact to produce exactly that: skipping the linger when a full batch is
    already queued is what removed the last observation of the disconnect.

    The pipelined burst is what creates the backlog — it drives the queue past
    `FLUSH_BATCH_MAX` so every drain exits full rather than empty.

    Asserted here is only that the process *terminates*, not that it exits zero: the
    module's shutdown handler also tears down RediSearch, which is beyond what this
    test is pinning. A hang is what the regression looked like.
    """

    def __init__(self):
        # The shutdown handler is only subscribed when RS_GLOBAL_DTORS is set (see
        # module_init.rs), i.e. the sanitizer/valgrind runs — without it nothing calls
        # `shutdown_flusher_thread` and there is no deadlock to have. RLTest snapshots
        # os.environ when it builds the runner, so this has to be set before Env(),
        # and is put back straight after so no later env inherits it.
        saved = os.environ.get('RS_GLOBAL_DTORS')
        os.environ['RS_GLOBAL_DTORS'] = '1'
        try:
            # A moduleArgs value distinct from every other class in this file, so
            # RLTest starts a fresh server for it rather than reusing one that was
            # spawned without RS_GLOBAL_DTORS.
            self.env, self.db = Env(moduleArgs="MAX_INFO_QUERIES 100000")
        finally:
            if saved is None:
                os.environ.pop('RS_GLOBAL_DTORS', None)
            else:
                os.environ['RS_GLOBAL_DTORS'] = saved
        self.conn = self.env.getConnection()

    def test01_shutdown_with_a_backlogged_flusher(self):
        # Only meaningful when RLTest owns the server process. In the CI image mode
        # (FALKORDB_TEST_IMAGE) the module runs in a container this test cannot signal
        # or reap, so there is nothing here to observe.
        proc = getattr(self.env.envRunner, 'masterProcess', None)
        if proc is None:
            self.env.skip()
            return

        # Pipelined so the entries arrive far faster than the flusher's ceiling of one
        # FLUSH_BATCH_MAX (256) per FLUSH_LINGER (5ms); a round trip per query would
        # keep the queue empty and never reach the state under test.
        for _ in range(8):
            pipe = self.conn.pipeline(transaction=False)
            for i in range(2000):
                pipe.execute_command("GRAPH.QUERY", "shutdownload", f"RETURN {i}")
            pipe.execute()

        # Straight into the shutdown, with the queue still backlogged. On the way out
        # the server never replies to this — it either goes away mid-command or, when
        # the bug is present, stops answering entirely — so every outcome here is an
        # exception, and the verdict is taken from the process below, not from this.
        try:
            self.conn.execute_command("SHUTDOWN", "NOSAVE")
        except Exception:
            pass

        deadline = time.time() + 30
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.1)
        alive = proc.poll() is None
        if alive:
            proc.kill()
        self.env.assertFalse(
            alive,
            message="server still running 30s after SHUTDOWN NOSAVE: the telemetry "
                    "flusher parked on the GIL while the main thread joined it",
        )
        # The server is gone; keep RLTest from trying to talk to it on the way out.
        self.env.envRunner.masterProcess = None
        self.env.envRunner.envIsUp = False
        self.env.envRunner.envIsHealthy = False

class testGraphInfoAof():
    """telemetry is node-local and never replicated, so it is not written to
       the AOF; replaying the AOF restores no telemetry and logs nothing

       C pins the second half by raising EFFECTS_THRESHOLD so that the AOF holds
       GRAPH.QUERY and the replay runs queries under the loading flag. This
       engine ignores that setting and always writes GRAPH.EFFECT, which logs
       nothing in the first place, so a replayed GRAPH.QUERY is not something
       these tests can produce"""

    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        # leave the AOF off for whatever runs next
        try:
            self.conn.config_set("appendonly", "no")
        except Exception:
            pass
        self.conn.flushall()

    def _wait_for_aof_rewrite(self):
        self.env.assertTrue(_wait_for(lambda:
            self.conn.info("persistence").get("aof_rewrite_in_progress", 0) == 0,
            timeout=30))

    def _start_empty_aof(self):
        # a fresh AOF whose base is an empty dataset, so DEBUG LOADAOF replays
        # only what the test wrote (see test_aof.py)
        self.conn.flushall()
        self.conn.config_set("appendonly", "no")
        self.conn.config_set("appendonly", "yes")
        self._wait_for_aof_rewrite()

    def test01_aof_holds_no_telemetry(self):
        stream = StreamName(self.graph)

        self._start_empty_aof()

        for _ in range(3):
            self.graph.query("CREATE ()")

        # the node logs its own telemetry locally
        self.env.assertTrue(_wait_for(lambda: self.conn.exists(stream) and
            self.conn.xinfo_stream(stream)["entries-added"] == 3))

        # the AOF holds the writes but none of the node-local telemetry
        # entries, so replaying it restores no telemetry stream
        self.env.assertEqual(self.conn.execute_command("DEBUG", "LOADAOF"), "OK")

        # well past the flusher's window: the replay logged nothing either
        time.sleep(1)
        self.env.assertFalse(self.conn.exists(stream))

        # the graph came back through the replay
        self.env.assertEqual(
            self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0], 3)

    def test02_aof_rewrite_keeps_live_telemetry_but_strips_the_file(self):
        """an AOF rewrite strips telemetry in the fork child only, as a
           background save does: the live server keeps its stream, and the
           rewritten file holds none of it"""

        stream = StreamName(self.graph)
        self._start_empty_aof()

        self.graph.query("CREATE ()")
        self.env.assertTrue(_wait_for(lambda: self.conn.exists(stream)))

        self.conn.execute_command("BGREWRITEAOF")
        # the rewrite is scheduled, not started, by the time the command returns
        self.env.assertTrue(_wait_for(lambda:
            self.conn.info("persistence")["aof_rewrite_scheduled"] == 0))
        self._wait_for_aof_rewrite()
        self.env.assertEqual(
            self.conn.info("persistence")["aof_last_bgrewrite_status"], "ok")
        self.env.assertTrue(self.conn.exists(stream))

        self.env.assertEqual(self.conn.execute_command("DEBUG", "LOADAOF"), "OK")
        self.env.assertFalse(self.conn.exists(stream))
        self.env.assertEqual(
            self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0], 1)

class testGraphInfoReplication():
    """telemetry streams are node-local: each node logs only the queries it
       serves, the writes are not replicated, and the streams are stripped from
       every snapshot so a divergent consumer-group cursor can never corrupt an
       RDB across a failover"""

    def __init__(self):
        # replication timing doesn't play well with Valgrind/sanitizers
        if SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env(env='oss', useSlaves=True, enableDebugCommand=True)
        self.master  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()

        # where each server is reached by the other one, for REPLICAOF: the
        # master's address is the one the replica already follows, and the
        # replica's is a sibling container under a CI image run, or the port
        # RLTest spawned it on locally
        info = self.replica.info("replication")
        self.master_address = (info["master_host"], info["master_port"])
        if hasattr(self.env, "replica_host"):
            self.replica_address = (self.env.replica_host, self.env.replica_port)
        else:
            self.replica_address = ("localhost", self.env.envRunner.slavePort)

    def tearDown(self):
        # test02 swaps the roles; put them back, so that whatever shares this
        # environment next does not inherit a read-only server, or a replica
        # that a failed test left detached
        if (self.master.info("replication")["role"] != "master" or
                self.replica.info("replication")["role"] != "slave"):
            self.master.execute_command("REPLICAOF", "NO", "ONE")
            self.replica.execute_command("REPLICAOF", *self.master_address)
            _wait_for(lambda:
                self.replica.info("replication")["master_link_status"] == "up",
                timeout=30)

    def _roles(self):
        """the current (master, replica) connections, test02 swaps them

           waits for the replica's link to be up: the first full sync is
           delayed by repl-diskless-sync-delay, and one that lands mid-test
           would replace the replica's own stream with the master's snapshot"""
        if self.master.info("replication")["role"] == "master":
            master, replica = self.master, self.replica
        else:
            master, replica = self.replica, self.master

        self.env.assertTrue(_wait_for(lambda:
            replica.info("replication")["master_link_status"] == "up", timeout=30))
        return master, replica

    def _address(self, conn):
        """the (host, port) another server reaches conn's server at"""
        return self.master_address if conn is self.master else self.replica_address

    def test01_streams_are_node_local(self):
        """each node logs only the queries it serves: the master its writes,
           the replica its own reads; neither appears in the other's stream"""

        master, replica = self._roles()
        master_graph  = Graph(master, GRAPH_ID)
        replica_graph = Graph(replica, GRAPH_ID)
        stream        = StreamName(master_graph)

        # the master serves a write; the replica applies it through replication
        # but does not log it, it is the master's query not the replica's own
        # (it arrives as GRAPH.EFFECT, which logs nothing whatever the dispatch
        # rule says: this pins the outcome, not `dispatch::logs_telemetry`)
        master_graph.query("CREATE ()")
        self.env.assertTrue(_wait_for(lambda:
            master.exists(stream) and master.xlen(stream) == 1))
        master.execute_command("WAIT", "1", "10000")

        # the replica serves a read-only query of its own
        replica_graph.ro_query("MATCH (n) RETURN count(n)")
        self.env.assertTrue(_wait_for(lambda:
            replica.exists(stream) and replica.xlen(stream) == 1))

        # the streams are independent: the master holds only its write, the
        # replica only its own read, neither carries the other's query
        master_q  = consumeStream(master,  self.env, stream, drop=False, n_items=1)
        replica_q = consumeStream(replica, self.env, stream, drop=False, n_items=1)
        self.env.assertEqual([q.Query for q in master_q],  ["CREATE ()"])
        self.env.assertEqual([q.Query for q in replica_q],
                ["MATCH (n) RETURN count(n)"])

    def test02_survives_failover_and_reload(self):
        """Redis replicates the telemetry stream's consumer-group cursor onto
           the replica's own (shorter) stream; stripping telemetry from every
           snapshot keeps a promoted node able to save, reload and full-sync,
           whatever state that cursor ends up in.

           note: the exact entries-read > entries-added state that breaks the
           loader is a timing-dependent replication artifact (Redis 8.10 clamps
           XGROUP SETID and rejects a lowering XSETID, so it can't be forced),
           so this asserts the deterministic properties: the group really
           replicated, and the promoted node survives save/reload/full-sync"""

        master, replica = self._roles()
        # a dedicated graph, so no earlier test's entries are in the stream
        master_graph  = Graph(master, "failover")
        replica_graph = Graph(replica, "failover")
        stream        = StreamName(master_graph)

        # the master writes its own entries
        for _ in range(3):
            master_graph.query("CREATE ()")
        self.env.assertTrue(_wait_for(lambda:
            master.exists(stream) and master.xlen(stream) == 3))
        master.execute_command("WAIT", "1", "10000")

        # the replica serves its own read-only queries first, so its node-local
        # stream exists and diverges from the master's (fewer entries, its own
        # IDs) before the master's group state replicates onto it
        for _ in range(2):
            replica_graph.ro_query("MATCH (n) RETURN count(n)")
        self.env.assertTrue(_wait_for(lambda:
            replica.exists(stream) and replica.xlen(stream) == 2))

        # an exporter reads the master's stream through a consumer group; the
        # group and its cursor replicate onto the replica's own (shorter) stream
        master.execute_command("XGROUP", "CREATE", stream, "exporter", "0")
        master.xreadgroup("exporter", "consumer", {stream: ">"}, count=100)
        master.execute_command("WAIT", "1", "10000")

        # the group really did replicate onto the replica's stream, so the
        # consumer-group state carried across the failover below is actually
        # present (not a silently-missing group the test would skip over)
        self.env.assertTrue(_wait_for(lambda:
            any(grp["name"] == "exporter"
                for grp in replica.xinfo_groups(stream))))

        #-----------------------------------------------------------------------
        # failover: the replica becomes the master
        #-----------------------------------------------------------------------

        replica.execute_command("REPLICAOF", "NO", "ONE")
        master.execute_command("REPLICAOF", *self._address(replica))
        master, replica = replica, master
        self.env.assertTrue(_wait_for(lambda:
            replica.info("replication")["master_link_status"] == "up", timeout=30))

        # the exporter reads from the promoted node, advancing the inherited
        # group cursor over the node's own entries (the group exists here, as
        # asserted before the failover, so this must not raise)
        master.xreadgroup("exporter", "consumer", {stream: ">"}, count=100)

        # the promoted node saves and reloads: the telemetry stream and its
        # replicated consumer-group state are stripped at save, so whatever the
        # counters are, nothing invalid reaches the RDB and the node loads
        self.env.assertEqual(master.execute_command("DEBUG", "RELOAD"), "OK")
        self.env.assertTrue(master.ping())
        self.env.assertFalse(master.exists(stream))

        # a full sync from the promoted node loads on the demoted node too
        sync_full = master.info("stats")["sync_full"]
        replica.execute_command("REPLICAOF", "NO", "ONE")
        replica.execute_command("REPLICAOF", *self._address(master))
        self.env.assertTrue(_wait_for(lambda:
            master.info("stats")["sync_full"] > sync_full and
            replica.info("replication")["master_link_status"] == "up", timeout=30))
        self.env.assertTrue(replica.ping())
