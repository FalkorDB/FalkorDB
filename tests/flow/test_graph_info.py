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

def consumeStream(conn, env, stream, drop=True, n_items=1, timeout=30):
    # wait for telemetry stream to be created
    t = 'none' # type of stream_key
    deadline = time.time() + timeout

    while t == 'none' and time.time() < deadline:
        time.sleep(0.2)
        t = conn.type(stream)

    env.assertEquals(t, "stream")
    if t != "stream":
        return []

    # convert stream events to LoggedQueries
    logged_queries = []
    streams = {stream: '0-0'}

    elapsed = 10
    while len(logged_queries) < n_items and elapsed > 0:
        # read messages from the stream
        messages = conn.xread(streams, block=0)

        if len(messages) > 0:
            # process each message received
            stream_messages = messages[0][1]
            for message_id, message_payload in stream_messages:
                logged_queries.append(LoggedQuery(message_payload))

            # update stream last ID
            streams[stream] = stream_messages[-1][0]

        time.sleep(0.2)
        elapsed -= 0.2

    # drop stream
    if drop:
        conn.delete(stream)

    # reverse order to match expected order of events
    logged_queries.reverse()

    return logged_queries

class testGraphInfo():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def assertLoggedQuery(self, logged_query, query, utilized_cache):
        # validate event values
        self.env.assertEquals(logged_query.Query, query)
        self.env.assertEquals(logged_query.UtilizedCache, utilized_cache)

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
        self.env.assertEquals(len(logged_queries), 3)
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
        self.env.assertEquals(len(logged_queries), 6)
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
        self.env.assertEquals(logged_query.Query[:2000], long_query[:2000])
        self.env.assertEquals(logged_query.Query[-3:], "...")

        # assert long param
        logged_query = logged_queries[1]
        self.env.assertLess(len(logged_query.Parameters), 4000)
        self.env.assertEquals(logged_query.Parameters[-3:], "...")

        # assert long param and query
        logged_query = logged_queries[0]
        self.env.assertLess(len(logged_query.Query), 4000)
        self.env.assertEquals(logged_query.Query[:2000], long_param_query[:2000])
        self.env.assertEquals(logged_query.Query[-3:], "...")
        self.env.assertLess(len(logged_query.Parameters), 4000)
        self.env.assertEquals(logged_query.Parameters[-3:], "...")

    def test04_delete_graph(self):
        """make sure reporting stream is deleted when graph is deleted"""

        # validate that stream exists
        stream_name = StreamName(self.graph)
        self.env.assertEquals(self.conn.type(stream_name), "stream")

        # make sure graph is deleted synchronously
        self.db.config_set("ASYNC_DELETE", "no")

        # delete graph
        self.graph.delete()

        # validate that stream was deleted
        self.env.assertEquals(self.conn.type(stream_name), "none")

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
        self.env.assertEquals(len(logged_queries), 1)

        # validate that stream exists
        self.env.assertEquals(self.conn.type(StreamName(old_graph)), "stream")

        # rename graph
        self.conn.rename(old_name, new_name)

        # issue query to create and populate stream
        new_graph.query("RETURN 1")

        # wait for stream to be created
        logged_queries = consumeStream(self.conn, self.env, StreamName(new_graph), drop=False)
        self.env.assertEquals(len(logged_queries), 1)

        # validate that stream was renamed
        self.env.assertEquals(self.conn.type(StreamName(old_graph)), "none")
        self.env.assertEquals(self.conn.type(StreamName(new_graph)), "stream")

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

        # issue threads
        for t in threads:
            t.start()
        
        # wait for graph to be created
        res = self.conn.type(GRAPH_ID)
        while res != "graphdata":
            res = self.conn.type(GRAPH_ID)

        # get waiting and running queries

        #-----------------------------------------------------------------------
        # validate running queries
        #-----------------------------------------------------------------------

        res = self.conn.execute_command("GRAPH.INFO", "RunningQueries", "WaitingQueries")
        while True:
            # validate response structure
            self.env.assertEquals(len(res), 4)
            self.env.assertEquals(res[0], "# Running queries")
            self.env.assertEquals(res[2], "# Waiting queries")
            if len(res[1]) > 0:
                break
            res = self.conn.execute_command("GRAPH.INFO", "RunningQueries", "WaitingQueries")

        running_queries = res[1]
        running_query = running_queries[0]
        self.env.assertEquals(running_query[0], "Received at")
        self.env.assertEquals(running_query[2], "Graph name")
        self.env.assertEquals(running_query[4], "Query")
        self.env.assertEquals(running_query[6], "Execution duration")
        self.env.assertEquals(running_query[8], "Replicated command")

        self.env.assertEquals(running_query[3], GRAPH_ID)
        self.env.assertTrue(running_query[5] == read_query or
                            running_query[5] == write_query1 or
                            running_query[5] == write_query2)
        self.env.assertEquals(running_query[9], False)

        #-----------------------------------------------------------------------
        # validate waiting queries
        #-----------------------------------------------------------------------

        res = self.conn.execute_command("GRAPH.INFO", "RunningQueries", "WaitingQueries")
        while True:
            # validate response structure
            self.env.assertEquals(len(res), 4)
            self.env.assertEquals(res[0], "# Running queries")
            self.env.assertEquals(res[2], "# Waiting queries")
            if len(res[3]) > 0:
                break
            res = self.conn.execute_command("GRAPH.INFO", "RunningQueries", "WaitingQueries")

        waiting_queries = res[3]
        waiting_query = waiting_queries[0]
        self.env.assertEquals(waiting_query[0], "Received at")
        self.env.assertEquals(waiting_query[2], "Graph name")
        self.env.assertEquals(waiting_query[4], "Query")
        self.env.assertEquals(waiting_query[6], "Wait duration")

        self.env.assertEquals(waiting_query[3], GRAPH_ID)
        self.env.assertTrue(waiting_query[5] == read_query or
                            waiting_query[5] == write_query1 or
                            waiting_query[5] == write_query2)

        # signal worker threads to stop
        alive = False

        # wait for all threads to complete
        for t in threads:
            t.join()

    def test08_cmd_info_off_at_runtime(self):
        """turning CMD_INFO off at runtime stops telemetry writes"""

        stream = StreamName(self.graph)

        try:
            # the cron task holds the GIL for its whole run, so once this
            # returns every later run sees CMD_INFO off
            self.conn.execute_command("GRAPH.CONFIG", "SET", "CMD_INFO", "no")
            self.conn.delete(stream)

            for _ in range(3):
                self.graph.query("RETURN 'off'")

            # the cron task runs at least every 3 seconds
            time.sleep(4)
            self.env.assertFalse(self.conn.exists(stream))
        finally:
            self.conn.execute_command("GRAPH.CONFIG", "SET", "CMD_INFO", "yes")

        # back on: new queries are written, the ones issued while off are not
        self.graph.query("RETURN 'on'")
        logged_queries = consumeStream(self.conn, self.env, stream, n_items=1)
        self.env.assertEquals([q.Query for q in logged_queries], ["RETURN 'on'"])

    def test09_cmd_info_back_on_before_cron_runs(self):
        """a query issued while CMD_INFO is off is not written, even when
           CMD_INFO is turned back on before the cron task's next run"""

        stream = StreamName(self.graph)
        self.conn.delete(stream)

        # no wait between the toggles: the cron task, which runs every 250ms
        # to 3 seconds, most likely sees CMD_INFO on
        try:
            self.conn.execute_command("GRAPH.CONFIG", "SET", "CMD_INFO", "no")
            self.graph.query("RETURN 'off'")
        finally:
            self.conn.execute_command("GRAPH.CONFIG", "SET", "CMD_INFO", "yes")

        self.graph.query("RETURN 'on'")
        logged_queries = consumeStream(self.conn, self.env, stream, n_items=1)
        self.env.assertEquals([q.Query for q in logged_queries], ["RETURN 'on'"])

    def test10_cmd_info_off_before_logged_queries_are_written(self):
        """queries logged while CMD_INFO is on, and not yet written when it is
           turned off, are dropped"""

        stream = StreamName(self.graph)

        # the read-only query below needs the graph to exist
        self.graph.query("RETURN 'create'")
        self.env.assertTrue(_wait_for(lambda: self.conn.exists(stream)))
        self.conn.delete(stream)

        try:
            try:
                # while writes are paused the cron task keeps the logged
                # queries instead of writing them
                self.conn.execute_command("CLIENT", "PAUSE", "30000", "WRITE")
                self.graph.ro_query("RETURN 'logged while on'")

                # the cron task runs at least every 3 seconds
                time.sleep(4)
                self.conn.execute_command("GRAPH.CONFIG", "SET", "CMD_INFO", "no")
            finally:
                self.conn.execute_command("CLIENT", "UNPAUSE")

            time.sleep(4)
            self.env.assertFalse(self.conn.exists(stream))
        finally:
            self.conn.execute_command("GRAPH.CONFIG", "SET", "CMD_INFO", "yes")

def _wait_for(condition, timeout=10):
    """poll condition until it holds or timeout seconds pass"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return False

class testGraphInfoOffAtLoad():
    """CMD_INFO off at module load, turned on at runtime"""

    def __init__(self):
        self.env, self.db = Env(moduleArgs='CMD_INFO no')
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_turned_on_at_runtime(self):
        """queries issued before CMD_INFO is turned on are not written"""

        stream = StreamName(self.graph)

        for _ in range(3):
            self.graph.query("RETURN 'off'")

        # the cron task runs at least every 3 seconds
        time.sleep(4)
        self.env.assertFalse(self.conn.exists(stream))

        try:
            self.conn.execute_command("GRAPH.CONFIG", "SET", "CMD_INFO", "yes")

            self.graph.query("RETURN 'on'")
            logged_queries = consumeStream(self.conn, self.env, stream, n_items=1)
            self.env.assertEquals([q.Query for q in logged_queries], ["RETURN 'on'"])
        finally:
            self.conn.execute_command("GRAPH.CONFIG", "SET", "CMD_INFO", "no")

class testGraphInfoAof():
    """queries replayed from the AOF are not logged again, the AOF already
       holds their entries"""

    def __init__(self):
        self.env, self.db = Env(useAof=True, enableDebugCommand=True)
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_aof_replay_writes_no_entries(self):
        stream = StreamName(self.graph)

        try:
            # replicate the writes as GRAPH.QUERY, so loading the AOF replays
            # the queries themselves
            self.db.config_set("EFFECTS_THRESHOLD", 999999)

            for _ in range(3):
                self.graph.query("CREATE ()")
            self.env.assertTrue(_wait_for(lambda: self.conn.exists(stream) and
                self.conn.xinfo_stream(stream)["entries-added"] == 3))

            self.env.assertEquals(self.conn.execute_command("DEBUG", "LOADAOF"), "OK")

            # the cron task runs at least every 3 seconds
            time.sleep(4)
            info = self.conn.xinfo_stream(stream)
            self.env.assertEquals(info["entries-added"], 3)
            self.env.assertEquals(info["length"], 3)
        finally:
            self.db.config_set("EFFECTS_THRESHOLD", 0)

def StreamState(conn, stream):
    """a stream's entry IDs and consumer-group counters, None if it doesn't exist"""
    if conn.type(stream) != 'stream':
        return None

    return {
        "ids"           : [entry[0] for entry in conn.xrange(stream)],
        "entries-added" : conn.xinfo_stream(stream)["entries-added"],
        "groups"        : [(g["name"], g["last-delivered-id"], g["entries-read"],
                            g["lag"]) for g in conn.xinfo_groups(stream)],
    }

class testGraphInfoReplication():
    """only the master writes the telemetry stream and its writes replicate,
       so replicas hold the same stream and consumer-group state stays valid
       across a failover"""

    def __init__(self):
        # replication timing doesn't play well with Valgrind/sanitizers
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env(env='oss', useSlaves=True, enableDebugCommand=True)
        self.master  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()

    def _roles(self):
        """the current (master, replica) connections, test03 swaps them

           waits for the replica's link to be up: the first full sync is
           delayed by repl-diskless-sync-delay, and one that lands mid-test
           would copy the master's stream instead of replicating its writes"""
        if self.master.info("replication")["role"] == "master":
            master, replica = self.master, self.replica
        else:
            master, replica = self.replica, self.master

        self.env.assertTrue(_wait_for(lambda:
            replica.info("replication")["master_link_status"] == "up", timeout=30))
        return master, replica

    def _assert_in_sync(self, master, replica, stream):
        """the replica's copy of the stream matches the master's"""
        master.execute_command("WAIT", "1", "10000")
        _wait_for(lambda: StreamState(replica, stream) == StreamState(master, stream))
        self.env.assertEquals(StreamState(replica, stream), StreamState(master, stream))

    def _port(self, conn):
        """the port of the server conn is connected to"""
        runner = self.env.envRunner
        return runner.port if conn is self.master else runner.slavePort

    def test01_replication_pause_defers_writes(self):
        """while replication traffic is paused the master keeps its logged
           queries, and writes them once the pause ends"""

        master, replica = self._roles()
        graph   = Graph(master, "pause")
        stream  = StreamName(graph)

        graph.query("CREATE ()")
        self.env.assertTrue(_wait_for(lambda: master.exists(stream)))
        self._assert_in_sync(master, replica, stream)

        try:
            # read-only queries still run during a write pause, and are logged
            # the pause outlasts the test, so only UNPAUSE ends it
            master.execute_command("CLIENT", "PAUSE", "30000", "WRITE")
            graph.ro_query("RETURN 1")

            # the cron task runs at least every 3 seconds; the master survives
            # the runs that fall inside the pause, and writes nothing in them
            time.sleep(4)
            self.env.assertTrue(master.ping())
            self.env.assertEquals(master.xlen(stream), 1)
        finally:
            master.execute_command("CLIENT", "UNPAUSE")

        self.env.assertTrue(_wait_for(lambda: master.xlen(stream) == 2))
        self._assert_in_sync(master, replica, stream)

    def test02_trim_replicates(self):
        """the master's trims replicate, replicas drop the same entries"""

        master, replica = self._roles()
        graph   = Graph(master, "trim")
        stream  = StreamName(graph)

        max_queries = master.execute_command("GRAPH.CONFIG", "GET", "MAX_INFO_QUERIES")[1]
        node_size   = master.config_get("stream-node-max-entries")["stream-node-max-entries"]

        try:
            # one entry per stream node, so the master's approximate trim drops
            # single entries; the replica keeps the default node size, where an
            # approximate trim of these few entries would drop none
            master.config_set("stream-node-max-entries", 1)

            graph.query("CREATE ()")
            self.env.assertTrue(_wait_for(lambda: master.exists(stream)))

            master.execute_command("GRAPH.CONFIG", "SET", "MAX_INFO_QUERIES", 2)
            for _ in range(6):
                graph.query("CREATE ()")

            self.env.assertTrue(_wait_for(lambda:
                StreamState(master, stream)["entries-added"] == 7))
            self.env.assertEquals(master.xlen(stream), 2)
            self._assert_in_sync(master, replica, stream)
        finally:
            master.execute_command("GRAPH.CONFIG", "SET", "MAX_INFO_QUERIES", max_queries)
            master.config_set("stream-node-max-entries", node_size)

    def test03_stream_replicates_and_survives_failover(self):
        master, replica = self._roles()
        master_graph  = Graph(master, GRAPH_ID)
        replica_graph = Graph(replica, GRAPH_ID)
        stream        = StreamName(master_graph)

        # each round runs queries on the master, waits for the master to write
        # them, then runs a read-only query on the replica, so that anything
        # the replica wrote would be newer than every master entry
        def traffic(master_queries, expected_len):
            for _ in range(master_queries):
                master_graph.query("CREATE ()")
            self.env.assertTrue(_wait_for(lambda:
                master.exists(stream) and master.xlen(stream) == expected_len))

            # read-only queries served by the replica are not written to its copy
            replica_graph.ro_query("MATCH (n) RETURN count(n)")
            time.sleep(4)  # the cron task runs at least every 3 seconds
            self._assert_in_sync(master, replica, stream)

        # the master writes the entries and they replicate under the same IDs
        traffic(1, 1)

        # an exporter reads the stream through a consumer group on the master
        master.execute_command("XGROUP", "CREATE", stream, "exporter", "0")
        traffic(2, 3)
        res = master.xreadgroup("exporter", "consumer", {stream: ">"}, count=100)
        self.env.assertEquals(sum(len(entries) for _, entries in res), 3)
        self._assert_in_sync(master, replica, stream)

        # one more entry the exporter hasn't read yet
        traffic(1, 4)

        #-----------------------------------------------------------------------
        # failover: the replica becomes the master
        #-----------------------------------------------------------------------

        # the replica serves one more query and is promoted straight away,
        # before its cron task's next run
        replica_graph.ro_query("MATCH (n) RETURN count(n)")
        replica.execute_command("REPLICAOF", "NO", "ONE")
        master.execute_command("REPLICAOF", "localhost", self._port(replica))
        master, replica = replica, master
        self.env.assertTrue(_wait_for(lambda:
            replica.info("replication")["master_link_status"] == "up"))

        # the queries the new master served as a replica were never logged:
        # its cron task runs at least every 3 seconds and writes nothing
        time.sleep(4)
        self.env.assertEquals(master.xinfo_stream(stream)["entries-added"], 4)

        # the exporter's next read delivers the one entry it hasn't seen,
        # and the group has read exactly what the stream received
        res = master.xreadgroup("exporter", "consumer", {stream: ">"}, count=100)
        self.env.assertEquals(sum(len(entries) for _, entries in res), 1)

        state = StreamState(master, stream)
        _, _, entries_read, lag = state["groups"][0]
        self.env.assertEquals(entries_read, state["entries-added"])
        self.env.assertEquals(lag, 0)

        # the new master's snapshot loads
        self.env.assertEquals(master.execute_command("DEBUG", "RELOAD"), "OK")
        self.env.assertEquals(StreamState(master, stream), state)

        # and so does a full sync from it
        sync_full = master.info("stats")["sync_full"]
        replica.execute_command("REPLICAOF", "NO", "ONE")
        replica.execute_command("REPLICAOF", "localhost", self._port(master))
        self.env.assertTrue(_wait_for(lambda:
            master.info("stats")["sync_full"] > sync_full and
            replica.info("replication")["master_link_status"] == "up", timeout=30))
        self._assert_in_sync(master, replica, stream)
