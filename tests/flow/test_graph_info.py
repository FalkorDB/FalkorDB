import time
import queue
import string
import random
import datetime
import threading
import multiprocessing
from common import *

GRAPH_ID ="info"

class LoggedQuery:
    def __init__(self, event):
        # make sure event contains all expected fields
        fields = ["Received at", "Query", "Total duration", "Wait duration",
                  "Execution duration", "Report duration", "Utilized cache",
                  "Write", "Timeout"]
        assert(all(field in event for field in fields))

        # cast and initialize
        self.received_at        = datetime.datetime.fromtimestamp(int(event['Received at']))
        self.query              = event['Query']
        self.total_duration     = float(event['Total duration'])
        self.wait_duration      = float(event['Wait duration'])
        self.execution_duration = float(event['Execution duration'])
        self.report_duration    = float(event['Report duration'])
        self.utilized_cache     = False if event['Utilized cache'] == '0' else True

        assert (self.TotalDuration >= (self.ExecutionDuration + self.ReportDuration))

    def __str__(self):
        return f"""ReceivedAt: {self.ReceivedAt}
                 Query: {self.Query}
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

class testGraphInfo(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def consumeStream(self, stream, drop=True, n_items=1):
        # wait for telemetry stream to be created
        t = 'none' # type of stream_key

        while t == 'none':
            t = self.conn.type(stream)

        self.env.assertEquals(t, "stream")

        # convert stream events to LoggedQueries
        logged_queries = []
        streams = {stream: '0-0'}

        elapsed = 10
        while len(logged_queries) < n_items and elapsed > 0:
            # read messages from the stream
            messages = self.conn.xread(streams, block=0)

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
            self.conn.delete(stream)

        # reverse order to match expected order of events
        logged_queries.reverse()

        return logged_queries

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
        logged_queries = self.consumeStream(StreamName(self.graph), n_items=3)

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
        logged_queries = self.consumeStream(StreamName(self.graph), n_items=6)

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
        logged_queries = self.consumeStream(StreamName(self.graph))

        # make sure number of logged queries is capped
        self.env.assertLess(len(logged_queries), 1200)

    def test03_long_query(self):
        """long queries are truncated with: ..."""

        # create a long string
        length = 4000
        long_str = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

        q = f"RETURN '{long_str}'"
        self.graph.query(q)

        # read stream
        logged_queries = self.consumeStream(StreamName(self.graph), drop=False)

        logged_query = logged_queries[0]

        self.env.assertEquals(logged_query.Query, q)

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
        logged_queries = self.consumeStream(StreamName(old_graph), drop=False)
        self.env.assertEquals(len(logged_queries), 1)

        # validate that stream exists
        self.env.assertEquals(self.conn.type(StreamName(old_graph)), "stream")

        # rename graph
        self.conn.rename(old_name, new_name)

        # issue query to create and populate stream
        new_graph.query("RETURN 1")

        # wait for stream to be created
        logged_queries = self.consumeStream(StreamName(new_graph), drop=False)
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

    # disabled, telemetry ttl crashes replicas
    def _test08_telemetry_ttl(self):
        """make sure telemetry keys have ttl"""

        # make sure graph exists
        self.graph.query("RETURN 1")

        # wait for telemetry stream to be created
        key_type = 'none' # type of stream_key
        telemetry_key = StreamName(self.graph)

        while key_type == 'none':
            key_type = self.conn.type(telemetry_key)

        # make sure we're dealing with a telemetry stream
        self.env.assertEquals(key_type, "stream")

        # make sure stream is associated with TTL
        ttl = self.conn.ttl(telemetry_key)
        self.env.assertGreater(ttl, 0)

