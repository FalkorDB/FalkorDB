from common import *
from index_utils import wait_for_indices_to_sync
from constraint_utils import *
import csv
import os
import struct
import tempfile
import threading
import time

GRAPH_ID = "replication"


def in_db(con, db):
    """A connection to the server behind `con`, on database `db`."""
    kwargs = con.connection_pool.connection_kwargs
    return redis.Redis(host=kwargs["host"], port=kwargs["port"], db=db)


def bulk_nodes(label, count, bad_last=False):
    """A GRAPH.BULK node token: `count` (:label {v: 7}) records, then, with
    `bad_last`, one record of an unknown property type, which fails the batch
    after the rest of it was built."""
    token = (label.encode() + b"\0" + struct.pack("=I", 1) + b"v\0" +
             struct.pack("=Bq", 4, 7) * count)
    if bad_last:
        token += struct.pack("=Bq", 99, 0)
    return token


# test to see if replication works as expected
# FalkorDB should replicate all write queries which had an effect on the
# underline graph, e.g. CREATE, DELETE, UPDATE operations as well as
# index creation and removal
# constraint creation and removal
# read queries shouldn't be replicated.

class testReplication(FlowTestsBase):

    def __init__(self):
        # skip test if we're running under sanitizer
        if SANITIZER:
            Environment.skip(None) # sanitizer is not working correctly with replication

        self.env, self.db = Env(env='oss', useSlaves=True)

    def test_CRUD_replication(self):
        # create a simple graph
        env = self.env
        source_con = env.getConnection()
        replica_con = env.getSlaveConnection()

        # enable write commands on slave, required as all FalkorDB
        # commands are registered as write commands
        replica_con.config_set("slave-read-only", "no")

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")

        # perform CRUD operations

        #-----------------------------------------------------------------------
        # create a simple graph
        #-----------------------------------------------------------------------

        src = Graph(source_con, GRAPH_ID)
        replica = Graph(replica_con, GRAPH_ID)

        q = """CREATE
                (s:L {id: $s_id, name: $s_name, height: $s_height}),
                (t:L {id: $t_id, name: $t_name, height: $t_height}),
                (s)-[e:R]->(t)"""

        params = {'s_id': 0,
                  's_name': 'abcd',
                  's_height': 178,
                  't_id': 1,
                  't_name': 'efgh',
                  't_height': 178
        }

        src.query(q, params)

        #-----------------------------------------------------------------------
        # create indices
        #-----------------------------------------------------------------------

        # create index
        create_node_range_index(src, 'L', 'id')

        # create full-text index
        create_node_fulltext_index(src, 'L', 'name')

        # add fields to existing index
        create_node_fulltext_index(src, 'L', 'title', 'desc', sync=True)

        # create full-text index with index config
        q = "CREATE FULLTEXT INDEX FOR (n:L1) ON (n.title, n.desc) OPTIONS {language: 'german', stopwords: ['a', 'b']}"
        src.query(q)

        #-----------------------------------------------------------------------
        # create constraints
        #-----------------------------------------------------------------------

        # create node unique constraint
        create_unique_node_constraint(src, "L", "id")

        # add another unique constraint
        create_unique_node_constraint(src, "L", "id", "name", sync=True)

        # add a unique constraint which is destined to fail
        q = """CREATE
               (:Actor {age: $age, name: $name}),
               (:Actor {age: $age, name: $name})"""

        params = {'age': 10, 'name': 'jerry'}

        result = src.query(q, params)
        self.env.assertEqual(result.nodes_created, 2)

        create_unique_node_constraint(src, "Actor", "age", sync=True)
        c = get_constraint(src, "UNIQUE", "LABEL", "Actor", "age")
        self.env.assertEqual(c.status, "FAILED")

        # update entity
        q = "MATCH (n:L {id:$id}) SET n.id = $new_id"
        params = {'id': 1, 'new_id': 2}
        result = src.query(q, params)
        self.env.assertEqual(result.properties_set, 1)

        # delete entity
        q = "MATCH (n:L {id:$id}) DELETE n"
        params = {'id': 0}
        result = src.query(q, params)
        self.env.assertEqual(result.nodes_deleted, 1)

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")

        # wait for index to sync in replica
        wait_for_indices_to_sync(replica)

        # make sure index is available on replica
        q = "MATCH (s:L {id:2}) RETURN s.name"
        plan = str(src.explain(q))
        replica_plan = str(replica.explain(q))
        env.assertContains("Index Scan", plan)
        env.assertEqual(replica_plan, plan)

        # issue query on both source and replica
        # make sure results are the same
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEqual(replica_result, result)

        # make sure node count on both primary and replica is the same
        q = "MATCH (n) RETURN count(n)"
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEqual(replica_result, result)

        # make sure nodes are in sync
        q = "MATCH (n) RETURN n ORDER BY n"
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEqual(replica_result, result)

        # remove label
        q = "MATCH (s:L {id:2}) REMOVE s:L"
        result = src.query(q)
        env.assertEqual(result.labels_removed, 1)

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")

        q = "MATCH (s:L {id:2}) RETURN s"
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEqual(len(result), 0)
        env.assertEqual(replica_result, result)

        # remove property
        q = "MATCH (s {id:$id}) SET s.id = $new_id RETURN s"
        params = {'id': 2, 'new_id': None}

        result = src.query(q, params)
        env.assertEqual(result.properties_removed, 1)

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")

        q = "MATCH (s {id:2}) RETURN s"
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEqual(len(result), 0)
        env.assertEqual(replica_result, result)

        # make sure both primary and replica have the same set of indexes
        q = "CALL db.indexes() YIELD label, properties, language, stopwords, entitytype"
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEqual(replica_result, result)

        # the comparison above is symmetric: it passes just as well when BOTH
        # sides lost the OPTIONS, so it cannot see an option dropped on the way
        # out or materialised into a default on the way in. Assert against the
        # values the statement above actually stated, on the replica, so that
        # what reached it is checked rather than merely agreed on.
        q = "CALL db.indexes() YIELD label, language, stopwords"
        l1 = [r for r in replica.ro_query(q).result_set if r[0] == 'L1']
        env.assertEqual(len(l1), 1)
        env.assertEqual(l1[0][1], 'german')
        env.assertEqual(sorted(l1[0][2]), ['a', 'b'])

        # drop fulltext index
        q = "CALL db.idx.fulltext.drop('L')"
        result = src.query(q)
        env.assertEqual(result.indices_deleted, 3)

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")
        
        # TODO: check if this sync is needed
        wait_for_indices_to_sync(src)
        wait_for_indices_to_sync(replica)

        # make sure both primary and replica have the same set of indexes
        q = "CALL db.indexes() YIELD label, properties, language, stopwords, entitytype"
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEqual(replica_result, result)

        # make sure both primary and replica have the same set of constraints
        origin_result = list_constraints(src)
        replica_result = list_constraints(replica)
        env.assertEqual(replica_result, origin_result)

        # drop constraint
        drop_unique_node_constraint(src, "L", "id")

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")

        # make sure both primary and replica have the same set of constraints
        origin_result = list_constraints(src)
        replica_result = list_constraints(replica)
        env.assertEqual(replica_result, origin_result)

        # drop failed constraint
        drop_unique_node_constraint(src, "Actor", "age")

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")

        # make sure both primary and replica have the same set of constraints
        origin_result = list_constraints(src)
        replica_result = list_constraints(replica)
        env.assertEqual(replica_result, origin_result)

    # GRAPH.BULK must reach replicas too.
    #
    # Regression for #2347: the background-writer path replicated with
    # RM_ReplicateVerbatim on a *thread-safe* context. That API propagates
    # ctx->client->argv, and the client behind a thread-safe context is a fake
    # pooled one carrying no argv, so a zero-argument command was propagated and
    # the entire batch was dropped — silently, with a healthy replication link and
    # no error anywhere. Every bulk load into a replicated deployment diverged, and
    # the data was lost on failover.
    #
    # Both bulk paths are covered because they take different branches: the BEGIN
    # batch that creates the graph, and an append batch (no BEGIN) into the graph
    # that batch just made. The loader CLI only ever produces the former, so the
    # append path is driven the way test_bulk_insertion.py does it — by reusing the
    # loader's own serialization with the BEGIN token suppressed.
    def test_bulk_replication(self):
        from falkordb_bulk_loader.bulk_insert import parse_schemas, process_entities
        from falkordb_bulk_loader.query_buffer import QueryBuffer
        from falkordb_bulk_loader.config import Config
        from falkordb_bulk_loader.label import Label

        env = self.env
        source_con = env.getConnection()
        replica_con = env.getSlaveConnection()
        graphname = "bulk_replication"

        def node_count(graph, label):
            """Count of :label nodes, reporting a wholly absent graph as 0.

            With the bug the replica never receives the BEGIN batch, so the key does
            not exist there at all and GRAPH.RO_QUERY errors. Map that to 0 so the
            failure reads as a count mismatch instead of a ResponseError."""
            try:
                q = f"MATCH (n:{label}) RETURN count(n)"
                return graph.ro_query(q).result_set[0][0]
            except ResponseError as e:
                if "empty key" in str(e):
                    return 0
                raise

        def wait_for_replica_offset(timeout=30):
            """Block until the replica has applied everything the master propagated.

            WAIT is NOT usable for this. It blocks on the calling client's `woff`,
            and Redis only advances `woff` when that client's *own* command
            propagated something:

                if (old_master_repl_offset != server.master_repl_offset)
                    c->woff = server.master_repl_offset;      // server.c

            The bulk batches are issued by the loader's worker process over its own
            connection, so `source_con`'s `woff` never covers them — WAIT would take
            the "already acked" fast path and return without waiting for the batches
            at all. Compare offsets directly instead."""
            target = source_con.info("replication")["master_repl_offset"]
            deadline = time.time() + timeout
            while time.time() < deadline:
                if replica_con.info("replication").get("slave_repl_offset", -1) >= target:
                    return
                time.sleep(0.05)
            raise AssertionError(
                f"replica did not reach master offset {target} within {timeout}s")

        # Private directory, and the whole body runs inside it so the CSVs are
        # removed even when an assertion fails.
        with tempfile.TemporaryDirectory() as tmp_dir:
            begin_csv  = os.path.join(tmp_dir, 'begin.csv')
            append_csv = os.path.join(tmp_dir, 'append.csv')
            with open(begin_csv, mode='w') as csv_file:
                out = csv.writer(csv_file)
                out.writerow(["v"])
                for i in range(100):
                    out.writerow([i])
            with open(append_csv, mode='w') as csv_file:
                out = csv.writer(csv_file)
                out.writerow(["v"])
                for i in range(100, 150):
                    out.writerow([i])

            # The replica MUST be attached and in sync before the first bulk batch.
            # RLTest's replica is still handshaking when the class starts
            # (master_link_status:down, master_repl_offset:0), and a replica that
            # attaches *after* the load gets the graph through a full RDB transfer —
            # which masks this bug completely, since the data then arrives no matter
            # what the command path propagated.
            #
            # WAIT is the right tool for *this* gate: a client with woff 0 blocks
            # until some replica reaches SLAVE_STATE_ONLINE and acks, which is
            # exactly "the replica is attached". It is the wrong tool after the load
            # — see wait_for_replica_offset.
            source_con.execute_command("WAIT", "1", "0")

            # Snapshot the full-sync counter so the assertions below can prove the
            # rows travelled as replicated commands rather than in an RDB snapshot.
            sync_full_before = source_con.info("stats")["sync_full"]

            config = Config(store_node_identifiers=True)

            # BEGIN path — creates the graph
            buf = QueryBuffer(graphname, self.db.connection, config)
            process_entities(parse_schemas(Label, buf, [], [('N', begin_csv)], config))
            buf.send_buffer()
            buf.wait_pool()

            # append path — no BEGIN, into the graph the batch above created
            buf = QueryBuffer(graphname, self.db.connection, config)
            buf.initial_query = False
            process_entities(parse_schemas(Label, buf, [], [('M', append_csv)], config))
            buf.send_buffer()
            buf.wait_pool()

            # Block until the replica has actually applied both batches.
            wait_for_replica_offset()

            src     = Graph(source_con, graphname)
            replica = Graph(replica_con, graphname)

            # No full resync happened, so anything the replica has, it got from the
            # replication stream. Pin this first: without it a re-synced replica
            # would satisfy the count assertions below with the fix reverted.
            env.assertEqual(source_con.info("stats")["sync_full"], sync_full_before)

            # Assert the absolute counts, not just master == replica. With the bug
            # both batches are missing on the replica, and a bare equality check
            # would also pass the day the master stops receiving the rows.
            for label, expected in (('N', 100), ('M', 50)):
                env.assertEqual(node_count(src, label), expected)
                env.assertEqual(node_count(replica, label), expected)

    # start `cmd` on `key` in a thread and return once it has been admitted and is
    # still running: admission creates the key on the main thread, before the
    # write runs on a worker
    def _start_slow_write(self, cmd, key, reply):
        import threading
        from redis.backoff import NoBackoff
        from redis.retry import Retry

        env = self.env
        # about a second of work inside one expression, so the write is still
        # matching when the caller acts on the key, while every operator emits
        # one row: GRAPH.RECORD keeps a trace entry per operator output row
        slow = ("WITH reduce(s = 0, i IN range(1, 8000) | "
                "s + reduce(t = 0, j IN range(1, 5000) | t + j)) AS c "
                "CREATE (:Slow {c: c})")

        def write():
            con = redis.Redis(env.host, env.port, socket_timeout=120,
                              retry=Retry(NoBackoff(), 0))
            try:
                reply.append(con.execute_command(cmd, key, slow))
            except redis.ResponseError as e:
                reply.append(e)

        writer = threading.Thread(target=write)
        writer.start()
        try:
            deadline = time.monotonic() + 10
            admitted = False
            while not admitted and time.monotonic() < deadline:
                admitted = self._bounded(env.getConnection(), deadline).exists(key) == 1
                if not admitted:
                    time.sleep(0.001)
            env.assertTrue(admitted)
            # admitted and still running, so the caller's change overlaps the write
            env.assertTrue(writer.is_alive())
        except BaseException:
            # the caller never gets the writer to join: do it here
            writer.join()
            raise
        return writer

    # run `action` while `args` runs on a worker, and return the command's reply,
    # or the error it got. This waits for the client to block, not for the key to
    # exist: a command on an existing graph finds its key there before it is even
    # sent, and the client stays blocked from admission until the worker is done
    def _while_queued(self, args, action):
        env = self.env
        reply = []

        def run():
            con = redis.Redis(env.host, env.port, socket_timeout=120,
                              decode_responses=True, retry=Retry(NoBackoff(), 0))
            try:
                reply.append(con.execute_command(*args))
            except Exception as e:  # a timeout too, so it reaches the assertions
                reply.append(e)

        sender = threading.Thread(target=run)
        sender.start()
        try:
            deadline = time.monotonic() + 10
            while True:
                clients = self._bounded(env.getConnection(), deadline).client_list()
                if any(c["cmd"].lower() == args[0].lower() and "b" in c["flags"]
                       for c in clients):
                    break
                if time.monotonic() > deadline:
                    raise AssertionError(f"{args[0]} never ran on a worker")
                time.sleep(0.001)
            action()
        finally:
            # joined even if the action raised, so the command cannot outlive the test
            sender.join()
        return reply[0]

    # a client for `con`'s server whose every command gives up at `deadline`:
    # the socket timeout is the time left (at most 5 s) and nothing is retried,
    # so no single command can outlast the caller's deadline
    @staticmethod
    def _bounded(con, deadline):
        kw = con.connection_pool.connection_kwargs
        left = max(0.1, min(5.0, deadline - time.monotonic()))
        return redis.Redis(kw["host"], kw["port"], socket_timeout=left,
                           socket_connect_timeout=left, retry=NO_RETRY)

    # until the replica has applied everything the master propagated, within
    # 30 s: every INFO, the primary's included, is bounded by that deadline. A
    # replica applies a replicated GRAPH.BULK inline, yielding as it goes, and
    # past busy-reply-threshold answers -BUSY until it is done
    def _wait_replica_caught_up(self, source_con, replica_con):
        deadline = time.monotonic() + 30
        target = self._bounded(source_con, deadline).info("replication")["master_repl_offset"]
        while time.monotonic() < deadline:
            try:
                info = self._bounded(replica_con, deadline).info("replication")
                if info.get("slave_repl_offset", -1) >= target:
                    return
            except (redis.TimeoutError, redis.ConnectionError):
                # the replica did not answer in the time left: poll again
                # until the deadline, then fail below
                continue
            except redis.ResponseError as e:
                if "BUSY" not in str(e):
                    raise
            time.sleep(0.05)
        self.env.assertTrue(False, message=f"replica did not reach offset {target} in 30 s")

    # a queued write is admitted under the key the client named and commits
    # later, from a worker. a RENAME in between moves the graph to another key.
    # the write was replicated under the key it was admitted at, which the
    # replica no longer had: GRAPH.EFFECT made a graph there, so the master kept
    # the new key with the write and the replica kept the old key with it and
    # the new key without. C replicates under the key the graph lives at
    def test_rename_during_queued_write(self):
        env = self.env
        source_con = env.getConnection()
        replica_con = env.getSlaveConnection()
        old, new = "rename_during_write_a", "rename_during_write_b"
        source_con.delete(old, new)
        # bounded: an unbounded WAIT never returns if the replica dropped
        env.assertEqual(source_con.execute_command("WAIT", "1", "10000"), 1)
        sync_full_before = source_con.info("stats")["sync_full"]

        reply = []
        writer = self._start_slow_write("GRAPH.QUERY", old, reply)
        try:
            env.assertTrue(source_con.rename(old, new))
        finally:
            # joined even if RENAME raised, so the write cannot outlive the test
            writer.join()
        env.assertEqual(len(reply), 1)
        env.assertFalse(isinstance(reply[0], Exception))

        self._wait_replica_caught_up(source_con, replica_con)

        # the write landed on the renamed graph, on both sides, and reached the
        # replica as a replicated command, not through a forced resync
        for con in (source_con, replica_con):
            env.assertEqual(con.exists(old), 0)
            env.assertEqual(con.exists(new), 1)
            res = Graph(con, new).ro_query("MATCH (n:Slow) RETURN count(n)")
            env.assertEqual(res.result_set[0][0], 1)
        env.assertEqual(source_con.info("stats")["sync_full"], sync_full_before)

    # the same, with MOVE: the graph leaves this database, so no key here holds
    # it, and the write is refused, as C refuses it. committed, it was
    # replicated under the key it was admitted at, which the replica had moved
    # away too: the replica made a new graph there and diverged
    def test_move_during_queued_write(self):
        env = self.env
        source_con = env.getConnection()
        replica_con = env.getSlaveConnection()
        key = "move_during_write"
        for con in (source_con, in_db(source_con, 1)):
            con.delete(key)
        source_con.execute_command("GRAPH.QUERY", key, "CREATE (:Seed)")
        env.assertEqual(source_con.execute_command("WAIT", "1", "10000"), 1)
        sync_full_before = source_con.info("stats")["sync_full"]

        slow = "UNWIND range(1, 20000000) AS x WITH count(x) AS c CREATE (:Slow {c: c})"
        reply = self._while_queued(("GRAPH.QUERY", key, slow),
                                   lambda: env.assertTrue(source_con.move(key, 1)))
        env.assertIsInstance(reply, ResponseError)
        env.assertContains("deleted or replaced", str(reply))
        self._wait_replica_caught_up(source_con, replica_con)

        for con in (source_con, replica_con):
            env.assertEqual(con.exists(key), 0)
            res = Graph(in_db(con, 1), key).ro_query(
                "MATCH (n) RETURN labels(n)[0], count(n)")
            env.assertEqual(res.result_set, [["Seed", 1]])
        env.assertEqual(source_con.info("stats")["sync_full"], sync_full_before)

    # a GRAPH.RECORD write whose key is deleted while it runs is undone, like the
    # same GRAPH.QUERY, and its trace ends with that refusal on the root. (A DEL
    # here is refused when the write escalates; `finish_write` refuses with the
    # same error and the same trace shape when only the keyspace has let go,
    # under a background flush, which a test cannot time reliably.)
    def test_record_refused_after_delete(self):
        env = self.env
        source_con = env.getConnection()
        replica_con = env.getSlaveConnection()
        key = "record_delete_during_write"
        source_con.delete(key)
        env.assertEqual(source_con.execute_command("WAIT", "1", "10000"), 1)

        reply = []
        writer = self._start_slow_write("GRAPH.RECORD", key, reply)
        try:
            env.assertEqual(source_con.delete(key), 1)
        finally:
            writer.join()
        env.assertEqual(len(reply), 1)
        env.assertFalse(isinstance(reply[0], Exception))
        trace = reply[0][0]
        # [operator, ok, value]: the last entry is the refusal, on the root
        env.assertEqual(trace[-1][0], 0)
        env.assertEqual(trace[-1][1], 0)
        env.assertContains("graph was deleted or replaced while the query was running",
                           trace[-1][2].decode())

        # nothing was published: no key on either side
        self._wait_replica_caught_up(source_con, replica_con)
        for con in (source_con, replica_con):
            env.assertEqual(con.exists(key), 0)

    # GRAPH.BULK builds a batch on a worker too, and replicates the command
    # itself, key and all. a BEGIN batch that fails is taken back out of the key
    # its graph was renamed to: taken out of the old one, it left the new one on
    # the master alone. one that loads is replicated under the key the graph
    # was renamed to: under the old one, it made the graph again at that key on
    # the replica, which never had the new one
    def test_rename_during_bulk(self):
        env = self.env
        source_con = env.getConnection()
        replica_con = env.getSlaveConnection()
        old, new = "rename_during_bulk_a", "rename_during_bulk_b"
        env.assertEqual(source_con.execute_command("WAIT", "1", "10000"), 1)
        sync_full_before = source_con.info("stats")["sync_full"]
        count = 2000000

        for bad_last in (True, False):
            source_con.delete(old, new)
            declared = count + 1 if bad_last else count
            reply = self._while_queued(
                ("GRAPH.BULK", old, "BEGIN", declared, 0, 1, 0,
                 bulk_nodes("N", count, bad_last)),
                lambda: env.assertTrue(source_con.rename(old, new)))
            self._wait_replica_caught_up(source_con, replica_con)

            if bad_last:
                env.assertIsInstance(reply, ResponseError)
                env.assertContains("bulk insert failed", str(reply))
            else:
                env.assertContains(f"{count} nodes created", reply)
            for con in (source_con, replica_con):
                env.assertEqual(con.exists(old), 0)
                env.assertEqual(con.exists(new), 0 if bad_last else 1)
                if not bad_last:
                    res = Graph(con, new).ro_query("MATCH (n:N) RETURN count(n)")
                    env.assertEqual(res.result_set[0][0], count)
        env.assertEqual(source_con.info("stats")["sync_full"], sync_full_before)

    # a bulk append whose graph is moved to another database is refused, as a
    # query is. committed, it was replicated under a key the replica no longer
    # had, which refused it as an empty key: the master's graph had the batch
    # and the replica's did not
    def test_move_during_bulk(self):
        env = self.env
        source_con = env.getConnection()
        replica_con = env.getSlaveConnection()
        key = "move_during_bulk"
        for con in (source_con, in_db(source_con, 1)):
            con.delete(key)
        source_con.execute_command("GRAPH.QUERY", key, "CREATE (:Seed)")
        env.assertEqual(source_con.execute_command("WAIT", "1", "10000"), 1)
        sync_full_before = source_con.info("stats")["sync_full"]
        count = 2000000

        reply = self._while_queued(
            ("GRAPH.BULK", key, count, 0, 1, 0, bulk_nodes("N", count)),
            lambda: env.assertTrue(source_con.move(key, 1)))
        env.assertIsInstance(reply, ResponseError)
        env.assertContains("deleted or replaced", str(reply))
        self._wait_replica_caught_up(source_con, replica_con)

        for con in (source_con, replica_con):
            env.assertEqual(con.exists(key), 0)
            res = Graph(in_db(con, 1), key).ro_query(
                "MATCH (n) RETURN labels(n)[0], count(n)")
            env.assertEqual(res.result_set, [["Seed", 1]])

    # a write can register a label with no entity to carry it: OPTIONAL MATCH
    # finds nothing, and SET n:L has interned L by then. the commit shipped no
    # payload, so the replica never heard of L. the next payload numbers its
    # labels from the master's dictionary, one past what the replica has: it
    # was refused, the replica resynced in full, and a master replaying its own
    # AOF exited on that record
    def test_schema_only_write_is_replicated(self):
        env = self.env
        source_con = env.getConnection()
        replica_con = env.getSlaveConnection()
        graphname = "schema_only_write"
        src = Graph(source_con, graphname)
        replica = Graph(replica_con, graphname)

        def labels(graph):
            return sorted(row[0] for row in graph.ro_query("CALL db.labels()").result_set)

        src.query("CREATE (:A)")
        source_con.execute_command("WAIT", "1", "0")
        sync_full_before = source_con.info("stats")["sync_full"]

        src.query("OPTIONAL MATCH (n:Nope) SET n:OnlyRegistered")
        source_con.execute_command("WAIT", "1", "0")
        env.assertContains("OnlyRegistered", labels(src))
        env.assertEqual(labels(replica), labels(src))

        # its label is numbered after OnlyRegistered on the master
        src.query("CREATE (:B)")
        source_con.execute_command("WAIT", "1", "0")
        env.assertEqual(labels(replica), labels(src))
        res = replica.ro_query("MATCH (n:B) RETURN count(n)")
        env.assertEqual(res.result_set[0][0], 1)
        # as replicated commands, not through a forced resync
        env.assertEqual(source_con.info("stats")["sync_full"], sync_full_before)
