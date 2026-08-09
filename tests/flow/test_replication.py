from common import *
from index_utils import wait_for_indices_to_sync
from constraint_utils import *
import csv
import os
import tempfile
import time

GRAPH_ID = "replication"


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
            # what the command path propagated. WAIT blocks until one replica acks.
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

            # the WAIT command forces master slave sync to complete
            source_con.execute_command("WAIT", "1", "0")

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

