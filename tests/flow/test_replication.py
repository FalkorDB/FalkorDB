from common import *
from index_utils import *
from constraint_utils import *
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
        # skip test if we're running under Valgrind
        if VALGRIND or SANITIZER:
            Environment.skip(None) # valgrind is not working correctly with replication

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

        s = Node(alias='s', labels='L', properties={'id': 0, 'name': 'abcd', 'height' : 178})
        t = Node(alias='t', labels='L', properties={'id': 1, 'name': 'efgh', 'height' : 178})
        e = Edge(s, 'R', t)

        src.query(f"CREATE {s}, {t}, {e}")

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
        q = "CALL db.idx.fulltext.createNodeIndex({label: 'L1', language: 'german', stopwords: ['a', 'b'] }, 'title', 'desc')"
        src.query(q)

        #-----------------------------------------------------------------------
        # create constraints
        #-----------------------------------------------------------------------

        # create node unique constraint
        create_unique_node_constraint(src, "L", "id")

        # add another unique constraint
        create_unique_node_constraint(src, "L", "id", "name", sync=True)

        # add a unique constraint which is destined to fail
        result = src.query("CREATE (:Actor {age: 10, name: 'jerry'}), (:Actor {age: 10, name: 'jerry'})")
        self.env.assertEquals(result.nodes_created, 2)

        create_unique_node_constraint(src, "Actor", "age", sync=True)
        c = get_constraint(src, "UNIQUE", "LABEL", "Actor", "age")
        self.env.assertEquals(c.status, "FAILED")

        # update entity
        q = "MATCH (n:L {id:1}) SET n.id = 2"
        result = src.query(q)
        self.env.assertEquals(result.properties_set, 1)

        # delete entity
        q = "MATCH (n:L {id:0}) DELETE n"
        result = src.query(q)
        self.env.assertEquals(result.nodes_deleted, 1)

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")

        # wait for index to sync in replica
        wait_for_indices_to_sync(replica)

        # make sure index is available on replica
        q = "MATCH (s:L {id:2}) RETURN s.name"
        plan = str(src.explain(q))
        replica_plan = str(replica.explain(q))
        env.assertIn("Index Scan", plan)
        env.assertEquals(replica_plan, plan)

        # issue query on both source and replica
        # make sure results are the same
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEquals(replica_result, result)

        # make sure node count on both primary and replica is the same
        q = "MATCH (n) RETURN count(n)"
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEquals(replica_result, result)

        # make sure nodes are in sync
        q = "MATCH (n) RETURN n ORDER BY n"
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEquals(replica_result, result)

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
        env.assertEquals(replica_result, result)

        # remove property
        q = "MATCH (s {id:2}) SET s.id = NULL RETURN s"
        result = src.query(q)
        env.assertEqual(result.properties_removed, 1)

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")

        q = "MATCH (s {id:2}) RETURN s"
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEqual(len(result), 0)
        env.assertEquals(replica_result, result)

        # make sure both primary and replica have the same set of indexes
        q = "CALL db.indexes() YIELD label, properties, language, stopwords, entitytype"
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEquals(replica_result, result)

        # drop fulltext index
        q = "CALL db.idx.fulltext.drop('L')"
        result = src.query(q)
        env.assertEquals(result.indices_deleted, 3)

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")
        
        # TODO: check if this sync is needed
        wait_for_indices_to_sync(src)
        wait_for_indices_to_sync(replica)

        # make sure both primary and replica have the same set of indexes
        q = "CALL db.indexes() YIELD label, properties, language, stopwords, entitytype"
        result = src.ro_query(q).result_set
        replica_result = replica.ro_query(q).result_set
        env.assertEquals(replica_result, result)

        # make sure both primary and replica have the same set of constraints
        origin_result = list_constraints(src)
        replica_result = list_constraints(replica)
        env.assertEquals(replica_result, origin_result)

        # drop constraint
        drop_unique_node_constraint(src, "L", "id")

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")

        # make sure both primary and replica have the same set of constraints
        origin_result = list_constraints(src)
        replica_result = list_constraints(replica)
        env.assertEquals(replica_result, origin_result)

        # drop failed constraint
        drop_unique_node_constraint(src, "Actor", "age")

        # the WAIT command forces master slave sync to complete
        source_con.execute_command("WAIT", "1", "0")

        # make sure both primary and replica have the same set of constraints
        origin_result = list_constraints(src)
        replica_result = list_constraints(replica)
        env.assertEquals(replica_result, origin_result)
