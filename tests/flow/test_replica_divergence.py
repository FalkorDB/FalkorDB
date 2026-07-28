from common import *
import time

GRAPH_ID = "replica_divergence"

# verifies that when a replicated GRAPH.EFFECT fails to apply on a replica
# because the replica has diverged from the master (e.g. an entity the
# effect targets no longer exists locally), the replica:
#   1. logs the divergence
#   2. forces a full resync with the master (not a partial PSYNC CONTINUE)
#   3. ends up with a dataset identical to the master's again


class testReplicaDivergence():
    def __init__(self):
        # replication timing doesn't play well with Valgrind/sanitizers
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        self.env, self.db = Env(env='oss', useSlaves=True)

    def test_forced_full_resync_on_effect_divergence(self):
        env = self.env
        master = env.getConnection()
        replica = env.getSlaveConnection()

        master_graph = Graph(master, GRAPH_ID)
        replica_graph = Graph(replica, GRAPH_ID)

        # force effects-based replication, so a small single-node delete
        # replicates as GRAPH.EFFECT rather than the raw GRAPH.QUERY
        self.db.config_set("EFFECTS_THRESHOLD", 0)

        # create a small graph and let the replica catch up
        master_graph.query("CREATE (:P {v: 1}), (:P {v: 2}), (:P {v: 3})")
        master.execute_command("WAIT", "1", "0")

        # pick a node we'll remove directly on the replica only, simulating
        # real divergence (e.g. a bug, manual intervention, or an earlier
        # undetected inconsistency) while the master keeps it
        res = master_graph.query(
            "MATCH (n:P) RETURN id(n) ORDER BY id(n) LIMIT 1")
        target_id = res.result_set[0][0]

        # allow direct writes on the replica and remove the node there only
        replica.config_set("slave-read-only", "no")
        replica_graph.query(f"MATCH (n) WHERE id(n) = {target_id} DELETE n")
        replica.config_set("slave-read-only", "yes")

        # sanity check: replica and master have diverged
        # (GRAPH.RO_QUERY, since the replica is read-only again by now)
        replica_count = replica_graph.ro_query(
            "MATCH (n:P) RETURN count(n)").result_set[0][0]
        env.assertEquals(replica_count, 2)

        # snapshot the master's full-sync counter before triggering the
        # divergence, so we can later confirm a FULLRESYNC actually
        # happened, not merely a reconnect that continues the stream
        sync_full_before = master.info()["sync_full"]

        # on the master, delete the very same node - this replicates as a
        # GRAPH.EFFECT DELETE_NODE effect targeting an id that no longer
        # exists on the replica
        master_graph.query(f"MATCH (n) WHERE id(n) = {target_id} DELETE n")

        # wait for the replica to detect the divergence, force a
        # REPLICAOF NO ONE / REPLICAOF cycle, and complete a fresh full
        # resync with the master
        deadline = time.time() + 30
        resynced = False
        while time.time() < deadline:
            info = master.info()
            if (info["sync_full"] > sync_full_before and
                    info["connected_slaves"] >= 1):
                resynced = True
                break
            time.sleep(0.5)

        env.assertTrue(resynced)

        # the replica's dataset should now match the master exactly again,
        # the earlier direct deletion included, since the full resync
        # replaced its entire dataset
        master.execute_command("WAIT", "1", "5000")

        expected = master_graph.query(
            "MATCH (n:P) RETURN id(n) ORDER BY id(n)").result_set
        actual = replica_graph.ro_query(
            "MATCH (n:P) RETURN id(n) ORDER BY id(n)").result_set
        env.assertEquals(expected, actual)

