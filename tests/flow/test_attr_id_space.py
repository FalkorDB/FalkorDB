import os

from common import *
from time import sleep

GRAPH_ID = "attr_id_space"


class testAttributeIdSpace(FlowTestsBase):
    def __init__(self):
        # replication timing doesn't play well with sanitizers
        if SANITIZER:
            Environment.skip(None)

        # This test repoints the replica with REPLICAOF and forces EFFECTS_THRESHOLD to 0,
        # both of which are process-wide. On CI's shared services container that would
        # reconfigure replication and effect routing for every other class in the cell, so
        # skip there rather than corrupt them.
        #
        # The classifier cannot route this for us on `main`: `useSlaves` is explicitly not
        # spawn-forcing, and the behavioural opt-out (`SPAWN_ONLY_FILES` in
        # tests/flow/test_matrix_split.py) arrives with #2450. Once that lands, add this
        # file there and drop the skip.
        if os.getenv("FALKORDB_USE_SERVICE"):
            Environment.skip(None)

        self.env, self.db = Env(env='oss', useSlaves=True)

    def test01_effect_after_full_sync_lands_on_the_right_attribute(self):
        # Regression for #2457.
        #
        # A property effect carries a bare attribute id with no name beside it. The id is
        # only meaningful against a dictionary, and the two sides used to build theirs
        # differently: a master that had never been reloaded numbered node and
        # relationship attributes in separate spaces, while a replica seeded by full sync
        # got the single unified list the RDB carries. The same id then meant a different
        # attribute on each side, so the value landed on the wrong one and the intended
        # one stayed stale — silently, and healed by any resync.
        #
        # Three things have to line up for it to bite, and all three are arranged here:
        #
        #   1. node attributes exist *before* the relationship attribute, so a per-store
        #      numbering would restart and collide;
        #   2. the replica is attached afterwards, so its dictionary comes from an RDB
        #      full sync rather than from replayed ADD_ATTRIBUTE effects;
        #   3. the write is replicated as an effect, not verbatim.
        env = self.env
        master = env.getConnection()
        replica_con = env.getSlaveConnection()
        graph = self.db.select_graph(GRAPH_ID)
        # Read the replica through the same parsed client the master uses, or the two
        # sides come back in different shapes (a dict vs the raw reply string) and any
        # comparison between them is meaningless.
        replica = Graph(replica_con, GRAPH_ID)

        # (3): force the effect path. At the 300us default, whether a write replicates
        # verbatim or as an effect depends on timing — and the verbatim path re-resolves
        # by name on the replica, which masks this entirely.
        self.db.config_set("EFFECTS_THRESHOLD", 0)

        # (1) node attributes first, then a relationship-only one.
        graph.query("CREATE (:N {a: 1, b: 2})")
        graph.query("CREATE (:N)-[:R {since: 1900}]->(:N)")

        # (2) attach now, so the replica is seeded by a full sync.
        replica_con.execute_command("REPLICAOF", "localhost", str(env.port))
        self._wait_for_sync(replica_con)

        # Sanity: the full sync itself is consistent. If this fails the test is not
        # exercising what it claims — the divergence below would predate the effect.
        synced = replica.ro_query("MATCH ()-[r:R]->() RETURN r.since").result_set[0][0]
        env.assertEquals(synced, 1900,
            message="full sync did not carry the edge property; the rest of this "
                    "test would be measuring the wrong thing")

        # The write whose effect the replica must interpret with the same numbering.
        graph.query("MATCH ()-[r:R]->() SET r.since = 9999")
        sleep(1)

        master_props = graph.query(
            "MATCH ()-[r:R]->() RETURN properties(r)").result_set[0][0]
        replica_props = replica.ro_query(
            "MATCH ()-[r:R]->() RETURN properties(r)").result_set[0][0]

        env.assertEquals(replica_props, master_props,
            message=f"replica diverged: master {master_props}, replica {replica_props} "
                    f"— an effect's attribute id resolved to a different attribute")

        # State the failure directly too, so a regression names itself rather than
        # showing up as a dict comparison.
        stale = replica.ro_query(
            "MATCH ()-[r:R]->() WHERE r.since = 9999 RETURN count(r)").result_set[0][0]
        env.assertEquals(stale, 1,
            message="the replica's edge did not receive the new `since` value")

        wrong = replica.ro_query(
            "MATCH ()-[r:R]->() WHERE r.a IS NOT NULL RETURN count(r)").result_set[0][0]
        env.assertEquals(wrong, 0,
            message="the value landed on `a` — a node attribute that should not exist "
                    "on an edge — which is the #2457 id-space collision")

    def _wait_for_sync(self, replica, timeout=30):
        while timeout > 0:
            info = replica.execute_command("INFO", "replication")
            if isinstance(info, dict):
                up = info.get("master_link_status") == "up"
            else:
                up = "master_link_status:up" in str(info)
            if up:
                sleep(0.5)
                return
            sleep(0.5)
            timeout -= 0.5
        raise RuntimeError("replica never reached master_link_status:up")
