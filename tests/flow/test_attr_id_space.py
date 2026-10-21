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

        # enableDebugCommand: test02 reloads via DEBUG RELOAD, which redis refuses
        # by default.
        self.env, self.db = Env(env='oss', useSlaves=True, enableDebugCommand=True)

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
                    "— an effect's attribute id resolved to a different attribute")

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

    # ── schema id stability ──────────────────────────────────────────────────
    #
    # Attribute, label and relationship-type ids all travel as bare integers in a
    # GRAPH.EFFECT — there is no name beside them to check against. So an id has to
    # mean the same thing after a reload, and on a replica seeded by full sync, as it
    # did on the master that stamped it. #2457 was the case where that stopped being
    # true for attributes; labels and types ride the same way ("label id N out of
    # range" is what a replica reports when it isn't true for them).
    #
    # The ids are not directly observable from a client, but they are exactly the row
    # order of these three procedures: each iterates its id-ordered vector.

    # Deliberately not alphabetical, and not creation-sorted by name. If any of the
    # three procedures ever started sorting its output, the order assertion in
    # `_schema_order` would stop reflecting ids — these names make that visible
    # instead of silently weakening the test.
    LABELS = ["Zeta", "Alpha", "Mid"]
    TYPES = ["ZLINK", "ALINK", "MLINK"]
    ATTRS = ["zprop", "aprop", "mprop"]

    def _schema_order(self, graph, expect_creation_order=False):
        """The three schema vectors in id order, as the procedures report them."""
        def col(proc, field):
            return [r[0] for r in graph.ro_query(
                f"CALL {proc}() YIELD {field} RETURN {field}").result_set]

        order = {
            "labels": col("db.labels", "label"),
            "types": col("db.relationshipTypes", "relationshipType"),
            "attrs": col("db.propertyKeys", "propertyKey"),
        }

        if expect_creation_order:
            # Proves the row order really is id order rather than something sorted:
            # these names were created in an order alphabetical sorting would not
            # reproduce.
            self.env.assertEquals(order["labels"], self.LABELS,
                message="db.labels() must report ids in creation order, not sorted; "
                        f"got {order['labels']}")
            self.env.assertEquals(order["types"], self.TYPES,
                message="db.relationshipTypes() must report ids in creation order; "
                        f"got {order['types']}")
            # Attributes include the ones the edges carry, appended after the node
            # ones, so compare only the prefix this test controls.
            self.env.assertEquals(order["attrs"][:len(self.ATTRS)], self.ATTRS,
                message="db.propertyKeys() must report ids in creation order; "
                        f"got {order['attrs']}")
        return order

    def _build_schema(self, graph):
        """Register labels, types and attributes in a known, non-alphabetical order."""
        for i, label in enumerate(self.LABELS):
            graph.query(f"CREATE (:{label} {{{self.ATTRS[i]}: {i}}})")
        for i, rel in enumerate(self.TYPES):
            graph.query(f"MATCH (a:{self.LABELS[0]}), (b:{self.LABELS[1]}) "
                        f"CREATE (a)-[:{rel} {{w: {i}}}]->(b)")

    def test02_schema_ids_survive_a_reload(self):
        # The RDB carries names as flat, ordered lists and the loader rebuilds the
        # numbering by reading them in order. If the encoder and decoder ever
        # disagreed on that order, every id already stamped into an effect — or into
        # an entity's stored span — would resolve to a different name after a reload.
        graph = self.db.select_graph(GRAPH_ID + "_reload")
        self._build_schema(graph)
        before = self._schema_order(graph, expect_creation_order=True)

        self.env.dumpAndReload()

        after = self._schema_order(graph)
        self.env.assertEquals(after, before,
            message=f"schema ids shifted across a reload: before {before}, after "
                    f"{after}. An id is meaningless on its own, so a shift silently "
                    "repoints every effect and every stored span")
        # And the data still reads back through those ids.
        for i, label in enumerate(self.LABELS):
            self.env.assertEquals(
                graph.ro_query(f"MATCH (n:{label}) RETURN n.{self.ATTRS[i]}")
                    .result_set[0][0], i,
                message=f"{label}.{self.ATTRS[i]} did not survive the reload")
        graph.delete()

    def test03_schema_ids_match_on_a_replica_after_full_sync(self):
        # Same invariant across the wire rather than across a restart: a replica
        # builds its tables from the RDB the master sends, so master and replica must
        # agree on the numbering before any effect referencing an id arrives.
        #
        # Detaching and emptying the replica first is what makes this a full-sync
        # test. Written the obvious way it is not one: test01 leaves the replica
        # attached, so the schema created below reaches it through the live
        # replication stream — where the numbering is rebuilt by replaying
        # ADD_ATTRIBUTE in order and the RDB decoder never runs. Ablating the
        # decoder's ordering left that version of this test passing.
        graph = self.db.select_graph(GRAPH_ID + "_sync")
        replica_con = self.env.getSlaveConnection()

        replica_con.execute_command("REPLICAOF", "NO", "ONE")
        replica_con.execute_command("FLUSHALL")
        # `sync_full` counts syncs a server has *served*, so it lives on the master.
        master_con = self.env.getConnection()
        syncs_before = self._full_sync_count(master_con)

        self._build_schema(graph)
        master = self._schema_order(graph, expect_creation_order=True)

        # Nothing on the replica yet, so whatever it reports afterwards came from the
        # sync rather than from having been there all along.
        self.env.assertEquals(replica_con.execute_command("KEYS", "*"), [],
            message="replica must start empty for this to be a full-sync test")

        replica_con.execute_command("REPLICAOF", "localhost", str(self.env.port))
        self._wait_for_sync(replica_con)

        self.env.assertGreater(self._full_sync_count(master_con), syncs_before,
            message="no full sync was performed, so this test would be measuring the "
                    "effect-replay path instead of the RDB decoder")

        replica = self._schema_order(Graph(replica_con, GRAPH_ID + "_sync"))
        self.env.assertEquals(replica, master,
            message="replica disagrees with master on schema ids after full sync: "
                    f"master {master}, replica {replica}. Every subsequent effect "
                    "carries these ids with no name to verify against")
        graph.delete()

    @staticmethod
    def _full_sync_count(con):
        """`sync_full` from INFO stats — how many full syncs this server has served.

        Master-side: a replica serves none of its own, so reading this off the replica
        always returns 0 and the assertion below would be vacuous.
        """
        info = con.execute_command("INFO", "stats")
        if isinstance(info, dict):
            return int(info.get("sync_full", 0))
        for line in str(info).splitlines():
            if line.startswith("sync_full:"):
                return int(line.split(":", 1)[1])
        return 0
