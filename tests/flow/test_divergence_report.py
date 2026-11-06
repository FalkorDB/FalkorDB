import os

from common import *
from time import sleep

GRAPH_ID = "divergence_report"


class testDivergenceReport(FlowTestsBase):
    """The log a replica leaves behind when it refuses an effect.

    A forced resync repairs the replica and destroys the evidence with it: the
    payload is freed when the command returns, the graph it would not apply to
    is replaced wholesale, and the master keeps no record of what it sent. So
    whatever the log says at that moment is all anyone will ever have, which
    makes the contents of these lines a behaviour worth testing rather than a
    formatting detail.
    """

    def __init__(self):
        # replication timing doesn't play well with sanitizers
        if SANITIZER:
            Environment.skip(None)

        # Makes a replica writable, process-wide, and ends by making that
        # replica resync. On CI's shared services container that would
        # reconfigure replication for every other class in the cell.
        if os.getenv("FALKORDB_USE_SERVICE"):
            Environment.skip(None)

        self.env, self.db = Env(env='oss', useSlaves=True)

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
        self.env.assertTrue(
            False, message="replica never reached master_link_status:up")

    def _replica_log(self):
        name = self.env.envRunner._getFileName("slave", ".log")
        with open(f"{self.env.logDir}/{name}") as f:
            return f.read()

    def _wait_for_log(self, needle, timeout=15, since=0):
        """Wait for `needle` to appear *after* byte `since`.

           The replica's log is cumulative across the tests in this class, so a
           match left by an earlier one satisfies the wait immediately and every
           assertion afterwards reads the wrong payload. Passing the length of
           the log taken before the write is what keeps each case looking at its
           own report.
        """
        while timeout > 0:
            log = self._replica_log()
            if needle in log[since:]:
                return log[since:]
            sleep(0.5)
            timeout -= 0.5
        return self._replica_log()[since:]

    def _setup(self, seed=None):
        """A synced master/replica pair, on the effect path, optionally seeded.

           `seed` runs *before* the collision below, so the entities it makes
           exist on both sides: an update to a node created in the same commit
           folds into its CreateNode and would never surface as UpdateNode.
        """
        replica_con = self.env.getSlaveConnection()
        graph = self.db.select_graph(GRAPH_ID)
        # Nothing to force: v3 replicates every write as effects. `EFFECTS_THRESHOLD`
        # survives as a deprecated no-op — stored and echoed back by GRAPH.CONFIG,
        # never read for a decision — so setting it here would only look like it
        # mattered.
        self._wait_for_sync(replica_con)
        if seed is not None:
            graph.query(seed)
            self._wait_for_sync(replica_con)
        return graph, replica_con

    def _report(self, graph, replica_con, rogue, write):
        """Refuse one payload and return the lines the replica logged for it.

           The collision: the replica takes the next schema id for a name the
           master has never heard of, and the master's own new name then claims
           the same id, so its ADD_SCHEMA contradicts what this replica holds and
           the whole buffer is refused. **`write` therefore has to introduce a
           new label, type or attribute** — that is what collides. Which record
           fails does not matter: the report renders every record the payload
           decoded, which is why one collision shows the record under test.

           `write` is a query string, or a callable for a command.
        """
        replica_con.execute_command("CONFIG", "SET", "replica-read-only", "no")
        Graph(replica_con, GRAPH_ID).query(f"CREATE (:{rogue})")
        since = len(self._replica_log())
        if callable(write):
            write()
        else:
            graph.query(write)
        lines = [
            l for l in self._wait_for_log("Diverged payload", since=since).splitlines()
            if "Diverged payload" in l
        ]
        self.env.assertTrue(len(lines) > 0, message="nothing was reported")
        return "\n".join(lines)

    def _assert_record(self, blob, kind, *fragments):
        """The record is named, and its contents survived into the line.

           Both halves matter. The name alone would pass a rendering that
           dropped every field — and the fields are what someone matches against
           the query that ran on the master.
        """
        env = self.env
        line = next((l for l in blob.splitlines() if kind in l), None)
        env.assertTrue(line is not None, message=f"no {kind} record in:\n{blob}")
        if line is None:
            return
        for fragment in fragments:
            env.assertContains(fragment, line)

    # ------------------------------------------------------------------
    # The report's shape. Not per-record: these are properties of the log
    # itself, and they only need one payload to be true of.
    # ------------------------------------------------------------------

    def test01_the_report_carries_records_bytes_and_a_resync(self):
        env = self.env
        graph, replica_con = self._setup("CREATE (:Seed)")

        replica = Graph(replica_con, GRAPH_ID)
        env.assertEqual(replica.ro_query("MATCH (n) RETURN count(n)").result_set[0][0], 1)

        blob = self._report(graph, replica_con, "RogueShape",
                            "CREATE (:Diverges {marker: 'find me', n: 42})")

        # What it says: the record rendering is the half a reader compares
        # against the query that ran on the master.
        self._assert_record(blob, "CreateNode", "find me", "42")

        # What arrived: the bytes are the half that still works when the
        # decoding is the thing that is wrong.
        env.assertContains("bytes[0x0000] ", blob)

        # Redis truncates a log message at LOG_MAX_LEN, silently. A line that
        # reaches it is a line being cut off with nothing to say so.
        too_long = [l for l in blob.splitlines() if len(l) > 1024]
        env.assertEqual(too_long, [], message="log lines hit the redis truncation limit")

        # And the guard still did its job: the report is an addition to the
        # resync, not a replacement for it.
        env.assertContains("Forced full resync", self._replica_log())

    # ------------------------------------------------------------------
    # One record kind each. Every kind renders through the same `{record:?}`,
    # but each carries different fields, and a field that stops rendering is
    # invisible in every other kind's line.
    # ------------------------------------------------------------------

    def test02_add_label(self):
        graph, replica_con = self._setup()
        blob = self._report(graph, replica_con, "RogueAddLabel",
                            "CREATE (:LabelUnderTest)")
        self._assert_record(blob, "AddLabel", "LabelUnderTest")

    def test03_add_rel_type(self):
        graph, replica_con = self._setup()
        blob = self._report(graph, replica_con, "RogueAddRel",
                            "CREATE (:RelSrc)-[:TYPE_UNDER_TEST]->(:RelDst)")
        self._assert_record(blob, "AddRelType", "TYPE_UNDER_TEST")

    def test04_add_attribute(self):
        graph, replica_con = self._setup()
        blob = self._report(graph, replica_con, "RogueAddAttr",
                            "CREATE (:AttrHolder {attr_under_test: 1})")
        self._assert_record(blob, "AddAttribute", "attr_under_test")

    def test05_create_node(self):
        graph, replica_con = self._setup()
        blob = self._report(graph, replica_con, "RogueCreateNode",
                            "CREATE (:CreatedNode {tag: 'created', k: 11})")
        self._assert_record(blob, "CreateNode", "created", "11")

    def test06_create_edge(self):
        graph, replica_con = self._setup()
        blob = self._report(graph, replica_con, "RogueCreateEdge",
                            "CREATE (:EdgeSrc)-[:EDGE_MADE {weight: 12}]->(:EdgeDst)")
        # `relation_id` and the endpoint columns are this record's own fields;
        # no other kind carries src/dst.
        self._assert_record(blob, "CreateEdge", "relation_id", "src", "dst", "12")

    def test07_update_node(self):
        graph, replica_con = self._setup("CREATE (:UpdSeed {k: 1})")
        blob = self._report(
            graph, replica_con, "RogueUpdNode",
            "MATCH (n:UpdSeed {k: 1}) SET n.updated_to = 'newvalue' CREATE (:UpdCollide)")
        self._assert_record(blob, "UpdateNode", "newvalue")

    def test08_update_edge(self):
        graph, replica_con = self._setup(
            "CREATE (:UeSrc)-[:UE_REL {w: 1}]->(:UeDst)")
        blob = self._report(
            graph, replica_con, "RogueUpdEdge",
            "MATCH ()-[r:UE_REL]->() SET r.w = 77 CREATE (:UeCollide)")
        self._assert_record(blob, "UpdateEdge", "relation_id", "77")

    def test09_set_labels(self):
        graph, replica_con = self._setup("CREATE (:SlSeed {k: 1})")
        # The label being set is itself new, so it is the collision.
        blob = self._report(graph, replica_con, "RogueSetLabels",
                            "MATCH (n:SlSeed {k: 1}) SET n:LabelAdded")
        self._assert_record(blob, "SetLabels", "labels")

    def test10_remove_labels(self):
        graph, replica_con = self._setup("CREATE (:RlSeed:LabelToDrop {k: 1})")
        blob = self._report(
            graph, replica_con, "RogueRemoveLabels",
            "MATCH (n:RlSeed {k: 1}) REMOVE n:LabelToDrop CREATE (:RlCollide)")
        self._assert_record(blob, "RemoveLabels", "labels")

    def test11_delete_node(self):
        graph, replica_con = self._setup("CREATE (:DnSeed {k: 1})")
        blob = self._report(
            graph, replica_con, "RogueDeleteNode",
            "MATCH (n:DnSeed {k: 1}) DELETE n CREATE (:DnCollide)")
        self._assert_record(blob, "DeleteNode", "ids")

    def test12_delete_edge(self):
        graph, replica_con = self._setup("CREATE (:DeSrc)-[:DE_REL]->(:DeDst)")
        blob = self._report(
            graph, replica_con, "RogueDeleteEdge",
            "MATCH ()-[r:DE_REL]->() DELETE r CREATE (:DeCollide)")
        self._assert_record(blob, "DeleteEdge", "relation_id", "src", "dst")

    def test13_create_index(self):
        graph, replica_con = self._setup()
        blob = self._report(graph, replica_con, "RogueCreateIdx",
                            "CREATE INDEX FOR (n:Indexed) ON (n.title)")
        self._assert_record(blob, "CreateIndex", "Indexed", "title")

    def test14_drop_index(self):
        graph, replica_con = self._setup(
            "CREATE INDEX FOR (n:DroppedIdx) ON (n.title)")
        # A drop introduces no schema, so the master's payload claims no id the
        # replica could already hold and the collision above cannot fire on one.
        # Dropping it on the replica first is what makes the master's drop
        # unapplicable there.
        replica_con.execute_command("CONFIG", "SET", "replica-read-only", "no")
        Graph(replica_con, GRAPH_ID).query(
            "DROP INDEX FOR (n:DroppedIdx) ON (n.title)")
        since = len(self._replica_log())
        graph.query("DROP INDEX FOR (n:DroppedIdx) ON (n.title)")
        blob = "\n".join(
            l for l in self._wait_for_log("Diverged payload", since=since).splitlines()
            if "Diverged payload" in l)
        self._assert_record(blob, "DropIndex", "DroppedIdx", "title")

    def test15_create_constraint(self):
        graph, replica_con = self._setup()
        con = self.env.getConnection()
        # MANDATORY rather than UNIQUE: UNIQUE is refused without a supporting
        # range index, and creating one first would spend the collision on the
        # index instead.
        blob = self._report(
            graph, replica_con, "RogueCreateCon",
            lambda: con.execute_command(
                "GRAPH.CONSTRAINT", "CREATE", GRAPH_ID,
                "MANDATORY", "NODE", "Constrained", "PROPERTIES", "1", "email"))
        # `status` is the field only a create carries, and the one thing v3
        # sends that C's v2 did not.
        self._assert_record(blob, "CreateConstraint", "Constrained", "email", "status")

    def test16_drop_constraint(self):
        graph, replica_con = self._setup()
        con = self.env.getConnection()
        con.execute_command(
            "GRAPH.CONSTRAINT", "CREATE", GRAPH_ID,
            "MANDATORY", "NODE", "Dropped", "PROPERTIES", "1", "gone")
        self._wait_for_sync(replica_con)

        # As in test14: the drop has to fail on the replica for its own reason.
        replica_con.execute_command("CONFIG", "SET", "replica-read-only", "no")
        replica_con.execute_command(
            "GRAPH.CONSTRAINT", "DROP", GRAPH_ID,
            "MANDATORY", "NODE", "Dropped", "PROPERTIES", "1", "gone")
        since = len(self._replica_log())
        con.execute_command(
            "GRAPH.CONSTRAINT", "DROP", GRAPH_ID,
            "MANDATORY", "NODE", "Dropped", "PROPERTIES", "1", "gone")
        blob = "\n".join(
            l for l in self._wait_for_log("Diverged payload", since=since).splitlines()
            if "Diverged payload" in l)
        # A drop mirrors the create *without* the status.
        self._assert_record(blob, "DropConstraint", "Dropped", "gone")
