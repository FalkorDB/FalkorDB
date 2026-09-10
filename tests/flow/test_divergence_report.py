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

        # Forces EFFECTS_THRESHOLD to 0 and makes a replica writable, both
        # process-wide, and ends by making that replica resync. On CI's shared
        # services container that would reconfigure replication for every other
        # class in the cell.
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

    def _wait_for_log(self, needle, timeout=15):
        while timeout > 0:
            log = self._replica_log()
            if needle in log:
                return log
            sleep(0.5)
            timeout -= 0.5
        return self._replica_log()

    def test01_a_refused_effect_is_logged_as_records_and_as_bytes(self):
        env = self.env
        replica_con = env.getSlaveConnection()
        graph = self.db.select_graph(GRAPH_ID)

        # Force the effect path. At the default threshold a write this small
        # replicates verbatim, and a verbatim query re-runs on the replica
        # rather than being decoded, so there would be no payload to report.
        self.db.config_set("EFFECTS_THRESHOLD", 0)

        graph.query("CREATE (:Seed)")
        self._wait_for_sync(replica_con)

        replica = Graph(replica_con, GRAPH_ID)
        synced = replica.ro_query("MATCH (n) RETURN count(n)").result_set[0][0]
        env.assertEquals(synced, 1)

        # Manufacture the divergence: a label the master does not know about,
        # taking the schema id the master is about to hand out. Writing to a
        # replica is exactly the operator mistake this guard exists to catch.
        replica_con.execute_command("CONFIG", "SET", "replica-read-only", "no")
        replica.query("CREATE (:OnlyOnTheReplica)")

        # The master's next label reuses that id, so its ADD_SCHEMA contradicts
        # what this replica already has.
        graph.query("CREATE (:Diverges {marker: 'find me', n: 42})")

        log = self._wait_for_log("Diverged payload")
        env.assertContains("Diverged payload", log)

        reported = [l for l in log.splitlines() if "Diverged payload" in l]
        env.assertTrue(
            len(reported) > 0,
            message=f"refused the effect but reported no payload; tail:\n{log[-2000:]}")

        # What it says. The record rendering is the half a reader compares
        # against the query that ran on the master, so the values that write
        # actually carried have to survive into it.
        env.assertTrue(
            any("CreateNode" in l for l in reported),
            message="no record was named:\n" + "\n".join(reported))
        env.assertTrue(
            any("find me" in l and "42" in l for l in reported),
            message="the record was named but its values were not, so the log "
                    "cannot be matched to a write:\n" + "\n".join(reported))

        # What arrived. The bytes are the half that still works when the
        # decoding is the thing that is wrong, and the half to diff against the
        # master's own record of what it sent.
        env.assertTrue(
            any("bytes[0x0000] " in l for l in reported),
            message="no raw bytes:\n" + "\n".join(reported))

        # Redis truncates a log message at LOG_MAX_LEN, silently. A line that
        # reaches it is a line being cut off with nothing to say so.
        too_long = [l for l in reported if len(l) > 1024]
        env.assertEqual(
            too_long, [],
            message=f"{len(too_long)} log lines hit the redis truncation limit")

        # And the guard still did its job: the report is an addition to the
        # resync, not a replacement for it.
        env.assertContains("Forced full resync", log)
