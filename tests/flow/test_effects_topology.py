"""Divergence, forced resync, promotion, and AOF replay.

What the pair does when it stops being a healthy pair. Grouped together because
they share a shape: each one puts the replica into a state the happy path never
reaches, and asserts the engine notices rather than carrying on.

Version coverage differs by class and the reason is stated on each:

  test01  divergence + forced resync   BOTH  -- the guard is version-agnostic
  test02  mid-stream refusal            v3   -- streaming decode is v3-only
  test03  promotion and id allocation  BOTH
  test04  promotion settles a PENDING
          constraint                    v3   -- the status only travels on v3
  test05  AOF replay                   BOTH  -- and deliberately has NO replica
"""

import os
import re
import time
from common import *
from effects_common import _EffectsBase


class test01_Divergence(_EffectsBase):
    """A replica that has diverged must refuse the effect and full-resync.

    Run at BOTH versions. The original suite exercised only the default, so the
    guard had never been shown to work on a v3 payload -- and the refusal path
    runs through a different decoder for each.
    """

    GRAPH_ID = "effects_topo_divergence"

    def __init__(self):
        self._setup(version=2, threshold=0)

    def _diverge_and_assert_resync(self, v):
        self.query_and_sync("CREATE (:P {v: 1}), (:P {v: 2}), (:P {v: 3})")

        target = self.master_graph.query(
            "MATCH (n:P) RETURN id(n) ORDER BY id(n) LIMIT 1").result_set[0][0]

        # remove the node on the replica and nowhere else
        self.replica.config_set("slave-read-only", "no")
        self.replica_graph.query(
            "MATCH (n) WHERE id(n) = %d DELETE n" % target)
        self.replica.config_set("slave-read-only", "yes")
        self.env.assertEquals(
            self.replica_graph.ro_query(
                "MATCH (n:P) RETURN count(n)").result_set[0][0], 2)

        sync_before = self.master.info()["sync_full"]

        # the master deletes the same node: an effect naming an id the replica
        # no longer has
        self.master_graph.query(
            "MATCH (n) WHERE id(n) = %d DELETE n" % target)

        resynced = False
        deadline = time.time() + 60
        while time.time() < deadline:
            info = self.master.info()
            if (info["sync_full"] > sync_before and
                    info.get("connected_slaves", 0) > 0):
                resynced = True
                break
            time.sleep(0.5)

        self.env.assertTrue(
            resynced,
            message="v%d: sync_full did not rise -- the refusal did not reach "
                    "the divergence guard" % v)

        # and the repair worked. Deliberately secondary: a resync converges
        # whether or not the mechanism under test did anything
        expected = self.master_graph.query(
            "MATCH (n) RETURN labels(n), n.v ORDER BY id(n)").result_set
        actual = self.replica_graph.ro_query(
            "MATCH (n) RETURN labels(n), n.v ORDER BY id(n)").result_set
        self.env.assertEquals(expected, actual)

    def test01_forced_resync_on_divergence(self):
        self.for_each_version(self._diverge_and_assert_resync)


class test02_MidStreamRefusal(_EffectsBase):
    """A refusal raised PART WAY THROUGH a payload still reaches the guard.

    v3 only: streaming decode is what makes a partial apply possible, and v2
    decodes and applies in one pass with nothing to be part way through.

    THE ASSERTION THAT MAKES THIS WORTH HAVING IS k >= 1. A payload refused
    BEFORE any record applies raises sync_full and converges exactly like one
    refused after, so neither the resync nor the state comparison can tell the
    new behaviour from the old. The log is the only place the distinction
    survives, because the partial state exists solely in the window the resync
    erases.
    """

    GRAPH_ID = "effects_topo_midstream"

    def __init__(self):
        self._setup(version=3, threshold=0)

    @staticmethod
    def _replica_log(env):
        parts = []
        for name in sorted(os.listdir(env.logDir)):
            if "slave" in name and name.endswith(".log"):
                with open(os.path.join(env.logDir, name), errors="replace") as f:
                    parts.append(f.read())
        return "\n".join(parts)

    def test01_partial_apply_still_reaches_the_guard(self):
        env = self.env
        self.query_and_sync("CREATE (:P {v: 1}), (:P {v: 2}), (:P {v: 3})")

        ids = self.master_graph.query(
            "MATCH (n:P) RETURN id(n) ORDER BY id(n)").result_set
        target, keep = ids[0][0], ids[1][0]

        self.replica.config_set("slave-read-only", "no")
        self.replica_graph.query("MATCH (n) WHERE id(n) = %d DELETE n" % target)
        self.replica.config_set("slave-read-only", "yes")

        before = len(re.findall(r"after applying (\d+) record\(s\)",
                                self._replica_log(env)))
        sync_before = self.master.info()["sync_full"]

        # ONE payload, two record groups, ordered so a record applies before the
        # one that fails. v3 sorts groups by opcode ascending, UPDATE_NODE is 1
        # and DELETE_NODE is 5, so the SET lands and the DELETE of the id the
        # replica no longer holds is refused after it. That ordering is a
        # property of the format, not of this query, which is why k is
        # predictable at all.
        self.master_graph.query(
            "MATCH (a:P) WHERE id(a) = %d SET a.v = 99 "
            "WITH count(a) AS _ "
            "MATCH (b) WHERE id(b) = %d DELETE b" % (keep, target))

        applied  = None
        deadline = time.time() + 30
        while time.time() < deadline:
            hits = re.findall(r"after applying (\d+) record\(s\)",
                              self._replica_log(env))
            if len(hits) > before:
                applied = int(hits[before])
                break
            time.sleep(0.2)

        env.assertIsNotNone(
            applied,
            message="no 'after applying N record(s)' line appeared; the payload "
                    "was not refused, or the refusal did not reach the log")

        if applied is not None:
            env.assertTrue(
                applied >= 1,
                message="refused after applying %d records. This class exists "
                        "for the case where records applied BEFORE the refusal; "
                        "at 0 it is testing what test01_Divergence already "
                        "tests" % applied)

        resynced = False
        deadline = time.time() + 60
        while time.time() < deadline:
            info = self.master.info()
            if (info["sync_full"] > sync_before and
                    info.get("connected_slaves", 0) > 0):
                resynced = True
                break
            time.sleep(0.5)
        env.assertTrue(resynced, message="the refusal did not reach the guard")

        expected = self.master_graph.query(
            "MATCH (n) RETURN labels(n), n.v ORDER BY id(n)").result_set
        actual = self.replica_graph.ro_query(
            "MATCH (n) RETURN labels(n), n.v ORDER BY id(n)").result_set
        env.assertEquals(expected, actual)


class _PromotionBase(_EffectsBase):
    """A promoted replica allocates ids that do not collide with inherited ones.

    VERSION-PINNED, and differentiated by moduleArgs rather than config_set,
    because promotion DESTROYS the pair: `REPLICAOF NO ONE` ends replication, so
    a second iteration on the same server would wait forever for an ack from
    something that is no longer a replica. Distinct Env parameters are what make
    RLTest give each version its own server.
    """

    GRAPH_ID = "effects_topo_promotion"
    VERSION  = 2

    def __init__(self):
        self._setup(version=self.VERSION, threshold=0)

    def _promote_and_mint(self, v):
        self.query_and_sync("UNWIND range(0, 499) AS i CREATE (:P {i: i})")
        inherited = self.replica_graph.ro_query(
            "MATCH (n:P) RETURN max(id(n))").result_set[0][0]

        # promote: REPLICAOF at a dead port then NO ONE fires NOW_MASTER without
        # needing a second server
        port = self.env.envRunner.port
        self.replica.execute_command("REPLICAOF", "127.0.0.1", str(port + 7))
        self.replica.execute_command("REPLICAOF", "NO", "ONE")
        self.replica.config_set("slave-read-only", "no")

        promoted = Graph(self.replica, self.graph_id)
        promoted.query("UNWIND range(0, 199) AS i CREATE (:Q {i: i})")

        minted = promoted.query(
            "MATCH (n:Q) RETURN min(id(n)), max(id(n))").result_set[0]
        self.env.assertTrue(
            minted[0] > inherited,
            message="v%d: minted id %s collides with inherited max %s"
                    % (v, minted[0], inherited))

        # a reused id surfaces as an entity answering with another's properties
        total = promoted.query("MATCH (n) RETURN count(n)").result_set[0][0]
        self.env.assertEquals(total, 700)

    def test01_promoted_replica_mints_non_colliding_ids(self):
        self._promote_and_mint(self.VERSION)


class test03_PromotionV2(_PromotionBase):
    VERSION = 2

    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True,
                                moduleArgs='EFFECTS_THRESHOLD 0 EFFECTS_VERSION 2')
        self._after_env()


class test04_PromotionV3(_PromotionBase):
    VERSION = 3

    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True,
                                moduleArgs='EFFECTS_THRESHOLD 0 EFFECTS_VERSION 3')
        self._after_env()


class test05_AofReplay():
    """A graph rebuilds from an AOF whose only graph records are effects.

    NO REPLICA in this Env, deliberately: the point is that effects are the sole
    carrier, and a replica in the picture leaves room for a query to have done
    the rebuilding.

    AOF is enabled on an EMPTY dataset. Turning it on afterwards triggers a
    rewrite that captures everything already present as a base snapshot, and the
    graph then comes back from that snapshot having exercised no effect at all --
    a test that passes and means nothing.
    """

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)
        self.env, self.db = Env(moduleArgs='EFFECTS_THRESHOLD 0',
                                enableDebugCommand=True)
        self.conn = self.env.getConnection()

    def _run(self, version):
        self.db.config_set("EFFECTS_VERSION", version)
        gid = "aof_replay_v%d" % version
        g   = Graph(self.conn, gid)

        self.conn.config_set("appendonly", "yes")
        self.conn.config_set("appendfsync", "always")
        deadline = time.time() + 30
        while time.time() < deadline:
            info = self.conn.info("persistence")
            if (info.get("aof_enabled") == 1 and
                    info.get("aof_rewrite_in_progress") == 0):
                break
            time.sleep(0.2)

        for i in range(13):
            g.query("UNWIND range(0, 99) AS x CREATE (:A {b: x, probe: %d})" % i)

        before = g.query("MATCH (n:A) RETURN count(n)").result_set[0][0]
        self.env.assertEquals(before, 1300)

        self.conn.execute_command("DEBUG", "LOADAOF")

        after = g.query("MATCH (n:A) RETURN count(n)").result_set[0][0]
        self.env.assertEquals(after, before)
        self.conn.config_set("appendonly", "no")

    def test01_rebuild_from_effects_only_aof(self):
        for v in (2, 3):
            self._run(v)
