"""Effects v3 -- the classes that take the primary/replica pair APART and put it back: a forced resync, a restarted primary, an AOF replay, a promotion.

This file is the reason the others do not need to worry about topology. It is also the only one with classes that skip under CI's shared-services mode, because two of them restart the primary through `env.envRunner`, which cannot restart a Docker service container.

See `effects_v3_common.py` for the shared fixture and why these are split.
"""

import time


from common import *
from constraint_utils import (create_unique_node_constraint)
from index_utils import (create_node_range_index, wait_for_indices_to_sync)

from effects_v3_common import MONITOR_MARK_KEY, _EffectsV3Base


class testEffectsV3_06c_DivergenceForcesResync(_EffectsV3Base):
    """A replica that cannot apply an effect must repair itself, not carry on.

    v3 detects far more divergence than v2 did — thirteen distinct checks
    against almost none — and detection without repair is worse than useless:
    Redis does not break a replication link over a module command's error
    reply, so the offset advances and the replica keeps serving data it has
    already proved wrong.

    The fix is C's: discard the cached replication state with `REPLICAOF NO
    ONE` so the reconnect cannot be satisfied by a partial resync, then
    reattach. `sync_full` on the master is what distinguishes that from a mere
    reconnect.
    """

    GRAPH_ID = "effects_v3_divergence"

    def __init__(self):
        self._setup()

    def test01_a_refused_effect_forces_a_full_resync(self):
        self.set_effects_config()
        self.query_and_sync("CREATE (:P {v: 1}), (:P {v: 2}), (:P {v: 3})")

        target = self.master_graph.ro_query(
            "MATCH (n:P) RETURN id(n) ORDER BY id(n) LIMIT 1").result_set[0][0]

        # Diverge the replica behind the master's back — a stand-in for the
        # bug or the earlier undetected inconsistency this guard exists for.
        self.replica.config_set("slave-read-only", "no")
        Graph(self.replica, self.GRAPH_ID).query(
            f"MATCH (n) WHERE id(n) = {target} DELETE n")
        self.replica.config_set("slave-read-only", "yes")
        self.env.assertEqual(
            self.replica_graph.ro_query("MATCH (n:P) RETURN count(n)").result_set,
            [[2]],
            message="the replica must actually be diverged for this to prove anything")

        full_before = self.master.info()["sync_full"]

        # The refusal this test is named for, counted before and after. Without
        # this the test passes with `refuse_recycled` deleted outright: the
        # id-space counter cross-check catches the same divergence independently
        # and resyncs, so `sync_full` moves either way and the assertion below
        # cannot tell which check fired. Measured — with `refuse_recycled`
        # stubbed to `Ok(())` the whole 73-test flow file stayed green, and only
        # a unit test noticed. Defence in depth is good; a test that documents
        # one mechanism and pins another is not.
        recycled_before = self.replica_log_count(b"already in the recycle bin")
        failures_before = self.effect_failures()

        # Deleting the same node on the master ships a DELETE_NODE naming an id
        # the replica no longer holds. v3 refuses it — "already in the recycle
        # bin" — which is exactly the detection that used to end in a shrug.
        self.master_graph.query(f"MATCH (n) WHERE id(n) = {target} DELETE n")

        deadline = time.time() + 60
        while time.time() < deadline:
            if self.master.info()["sync_full"] > full_before:
                break
            time.sleep(0.1)
        else:
            raise AssertionError(
                f"no full resync within 60s; sync_full still {full_before}")

        # And the resync actually repaired it, rather than merely happening.
        self.wait_for_replica_link()
        deadline = time.time() + 60
        while time.time() < deadline:
            m = self.master_graph.ro_query("MATCH (n:P) RETURN count(n)").result_set
            r = self.replica_graph.ro_query("MATCH (n:P) RETURN count(n)").result_set
            if m == r:
                break
            time.sleep(0.1)
        self.assert_agree("MATCH (n:P) RETURN count(n)", [[2]])
        self.assert_graph_eq()

        # Assert the strongest thing this mode can see. With a readable log that
        # is the MECHANISM — which check refused the buffer — and it is the
        # assertion this test exists for. CI's services mode has no readable
        # log (`env.log_path` is None), so there it falls back to the outcome:
        # a GRAPH.EFFECT was refused at all. That is weaker and says so, rather
        # than passing vacuously, which is what returning 0 used to do.
        recycled_after = self.replica_log_count(b"already in the recycle bin")
        if recycled_after is None:
            self.env.assertGreater(
                self.effect_failures(), failures_before,
                message="no GRAPH.EFFECT was refused on the replica, so nothing "
                        "detected the divergence. (Server logs are unreadable in "
                        "this mode, so which check refused it is unchecked here — "
                        "run locally or in spawn mode to pin the mechanism.)")
        else:
            self.env.assertGreater(
                recycled_after, recycled_before,
                message="the resync happened, but not for the reason this test "
                        "names: no DELETE_NODE was refused for an id in the "
                        "recycle bin. Some other divergence check fired instead")

    def test02_a_client_sent_effect_does_not_force_a_resync(self):
        # The guard keys off the command being replayed, not off the failure.
        # `GRAPH.EFFECT` is a payload any client can send, so treating a
        # malformed one as divergence would hand anyone a way to resync — or
        # shut down — a replica at will. C guards this path unconditionally;
        # this is a deliberate difference.
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]

        # A v3 buffer whose opcode is nonsense: refused by the decoder.
        try:
            self.master.execute_command(
                "GRAPH.EFFECT", self.GRAPH_ID, b"\x03\x00\xff\xff\xff\xff")
            self.env.assertTrue(False, message="a malformed effect must be refused")
        except ResponseError:
            pass

        # Nothing was scheduled: no resync, and the server is still here.
        time.sleep(1)
        self.env.assertEqual(self.master.info()["sync_full"], full_before)
        self.env.assertEqual(self.master.ping(), True)
        self.env.assertEqual(self.replica.ping(), True)


class testEffectsV3_06d_FirstReplicaAttach(_EffectsV3Base):
    """A write that lands while a replica is doing its first full sync must
    still reach that replica.

    `REPLICATION_CONSUMERS` (`src/graph_core.rs`) is a process-global sticky
    flag, and `execute_query_write` reads it to decide whether to build an
    effects buffer at all. Nothing latches it until Redis fires
    `ReplicaChange`, and Redis fires that from `replicaPutOnline` — *after* the
    snapshot has been delivered. So every write between the sync's fork and the
    replica coming online runs with `build_effects` false and produces no
    buffer.

    That used to be harmless: `replicate_effects` fell back to
    `ctx.replicate("GRAPH.QUERY", query)` when there was no buffer, so the
    write still propagated, as a replay. The fallback went with v2. Now nothing
    is propagated, and nothing notices: no effect ever arrives to be refused,
    so the divergence guard never runs, the link reports healthy, and the
    replica is permanently short one write.

    The window is the fork's snapshot, so `rdb-key-save-delay` widens it and
    `repl-diskless-sync-delay 0` stops the master idling *before* the fork —
    a write in that earlier window would be captured by the snapshot and prove
    nothing.

    Needs a master process that has never had a replica, which is why it
    restarts one: the flag is never cleared. That is also why it runs late and
    puts the topology back in a `finally`.
    """

    GRAPH_ID = "effects_v3_first_attach"

    # Per key, in microseconds, applied inside the forked child. One graph key,
    # so this is the whole width of the window.
    SAVE_DELAY_US = 3_000_000

    def __init__(self):
        # `_fresh_master` restarts the master through
        # `env.envRunner.stopEnv`/`startEnv`, which cannot restart a Docker
        # service container. Under CI's shared services topology the restart
        # is silently a no-op, so `REPLICATION_CONSUMERS` stays latched from
        # whichever class ran first and the premise of this test -- a master
        # process that has never had a replica -- cannot be established.
        #
        # Skipped rather than relaxed. The three `sync_full` assertions are
        # absolute (0, then 1, then still 1) precisely because they are the
        # thing that detects an unlatched flag, and they did their job: on
        # the shared container they read 2, 3 and 3 and failed on all three
        # CI shards. Rewriting them as deltas would make this class green
        # everywhere while the mechanism it exists to test never engages,
        # which is worse than not running it.
        if os.getenv("FALKORDB_USE_SERVICE"):
            Environment.skip(None)

        self._setup()

    def test01_a_write_during_the_first_full_sync_is_not_lost(self):
        info = self.replica.info("replication")
        host, port = info["master_host"], info["master_port"]
        try:
            self._write_during_first_sync(host, port)
        finally:
            # Hand the topology back for the classes that follow, whatever
            # happened above.
            try:
                self.master.config_set("rdb-key-save-delay", 0)
            except Exception:
                pass
            self.replica.execute_command("REPLICAOF", host, port)
            self.wait_for_replica_link()

    def _fresh_master(self, host, port):
        """Restart the master so `REPLICATION_CONSUMERS` starts false again.

        The replica is detached first: left attached it would reconnect the
        instant the master came back and latch the flag before the test could
        do anything.
        """
        self.replica.execute_command("REPLICAOF", "NO", "ONE")
        self.env.envRunner.stopEnv(masters=True, slaves=False)
        self.env.envRunner.startEnv(masters=True, slaves=False)
        # The pooled connections died with the old process, and these clients
        # are built with no retry policy on purpose (see common.NO_RETRY), so a
        # stale one raises rather than reconnecting. Take fresh ones.
        self.master = self.env.getConnection()
        self.master_graph  = Graph(self.master,  self.GRAPH_ID)
        self.replica_graph = Graph(self.replica, self.GRAPH_ID)
        self.master.flushall()
        # A restarted process is back at the compression default, so there is
        # nothing to set — and `self.db` still wraps the dead pool.
        self.env.assertEqual(
            self.master.info("replication")["role"], "master",
            message="the restarted master must not have come back as a replica")
        self.env.assertEqual(
            self.master.info()["sync_full"], 0,
            message="a master that has already served a full sync has the flag latched")

    def _write_during_first_sync(self, host, port):
        self._fresh_master(host, port)

        # The seed registers the label and the attribute before the fork, on
        # purpose: the write inside the window then introduces no new schema and
        # no new attribute id, so no *later* effect can trip a divergence check
        # on its behalf. That is what makes the loss permanent rather than
        # something the guard eventually repairs with a resync.
        #
        # It is also the control. One key, so the child's per-key delay is the
        # whole window; if the seed is missing too then the sync itself failed
        # and nothing here says anything about propagation.
        self.master_graph.query("CREATE (:Seed {v: 1})")
        self.master.config_set("repl-diskless-sync-delay", 0)
        self.master.config_set("rdb-key-save-delay", self.SAVE_DELAY_US)

        self.replica.execute_command("REPLICAOF", host, port)

        # Wait for the child to exist. `rdb_bgsave_in_progress` rather than the
        # replica's `master_sync_in_progress`, which is already 1 while the
        # master is still deciding to fork — a write in *that* window would be
        # captured by the snapshot and prove nothing.
        deadline = time.time() + 60
        while time.time() < deadline:
            if int(self.master.info("persistence")["rdb_bgsave_in_progress"]) == 1:
                break
            time.sleep(0.005)
        else:
            raise AssertionError("the master never forked for the replica's full sync")

        self.master_graph.query("CREATE (:Seed {v: 2})")

        # Still forking, so the write really did land after the snapshot. An
        # assertion rather than a branch: had the child already finished, the
        # snapshot would carry the write and the rest would pass for the wrong
        # reason.
        self.env.assertEqual(
            int(self.master.info("persistence")["rdb_bgsave_in_progress"]), 1,
            message="the write must land while the snapshot is still being taken")
        self.master.config_set("rdb-key-save-delay", 0)

        self.wait_for_replica_link()
        self.wait_for_replica_offset()

        self.env.assertEqual(
            self.master.info()["sync_full"], 1,
            message="this must be the first full sync this master process served")

        self.assert_agree("MATCH (n:Seed) RETURN count(n), sum(n.v)", [[2, 3]])
        self.assert_graph_eq()

        # And it stays wrong. Three more writes of the same shape, each of which
        # the replica applies without complaint, because none of them names
        # anything it does not already hold — so the offsets keep matching, the
        # link keeps reporting healthy, and nothing ever forces the resync that
        # would repair it.
        for v in (3, 4, 5):
            self.query_and_sync(f"CREATE (:Seed {{v: {v}}})")
        self.assert_agree("MATCH (n:Seed) RETURN count(n), sum(n.v)", [[5, 15]])
        self.env.assertEqual(
            self.master.info()["sync_full"], 1,
            message="nothing detected the drift, so no second full sync happened")


class testEffectsV3_06e_AofReplay(_EffectsV3Base):
    """A graph rebuilds from an AOF whose only graph records are effects.

    Effects go to the AOF as well as to the replication stream —
    `execute_query_write` builds a buffer when `ContextFlags::AOF` is set, quite
    apart from whether a replica is attached — and `GRAPH.EFFECT` re-propagates
    verbatim, so it lands in the AOF of every node that applies one.

    That path is newly load-bearing. Query replay used to be an alternative
    recording, chosen per query by `EFFECTS_THRESHOLD`; now a graph's entire
    history on disk is the RDB base plus a stream of `GRAPH.EFFECT`, and
    replaying it is the only way an AOF-configured instance comes back. It is
    also the one place a failure to apply is fatal rather than recoverable:
    `divergence_guard::on_failure` calls `exit(1)` under `LOADING`, because a
    resync cannot repair state that is already baked into local disk.

    AOF is enabled here on an empty dataset on purpose, so the base file holds
    nothing and every byte of the graph has to arrive as an effect.

    Left with `appendonly no`, by `test09`: it is server state, and the classes
    that follow share this server.
    """

    GRAPH_ID = "effects_v3_aof"

    # Mixed on purpose: creates, a multi-commit query, an update, a removal, a
    # delete that frees ids, an edge, a self-loop, and schema DDL — so the AOF
    # holds every record type the emitter writes rather than only CREATE_NODE.
    WORKLOAD = [
        """UNWIND range(1, 3000) AS i
           CREATE (:A {id: i, s: 'v' + i, l: [i, i + 1], f: i * 1.5, b: i % 2 = 0})""",
        "MATCH (a:A), (b:A) WHERE b.id = a.id + 1 AND a.id <= 500 CREATE (a)-[:R {w: a.id}]->(b)",
        "MATCH (a:A {id: 1}) CREATE (a)-[:SELF]->(a)",
        "MATCH (n:A) WHERE n.id % 3 = 0 SET n.s = NULL, n:Extra, n.f = n.f + 0.5",
        "MATCH (n:A) WHERE n.id % 7 = 0 DETACH DELETE n",
        # removals in both of the shapes that reach the wire as a null value
        # row: a whole-shape node SET, and an edge property
        "MATCH (n:A) WHERE n.id % 13 = 0 SET n = {id: n.id, only: true}",
        "MATCH ()-[e:R]->() SET e.tag = 'x'",
        "MATCH ()-[e:R]->() WHERE e.w % 4 = 0 SET e.tag = NULL",
        "UNWIND range(1, 200) AS i MERGE (:Merged {i: i})",
        "CREATE (:Geo {p: point({latitude: 1.5, longitude: 2.5}), d: date('2024-01-15')})",
        "CREATE INDEX FOR (n:A) ON (n.id)",
    ]

    PROBES = [
        "MATCH (n) RETURN count(n)",
        "MATCH (n:A) RETURN count(n), count(n.s), sum(n.id), sum(n.f), count(n.b)",
        "MATCH (n:A) RETURN sum(size(n.l))",
        "MATCH (n:Extra) RETURN count(n)",
        "MATCH (n:Merged) RETURN count(n), sum(n.i)",
        "MATCH ()-[e:R]->() RETURN count(e), sum(e.w), count(e.tag)",
        "MATCH (n:A) RETURN count(n.only), count(n.l), count(n.f)",
        "MATCH (a)-[e:SELF]->(b) RETURN count(e), ID(a) = ID(b)",
        "MATCH (n:Geo) RETURN n.p, toString(n.d)",
        "MATCH (n:A) WHERE n.id > 2990 RETURN count(n), collect(n.id)",
        "CALL db.labels() YIELD label RETURN label ORDER BY label",
        "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey",
        "CALL db.indexes() YIELD label, properties RETURN label, properties ORDER BY label",
        "CALL db.constraints() YIELD type, label, properties, status RETURN type, label, properties, status",
    ]

    def __init__(self):
        # Same root cause as `testEffectsV3_06d_FirstReplicaAttach`: this class
        # stops and starts the master through `env.envRunner`, which cannot
        # restart a Docker service container, and `enableDebugCommand` does not
        # get it a private one (`common.py:564`).
        #
        # Weaker evidence than 06d's, and worth saying so: 06d was OBSERVED
        # failing on three CI shards, whereas this class has never actually run
        # in CI -- `FAIL_FAST=1` stopped the file at 06d, so everything after it
        # is unreported rather than green. This skip is from the mechanism, not
        # from a red run. It is keyed on `FALKORDB_USE_SERVICE` only, so CI's
        # spawn mode -- one fresh container per `Env()` call -- still exercises
        # it, as does local dev.
        if os.getenv("FALKORDB_USE_SERVICE"):
            Environment.skip(None)

        self._setup(enable_debug=True)

    #-------------------------------------------------------------------------
    # helpers
    #-------------------------------------------------------------------------

    def enable_aof(self):
        """Turn on `appendonly` and wait for the initial rewrite to land.

        Enabling it forks to write the base file; a `DEBUG LOADAOF` issued
        before that finished would load a manifest still being assembled.
        """
        self.master.config_set("appendonly", "yes")
        self.wait_for_rewrite()
        info = self.master.info("persistence")
        self.env.assertEqual(int(info["aof_enabled"]), 1)

    def wait_for_rewrite(self, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            info = self.master.info("persistence")
            if (int(info["aof_rewrite_in_progress"]) == 0
                    and int(info.get("aof_rewrite_scheduled", 0)) == 0):
                self.env.assertEqual(
                    info["aof_last_bgrewrite_status"], "ok",
                    message="the AOF rewrite failed; nothing below can be trusted")
                return
            time.sleep(0.05)
        raise AssertionError("the AOF rewrite did not finish")

    def snapshot(self):
        return [self.master_graph.ro_query(p).result_set for p in self.PROBES]

    def reload_from_aof(self):
        """Throw the dataset away and rebuild it from the AOF.

        `DEBUG LOADAOF` flushes the AOF buffer, empties the keyspace and
        replays the manifest — the same code path a restart takes, without the
        restart. It does not propagate, so the replica keeps the copy it
        already had and stays a second, independent witness.
        """
        self.master.execute_command("DEBUG", "LOADAOF")

    #-------------------------------------------------------------------------
    # tests
    #-------------------------------------------------------------------------

    def test01_a_graph_rebuilds_from_an_effects_only_aof(self):
        self.set_effects_config()
        self.enable_aof()

        for q in self.WORKLOAD:
            self.query_and_sync(q)
        create_node_range_index(self.master_graph, 'Merged', 'i', sync=True)
        create_unique_node_constraint(self.master_graph, 'Merged', 'i')
        self.wait_for_constraint_settled(self.master_graph, 'Merged')
        self.wait_for_replica_offset()
        wait_for_indices_to_sync(self.master_graph)

        before = self.snapshot()
        # not vacuously true: the workload has to have produced something
        self.env.assertEqual(before[0], [[3000 - 428 + 200 + 1]])

        self.reload_from_aof()

        # Still serving, which is the first thing to check: a failure to apply
        # while `LOADING` is `exit(1)` by design, so a broken replay shows up
        # as a dead server rather than as a wrong answer.
        self.env.assertEqual(self.master.ping(), True)
        wait_for_indices_to_sync(self.master_graph)
        after = self.snapshot()
        for probe, b, a in zip(self.PROBES, before, after):
            self.env.assertEqual(a, b, message=f"changed across the reload: {probe}")

        # And the replica — which never reloaded — agrees with what came back
        # off disk, so the AOF and the replication stream carried the same
        # thing.
        self.assert_graph_eq()

    def test02_a_rewrite_compacts_and_the_effects_after_it_still_replay(self):
        # After `BGREWRITEAOF` the base file is an RDB of the current state and
        # the incremental file starts empty, so this is the *other* AOF shape:
        # a base that carries the graph plus effects layered on top of it. Both
        # have to replay, in that order.
        self.set_effects_config()
        self.master.execute_command("BGREWRITEAOF")
        self.wait_for_rewrite()

        self.query_and_sync(
            "UNWIND range(1, 500) AS i CREATE (:Post {i: i, s: 'p' + i})")
        self.query_and_sync("MATCH (n:Post) WHERE n.i % 5 = 0 SET n.s = NULL")
        self.query_and_sync("MATCH (n:Post) WHERE n.i % 11 = 0 DELETE n")
        self.query_and_sync("MATCH (a:Post {i: 1}), (b:Post {i: 2}) CREATE (a)-[:P]->(b)")

        extra = ["MATCH (n:Post) RETURN count(n), count(n.s), sum(n.i)",
                 "MATCH ()-[e:P]->() RETURN count(e)"]
        before = self.snapshot() + [
            self.master_graph.ro_query(p).result_set for p in extra]

        self.reload_from_aof()
        self.env.assertEqual(self.master.ping(), True)
        wait_for_indices_to_sync(self.master_graph)
        after = self.snapshot() + [
            self.master_graph.ro_query(p).result_set for p in extra]
        for probe, b, a in zip(self.PROBES + extra, before, after):
            self.env.assertEqual(a, b, message=f"changed across the reload: {probe}")
        self.assert_graph_eq()

    def test03_a_graph_deleted_and_rebuilt_replays_in_order(self):
        # `GRAPH.DELETE` replicates verbatim while the writes around it are
        # effects, so the AOF holds a mix — and a replay that applied the
        # effects against the pre-delete graph would rebuild the wrong thing.
        self.set_effects_config()
        key = "effects_v3_aof_recreate"
        g = Graph(self.master, key)
        g.query("CREATE (:Old {v: 1}), (:Old {v: 2})")
        self.master.execute_command("GRAPH.DELETE", key)
        g.query("UNWIND range(1, 50) AS i CREATE (:New {i: i})")
        self.wait_for_replica_offset()

        before = g.ro_query(
            "MATCH (n) RETURN count(n), labels(n)[0], sum(n.i)").result_set
        self.env.assertEqual(before, [[50, 'New', 1275]])

        self.reload_from_aof()
        self.env.assertEqual(self.master.ping(), True)
        self.env.assertEqual(
            Graph(self.master, key).ro_query(
                "MATCH (n) RETURN count(n), labels(n)[0], sum(n.i)").result_set,
            before)
        # the deleted label is gone rather than resurrected by the replay
        self.env.assertEqual(
            Graph(self.master, key).ro_query(
                "CALL db.labels() YIELD label RETURN label").result_set,
            [['New']])

    def test08_a_poisoned_aof_takes_the_server_down_rather_than_loading_it(self):
        """An effect that will not apply while `LOADING` is `exit(1)`.

        The other arm of `divergence_guard::on_failure` is a forced resync, and
        it cannot help here: the divergence is already baked into this
        instance's own disk, and continuing would replay the rest of the file
        against a dataset that is known wrong. So the guard refuses to come up.
        `testEffectsV3_06c_DivergenceForcesResync` covers the resync arm; this
        is the only way to reach the other one, because `LOADING` is not a flag
        a client can set.

        The effect is injected into the AOF by hand — appended as RESP after a
        clean shutdown, so nothing races the server's own writer. It names a
        label id the replaying graph would not assign, which is `IdMismatch`:
        well-formed bytes describing a graph this instance does not have, which
        is the failure v3 exists to make loud.
        """
        # A dead master must not drag a resyncing replica with it.
        info = self.replica.info("replication")
        host, port = info["master_host"], info["master_port"]
        self.replica.execute_command("REPLICAOF", "NO", "ONE")

        runner = self.env.envRunner
        try:
            self._poisoned_start_must_exit(runner)
        finally:
            # Bring the master back however the above ended, then the link.
            if runner.masterProcess is None:
                runner.startEnv(masters=True, slaves=False)
            self.master = self.env.getConnection()
            self.master.flushall()
            self.master_graph  = Graph(self.master,  self.GRAPH_ID)
            self.replica_graph = Graph(self.replica, self.GRAPH_ID)
            self.replica.execute_command("REPLICAOF", host, port)
            self.wait_for_replica_link()

        # The pair still works, so the recovery was real and not a husk.
        self.query_and_sync("CREATE (:AfterPoison {v: 1})")
        self.assert_agree("MATCH (n:AfterPoison) RETURN count(n)", [[1]])

    def _poisoned_start_must_exit(self, runner):
        runner.stopEnv(masters=True, slaves=False)

        incr = self._aof_incr_path(runner)
        clean_len = os.path.getsize(incr)
        with open(incr, "ab") as f:
            f.write(self._resp(b"GRAPH.EFFECT", self.GRAPH_ID.encode(),
                               self._bad_schema_effect()))

        log_before = self._log_size(runner)
        # Spawned here rather than through `startEnv`, which is built to wait
        # for a server that comes up: this one must not, and the exit code is
        # the assertion.
        args = list(runner.masterCmdArgs) + ["--appendonly", "yes"]
        proc = subprocess.Popen(args, cwd=runner.dbDirPath,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            rc = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise AssertionError(
                "the server came up on a poisoned AOF instead of refusing to")
        finally:
            # Whatever happened, the file goes back to what the server wrote,
            # so the recovery start in the caller's `finally` has a loadable
            # AOF.
            with open(incr, "r+b") as f:
                f.truncate(clean_len)

        self.env.assertEqual(rc, 1, message=f"expected exit(1), got {rc}")

        # And it exited for *this* reason. An exit code alone would also be
        # satisfied by a server that failed to bind its port.
        tail = self._log_tail(runner, log_before)
        self.env.assertContains("Diverged applying GRAPH.EFFECT", tail)
        self.env.assertContains("while loading from disk", tail)
        self.env.assertContains("shutting down", tail)

    #-------------------------------------------------------------------------
    # poisoned-AOF plumbing
    #-------------------------------------------------------------------------

    @staticmethod
    def _resp(*args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        return out

    @staticmethod
    def _bad_schema_effect():
        """A v3 buffer holding one `ADD_SCHEMA` record for node label id 9999.

        `u8 3 · u8 0 · u32 9 · u32 0 · i32 id · u64 len+1 · name · NUL`. The id
        is far past anything the replay could have reached, so the apply path
        refuses it with `IdMismatch` — the bytes are well formed, which is the
        point: a decode error would prove nothing about divergence.
        """
        name = b"PoisonedLabel"
        return (b"\x03\x00"
                + b"\x09\x00\x00\x00"
                + b"\x00\x00\x00\x00"
                + (9999).to_bytes(4, "little", signed=True)
                + (len(name) + 1).to_bytes(8, "little") + name + b"\x00")

    @staticmethod
    def _aof_incr_path(runner):
        d = os.path.join(runner.dbDirPath, "appendonlydir")
        incr = sorted(f for f in os.listdir(d) if f.endswith(".incr.aof"))
        if not incr:
            raise AssertionError(f"no incremental AOF under {d}: {os.listdir(d)}")
        return os.path.join(d, incr[-1])

    @staticmethod
    def _master_log(runner):
        name = runner._getFileName('master', '.log')
        for base in (runner.dbDirPath, getattr(runner, 'outputFilesFormat', None)):
            if base and os.path.exists(os.path.join(str(base), name)):
                return os.path.join(str(base), name)
        return os.path.join(runner.dbDirPath, name)

    def _log_size(self, runner):
        path = self._master_log(runner)
        return os.path.getsize(path) if os.path.exists(path) else 0

    def _log_tail(self, runner, offset):
        path = self._master_log(runner)
        if not os.path.exists(path):
            raise AssertionError(f"no master log at {path}")
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode(errors="replace")

    def test09_leave_appendonly_off(self):
        # Server state, and the classes after this one share the server.
        self.master.config_set("appendonly", "no")
        self.env.assertEqual(
            int(self.master.info("persistence")["aof_enabled"]), 0)


#-----------------------------------------------------------------------------
# 8. the same write produces the same bytes
#-----------------------------------------------------------------------------


class testEffectsV3_07_PromotedReplica(_EffectsV3Base):
    """A replica promoted while holding a constraint UNDER CONSTRUCTION must
    settle it itself.

    A replica never validates — it installs the status the primary announced.
    That is right while it is a replica and wrong the moment it is promoted: a
    constraint the old primary was still building becomes this node's to
    finish, and until it does the constraint is neither enforcing nor failed,
    so writes that should be rejected are not.

    The window this test needs is the gap between the two announcements. A
    UNIQUE constraint over three indexed properties of a million nodes spends
    about 400ms there (measured on a release build), against roughly 1.5ms to
    confirm the replica has the first announcement and cut the link — so the
    promotion lands inside it with two orders of magnitude to spare. A slower
    build widens the gap without widening the two round trips, so the margin
    only grows. MANDATORY will not do: its validation is a bare scan and
    settles in under 4ms even at a million nodes.

    This class runs last because it takes the topology apart. It puts the link
    back before returning either way.
    """

    GRAPH_ID = "effects_v3_promotion"

    N = 1_000_000
    PROPS = ('a', 'b', 'c')

    def __init__(self):
        self._setup()
        info = self.replica.info("replication")
        self.replicaof_host = info["master_host"]
        self.replicaof_port = info["master_port"]

    def test01_promotion_settles_an_inherited_constraint(self):
        try:
            self._promote_mid_validation()
        finally:
            # Hand the topology back. A later class — or a rerun in a reused
            # env — gets a replica again, and the resync wipes whatever this
            # test left on the promoted node.
            self.replica.execute_command(
                "REPLICAOF", self.replicaof_host, self.replicaof_port)
            self.wait_for_replica_link()

    def _promote_mid_validation(self):
        self.set_effects_config()
        props = ", ".join(f"{p}: x" for p in self.PROPS)
        self.query_and_sync(
            f"UNWIND range(1, {self.N}) AS x CREATE (:Huge {{{props}}})")
        for p in self.PROPS:
            create_node_range_index(self.master_graph, 'Huge', p)
        wait_for_indices_to_sync(self.master_graph)
        self.wait_for_replica_offset()
        wait_for_indices_to_sync(self.replica_graph)
        self.assert_agree("MATCH (n:Huge) RETURN count(n)", [[self.N]])

        # Create the constraint and cut the link as soon as the replica has the
        # first announcement — before the settled one can reach it.
        #
        # One constraint, because one is all this test needs — not because two
        # is impossible. Two `GRAPH.CONSTRAINT CREATE`s issued concurrently on
        # different labels of the same graph *do* both answer PENDING and sit
        # UNDER CONSTRUCTION together (measured: two 400k-node labels, both
        # observed pending at once), so `settle_async_constraint`'s
        # `for c in announce` really can iterate more than once — reached
        # through `enforce_pending_constraints_after_promotion`, which collects
        # every pending constraint of a graph into one slice. That case is
        # still uncovered; it needs a replica holding two pending constraints
        # at the instant it is promoted.
        res = self.master_graph.execute_command(
            "GRAPH.CONSTRAINT", "CREATE", self.GRAPH_ID, "UNIQUE", "NODE",
            "Huge", "PROPERTIES", len(self.PROPS), *self.PROPS)
        self.env.assertEqual(res, 'PENDING')
        self.wait_for_replica_offset()
        inherited = self.constraint_rows(self.replica_graph, 'Huge')
        self.replica.execute_command("REPLICAOF", "NO", "ONE")

        # What it was holding at the instant it stopped being a replica. If
        # this is not UNDER CONSTRUCTION the promotion had nothing to finish
        # and the rest of the test would prove nothing, so it is an assertion
        # rather than a branch.
        self.env.assertEqual(
            inherited, [['UNIQUE', 'Huge', list(self.PROPS), 'NODE', 'UNDER CONSTRUCTION']])
        self.env.assertEqual(self.replica.info("replication")["role"], "master")

        # It settles on its own, as a node that is no longer a replica: there
        # is nothing left to learn the status from.
        settled = self.wait_for_constraint_settled(self.replica_graph, 'Huge')
        self.env.assertEqual(
            settled, [['UNIQUE', 'Huge', list(self.PROPS), 'NODE', 'OPERATIONAL']])
        self.env.assertEqual(self.replica.info("replication")["role"], "master")
        # exactly one constraint — the promotion hook updates in place
        self.env.assertEqual(
            self.replica_graph.ro_query(
                "CALL db.constraints() YIELD status RETURN count(1)"
            ).result_set, [[1]])

        # the old primary reached the same answer independently
        self.env.assertEqual(
            self.wait_for_constraint_settled(self.master_graph, 'Huge'),
            [['UNIQUE', 'Huge', list(self.PROPS), 'NODE', 'OPERATIONAL']])

        # and the promoted node enforces it, which is the point of finishing it
        promoted = Graph(self.replica, self.GRAPH_ID)
        try:
            promoted.query("CREATE (:Huge {a: 1, b: 1, c: 1})")
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("unique constraint violation", str(e))
