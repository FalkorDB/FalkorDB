import os
import time
from common import *

GRAPH_ID = "effects_v3_aof"

# A graph rebuilds from an AOF whose only graph records are effects.
#
# WHY THIS IS DIFFERENT FROM EVERY OTHER v3 TEST. All of them go through
# replication: a master emits, a replica applies, and the two are compared. The
# AOF is a SECOND CONSUMER OF THE SAME BYTES with no replica involved at all --
# GRAPH.EFFECT re-propagates verbatim, so it lands in the AOF of any node that
# applies one, and an AOF-configured instance comes back only by replaying it.
#
# It is also the one place where failing to apply is unrecoverable. A replica
# that cannot apply an effect asks for a resync; an instance replaying its own
# disk has nothing to resync FROM, so a refusal there is fatal rather than
# repairable.
#
# AOF IS ENABLED ON AN EMPTY DATASET ON PURPOSE, and this is the load-bearing
# detail. Turning appendonly on triggers a rewrite that captures everything
# already present as a base snapshot. Enable it after the writes and the graph
# comes back from that snapshot, exercising no effect at all -- the test would
# pass and mean nothing.
#
# Ported from the intention of Rust's testEffectsV3_06e_AofReplay. Their class
# skips itself when FALKORDB_USE_SERVICE is set, because restarting needs a
# private server rather than a shared container. That variable does not exist on
# this branch, so the guard is not carried over -- porting it would have added a
# condition that is never true and reads as coverage.


class testEffectsV3Aof():
    # Mixed on purpose, so the AOF holds every record shape the emitter writes
    # rather than only CREATE_NODE: creates, an edge, a self-loop, updates, a
    # label add, property removal in both shapes that reach the wire as a null
    # value row, a delete that frees ids, a MERGE, and schema DDL.
    WORKLOAD = [
        """UNWIND range(1, 1500) AS i
           CREATE (:A {id: i, s: 'v' + i, l: [i, i + 1],
                       f: i * 1.5, b: i % 2 = 0})""",
        """MATCH (a:A), (b:A) WHERE b.id = a.id + 1 AND a.id <= 300
           CREATE (a)-[:R {w: a.id}]->(b)""",
        "MATCH (a:A {id: 1}) CREATE (a)-[:SELF]->(a)",
        "MATCH (n:A) WHERE n.id % 3 = 0 SET n.s = NULL, n:Extra, n.f = n.f + 0.5",
        "MATCH (n:A) WHERE n.id % 7 = 0 DELETE n",
        "MATCH (n:A) WHERE n.id % 13 = 0 SET n = {id: n.id, only: true}",
        "MATCH ()-[e:R]->() SET e.tag = 'x'",
        "MATCH ()-[e:R]->() WHERE e.w % 4 = 0 SET e.tag = NULL",
        "UNWIND range(1, 200) AS i MERGE (:Merged {i: i})",
        "CREATE (:Geo {p: point({latitude: 1.5, longitude: 2.5})})",
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
        "MATCH (n:Geo) RETURN n.p",
        "MATCH (n:A) WHERE n.id > 1490 RETURN count(n), collect(n.id)",
        "CALL db.labels() YIELD label RETURN label ORDER BY label",
        "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey "
        "ORDER BY propertyKey",
        "CALL db.indexes() YIELD label, properties RETURN label, properties "
        "ORDER BY label",
    ]

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        # EFFECTS_THRESHOLD 0 puts every write on the effects path;
        # EFFECTS_VERSION 3 because C still emits v2 by default. No replica: the
        # whole point is that this path does not need one.
        self.env, self.db = Env(env='oss',
                moduleArgs='EFFECTS_THRESHOLD 0 EFFECTS_VERSION 3')
        self.con = self.env.getConnection()

    def _enable_aof_on_an_empty_dataset(self):
        # appendfsync always so a query's records are on disk by the time its
        # reply returns, rather than relying on timing
        self.con.config_set("appendonly", "yes")
        self.con.config_set("appendfsync", "always")

        deadline = time.time() + 60
        while time.time() < deadline:
            info = self.con.info("persistence")
            if (info.get("aof_enabled") == 1 and
                    info.get("aof_rewrite_in_progress") == 0):
                return
            time.sleep(0.2)

        raise Exception("AOF never finished its initial rewrite")

    def _aof_bytes(self):
        """Everything in the incremental AOF files, concatenated.

        Located through the manifest rather than by guessing Redis' file
        naming -- the same approach test_replica_divergence.py takes, and for
        the same reason.
        """
        d = self.con.config_get("dir")["dir"]
        sub = self.con.config_get("appenddirname")["appenddirname"]
        base = self.con.config_get("appendfilename")["appendfilename"]
        aof_dir = os.path.join(d, sub)

        out = b""
        with open(os.path.join(aof_dir, base + ".manifest")) as f:
            for line in f:
                toks = line.split()
                kv = dict(zip(toks[0::2], toks[1::2]))
                # type 'i' is an incremental file; 'b' is the base snapshot,
                # which is what we are asserting does NOT carry the graph
                if kv.get("type") != "i":
                    continue
                p = os.path.join(aof_dir, kv["file"])
                if os.path.exists(p):
                    with open(p, "rb") as fh:
                        out += fh.read()
        return out

    def _probe_all(self):
        g = Graph(self.con, GRAPH_ID)
        return [g.ro_query(q).result_set for q in self.PROBES]

    def test01_a_graph_rebuilds_from_an_effects_only_aof(self):
        self._enable_aof_on_an_empty_dataset()

        g = Graph(self.con, GRAPH_ID)
        for q in self.WORKLOAD:
            g.query(q)

        # THE MECHANISM CHECK, and it is not optional. Rust's version rests on
        # effects reaching the AOF whether or not a replica is attached. That is
        # a claim about their emitter, so it is verified here rather than
        # assumed: if C only built an effects buffer when a replica existed, the
        # AOF would hold GRAPH.QUERY instead and the graph would come back by
        # query replay -- passing this test while exercising nothing it is for.
        aof = self._aof_bytes()
        n_effect = aof.count(b"GRAPH.EFFECT")
        n_query  = aof.count(b"GRAPH.QUERY")

        self.env.assertTrue(n_effect > 0,
                message=f"no GRAPH.EFFECT in the AOF ({len(aof)} bytes, "
                        f"{n_query} GRAPH.QUERY): effects are not reaching the "
                        f"AOF without a replica attached, so a rebuild here "
                        f"would be query replay and this test would prove "
                        f"nothing")

        before = self._probe_all()
        self.env.assertTrue(before[0][0][0] > 0,
                message="the workload produced an empty graph")

        # restart: with appendonly on, the instance comes back from the AOF
        self.env.stop()
        self.env.start()
        self.con = self.env.getConnection()

        after = self._probe_all()

        for q, a, b in zip(self.PROBES, before, after):
            self.env.assertEqual(a, b,
                    message=f"AOF replay did not reproduce the graph.\n"
                            f"      probe: {q}\n"
                            f"      before: {a}\n"
                            f"      after:  {b}")

        # and the effects really were the carrier: the base snapshot was taken
        # when the dataset was empty, so it cannot be what rebuilt this
        self.env.assertTrue(n_effect > 0)
