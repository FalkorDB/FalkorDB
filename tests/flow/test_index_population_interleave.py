import random
import time

from common import *

# Deterministic interleavings of background index population with writers
# (#3095).
#
# A population batch reads the *committed* graph, while a writer publishes its
# index documents during the query, before its MVCC commit. A batch that runs
# in that window used to overwrite the writer's newer document with the value
# it had just replaced, or resurrect a deleted entity.
#
# Needs a module built with `--features index-population-hooks`, which adds
# `GRAPH.DEBUG INDEX_POPULATION` (batch size, pause/step, and a per-writer
# window that waits for population batches); the test is skipped otherwise:
#
#   cargo build --release --features index-population-hooks
#   RELEASE=1 TEST=test_index_population_interleave ./flow.sh
#
# Each case builds a small graph, pauses population with a batch size of 1-5,
# starts CREATE INDEX and then follows a seeded schedule of STEP (one batch),
# WRITE (a committed SET / REMOVE / DELETE / CREATE / SET+DELETE / a write that
# fails after publishing) and WINDOW (a write during which one batch runs
# between its publish and its commit). Afterwards every value's index scan is
# compared with a full scan.

VALUES = 6
CASES = int(os.getenv("POP_INTERLEAVE_CASES", "150"))


class testIndexPopulationInterleave(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        try:
            self.env.cmd("GRAPH.DEBUG", "INDEX_POPULATION", "BATCHES")
            self.hooks = True
        except ResponseError:
            self.hooks = False

    def dbg(self, *args):
        return self.env.cmd("GRAPH.DEBUG", "INDEX_POPULATION", *args)

    def wait_batches(self, target, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if int(self.dbg("BATCHES")) >= target:
                return
            time.sleep(0.0005)
        self.env.assertGreaterEqual(int(self.dbg("BATCHES")), target,
                                    message="a population batch did not run", depth=1)

    def operational(self, g):
        statuses = g.query("CALL db.indexes() YIELD status RETURN status").result_set
        return bool(statuses) and all(s[0] == "OPERATIONAL" for s in statuses)

    def write_query(self, rng, g, kind, ents, counter, n_nodes):
        node = kind == "node"
        prop = "v" if node else "w"
        x = rng.randrange(VALUES)
        op = rng.choice(["set", "set", "set", "remove", "delete", "create", "create", "setdelete", "fail"])
        if not ents:
            op = "create"
        k = rng.choice(ents) if ents else None
        pat = f"(n:L {{k: {k}}})" if node else f"()-[n:R {{k: {k}}}]->()"
        if op == "set":
            return f"MATCH {pat} SET n.{prop} = {x}"
        if op == "remove":
            return f"MATCH {pat} REMOVE n.{prop}"
        if op == "delete":
            return f"MATCH {pat} DELETE n"
        if op == "setdelete":
            return f"MATCH {pat} SET n.{prop} = {x} WITH n DELETE n"
        if op == "fail":
            # publishes its documents, then fails: the index must be rolled back
            return f"MATCH {pat} SET n.{prop} = {x} WITH n RETURN 1 / (n.k - n.k)"
        counter[0] += 1
        if node:
            return f"CREATE (:L {{k: {counter[0]}, v: {x}}})"
        a, b = rng.randrange(n_nodes), rng.randrange(n_nodes)
        return (f"MATCH (a:N {{i: {a}}}), (b:N {{i: {b}}}) "
                f"CREATE (a)-[:R {{k: {counter[0]}, w: {x}}}]->(b)")

    def run_case(self, kind, seed):
        rng = random.Random(f"{kind}{seed}")
        node = kind == "node"
        g = self.db.select_graph(f"pop_{kind}")
        n = rng.randint(8, 40)
        if node:
            for i in range(n):
                v = f", v: {rng.randrange(VALUES)}" if rng.random() < 0.9 else ""
                g.query(f"CREATE (:L {{k: {i}{v}}})")
        else:
            g.query("UNWIND range(0, $n - 1) AS i CREATE (:N {i: i})", {"n": n})
            for k in range(n):
                a, b = rng.randrange(n), rng.randrange(n)
                v = f", w: {rng.randrange(VALUES)}" if rng.random() < 0.9 else ""
                g.query(f"MATCH (a:N {{i: {a}}}), (b:N {{i: {b}}}) CREATE (a)-[:R {{k: {k}{v}}}]->(b)")

        ents_q = "MATCH (n:L) RETURN n.k" if node else "MATCH ()-[n:R]->() RETURN n.k"
        counter = [1000]

        def write(window):
            q = self.write_query(rng, g, kind, [r[0] for r in g.query(ents_q).result_set], counter, n)
            fails = "1 / (n.k - n.k)" in q
            if window:
                self.dbg("WINDOW", 1)
            try:
                g.query(q)
                self.env.assertFalse(fails, message="the write built to fail succeeded", depth=1)
            except ResponseError as e:
                # only the write built to fail may fail
                self.env.assertTrue(fails, message=f"unexpected error: {e}", depth=1)
            if window:
                self.dbg("WINDOW", 0)

        self.dbg("WINDOW", 0)
        self.dbg("BATCH_SIZE", rng.choice([1, 2, 3, 5]))
        self.dbg("PAUSE")
        g.query("CREATE INDEX FOR (n:L) ON (n.v)" if node else "CREATE INDEX FOR ()-[r:R]-() ON (r.w)")
        for _ in range(400):
            a = rng.random()
            if a < 0.35:
                done = int(self.dbg("BATCHES"))
                self.dbg("STEP", 1)
                self.wait_batches(done + 1)
            elif a < 0.65:
                write(False)
            else:
                write(True)
            if self.operational(g):
                break
        self.dbg("RESUME")
        deadline = time.time() + 30
        while not self.operational(g) and time.time() < deadline:
            time.sleep(0.01)
        self.dbg("BATCH_SIZE", 0)

        full = "MATCH (n:L) RETURN n.k, n.v" if node else "MATCH ()-[n:R]->() RETURN n.k, n.w"
        truth = {tuple(r) for r in g.query(full).result_set if r[1] is not None}
        for x in range(VALUES):
            q = (f"MATCH (n:L) WHERE n.v = {x} RETURN n.k, n.v" if node
                 else f"MATCH ()-[n:R]->() WHERE n.w = {x} RETURN n.k, n.w")
            self.env.assertIn("Index Scan", str(g.explain(q)), depth=1)
            got = {tuple(r) for r in g.query(q).result_set}
            want = {t for t in truth if t[1] == x}
            self.env.assertEqual(got, want, message=f"{kind} seed {seed}, value {x}", depth=1)
        g.delete()

    def check_windows(self, fired_before):
        # every window must have seen its population batch run, and a good
        # share of the cases must have had one at all, or the schedule did not
        # exercise the interleaving
        fired, timed_out = (int(x) for x in self.dbg("WINDOWS"))
        self.env.assertEqual(timed_out, 0, message="a window gave up waiting for a batch")
        self.env.assertGreaterEqual(fired - fired_before, CASES // 2, depth=1)

    def test01_node_index_matches_scan(self):
        if not self.hooks:
            self.env.skip()
            return
        fired_before = int(self.dbg("WINDOWS")[0])
        for seed in range(CASES):
            self.run_case("node", seed)
        self.check_windows(fired_before)

    def test02_edge_index_matches_scan(self):
        if not self.hooks:
            self.env.skip()
            return
        fired_before = int(self.dbg("WINDOWS")[0])
        for seed in range(CASES):
            self.run_case("edge", seed)
        self.check_windows(fired_before)
