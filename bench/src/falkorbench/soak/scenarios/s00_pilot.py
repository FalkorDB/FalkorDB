"""s00_pilot — template scenario: small social graph, mixed read/write.

Every scenario script follows this CLI contract:
  python -I s00_pilot.py --module <path> --port <port> --out <dir> --duration <sec> --scale <float>
--scale 1.0 = the full soak size; smoke runs use e.g. 0.05.
It must: load data, run timed load through soakkit.run_load, record correctness
checks with Checks, monitor the server with Monitor, and finish with analyze().
"""
import argparse
import time
from pathlib import Path

from falkorbench.soak import soakkit as sk

ap = argparse.ArgumentParser()
sk.add_engine_args(ap)  # --module PATH | --image REF
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--duration", type=float, default=60)
ap.add_argument("--scale", type=float, default=0.05)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

G = "pilot"
N = max(1000, int(200_000 * a.scale))
DEG = 10

srv = sk.Server(a.module, a.port, out)
mon = sk.Monitor(srv, out)
mon.start()
checks = sk.Checks(out)
r = srv.client()

# ---------------- load
mon.mark("ingest")
t = time.time()
B = 5000
for i in range(0, N, B):
    sk.query(r, G, "UNWIND range($a, $b) AS i CREATE (:User {id: i, age: i % 80})",
             {"a": i, "b": min(N, i + B) - 1})
sk.query(r, G, "CREATE INDEX FOR (u:User) ON (u.id)")
for i in range(0, N, B):
    sk.query(r, G, f"UNWIND range($a, $b) AS i MATCH (u:User {{id: i}}) "
                   f"UNWIND range(1, {DEG}) AS k MATCH (v:User {{id: (i * 7919 + k * 104729) % $n}}) "
                   f"CREATE (u)-[:FOLLOWS]->(v)", {"a": i, "b": min(N, i + B) - 1, "n": N})
ingest_s = time.time() - t
nodes = sk.scalar(sk.query(r, G, "MATCH (u:User) RETURN count(u)"))
edges = sk.scalar(sk.query(r, G, "MATCH ()-[e:FOLLOWS]->() RETURN count(e)"))
checks.add("ingest node count", nodes == N, f"{nodes} != {N}", "critical")
checks.add("ingest edge count", edges == N * DEG, f"{edges} != {N * DEG}", "critical")


# ---------------- workload
def read_op(ctx):
    k = ctx.rng.randrange(N)
    if ctx.rng.random() < 0.7:
        sk.query(ctx.r, G, "MATCH (u:User {id: $id})-[:FOLLOWS]->(v) RETURN v.id", {"id": k}, ro=True)
        return "1hop"
    sk.query(ctx.r, G, "MATCH (u:User {id: $id})-[:FOLLOWS*2]->(v) RETURN count(DISTINCT v)",
             {"id": k}, ro=True)
    return "2hop"


def write_op(ctx):
    k = ctx.rng.randrange(N)
    sk.query(ctx.r, G, "MATCH (u:User {id: $id}) SET u.seen = coalesce(u.seen, 0) + 1", {"id": k})
    return "set_prop"


mon.mark("soak")
sk.run_load(srv, out, [sk.WorkerSpec(read_op, rate=200, count=4, name="reader"),
                       sk.WorkerSpec(write_op, rate=50, count=1, name="writer")], a.duration)

# ---------------- post checks
mon.mark("verify")
seen = sk.scalar(sk.query(r, G, "MATCH (u:User) RETURN sum(coalesce(u.seen, 0))"))
summ_partial = sk.analyze(out, srv)
ok_writes = summ_partial["ops"].get("set_prop", {}).get("count", 0) - \
    summ_partial["ops"].get("set_prop", {}).get("errors", 0)
checks.add("every acknowledged write is visible", seen == ok_writes, f"seen={seen} acked={ok_writes}",
           "critical")
dt = srv.restart(save=True)
r = srv.client()
checks.add("RDB round-trip preserves counts",
           sk.scalar(sk.query(r, G, "MATCH ()-[e]->() RETURN count(e)")) == N * DEG, "", "critical")
checks.add("server alive at end", srv.alive(), "", "critical")
mon.stop()
s = sk.analyze(out, srv)
s["meta"] = {"scenario": "s00_pilot", "nodes": N, "edges": N * DEG, "ingest_s": ingest_s,
             "restart_s": dt, "duration": a.duration, "scale": a.scale}
(out / "summary.json").write_text(sk.json.dumps(s, indent=1, default=str))
srv.stop()
sk.print_summary(s)
