"""s09_replication — primary + replica as in a production HA deployment.

CLI: --module --port --out --duration --scale (see s00_pilot). Primary on --port, replica on --port+1
(--port+2 reserved/unused).
Phases: ingest on primary -> replica attaches (full sync) -> soak (writes on primary incl.
rand()/timestamp(), MERGE, deletes, CREATE/DROP INDEX; reads on replica; marker lag probes;
ledger writer using WAIT) with quiesce checksum comparisons, a replica restart + full resync
mid-load -> final quiesce compare -> failover drill (REPLICAOF NO ONE under load, every
WAIT-confirmed write must be present).
"""
import argparse
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import soakkit as sk  # noqa: E402

ap = argparse.ArgumentParser()
sk.add_engine_args(ap)  # --module PATH | --image REF
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--duration", type=float, default=60)
ap.add_argument("--scale", type=float, default=0.05)
a = ap.parse_args()
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
(out / "replica").mkdir(exist_ok=True)

G = "repl"
U = max(2000, int(200_000 * a.scale))      # users
P = max(8000, int(800_000 * a.scale))      # posts
FOLLOW_DEG = 8
LIKES_PER_POST = 3
PORT_R = a.port + 1

PAUSE = mp.Event()          # set => writers idle (quiesce); inherited by forked workers
T_SOAK = mp.Value("d", 0.0)
REPL_DOWN = mp.Value("i", 0)  # 1 while the replica is restarting / resyncing
WRITE_RATES = float(os.environ.get("S09_WRITE_SCALE", "3.0"))

checks = None


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


# ---------------------------------------------------------------- servers
srv = sk.Server(a.module, a.port, out, extra=["--repl-diskless-sync-delay", "0"], name="primary")
mon = sk.Monitor(srv, out)
mon.start()
checks = sk.Checks(out)
r = srv.client()
rep = None
rmon = None

import atexit  # noqa: E402
atexit.register(lambda: [x.stop() for x in (rep, srv) if x is not None])


def info(c, section="replication"):
    return c.info(section)


def wait_link_up(timeout):
    t = time.time()
    rc = sk.connect(PORT_R)
    while time.time() - t < timeout:
        try:
            i = rc.info("replication")
            if i.get("master_link_status") == "up" and not i.get("master_sync_in_progress"):
                return time.time() - t
        except Exception:
            pass
        time.sleep(0.25)
    return None


def converge(timeout=120):
    """Writers are paused: wait until replica offset == primary offset. Returns seconds or None."""
    t = time.time()
    rc = sk.connect(PORT_R)
    while time.time() - t < timeout:
        try:
            pi = r.info("replication")
            ri = rc.info("replication")
            if r.execute_command("WAIT", 1, 2000) >= 1 and \
                    ri.get("slave_repl_offset") == pi.get("master_repl_offset"):
                return time.time() - t
        except Exception:
            pass
        time.sleep(0.2)
    return None


# ---------------------------------------------------------------- checksum
CS_QUERIES = {}
for lbl in ["User", "Post", "Evt", "Ack", "Marker"]:
    CS_QUERIES[f"cnt:{lbl}"] = f"MATCH (n:{lbl}) RETURN count(n)"
    CS_QUERIES[f"ids:{lbl}"] = f"MATCH (n:{lbl}) RETURN sum(id(n))"
for t in ["FOLLOWS", "POSTED", "LIKES", "EMITTED"]:
    CS_QUERIES[f"cnt:{t}"] = f"MATCH ()-[e:{t}]->() RETURN count(e)"
    CS_QUERIES[f"ids:{t}"] = f"MATCH ()-[e:{t}]->() RETURN sum(id(e))"
CS_QUERIES.update({
    "user_props": "MATCH (u:User) RETURN sum(u.age), sum(coalesce(u.score,0)), sum(coalesce(u.visits,0)), "
                  "sum(coalesce(u.touched,0)), count(u.score), sum(size(u.name))",
    "post_props": "MATCH (p:Post) RETURN sum(p.len), sum(p.author), sum(size(p.title))",
    "evt_props": "MATCH (e:Evt) RETURN sum(e.r), sum(e.ts), sum(toInteger(e.tag)), sum(e.id % 1000)",
    "marker_props": "MATCH (m:Marker) RETURN sum(m.v), sum(m.id)",
    "ack_props": "MATCH (k:Ack) RETURN sum(k.id % 100000), max(k.id)",
    "follows_w": "MATCH ()-[e:FOLLOWS]->() RETURN sum(coalesce(e.w,0)), count(e.w)",
    "emitted_w": "MATCH ()-[e:EMITTED]->() RETURN sum(e.w)",
    "follow_struct": "MATCH (a:User)-[:FOLLOWS]->(b:User) RETURN sum((a.id % 1000) * (b.id % 997))",
    "likes_struct": "MATCH (u:User)-[:LIKES]->(p:Post) RETURN sum((u.id % 1000) * (p.id % 991))",
    "posted_struct": "MATCH (u:User)-[:POSTED]->(p:Post) RETURN sum((u.id % 1000) * (p.id % 983))",
    "emit_struct": "MATCH (u:User)-[:EMITTED]->(e:Evt) RETURN sum((u.id % 1000) * (e.id % 977))",
    "indexes": "CALL db.indexes() YIELD label, properties, entitytype RETURN label, properties, entitytype",
})


def norm(c):
    if isinstance(c, list) and len(c) == 2 and isinstance(c[0], int):
        if c[0] == 6:
            return tuple(norm(x) for x in c[1])
        return sk.cval(c)
    return c


def checksum(port):
    c = sk.connect(port)
    res = {}
    for k, q in CS_QUERIES.items():
        try:
            rs = sk.rows(sk.query(c, G, q, ro=True))
            v = [tuple(norm(x) for x in row) for row in rs]
            if k == "indexes":
                v = sorted(v, key=repr)
            res[k] = v
        except Exception as e:
            res[k] = f"ERR {e}"
    return res


def compare(tag, id_keys_major=True):
    """Both sides checksummed concurrently. Returns (ok, ok_ids, details)."""
    box = {}
    ths = [threading.Thread(target=lambda p=p: box.__setitem__(p, checksum(p))) for p in (a.port, PORT_R)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    p, q = box[a.port], box[PORT_R]
    diff = [k for k in CS_QUERIES if p[k] != q[k]]
    idd = [k for k in diff if k.startswith("ids:")]
    other = [k for k in diff if not k.startswith("ids:")]
    det = "; ".join(f"{k}: primary={str(p[k])[:100]} replica={str(q[k])[:100]}" for k in diff[:6])
    errs = [k for k in CS_QUERIES if str(p[k]).startswith("ERR") or str(q[k]).startswith("ERR")]
    checks.add(f"{tag}: replica content == primary (counts/props/structure/indexes)", not other, det, "critical")
    checks.add(f"{tag}: internal ids (sum id(n)/id(e)) identical", not idd, det, "major")
    checks.add(f"{tag}: checksum queries ran without error", not errs,
               f"{errs}: {[str(p[k])[:100] for k in errs[:2]]}", "major")
    return not other, p


def quiesce_compare(tag):
    PAUSE.set()
    time.sleep(1.5)  # in-flight writes finish
    t = time.time()
    cv = converge()
    ok = cv is not None
    checks.add(f"{tag}: replica caught up (offsets equal) after writes paused", ok,
               f"no convergence in 120s", "critical")
    ok1, base = compare(tag)
    if not ok1:  # allow for a laggy apply: retry once after 5 s
        time.sleep(5)
        ok2, _ = compare(tag + " (retry)")
    paused = time.time() - t + 1.5
    log(f"quiesce {tag}: converge={cv} paused={paused:.1f}s")
    PAUSE.clear()
    return cv, paused


# ---------------------------------------------------------------- ingest (primary alone)
mon.mark("ingest")
t_ing = time.time()
B = 5000


def q(c, s, p=None, **kw):
    return sk.query(c, G, s, p, **kw)


for i in range(0, U, B):
    q(r, "UNWIND range($a, $b) AS i CREATE (:User {id: i, age: 18 + i % 60, name: 'user' + toString(i), "
         "score: i % 1000})", {"a": i, "b": min(U, i + B) - 1})
q(r, "CREATE INDEX FOR (u:User) ON (u.id)")
q(r, "CREATE INDEX FOR (p:Post) ON (p.id)")
q(r, "CREATE INDEX FOR (e:Evt) ON (e.id)")
q(r, "CREATE INDEX FOR (m:Marker) ON (m.id)")
for i in range(0, P, B):
    q(r, "UNWIND range($a, $b) AS i CREATE (:Post {id: i, author: (i * 7) % $u, len: i % 280, "
         "title: 'post' + toString(i)})", {"a": i, "b": min(P, i + B) - 1, "u": U})
BU = 2000
for i in range(0, U, BU):
    q(r, f"UNWIND range($a, $b) AS i MATCH (u:User {{id: i}}) UNWIND range(1, {FOLLOW_DEG}) AS k "
         f"MATCH (v:User {{id: (i * 7919 + k * 104729) % $n}}) CREATE (u)-[:FOLLOWS {{w: k}}]->(v)",
      {"a": i, "b": min(U, i + BU) - 1, "n": U})
for i in range(0, P, B):
    q(r, "UNWIND range($a, $b) AS i MATCH (p:Post {id: i}) WITH p MATCH (u:User {id: p.author}) "
         "CREATE (u)-[:POSTED]->(p)", {"a": i, "b": min(P, i + B) - 1})
for i in range(0, P, 1000):
    q(r, f"UNWIND range($a, $b) AS i MATCH (p:Post {{id: i}}) UNWIND range(1, {LIKES_PER_POST}) AS k "
         f"MATCH (u:User {{id: (i * 31 + k * 977) % $n}}) CREATE (u)-[:LIKES]->(p)",
      {"a": i, "b": min(P, i + 1000) - 1, "n": U})
ingest_s = time.time() - t_ing
nn = U + P
ee = U * FOLLOW_DEG + P + P * LIKES_PER_POST
n_ = sk.scalar(q(r, "MATCH (n) RETURN count(n)"))
e_ = sk.scalar(q(r, "MATCH ()-[e]->() RETURN count(e)"))
checks.add("ingest node count", n_ == nn, f"{n_} != {nn}", "critical")
checks.add("ingest edge count", e_ == ee, f"{e_} != {ee}", "critical")
log(f"ingest {ingest_s:.0f}s nodes={n_} edges={e_}")

# ---------------------------------------------------------------- replica attach (full sync of large graph)
mon.mark("initial-sync")
rep = sk.Server(a.module, PORT_R, out / "replica", extra=["--replicaof", "127.0.0.1", str(a.port)],
                name="replica")
rmon = sk.Monitor(rep, out / "replica")
rmon.start()
rmon.mark("initial-sync")
init_sync_s = wait_link_up(900)
checks.add("replica initial full sync completes (<=900s)", init_sync_s is not None, "", "critical")
log(f"initial sync {init_sync_s}")
rc = sk.connect(PORT_R)
ro_ok = True
try:
    sk.query(rc, G, "MATCH (n:User) RETURN count(n)", ro=True)
except Exception as e:
    ro_ok = False
    log("RO_QUERY on replica failed:", e)
checks.add("GRAPH.RO_QUERY works on replica", ro_ok, "", "critical")
has_cat = False
if init_sync_s is not None:
    converge(60)
    compare("after initial sync")

# ---------------------------------------------------------------- workload fns
OUT = str(out)
EVT_BASE = 10_000_000


def pw(ctx):
    """pause gate for writers; returns True if paused."""
    if PAUSE.is_set():
        time.sleep(0.01)
        return True
    return False


def w_create_evt(ctx):
    if pw(ctx):
        return "_paused"
    st = ctx.state
    n = st.get("n", 0)
    st["n"] = n + 1
    eid = ctx.wid * EVT_BASE + n
    st.setdefault("mine", []).append(eid)
    q(ctx.r, "MATCH (u:User {id: $u}) CREATE (u)-[:EMITTED {w: toInteger(rand() * 1000)}]->"
             "(:Evt {id: $id, r: toInteger(rand() * 1000000), ts: timestamp(), "
             "tag: toString(toInteger(rand() * 1000000))})", {"u": ctx.rng.randrange(U), "id": eid})
    return "w_create_evt"


def w_delete_evt(ctx):
    """delete Evt nodes created by the creator workers (wids 0,1): a window of ids that exist by now."""
    if pw(ctx):
        return "_paused"
    est = int((time.time() - T_SOAK.value) * 30)
    if est < 100:
        time.sleep(0.02)
        return "_idle"
    lo = ctx.rng.randrange(2) * EVT_BASE + ctx.rng.randrange(est)
    q(ctx.r, "MATCH (e:Evt) WHERE e.id >= $lo AND e.id < $hi WITH e LIMIT 1 DETACH DELETE e",
      {"lo": lo, "hi": lo + 200})
    return "w_delete_evt"


def w_update(ctx):
    if pw(ctx):
        return "_paused"
    q(ctx.r, "MATCH (u:User {id: $id}) SET u.score = toInteger(rand() * 1000000000), u.touched = timestamp()",
      {"id": ctx.rng.randrange(U)})
    return "w_update_rand"


def w_merge(ctx):
    if pw(ctx):
        return "_paused"
    q(ctx.r, "MERGE (u:User {id: $id}) ON CREATE SET u.age = 20 + $id % 50, u.name = 'user' + toString($id), "
             "u.score = toInteger(rand() * 1000) ON MATCH SET u.visits = coalesce(u.visits, 0) + 1",
      {"id": ctx.rng.randrange(int(U * 1.05))})
    return "w_merge"


def w_follow(ctx):
    if pw(ctx):
        return "_paused"
    if ctx.rng.random() < 0.6:
        q(ctx.r, "MATCH (a:User {id: $a}), (b:User {id: $b}) CREATE (a)-[:FOLLOWS {w: toInteger(rand() * 100)}]->(b)",
          {"a": ctx.rng.randrange(U), "b": ctx.rng.randrange(U)})
        return "w_follow"
    q(ctx.r, "MATCH (a:User {id: $a})-[f:FOLLOWS]->() WITH f LIMIT 1 DELETE f", {"a": ctx.rng.randrange(U)})
    return "w_unfollow"


def w_like(ctx):
    if pw(ctx):
        return "_paused"
    q(ctx.r, "MATCH (u:User {id: $u}), (p:Post {id: $p}) CREATE (u)-[:LIKES]->(p)",
      {"u": ctx.rng.randrange(U), "p": ctx.rng.randrange(P)})
    return "w_like"


SCHEMA_SEQ = ["CREATE INDEX FOR (e:Evt) ON (e.r)", "CREATE INDEX FOR (u:User) ON (u.age)",
              "CREATE INDEX FOR (p:Post) ON (p.author)", "DROP INDEX ON :Evt(r)",
              "DROP INDEX ON :User(age)", "DROP INDEX ON :Post(author)"]


def w_schema(ctx):
    if pw(ctx):
        return "_paused"
    i = ctx.state.get("i", 0)
    ctx.state["i"] = i + 1
    try:
        q(ctx.r, SCHEMA_SEQ[i % len(SCHEMA_SEQ)])
    except Exception as e:
        # index already exists / does not exist after a drop at an odd moment: not a replication matter
        ctx.rec.record("w_schema_err", 0, ok=False, err=str(e))
        return "w_schema_noop"
    return "w_schema"


def w_ledger(ctx, label="Ack"):
    if pw(ctx):
        return "_paused"
    st = ctx.state
    i = st.get("i", 0)
    st["i"] = i + 1
    k = ctx.wid * EVT_BASE + i
    q(ctx.r, f"CREATE (:{label} {{id: $id}})", {"id": k})
    if i % 3 == 2:
        n = ctx.r.execute_command("WAIT", 1, 2000)
        if n >= 1:
            with open(f"{OUT}/ledger-{label}-{ctx.wid}.txt", "a") as f:
                f.write(f"{k}\n")
        return "w_ledger_wait"
    return "w_ledger"


def w_drill(ctx):
    return w_ledger(ctx, "Drill")


def r_replica(ctx):
    if REPL_DOWN.value:
        time.sleep(0.2)
        ctx.state.pop("c", None)
        return "_gap"
    c = ctx.state.get("c")
    if c is None:
        c = ctx.state["c"] = sk.connect(PORT_R)
    x = ctx.rng.random()
    try:
        if x < 0.45:
            sk.query(c, G, "MATCH (u:User {id: $id})-[:FOLLOWS]->(v) RETURN v.id, v.age", {"id": ctx.rng.randrange(U)}, ro=True)
            op = "rep_1hop"
        elif x < 0.7:
            sk.query(c, G, "MATCH (u:User {id: $id})-[:FOLLOWS*2]->(v) RETURN count(DISTINCT v)",
                     {"id": ctx.rng.randrange(U)}, ro=True)
            op = "rep_2hop"
        elif x < 0.9:
            sk.query(c, G, "MATCH (p:Post {id: $id})<-[:LIKES]-(u) RETURN count(u)", {"id": ctx.rng.randrange(P)}, ro=True)
            op = "rep_post_likes"
        else:
            sk.query(c, G, "MATCH (u:User {id: $id})-[:POSTED]->(p)<-[:LIKES]-(f) RETURN count(DISTINCT f)",
                     {"id": ctx.rng.randrange(U)}, ro=True)
            op = "rep_author_fans"
    except Exception:
        if REPL_DOWN.value:  # restart raced this query
            ctx.state.pop("c", None)
            return "_gap"
        raise
    return op


def lag_init(ctx):
    ctx.state["c"] = sk.connect(PORT_R)


def w_marker(ctx):
    """write marker on primary, time until visible on replica."""
    if pw(ctx) or REPL_DOWN.value:
        time.sleep(0.1)
        return "_gap"
    st = ctx.state
    seq = st.get("seq", 0) + 1
    st["seq"] = seq
    t0 = time.time()
    q(ctx.r, "MERGE (m:Marker {id: $id}) SET m.v = $v", {"id": ctx.wid, "v": seq})
    t1 = time.time()
    ctx.rec.record("marker_write", (t1 - t0) * 1e6)
    while True:
        try:
            v = sk.scalar(sk.query(st["c"], G, "MATCH (m:Marker {id: $id}) RETURN m.v", {"id": ctx.wid}, ro=True))
        except Exception:
            if REPL_DOWN.value:
                st["c"] = sk.connect(PORT_R)
                return "_gap"
            raise
        if v is not None and v >= seq:
            ctx.rec.record("repl_lag", (time.time() - t1) * 1e6)
            return "marker_cycle"
        if time.time() - t1 > 60:
            raise RuntimeError("marker not visible on replica in 60s")
        time.sleep(0.0005)


# ---------------------------------------------------------------- soak
S = a.scale
D = a.duration
wr = WRITE_RATES
specs = [
    sk.WorkerSpec(w_create_evt, rate=40 * wr, count=2, name="wcreate"),
    sk.WorkerSpec(w_delete_evt, rate=25 * wr, count=2, name="wdelete"),
    sk.WorkerSpec(w_update, rate=40 * wr, count=2, name="wupdate"),
    sk.WorkerSpec(w_merge, rate=20 * wr, count=1, name="wmerge"),
    sk.WorkerSpec(w_follow, rate=30 * wr, count=1, name="wfollow"),
    sk.WorkerSpec(w_like, rate=30 * wr, count=1, name="wlike"),
    sk.WorkerSpec(w_schema, rate=1 / max(8.0, min(20.0, D / 8)), count=1, name="wschema"),
    sk.WorkerSpec(w_ledger, rate=30, count=1, name="wledger"),
    sk.WorkerSpec(w_marker, rate=5, count=2, name="wmarker", init=lag_init),
    sk.WorkerSpec(r_replica, rate=150, count=4, name="rrep"),
]
mon.mark("soak")
rmon.mark("soak")
lt = threading.Thread(target=sk.run_load, args=(srv, out, specs, D), daemon=True)
t_soak = time.time()
T_SOAK.value = t_soak
lt.start()
resync = {}


def sleep_until(frac):
    dt = t_soak + D * frac - time.time()
    if dt > 0:
        time.sleep(dt)


cvs = []
sleep_until(0.20)
cv, pz = quiesce_compare("mid-soak Q1")
cvs.append(cv)
sleep_until(0.40)
if rep and lt.is_alive():
    # replica restart with full resync under live write load
    log("replica restart")
    mon.mark("replica-resync")
    rmon.mark("replica-resync")
    REPL_DOWN.value = 1
    time.sleep(0.5)
    pi0 = r.info("stats")
    rep.stop()
    for f in (out / "replica").glob("replica-*.rdb"):
        f.unlink()  # a replica with no usable local dataset: forces FULL resync, not PSYNC continuation
    t = time.time()
    rep.start()
    rs = wait_link_up(900)
    resync["restart_to_link_up_s"] = rs
    resync["restart_total_s"] = time.time() - t
    pi1 = r.info("stats")
    resync["sync_full_delta"] = pi1.get("sync_full", 0) - pi0.get("sync_full", 0)
    resync["sync_partial_ok_delta"] = pi1.get("sync_partial_ok", 0) - pi0.get("sync_partial_ok", 0)
    REPL_DOWN.value = 0
    mon.mark("soak")
    rmon.mark("soak")
    log("resync", resync)
    checks.add("replica full resync after restart completes (<=900s)", rs is not None, str(resync), "critical")
    checks.add("replica restart triggered a FULL resync", resync["sync_full_delta"] >= 1, str(resync), "minor")
sleep_until(0.62)
cv, pz = quiesce_compare("mid-soak Q2 (after resync)")
cvs.append(cv)
lt.join(D * 2 + 600)
mon.mark("verify")
rmon.mark("verify")
time.sleep(1.0)

# ---------------------------------------------------------------- final quiesce compare
PAUSE.set()
cv = converge(300)
checks.add("final: replica caught up", cv is not None, "", "critical")
cvs.append(cv)
ok_final, final_base = compare("final")
if not ok_final:
    time.sleep(5)
    converge(60)
    ok_final, final_base = compare("final (retry)")
PAUSE.clear()

ledger_ok_ids = {}
for f in out.glob("ledger-Ack-*.txt"):
    ids = [int(x) for x in f.read_text().split()]
    ledger_ok_ids[f.name] = (len(ids), max(ids) if ids else None)

# ---------------------------------------------------------------- failover drill
mon.mark("failover")
rmon.mark("failover")
drill = {}
DD = 14.0
dt_ = threading.Thread(target=sk.run_load, args=(srv, out, [sk.WorkerSpec(w_drill, rate=150, count=2, name="drill")], DD),
                       kwargs={"tag": "drill_"}, daemon=True)
dt_.start()
time.sleep(DD * 0.5)
rc = sk.connect(PORT_R)
t = time.time()
drill_resp = rc.execute_command("REPLICAOF", "NO", "ONE")
drill["promote_s"] = time.time() - t
dt_.join(DD + 600)
role = rc.info("replication").get("role")
checks.add("failover: replica promoted (role:master)", role == "master", role, "critical")
bad = []
conf_total = 0
for f in sorted(out.glob("ledger-Drill-*.txt")):
    wid = int(f.stem.split("-")[-1])
    ids = [int(x) for x in f.read_text().split()]
    if not ids:
        continue
    last = max(ids)
    n_conf = last - wid * EVT_BASE + 1  # sequential ids 0..i: all up to last confirmed by that connection
    conf_total += n_conf
    have = sk.scalar(sk.query(rc, G, "MATCH (k:Drill) WHERE k.id >= $lo AND k.id <= $hi RETURN count(k)",
                              {"lo": wid * EVT_BASE, "hi": last}))
    if have != n_conf:
        bad.append(f"worker {wid}: confirmed up to {last} ({n_conf} writes) but promoted replica has {have}")
checks.add("failover: promoted replica has every WAIT-confirmed write", not bad and conf_total > 0,
           f"confirmed={conf_total} {bad}", "critical")
drill["confirmed_writes"] = conf_total
prim_drill = sk.scalar(sk.query(r, G, "MATCH (k:Drill) RETURN count(k)"))
rep_drill = sk.scalar(sk.query(rc, G, "MATCH (k:Drill) RETURN count(k)"))
drill["drill_nodes_primary"] = prim_drill
drill["drill_nodes_promoted"] = rep_drill
# promoted replica must still hold the full final graph and accept writes
rp = checksum(PORT_R)
lost = [k for k in CS_QUERIES if k in final_base and k not in ("cnt:Drill",) and
        not k.startswith("ids:") and k not in ("ack_props",) and rp[k] != final_base[k]]
checks.add("failover: promoted replica still holds the pre-drill graph", not lost, str(lost[:5]), "critical")
try:
    sk.query(rc, G, "CREATE (:AfterFailover {t: timestamp()})")
    w_ok = sk.scalar(sk.query(rc, G, "MATCH (n:AfterFailover) RETURN count(n)")) == 1
except Exception as e:
    w_ok = False
    log("write on promoted replica:", e)
checks.add("failover: promoted replica accepts writes", w_ok, "", "critical")

checks.add("primary alive at end", srv.alive(), "", "critical")
checks.add("replica alive at end", rep.alive(), "", "critical")
for nm, sv in (("primary", srv), ("replica", rep)):
    cr = sv.crashed()
    checks.add(f"{nm} log has no crash/panic markers", not cr, "; ".join(cr[:3]), "critical")
rlog = rep.log.read_text(errors="replace") if rep.log.exists() else ""
dv = [l for l in rlog.splitlines() if "diverg" in l.lower() or "failed to apply" in l.lower()]
checks.add("replica log has no divergence/apply-failure lines", not dv, "; ".join(dv[:3]), "critical")

mon.stop()
rmon.stop()
rsum = sk.analyze(out / "replica", rep)
s = sk.analyze(out, srv)
lag = s["ops"].get("repl_lag", {})
if lag:
    checks.add("replication lag p99 < 1 s", lag["p99_ms"] < 1000, f"p99={lag['p99_ms']:.1f}ms", "major")
    checks.add("no marker timeouts", s["ops"].get("marker_cycle", {}).get("errors", 0) == 0,
               str(s["top_errors"][:3]), "major")
else:
    checks.add("replication lag measured", False, "no repl_lag samples", "major")
for op in ("rep_1hop", "rep_2hop", "rep_post_likes", "rep_author_fans"):
    d = s["ops"].get(op)
    if d:
        checks.add(f"{op} errors == 0", d["errors"] == 0, f"{d['errors']}/{d['count']}", "major")
for op in ("w_create_evt", "w_delete_evt", "w_update_rand", "w_merge", "w_follow", "w_like", "w_ledger"):
    d = s["ops"].get(op)
    if d:
        checks.add(f"{op} errors == 0", d["errors"] == 0, f"{d['errors']}/{d['count']}", "major")
sk.analyze(out, srv)  # refresh with checks
s = sk.analyze(out, srv)
s["meta"] = {"scenario": "s09_replication", "users": U, "posts": P, "nodes": nn, "edges": ee,
             "ingest_s": ingest_s, "initial_sync_s": init_sync_s, "resync": resync, "drill": drill,
             "repl_lag_p50_ms": lag.get("p50_ms"), "repl_lag_p95_ms": lag.get("p95_ms"),
             "repl_lag_p99_ms": lag.get("p99_ms"), "repl_lag_max_ms": lag.get("max_ms"),
             "repl_lag_samples": lag.get("count"), "converge_s": cvs,
             "replica_rss_max_mb": rsum["monitor"].get("rss_max_mb"),
             "replica_rss_end_mb": rsum["monitor"].get("rss_end_mb"),
             "replica_cpu_avg_pct": rsum["monitor"].get("cpu_avg_pct"),
             "replica_used_mem_end_mb": rsum["monitor"].get("used_mem_end_mb"),
             "duration": D, "scale": a.scale, "ledger": ledger_ok_ids}
(out / "summary.json").write_text(sk.json.dumps(s, indent=1, default=str))
rep.stop()
srv.stop()
sk.print_summary(s)
print("meta:", sk.json.dumps(s["meta"], default=str))
