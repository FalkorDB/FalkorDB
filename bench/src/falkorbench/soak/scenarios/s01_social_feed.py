"""s01_social_feed — LDBC-SNB-like social network backend, 95% read / 5% write, zipfian hubs.

  python -I s01_social_feed.py --module <path> --port <port> --out <dir> --duration <sec> --scale <float>

Model (scale 1.0): 1M Person, 1.5M Post, 0.5M Comment, 3M KNOWS (power-law, hubs with thousands of
friends), 2M HAS_CREATOR, 1.5M LIKES, 0.5M REPLY_OF (~7M edges). Range indexes on Person.id, Post.id.
Python holds the full ground truth (adjacency, posts per person, likes) and the writers log every
acknowledged write so global counts, friend/post/like counts, feeds, mutual-friend counts, fof
recommendations and shortest paths are verified exactly before and after the soak.
Persons with id%10==0 never receive friendship writes, so their friend counts are exact live.
"""
import argparse
import heapq
import json
import random
import time
from array import array
from collections import Counter, deque
from pathlib import Path

from falkorbench.soak import soakkit as sk

ap = argparse.ArgumentParser()
sk.add_engine_args(ap)  # --module PATH | --image REF
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--duration", type=float, default=60)
ap.add_argument("--scale", type=float, default=0.05)
ap.add_argument("--read-rate", type=float, default=None, help="total read ops/s override")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

G = "social"
N = max(2000, int(1_000_000 * a.scale))        # persons
NPOST = int(1.5 * N)
NCOMM = N // 2
NKNOWS = 3 * N
NLIKE = int(1.5 * N)
T0 = 1_500_000_000_000                         # creationDate base (ms)
NCITY = 500
NHUB = 200
SEED = 20260101


def pname(i): return f"Person{i}"
def pcity(i): return f"City{i % NCITY}"
def pbday(i): return 7000 + (i * 37) % 12000
def pcd(i): return T0 + i * 1000
def postcd(i): return T0 + i * 100


# ---------------------------------------------------------------- generate ground truth
t_gen = time.time()
rng = random.Random(SEED)
EDGES = set()           # a*N+b with a<b
ADJ = [[] for _ in range(N)]
n_try = 0
while len(EDGES) < NKNOWS:
    n_try += 1
    x = rng.random()
    p = rng.random()
    u = int(N * rng.random() ** 3) if rng.random() < 0.15 else rng.randrange(N)
    v = int(N * x ** 3) if p < 0.5 else rng.randrange(N)
    if u == v:
        continue
    if u > v:
        u, v = v, u
    k = u * N + v
    if k in EDGES:
        continue
    EDGES.add(k)
    ADJ[u].append(v)
    ADJ[v].append(u)
KNOWS_LIST = [(k // N, k % N) for k in EDGES]
rng.shuffle(KNOWS_LIST)

# posts: creator skewed mildly (hubs post more); post id i has creationDate postcd(i)
POST_CREATOR = array("i", [0]) * NPOST
POSTS = [[] for _ in range(N)]                  # person -> [(cd, id)]
for i in range(NPOST):
    c = int(N * rng.random() ** 1.5)
    POST_CREATOR[i] = c
    POSTS[c].append((postcd(i), i))
COMMENTS = [(i, rng.randrange(N), rng.randrange(NPOST)) for i in range(NCOMM)]   # id, creator, post
LIKESET = set()
LIKES0 = []
while len(LIKES0) < NLIKE:
    p = rng.randrange(N)
    m = int(NPOST * rng.random() ** 1.3)
    k = p * NPOST + m
    if k in LIKESET:
        continue
    LIKESET.add(k)
    LIKES0.append((p, m))
LIKES_OUT = Counter(p for p, _ in LIKES0)
HUBS = sorted(range(N), key=lambda i: -len(ADJ[i]))[:NHUB]
STABLE_HUBS = [h for h in HUBS if h % 10 == 0]
gen_s = time.time() - t_gen
print(f"ground truth generated in {gen_s:.1f}s: N={N} knows={len(EDGES)} maxdeg={len(ADJ[HUBS[0]])} "
      f"hub[{NHUB}] mindeg={len(ADJ[HUBS[-1]])}", flush=True)

srv = sk.Server(a.module, a.port, out)
mon = sk.Monitor(srv, out)
mon.start()
checks = sk.Checks(out)
r = srv.client()


def q(cy, params=None, ro=False, r_=None):
    return sk.query(r_ or r, G, cy, params, ro=ro)


# ---------------------------------------------------------------- ingest
mon.mark("ingest")
t_ing = time.time()
q("CREATE INDEX FOR (p:Person) ON (p.id)")
q("CREATE INDEX FOR (p:Post) ON (p.id)")
B = 4000
TXT = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do"
for i in range(0, N, B):
    rows_ = [[j, pname(j), pbday(j), pcity(j), pcd(j)] for j in range(i, min(N, i + B))]
    q("UNWIND $r AS x CREATE (:Person {id: x[0], name: x[1], birthday: x[2], city: x[3], "
      "creationDate: x[4]})", {"r": rows_})
t_nodes = time.time() - t_ing
for i in range(0, len(KNOWS_LIST), B):
    q("UNWIND $r AS x MATCH (a:Person {id: x[0]}), (b:Person {id: x[1]}) "
      "CREATE (a)-[:KNOWS {since: 1}]->(b)", {"r": [list(e) for e in KNOWS_LIST[i:i + B]]})
t_knows = time.time() - t_ing
t_k = time.time() - t_ing
for i in range(0, NPOST, B):
    rows_ = [[j, postcd(j), TXT, POST_CREATOR[j]] for j in range(i, min(NPOST, i + B))]
    q("UNWIND $r AS x MATCH (p:Person {id: x[3]}) "
      "CREATE (:Post {id: x[0], creationDate: x[1], content: x[2]})-[:HAS_CREATOR]->(p)", {"r": rows_})
for i in range(0, NCOMM, B):
    rows_ = [[c[0], T0 + c[0] * 100 + 50, TXT, c[1], c[2]] for c in COMMENTS[i:i + B]]
    q("UNWIND $r AS x MATCH (p:Person {id: x[3]}), (m:Post {id: x[4]}) "
      "CREATE (c:Comment {id: x[0], creationDate: x[1], content: x[2]}) "
      "CREATE (c)-[:HAS_CREATOR]->(p) CREATE (c)-[:REPLY_OF]->(m)", {"r": rows_})
for i in range(0, NLIKE, B):
    q("UNWIND $r AS x MATCH (p:Person {id: x[0]}), (m:Post {id: x[1]}) "
      "CREATE (p)-[:LIKES {at: 1}]->(m)", {"r": [list(e) for e in LIKES0[i:i + B]]})
ingest_s = time.time() - t_ing
print(f"ingest {ingest_s:.1f}s (persons {t_nodes:.1f}s, +knows {t_knows - t_nodes:.1f}s, rest {ingest_s - t_knows:.1f}s)", flush=True)


def cnt(cy):
    return sk.scalar(q(cy))


def global_counts():
    return {
        "person": cnt("MATCH (n:Person) RETURN count(n)"),
        "post": cnt("MATCH (n:Post) RETURN count(n)"),
        "comment": cnt("MATCH (n:Comment) RETURN count(n)"),
        "knows": cnt("MATCH ()-[e:KNOWS]->() RETURN count(e)"),
        "likes": cnt("MATCH ()-[e:LIKES]->() RETURN count(e)"),
        "has_creator": cnt("MATCH ()-[e:HAS_CREATOR]->() RETURN count(e)"),
        "reply_of": cnt("MATCH ()-[e:REPLY_OF]->() RETURN count(e)"),
    }


TRUTH = {"person": N, "post": NPOST, "comment": NCOMM, "knows": len(EDGES), "likes": NLIKE,
         "has_creator": NPOST + NCOMM, "reply_of": NCOMM}
gc = global_counts()
for k, v in TRUTH.items():
    checks.add(f"ingest count {k}", gc[k] == v, f"{gc[k]} != {v}", "critical")

# indexes used? (minor)
try:
    plan = q("MATCH (p:Person {id: 5}) RETURN p.name", ro=False, r_=r) and r.execute_command(
        "GRAPH.EXPLAIN", G, "MATCH (p:Person {id: 5}) RETURN p.name")
    ptxt = b" ".join(plan).decode().lower() if isinstance(plan, list) else str(plan).lower()
    checks.add("Person.id lookup uses an index", "index" in ptxt, ptxt[:200], "minor")
except Exception as e:
    checks.add("Person.id lookup uses an index", False, f"explain failed: {e}", "minor")


# ---------------------------------------------------------------- verification helpers
def cells(res):
    return [[sk.cval(c) for c in row] for row in sk.rows(res)]


def py_feed(p, k=20):
    return [i for _, i in heapq.nlargest(k, (x for f in ADJ[p] for x in POSTS[f]))]


def q_feed(p, r_=None):
    res = q("MATCH (p:Person {id: $id})-[:KNOWS]-(f)<-[:HAS_CREATOR]-(m:Post) "
            "RETURN m.id, m.creationDate ORDER BY m.creationDate DESC LIMIT 20", {"id": p}, ro=True, r_=r_)
    return [x[0] for x in cells(res)]


def py_fof(p, k=10):
    fr = set(ADJ[p])
    c = Counter()
    for f in ADJ[p]:
        for x in ADJ[f]:
            if x != p and x not in fr:
                c[x] += 1
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


FOF_Q = ("MATCH (p:Person {id: $id})-[:KNOWS]-(f)-[:KNOWS]-(fof) "
         "WHERE fof <> p AND NOT (p)-[:KNOWS]-(fof) "
         "RETURN fof.id AS id, count(f) AS mutual ORDER BY mutual DESC, id ASC LIMIT 10")


def q_fof(p, r_=None):
    return [(x[0], x[1]) for x in cells(q(FOF_Q, {"id": p}, ro=True, r_=r_))]


def py_dist(s, t, maxd=4):
    if s == t:
        return 0
    seen = {s: 0}
    dq = deque([s])
    while dq:
        u = dq.popleft()
        if seen[u] >= maxd:
            continue
        for w in ADJ[u]:
            if w not in seen:
                seen[w] = seen[u] + 1
                if w == t:
                    return seen[w]
                dq.append(w)
    return None


SP_Q = ("MATCH (a:Person {id: $a}), (b:Person {id: $b}) "
        "WITH a, b, shortestPath((a)-[:KNOWS*..4]-(b)) AS p RETURN length(p)")


def q_sp(s, t, r_=None):
    rs = sk.rows(q(SP_Q, {"a": s, "b": t}, ro=True, r_=r_))
    return sk.cval(rs[0][0]) if rs else None


vr = random.Random(777)
SAMPLE = sorted(set(vr.sample(range(N), 150) + HUBS[:25] + vr.sample(HUBS, 10)))
PAIRS = [(vr.randrange(N), vr.randrange(N)) for _ in range(8)] + [(HUBS[0], HUBS[1]), (HUBS[2], vr.randrange(N)),
                                                               (HUBS[0], vr.choice(ADJ[HUBS[0]]))]
FEEDP = vr.sample([p for p in range(N) if 0 < len(ADJ[p]) < 40], 20) + HUBS[:3] + vr.sample(HUBS, 2)
FOFP = vr.sample([p for p in range(N) if 0 < len(ADJ[p]) < 25], 12) + vr.sample(HUBS[20:], 2)


def verify(tag, sev="critical", extra_tol=0):
    """Compare DB to the Python truth for the sample set. extra_tol: allowed slack in counts."""
    bad = []
    for p in SAMPLE:
        fc = cnt(f"MATCH (p:Person {{id: {p}}})-[:KNOWS]-(f) RETURN count(f)")
        if abs(fc - len(ADJ[p])) > extra_tol:
            bad.append(f"friends({p}) db={fc} py={len(ADJ[p])}")
        pc = cnt(f"MATCH (p:Person {{id: {p}}})<-[:HAS_CREATOR]-(m:Post) RETURN count(m)")
        if abs(pc - len(POSTS[p])) > extra_tol:
            bad.append(f"posts({p}) db={pc} py={len(POSTS[p])}")
        lc = cnt(f"MATCH (p:Person {{id: {p}}})-[:LIKES]->(m) RETURN count(m)")
        if abs(lc - LIKES_OUT.get(p, 0)) > extra_tol:
            bad.append(f"likes({p}) db={lc} py={LIKES_OUT.get(p, 0)}")
    checks.add(f"{tag}: friend/post/like counts for {len(SAMPLE)} sampled persons", not bad,
               "; ".join(bad[:6]) + f" ({len(bad)} bad)", sev)
    bad = []
    for x, y in PAIRS:
        py = len(set(ADJ[x]) & set(ADJ[y]))
        db = cnt(f"MATCH (a:Person {{id: {x}}})-[:KNOWS]-(m)-[:KNOWS]-(b:Person {{id: {y}}}) "
                 f"RETURN count(DISTINCT m)")
        if abs(db - py) > extra_tol:
            bad.append(f"mutual({x},{y}) db={db} py={py}")
    checks.add(f"{tag}: mutual-friend counts for {len(PAIRS)} fixed pairs", not bad, "; ".join(bad), sev)
    bad = []
    for p in FEEDP:
        db, py = q_feed(p), py_feed(p)
        if db != py and not extra_tol:
            bad.append(f"feed({p}) db={db[:5]} py={py[:5]}")
    checks.add(f"{tag}: home feed top-20 for {len(FEEDP)} persons", not bad, "; ".join(bad[:3]), sev)
    bad = []
    for p in FOFP:
        db, py = q_fof(p), py_fof(p)
        if db != py and not extra_tol:
            bad.append(f"fof({p}) db={db[:4]} py={py[:4]}")
    checks.add(f"{tag}: friend-of-friend top-10 for {len(FOFP)} persons", not bad, "; ".join(bad[:3]), sev)
    bad = []
    for x, y in (PAIRS if SP_OK else []):
        db, py = q_sp(x, y), py_dist(x, y)
        if db != py and not extra_tol:
            bad.append(f"sp({x},{y}) db={db} py={py}")
    checks.add(f"{tag}: bounded shortest path for {len(PAIRS)} pairs", not bad, "; ".join(bad), sev)


SP_OK = True
try:
    q_sp(PAIRS[0][0], PAIRS[0][1])
except Exception as e:  # noqa: BLE001 - e.g. the C reference has no undirected shortestPath
    SP_OK = False
    checks.add("undirected shortestPath supported", False, str(e)[:200], "minor")
mon.mark("verify-pre")
verify("pre-soak")

# ---------------------------------------------------------------- workload
NREAD = 4 if a.scale <= 0.1 else 8
NWRITE = 2
READ_RATE = a.read_rate or 800.0     # total reads/s (calibrated on smoke; hub ops are separate op names)
WRITE_RATE = READ_RATE * 0.05 / 0.95


def zipf_hub(rng):
    """Zipfian rank over the hub list (weight 1/rank)."""
    return HUBS[min(NHUB - 1, int(NHUB ** rng.random()) - 1)]


def key(ctx, p_hub):
    return zipf_hub(ctx.rng) if ctx.rng.random() < p_hub else ctx.rng.randrange(N)


def wrong(ctx, op, msg):
    ctx.rec.record("_wrong_result", 0, ok=False, err=f"{op}: {msg}")


def read_op(ctx):
    rg = ctx.rng
    x = rg.random()
    if x < 0.34:
        p = key(ctx, 0.20)
        rs = cells(q("MATCH (p:Person {id: $id}) RETURN p.id, p.name, p.birthday, p.city, p.creationDate",
                     {"id": p}, ro=True, r_=ctx.r))
        if len(rs) != 1 or rs[0] != [p, pname(p), pbday(p), pcity(p), pcd(p)]:
            wrong(ctx, "profile", f"id={p} got {rs[:1]}")
        return "profile"
    if x < 0.46:
        p = key(ctx, 0.20)
        v = sk.scalar(q("MATCH (p:Person {id: $id})-[:KNOWS]-(f) RETURN count(f)", {"id": p}, ro=True, r_=ctx.r))
        if p % 10 == 0 and v != len(ADJ[p]):
            wrong(ctx, "friend_count", f"id={p} got {v} want {len(ADJ[p])}")
        return "friend_count_hub" if len(ADJ[p]) > 500 else "friend_count"
    if x < 0.66:
        hub = rg.random() < 0.03
        p = zipf_hub(rg) if hub else rg.randrange(N)
        rs = cells(q("MATCH (p:Person {id: $id})-[:KNOWS]-(f)<-[:HAS_CREATOR]-(m:Post) "
                     "RETURN m.id, m.creationDate ORDER BY m.creationDate DESC LIMIT 20", {"id": p},
                     ro=True, r_=ctx.r))
        cds = [x_[1] for x_ in rs]
        if len(rs) > 20 or cds != sorted(cds, reverse=True):
            wrong(ctx, "feed", f"id={p} not ordered/limited")
        return "home_feed_hub" if hub else "home_feed"
    if x < 0.82:
        hub = rg.random() < 0.01
        p = zipf_hub(rg) if hub else rg.randrange(N)
        rs = cells(q(FOF_Q, {"id": p}, ro=True, r_=ctx.r))
        ms = [x_[1] for x_ in rs]
        fr = set(ADJ[p])
        if len(rs) > 10 or ms != sorted(ms, reverse=True) or any(x_[0] == p for x_ in rs):
            wrong(ctx, "fof", f"id={p} bad ordering/self")
        elif p % 10 == 0 and not hub and any(x_[0] in fr for x_ in rs):
            wrong(ctx, "fof", f"id={p} recommends an existing friend")
        return "fof_recommend_hub" if hub else "fof_recommend"
    # shortest path (module without undirected shortestPath: serve a profile read instead)
    if not SP_OK:
        p = key(ctx, 0.2)
        q("MATCH (p:Person {id: $id}) RETURN p.id, p.name", {"id": p}, ro=True, r_=ctx.r)
        return "profile_light"
    s_ = key(ctx, 0.10)
    t_ = key(ctx, 0.10)
    if s_ == t_:
        t_ = (t_ + 1) % N
    v = q_sp(s_, t_, r_=ctx.r)
    if v is not None and not (1 <= v <= 4):
        wrong(ctx, "shortest_path", f"({s_},{t_}) len {v}")
    return "shortest_path"


def w_init(ctx):
    ctx.state["k"] = 0
    ctx.state["ptr"] = 0
    ctx.state["added"] = set()
    ctx.state["log"] = open(out / f"wlog-{ctx.wid}.jsonl", "w")


def wlog(ctx, d):
    f = ctx.state["log"]
    f.write(json.dumps(d) + "\n")
    f.flush()


def write_op(ctx):
    rg = ctx.rng
    wk = ctx.wid - NREAD
    x = rg.random()
    st = ctx.state
    if x < 0.35:                                     # new post
        st["k"] += 1
        pid = NPOST + (wk + 1) * 10_000_000 + st["k"]
        person = key(ctx, 0.05)
        cd = T0 + 10**11 + int(time.time() * 1000) * 1000 + wk * 10 + (st["k"] % 10)   # later than all base posts
        res = q("MATCH (p:Person {id: $p}) CREATE (:Post {id: $id, creationDate: $cd, content: $t})"
                "-[:HAS_CREATOR]->(p)", {"p": person, "id": pid, "cd": cd, "t": TXT}, r_=ctx.r)
        wlog(ctx, {"op": "post", "p": person, "id": pid, "cd": cd})
        return "w_new_post"
    if x < 0.70:                                     # new like (person in own residue class)
        for _ in range(20):
            person = (rg.randrange(N // NWRITE) * NWRITE + wk) % N
            m = int(NPOST * rg.random() ** 1.3)
            k = person * NPOST + m
            if k not in LIKESET and k not in st["added"]:
                break
        else:
            return "w_skip"
        st["added"].add(k)
        q("MATCH (p:Person {id: $p}), (m:Post {id: $m}) CREATE (p)-[:LIKES {at: 2}]->(m)",
          {"p": person, "m": m}, r_=ctx.r)
        wlog(ctx, {"op": "like", "p": person, "m": m})
        return "w_new_like"
    if x < 0.85:                                     # new friendship (never touches id%10==0)
        for _ in range(50):
            u = (rg.randrange(N // NWRITE) * NWRITE + wk) % N
            v = zipf_hub(rg) if rg.random() < 0.3 else rg.randrange(N)
            if u % 10 == 0 or v % 10 == 0 or u == v:
                continue
            lo, hi = (u, v) if u < v else (v, u)
            k = lo * N + hi
            if k in EDGES or k in st["added"]:
                continue
            break
        else:
            return "w_skip"
        st["added"].add(k)
        q("MATCH (a:Person {id: $a}), (b:Person {id: $b}) CREATE (a)-[:KNOWS {since: 2}]->(b)",
          {"a": lo, "b": hi}, r_=ctx.r)
        wlog(ctx, {"op": "friend", "a": lo, "b": hi})
        return "w_new_friendship"
    # delete an old like owned by this writer (index i of LIKES0 with i % NWRITE == wk)
    i = st["ptr"] * NWRITE + wk
    st["ptr"] += 1
    if i >= len(LIKES0):
        return "w_skip"
    person, m = LIKES0[i]
    res = q("MATCH (p:Person {id: $p})-[l:LIKES]->(m:Post {id: $m}) DELETE l", {"p": person, "m": m}, r_=ctx.r)
    st_txt = b" ".join(x_ for x_ in res[-1] if isinstance(x_, bytes)) if isinstance(res, list) and res else b""
    if b"Relationships deleted: 1" not in st_txt:
        wrong(ctx, "w_delete_like", f"nothing deleted for ({person},{m}): {st_txt[:80]}")
    wlog(ctx, {"op": "unlike", "p": person, "m": m})
    return "w_delete_like"


mon.mark("soak")
sk.run_load(srv, out, [
    sk.WorkerSpec(read_op, rate=READ_RATE / NREAD, count=NREAD, name="reader"),
    sk.WorkerSpec(write_op, rate=WRITE_RATE / NWRITE, count=NWRITE, name="writer", init=w_init),
], a.duration)

# ---------------------------------------------------------------- post checks
mon.mark("verify")
alive = srv.alive()
checks.add("server alive after soak", alive, "", "critical")
part = sk.analyze(out, srv)
werr = sum(part["ops"].get(o, {}).get("errors", 0) for o in
           ("w_new_post", "w_new_like", "w_new_friendship", "w_delete_like", "writer"))
nwrong = part["ops"].get("_wrong_result", {}).get("count", 0)
checks.add("no wrong results observed live (profile/friend_count/feed/fof/sp invariants)", nwrong == 0,
           f"{nwrong} wrong; " + str([e for e in part["top_errors"] if "wrong" in e[0] or ":" in e[0]][:3]),
           "critical")
rerr = sum(part["ops"].get(o, {}).get("errors", 0) for o in part["ops"] if not o.startswith("w")
           and o not in ("_wrong_result", "_missed_schedule"))
checks.add("no read errors during soak", rerr == 0, f"{rerr} read errors; {part['top_errors'][:3]}", "major")
checks.add("no write errors during soak", werr == 0, f"{werr} write errors; {part['top_errors'][:3]}", "major")
missed = part["ops"].get("_missed_schedule", {}).get("count", 0)
checks.add("open-loop schedule kept (no dropped ops)", missed == 0, f"{missed} ops dropped by client lag", "major")

# apply the writers' ack log to the truth
acked = Counter()
for f in sorted(out.glob("wlog-*.jsonl")):
    for line in f.read_text().splitlines():
        d = json.loads(line)
        o = d["op"]
        acked[o] += 1
        if o == "post":
            POSTS[d["p"]].append((d["cd"], d["id"]))
            TRUTH["post"] += 1
            TRUTH["has_creator"] += 1
        elif o == "like":
            LIKES_OUT[d["p"]] += 1
            TRUTH["likes"] += 1
        elif o == "unlike":
            LIKES_OUT[d["p"]] -= 1
            TRUTH["likes"] -= 1
        elif o == "friend":
            ADJ[d["a"]].append(d["b"])
            ADJ[d["b"]].append(d["a"])
            EDGES.add(d["a"] * N + d["b"])
            TRUTH["knows"] += 1
print("acked writes:", dict(acked), flush=True)
# The log is written after the ack; a crash between ack and log (or an errored write that did
# apply) would shift counts by at most werr + #workers.
tol = werr + NWRITE if werr else 0
gc = global_counts()
for k, v in TRUTH.items():
    checks.add(f"post-soak global count {k}", abs(gc[k] - v) <= tol, f"db={gc[k]} truth={v}", "critical")
verify("post-soak", extra_tol=tol)

# durability: RDB round trip
mon.mark("restart")
try:
    restart_s = srv.restart(save=True)
    r = srv.client()
    mon.mark("verify-post-restart")
    gc2 = global_counts()
    checks.add("RDB round-trip preserves all counts", all(abs(gc2[k] - gc[k]) == 0 for k in gc),
               f"{gc} vs {gc2}", "critical")
    verify("post-restart", extra_tol=tol)
except Exception as e:  # noqa: BLE001
    restart_s = None
    checks.add("restart round trip", False, f"{type(e).__name__}: {e}", "critical")
checks.add("server alive at end", srv.alive(), "", "critical")
mon.stop()
s = sk.analyze(out, srv)
s["meta"] = {"scenario": "s01_social_feed", "scale": a.scale, "persons": N, "posts": TRUTH["post"],
             "comments": NCOMM, "knows": TRUTH["knows"], "likes": TRUTH["likes"],
             "edges_total": TRUTH["knows"] + TRUTH["likes"] + TRUTH["has_creator"] + TRUTH["reply_of"],
             "max_friend_degree": len(ADJ[HUBS[0]]), "ingest_s": ingest_s, "ground_truth_gen_s": gen_s,
             "restart_s": restart_s, "duration": a.duration, "read_rate": READ_RATE,
             "write_rate": WRITE_RATE, "acked_writes": dict(acked)}
(out / "summary.json").write_text(sk.json.dumps(s, indent=1, default=str))
srv.stop()
sk.print_summary(s)
