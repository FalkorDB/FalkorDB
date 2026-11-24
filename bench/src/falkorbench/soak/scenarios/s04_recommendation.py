"""s04_recommendation — e-commerce recommendation engine, aggregation-heavy OLTP.

CLI: python -I s04_recommendation.py --module M --port P --out DIR --duration S --scale F
Full scale (1.0): 500k users, 100k products, 200 categories, 2k brands,
~2M PURCHASED + ~3.2M VIEWED + ~1.2M RATED edges (~6.5M interaction edges).

Ground truth: the whole interaction data set is generated deterministically with numpy;
acknowledged writes are logged per worker (only after the server acknowledged them) and
applied to the in-Python model afterwards. Exact co-purchase top-10 (distinct-customer
count and raw path-multiplicity count), per-product avg rating, price range counts,
price spot values and per-type edge counts are compared at start (after ingest) and
at the end (after the soak).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import soakkit as sk  # noqa: E402

ap = argparse.ArgumentParser()
sk.add_engine_args(ap)  # --module PATH | --image REF
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--duration", type=float, default=60)
ap.add_argument("--scale", type=float, default=0.05)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

G = "reco"
SC = a.scale
U = max(2000, int(500_000 * SC))
P = max(1000, int(100_000 * SC))
NC = max(20, int(200 * min(1.0, max(SC, 0.1)) ** 0.5))   # categories
NB = max(20, int(2000 * SC))                              # brands
N_PUR = int(4 * U)       # target before dedupe (sized for <= 8 min ingest on Rust: ~15k edges/s)
N_VIEW = int(6.5 * U)
N_RATE = int(2.5 * U)
T0 = 1_600_000_000

rng = np.random.default_rng(20261118)


def skew(n, size):
    """product index with popularity skew (id 0 hottest)."""
    return np.minimum((rng.random(size) ** 2 * P).astype(np.int64), P - 1)


def gen_pairs(n):
    u = rng.integers(0, U, n)
    p = skew(P, n)
    key = np.unique(u * P + p)
    return (key // P), (key % P)


pu, pp = gen_pairs(N_PUR)
pq = rng.integers(1, 6, len(pu))
pt = T0 + rng.integers(0, 10**7, len(pu))
vu, vp = gen_pairs(N_VIEW)
vt = T0 + rng.integers(0, 10**7, len(vu))
ru, rp = gen_pairs(N_RATE)
rr = rng.integers(1, 6, len(ru))
rt = T0 + rng.integers(0, 10**7, len(ru))
price = rng.integers(100, 50_000, P)          # cents; ints keep comparisons exact
cat_of = lambda p: (p * 7919) % NC             # noqa: E731
brand_of = lambda p: (p * 31) % NB             # noqa: E731

# verification set: products across the popularity spectrum
vset = sorted({min(P - 1, x) for x in (0, 1, 3, 10, 30, 100, 300, 1000, 3000, P // 4, P // 3, P // 2,
                                       2 * P // 3, P - 1)})

srv = sk.Server(a.module, a.port, out)
mon = sk.Monitor(srv, out)
mon.start()
checks = sk.Checks(out)
r = srv.client()


def rawq(params: dict, q: str, ro=False):
    pre = " ".join(f"{k}={v}" for k, v in params.items())
    cmd = "GRAPH.RO_QUERY" if ro else "GRAPH.QUERY"
    return r.execute_command(cmd, G, f"CYPHER {pre} {q}", "--compact")


def fmt_rows(cols):
    n = len(cols[0])
    cl = [c.tolist() for c in cols]
    fm = "[" + ",".join(["%d"] * len(cl)) + "]"
    return "[" + ",".join(fm % tuple(c[i] for c in cl) for i in range(n)) + "]"


# ---------------------------------------------------------------- ingest
mon.mark("ingest")
t_ing = time.time()
sk.query(r, G, "CREATE INDEX FOR (u:User) ON (u.id)")
sk.query(r, G, "CREATE INDEX FOR (p:Product) ON (p.id)")
sk.query(r, G, "CREATE INDEX FOR (c:Category) ON (c.id)")
sk.query(r, G, "CREATE INDEX FOR (b:Brand) ON (b.id)")
sk.query(r, G, "CREATE INDEX FOR (p:Product) ON (p.price)")
B = 5000
for i in range(0, U, B):
    sk.query(r, G, "UNWIND range($a, $b) AS i CREATE (:User {id: i, name: 'user-' + toString(i), age: 18 + i % 60})",
             {"a": i, "b": min(U, i + B) - 1})
pl = price.tolist()
for i in range(0, P, B):
    rowsl = "[" + ",".join("[%d,%d]" % (j, pl[j]) for j in range(i, min(P, i + B))) + "]"
    rawq({"rows": rowsl}, "UNWIND $rows AS x CREATE (:Product {id: x[0], price: x[1], rank: x[0], "
                          "name: 'product-' + toString(x[0])})")
sk.query(r, G, "UNWIND range(0, $n) AS i CREATE (:Category {id: i, name: 'cat-' + toString(i)})", {"n": NC - 1})
sk.query(r, G, "UNWIND range(0, $n) AS i CREATE (:Brand {id: i, name: 'brand-' + toString(i)})", {"n": NB - 1})
for i in range(0, P, B):
    sk.query(r, G, f"UNWIND range($a, $b) AS i MATCH (p:Product {{id: i}}) "
                   f"MATCH (c:Category {{id: (i * 7919) % {NC}}}) MATCH (b:Brand {{id: (i * 31) % {NB}}}) "
                   f"CREATE (p)-[:IN_CATEGORY]->(c), (p)-[:BY_BRAND]->(b)",
             {"a": i, "b": min(P, i + B) - 1})


def load_edges(label, cols, props, bs=8000):
    n = len(cols[0])
    pr = ", ".join(f"{k}: x[{j + 2}]" for j, k in enumerate(props))
    q = (f"UNWIND $rows AS x MATCH (u:User {{id: x[0]}}) MATCH (p:Product {{id: x[1]}}) "
         f"CREATE (u)-[:{label} {{{pr}}}]->(p)")
    for i in range(0, n, bs):
        rawq({"rows": fmt_rows([c[i:i + bs] for c in cols])}, q)


t_e = time.time()
load_edges("PURCHASED", [pu, pp, pq, pt], ["qty", "ts"])
load_edges("VIEWED", [vu, vp, vt], ["ts"])
load_edges("RATED", [ru, rp, rr, rt], ["rating", "ts"])
edge_load_s = time.time() - t_e
ingest_s = time.time() - t_ing
n_edges = len(pu) + len(vu) + len(ru)
print(f"ingest {ingest_s:.1f}s edges={n_edges} ({n_edges / edge_load_s:.0f} edges/s)", flush=True)


def cnt(q):
    return sk.scalar(sk.query(r, G, q, ro=True))


checks.add("ingest users", cnt("MATCH (u:User) RETURN count(u)") == U, "", "critical")
checks.add("ingest products", cnt("MATCH (u:Product) RETURN count(u)") == P, "", "critical")
checks.add("ingest PURCHASED", cnt("MATCH ()-[e:PURCHASED]->() RETURN count(e)") == len(pu), "", "critical")
checks.add("ingest VIEWED", cnt("MATCH ()-[e:VIEWED]->() RETURN count(e)") == len(vu), "", "critical")
checks.add("ingest RATED", cnt("MATCH ()-[e:RATED]->() RETURN count(e)") == len(ru), "", "critical")
checks.add("ingest IN_CATEGORY", cnt("MATCH ()-[e:IN_CATEGORY]->() RETURN count(e)") == P, "", "critical")

# ---------------------------------------------------------------- ground truth
ALSO_D = ("MATCH (x:Product {id: $x})<-[:PURCHASED]-(u:User)-[:PURCHASED]->(o:Product) WHERE o <> x "
          "RETURN o.id AS id, count(DISTINCT u) AS c ORDER BY c DESC, id ASC LIMIT 10")
# Raw path multiplicity. NOTE: count(*) over a pattern whose edge aliases are unused collapses
# parallel edges in both engines (C parity), so the edges are referenced through properties.
ALSO_M = ("MATCH (x:Product {id: $x})<-[a:PURCHASED]-(u:User)-[b:PURCHASED]->(o:Product) WHERE o <> x "
          "RETURN o.id AS id, count(a.qty + b.qty) AS c ORDER BY c DESC, id ASC LIMIT 10")


def top10(counts):
    nz = np.nonzero(counts)[0]
    order = sorted(nz.tolist(), key=lambda i: (-counts[i], i))[:10]
    return [(i, int(counts[i])) for i in order]


def expected_also(x, PU, PP, distinct):
    if distinct:
        key = np.unique(PU * P + PP)
        PU, PP = key // P, key % P
    mult = np.zeros(U, dtype=np.int64)
    bu, bc = np.unique(PU[PP == x], return_counts=True)
    mult[bu] = bc
    sel = (mult[PU] > 0) & (PP != x)
    cnts = np.bincount(PP[sel], weights=mult[PU[sel]], minlength=P)
    return top10(cnts.astype(np.int64))


def verify(tag, PU, PP, RU, RP, RR, PRICE, view_total, purch_total, sev="critical"):
    bad = []
    for x in vset:
        for q, dist in ((ALSO_D, True), (ALSO_M, False)):
            got = [(sk.cval(c[0]), sk.cval(c[1])) for c in sk.rows(sk.query(r, G, q, {"x": x}, ro=True))]
            exp = expected_also(x, PU, PP, dist)
            if got != exp:
                bad.append(f"x={x} distinct={dist} got={got[:4]}.. exp={exp[:4]}..")
    checks.add(f"{tag}: co-purchase top-10 exact ({len(vset)} products x 2 forms)", not bad, "; ".join(bad[:3]), sev)
    # avg rating per product (sample = verify set + 36 pseudo-random)
    ids = sorted(set(vset) | set(np.random.default_rng(5).integers(0, P, 36).tolist()))
    res = sk.rows(sk.query(r, G, "UNWIND $ids AS id MATCH (:User)-[r:RATED]->(p:Product {id: id}) "
                                 "RETURN id, avg(r.rating), count(r)", {"ids": ids}, ro=True))
    got = {sk.cval(c[0]): (sk.cval(c[1]), sk.cval(c[2])) for c in res}
    sums = np.bincount(RP, weights=RR, minlength=P)
    cn = np.bincount(RP, minlength=P)
    bad = []
    for i in ids:
        if cn[i] == 0:
            if i in got:
                bad.append(f"{i} unexpected")
            continue
        g = got.get(i)
        if g is None or g[1] != cn[i] or abs(g[0] - sums[i] / cn[i]) > 1e-6:
            bad.append(f"{i}: got={g} exp=({sums[i] / cn[i]:.6f},{cn[i]})")
    checks.add(f"{tag}: avg rating + count per product", not bad, "; ".join(bad[:3]), sev)
    # price range scans through the index
    bad = []
    for lo, hi in ((100, 2000), (20000, 20500), (45000, 50000), (7000, 7001)):
        got = cnt(f"MATCH (p:Product) WHERE p.price >= {lo} AND p.price <= {hi} RETURN count(p)")
        exp = int(((PRICE >= lo) & (PRICE <= hi)).sum())
        if got != exp:
            bad.append(f"[{lo},{hi}] got={got} exp={exp}")
    sample = np.random.default_rng(7).integers(0, P, 200).tolist()
    res = sk.rows(sk.query(r, G, "UNWIND $ids AS id MATCH (p:Product {id: id}) RETURN id, p.price",
                           {"ids": sample}, ro=True))
    for c in res:
        if sk.cval(c[1]) != int(PRICE[sk.cval(c[0])]):
            bad.append(f"price {sk.cval(c[0])}")
    checks.add(f"{tag}: price range counts + spot prices", not bad, "; ".join(bad[:3]), sev)
    bad = []
    for lbl, want in (("PURCHASED", purch_total), ("VIEWED", view_total), ("RATED", len(RU))):
        got = cnt(f"MATCH ()-[e:{lbl}]->() RETURN count(e)")
        if got != want:
            bad.append(f"{lbl} got={got} exp={want}")
    checks.add(f"{tag}: edge counts per type", not bad, "; ".join(bad), sev)


mon.mark("verify0")
t_v = time.time()
verify("start", pu, pp, ru, rp, rr, price, len(vu), len(pu))
verify0_s = time.time() - t_v

# ---------------------------------------------------------------- workload
import os  # noqa: E402
# smoke: 4 readers x 40/s + 28 writes/s; full scale: 10 readers x 100/s + 175 writes/s (~15% writes).
# Several moderate workers (not a few fast ones) so one slow aggregation does not queue its siblings.
NREAD = int(os.environ.get("S04_NREAD", 4 if SC < 0.3 else 10))
READ_RATE = float(os.environ.get("S04_READ_RATE", 40 if SC < 0.3 else 100))   # ops/s per reader worker
WRITE_RATE = float(os.environ.get("S04_WRITE_RATE", 28 if SC < 0.3 else 175))  # ops/s, single writer
LOGD = out / "wlog"
LOGD.mkdir(exist_ok=True)
for f in LOGD.glob("*.log"):
    f.unlink()

R_OPS = [("also_bought", 15), ("collab_filter", 8), ("category_top_sellers", 6), ("category_ratings", 6),
         ("user_feed", 15), ("price_range", 15), ("user_profile", 15), ("batch_products", 10),
         ("batch_user_counts", 5), ("product_rating", 5)]
R_NAMES = [n for n, _ in R_OPS]
R_W = [w for _, w in R_OPS]

Q = {
    "also_bought": ALSO_D,
    "collab_filter": (
        "MATCH (u:User {id: $u})-[:PURCHASED]->(p:Product)<-[:PURCHASED]-(o:User) WHERE o <> u "
        "WITH u, o, count(p) AS shared ORDER BY shared DESC LIMIT 50 "
        "MATCH (o)-[:PURCHASED]->(rec:Product) WHERE NOT (u)-[:PURCHASED]->(rec) "
        "RETURN rec.id AS id, sum(shared) AS score ORDER BY score DESC, id ASC LIMIT 10"),
    "category_top_sellers": (
        "MATCH (c:Category {id: $c})<-[:IN_CATEGORY]-(p:Product)<-[pu:PURCHASED]-(:User) "
        "RETURN p.id AS id, count(pu) AS sold, sum(pu.qty) AS units ORDER BY sold DESC, id ASC LIMIT 10"),
    "category_ratings": (
        "MATCH (c:Category {id: $c})<-[:IN_CATEGORY]-(p:Product)<-[r:RATED]-(:User) "
        "WITH p, avg(r.rating) AS a, count(r) AS n WHERE n >= 3 "
        "RETURN count(p), avg(a), max(n), collect(p.id)[0..5]"),
    "user_feed": (
        "MATCH (u:User {id: $u})-[:VIEWED]->(:Product)-[:IN_CATEGORY]->(c:Category) "
        "WITH u, c, count(*) AS views ORDER BY views DESC, c.id ASC LIMIT 3 "
        "MATCH (c)<-[:IN_CATEGORY]-(rec:Product) WHERE NOT (u)-[:PURCHASED]->(rec) "
        "WITH rec ORDER BY rec.rank LIMIT 20 "
        "OPTIONAL MATCH (rec)-[:BY_BRAND]->(b:Brand) RETURN rec.id, rec.price, b.name"),
    "price_range": (
        "MATCH (p:Product) WHERE p.price >= $lo AND p.price <= $hi "
        "RETURN p.id, p.price ORDER BY p.price, p.id LIMIT 20"),
    "user_profile": (
        "MATCH (u:User {id: $u}) OPTIONAL MATCH (u)-[r:RATED]->(:Product) "
        "WITH u, count(r) AS nr, avg(r.rating) AS ar OPTIONAL MATCH (u)-[pu:PURCHASED]->(q:Product) "
        "RETURN u.id, nr, ar, count(q), sum(pu.qty)"),
    "batch_products": (
        "UNWIND $ids AS id MATCH (p:Product {id: id}) OPTIONAL MATCH (p)-[:BY_BRAND]->(b:Brand) "
        "RETURN p.id, p.price, b.name"),
    "batch_user_counts": (
        "UNWIND $ids AS id MATCH (u:User {id: id}) OPTIONAL MATCH (u)-[v:VIEWED]->(:Product) "
        "RETURN id, count(v)"),
    "product_rating": (
        "MATCH (:User)-[r:RATED]->(p:Product {id: $p}) RETURN avg(r.rating), count(r)"),
}


def hot_product(rng_):
    return min(int(rng_.random() ** 2 * P), P - 1)


def read_op(ctx):
    rg = ctx.rng
    op = rg.choices(R_NAMES, R_W)[0]
    prm = {}
    if op == "also_bought":
        prm = {"x": hot_product(rg)}
    elif op in ("collab_filter", "user_feed", "user_profile"):
        prm = {"u": rg.randrange(U)}
    elif op in ("category_top_sellers", "category_ratings"):
        prm = {"c": rg.randrange(NC)}
    elif op == "price_range":
        lo = rg.randrange(100, 49_000)
        prm = {"lo": lo, "hi": lo + rg.choice((10, 50, 200))}
    elif op == "batch_products":
        prm = {"ids": [hot_product(rg) for _ in range(100)]}
    elif op == "batch_user_counts":
        prm = {"ids": [rg.randrange(U) for _ in range(100)]}
    elif op == "product_rating":
        prm = {"p": hot_product(rg)}
    try:
        sk.query(ctx.r, G, Q[op], prm, ro=True)
    except Exception as e:
        e.op = op
        raise
    return op


W_OPS = ["write_view", "write_purchase", "write_rating", "write_price"]
W_W = [55, 25, 15, 5]


def w_init(ctx):
    ctx.state["log"] = open(LOGD / f"w{ctx.wid}.log", "a")


def write_op(ctx):
    rg = ctx.rng
    op = rg.choices(W_OPS, W_W)[0]
    lg = ctx.state["log"]
    try:
        if op in ("write_view", "write_purchase"):
            uu = rg.randrange(U)
            # 5% of writes target the verification set so the checks see them
            pr_ = rg.choice(vset) if rg.random() < 0.05 else hot_product(rg)
            ts = T0 + 10**7 + rg.randrange(10**6)
            if op == "write_view":
                sk.query(ctx.r, G, "MATCH (u:User {id: $u}), (p:Product {id: $p}) "
                                   "CREATE (u)-[:VIEWED {ts: $ts}]->(p)", {"u": uu, "p": pr_, "ts": ts})
                lg.write(f"V {uu} {pr_}\n")
            else:
                qty = rg.randrange(1, 6)
                sk.query(ctx.r, G, "MATCH (u:User {id: $u}), (p:Product {id: $p}) "
                                   "CREATE (u)-[:PURCHASED {qty: $q, ts: $ts}]->(p)",
                         {"u": uu, "p": pr_, "q": qty, "ts": ts})
                lg.write(f"P {uu} {pr_}\n")
        elif op == "write_rating":
            # each rated edge is owned by exactly one writer (single writer worker here)
            j = rg.randrange(len(ru))
            v = rg.randrange(1, 6)
            res = sk.query(ctx.r, G, "MATCH (u:User {id: $u})-[r:RATED]->(p:Product {id: $p}) "
                                     "SET r.rating = $v RETURN count(r)",
                           {"u": int(ru[j]), "p": int(rp[j]), "v": v})
            if sk.scalar(res) != 1:
                raise RuntimeError(f"rating update matched {sk.scalar(res)} edges")
            lg.write(f"R {j} {v}\n")
        else:
            p = rg.randrange(P)
            v = rg.randrange(100, 50_000)
            sk.query(ctx.r, G, "MATCH (p:Product {id: $p}) SET p.price = $v", {"p": p, "v": v})
            lg.write(f"C {p} {v}\n")
        lg.flush()
    except Exception as e:
        e.op = op
        raise
    return op


mon.mark("soak")
t_soak = time.time()
sk.run_load(srv, out, [sk.WorkerSpec(read_op, rate=READ_RATE, count=NREAD, name="reader"),
                       sk.WorkerSpec(write_op, rate=WRITE_RATE, count=1, name="writer", init=w_init)],
            a.duration)
soak_s = time.time() - t_soak

# ---------------------------------------------------------------- final verification
mon.mark("verify")
for _ in range(10):
    if srv.alive():
        break
    time.sleep(1)
checks.add("server alive after soak", srv.alive(), "", "critical")
if srv.alive():
    FPU, FPP, FRU, FRP, FRR, FPRICE = list(pu), list(pp), ru, rp, rr.copy(), price.copy()
    vlog = []
    nview = nres = 0
    adds_u, adds_p = [], []
    for f in sorted(LOGD.glob("*.log")):
        for line in f.read_text().splitlines():
            t = line.split()
            if len(t) < 3:
                continue
            if t[0] == "V":
                nview += 1
            elif t[0] == "P":
                adds_u.append(int(t[1]))
                adds_p.append(int(t[2]))
            elif t[0] == "R":
                FRR[int(t[1])] = int(t[2])
            elif t[0] == "C":
                FPRICE[int(t[1])] = int(t[2])
    PU2 = np.concatenate([pu, np.array(adds_u, dtype=np.int64)])
    PP2 = np.concatenate([pp, np.array(adds_p, dtype=np.int64)])
    t_v = time.time()
    verify("end", PU2, PP2, ru, rp, FRR, FPRICE, len(vu) + nview, len(PU2))
    verify1_s = time.time() - t_v
    print(f"acked writes: views={nview} purchases={len(adds_u)}", flush=True)
else:
    verify1_s = 0
    nview = 0
    adds_u = []

mon.stop()
s = sk.analyze(out, srv)
s["meta"] = {"scenario": "s04_recommendation", "scale": SC, "users": U, "products": P, "categories": NC,
             "brands": NB, "purchased": len(pu), "viewed": len(vu), "rated": len(ru),
             "interaction_edges": n_edges, "ingest_s": ingest_s, "edge_load_s": edge_load_s,
             "edges_per_s": n_edges / edge_load_s, "verify_start_s": verify0_s, "verify_end_s": verify1_s,
             "duration": a.duration, "soak_s": soak_s, "acked_views": nview, "acked_purchases": len(adds_u),
             "read_workers": NREAD, "read_rate_per_worker": READ_RATE, "write_rate": WRITE_RATE}
(out / "summary.json").write_text(sk.json.dumps(s, indent=1, default=str))
srv.stop()
sk.print_summary(s)
