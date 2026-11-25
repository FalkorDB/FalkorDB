"""s10_schema_under_load — online schema changes on a live graph.

Continuous reads (point traversals + Python-ground-truth verification queries) and writes
(Live label) run while one churn worker repeatedly creates/drops range indexes (node, edge,
large labels, async build), full-text indexes and unique/mandatory constraints.

Correctness: indexed result == scan result == Python ground truth before/during/after build
and after drop; EXPLAIN shows index use iff operational; unique/mandatory enforced once
OPERATIONAL; creation FAILs on violating data; no violation slips in during construction.

CLI: --module --port --out --duration --scale   (scale 1.0 = ~2M nodes / ~5.4M edges)
"""
import argparse
import json
import threading
import time
from collections import Counter
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
G = "schema"
sc = a.scale
NP = max(2000, int(700_000 * sc))    # Person
NR = max(1500, int(500_000 * sc))    # Product
NO = max(1500, int(500_000 * sc))    # Order
NA = max(1000, int(200_000 * sc))    # Acct (constraint playground)
NL = max(500, int(50_000 * sc))      # Live (written concurrently)
BOUGHT_DEG = 4                       # Person->Product, prop qty
KNOWS_DEG = 2                        # Person->Person, prop since
N_NODES = NP + NR + NO + NA + NL
N_EDGES = NP * BOUGHT_DEG + NP * KNOWS_DEG + NO + NO + NA
TIMEOUT_BUILD = 900                  # s, any single build/constraint wait


# ------------------------------------------------------------------ ground truth (pure functions of id)
def p_age(i): return i % 90
def p_city(i): return (i * 31) % 1000
def p_email(i): return f"p{i}@x.io"
def p_name(i): return f"w{i % 500} t{i % 13}"
def r_cat(j): return (j * 7) % 200
def o_status(o): return o % 5
def b_qty(i, k): return (i + k) % 10 + 1
def kn_since(i, k): return 2000 + (i + k * 3) % 25
def b_tgt(i, k): return (i * 7919 + k * 104729) % NR
def kn_tgt(i, k): return (i * 31 + k * 977 + 1) % NP


T_AGE = Counter(p_age(i) for i in range(NP))
T_CITY = Counter(p_city(i) for i in range(NP))
T_CAT = Counter(r_cat(j) for j in range(NR))
T_STATUS = Counter(o_status(o) for o in range(NO))
T_QTY = Counter(b_qty(i, k) for i in range(NP) for k in range(BOUGHT_DEG))
T_SINCE = Counter(kn_since(i, k) for i in range(NP) for k in range(KNOWS_DEG))
T_NAME_W = Counter(i % 500 for i in range(NP))
T_NAME_T = Counter(i % 13 for i in range(NP))
T_NAME_WT = Counter((i % 500, i % 13) for i in range(NP))

# spec: kind, label/rel, prop, value table (value -> count)
SPECS = {
    "person_age": dict(kind="node", label="Person", prop="age", tab=T_AGE),
    "person_city": dict(kind="node", label="Person", prop="city", tab=T_CITY),
    "product_cat": dict(kind="node", label="Product", prop="cat", tab=T_CAT),
    "order_status": dict(kind="node", label="Order", prop="status", tab=T_STATUS),
    "bought_qty": dict(kind="edge", label="BOUGHT", prop="qty", tab=T_QTY),
    "knows_since": dict(kind="edge", label="KNOWS", prop="since", tab=T_SINCE),
}


def eq_q(sp, lit=False, v=None):
    val = "$v" if not lit else str(v)
    if sp["kind"] == "node":
        return f"MATCH (n:{sp['label']}) WHERE n.{sp['prop']} = {val} RETURN count(n)"
    return f"MATCH ()-[r:{sp['label']}]->() WHERE r.{sp['prop']} = {val} RETURN count(r)"


def range_q(sp):
    if sp["kind"] == "node":
        return (f"MATCH (n:{sp['label']}) WHERE n.{sp['prop']} >= $a AND n.{sp['prop']} <= $b "
                f"RETURN count(n)")
    return (f"MATCH ()-[r:{sp['label']}]->() WHERE r.{sp['prop']} >= $a AND r.{sp['prop']} <= $b "
            f"RETURN count(r)")


def explain(r, q):
    res = r.execute_command("GRAPH.EXPLAIN", G, q)
    return "\n".join(x.decode() if isinstance(x, bytes) else str(x) for x in res)


def idx_rows(r):
    res = sk.query(r, G, "CALL db.indexes() YIELD label, properties, entitytype, status "
                         "RETURN label, properties, entitytype, status", ro=True)
    o = []
    for row in sk.rows(res):
        label = sk.cval(row[0])
        props = [sk.cval(x) for x in row[1][1]] if isinstance(row[1], list) else []
        o.append((label, props, sk.cval(row[2]), sk.cval(row[3])))
    return o


def idx_status(r, label, prop, etype):
    """status string of the index row for (label, etype) listing prop; None if absent."""
    for lb, props, et, st in idx_rows(r):
        if lb == label and et == etype and prop in props:
            return st
    return None


def wait_dropped(rr, label, prop, etype, timeout=60):
    """DROP INDEX may complete asynchronously (C): poll until the field leaves db.indexes()."""
    t = time.time()
    while time.time() - t < timeout:
        if idx_status(rr, label, prop, etype) is None:
            return time.time() - t
        time.sleep(0.01)
    return None


DROP_LAT = []


def con_rows(r):
    res = sk.query(r, G, "CALL db.constraints() YIELD type, label, properties, entitytype, status "
                         "RETURN type, label, properties, entitytype, status", ro=True)
    o = []
    for row in sk.rows(res):
        o.append((sk.cval(row[0]), sk.cval(row[1]), tuple(sk.cval(x) for x in row[2][1]),
                  sk.cval(row[3]), sk.cval(row[4])))
    return o


def con_status(r, ctype, label, prop, etype):
    for t, lb, props, et, st in con_rows(r):
        if t == ctype and lb == label and props == (prop,) and et == etype:
            return st
    return None


def stat(res, key):
    if isinstance(res, list) and res and isinstance(res[-1], list):
        for s in res[-1]:
            s = s.decode() if isinstance(s, bytes) else s
            if s.startswith(key + ":"):
                return int(float(s.split(":")[1].split()[0]))
    return 0


# ------------------------------------------------------------------ findings (written by workers)
def finding(name, ok, detail="", sev="major"):
    with open(out / f"findings-{threading.get_native_id()}-{id(name) % 7}.jsonl", "a") as f:
        f.write(json.dumps({"name": name, "ok": bool(ok), "detail": str(detail)[:600], "sev": sev}) + "\n")


def event(**kw):
    with open(out / f"events-{kw.get('who', 'x')}.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")


srv = sk.Server(a.module, a.port, out)
mon = sk.Monitor(srv, out)
mon.start()
checks = sk.Checks(out)
r = srv.client()
IS_RUST = srv.engine() == "rust"


def Q(q, p=None, ro=False):
    return sk.query(r, G, q, p, ro=ro)


# ------------------------------------------------------------------ ingest
mon.mark("ingest")
t0 = time.time()
B = 10000


def batches(n):
    for i in range(0, n, B):
        yield i, min(n, i + B) - 1


for lo, hi in batches(NP):
    Q("UNWIND range($a,$b) AS i CREATE (:Person {id:i, age:i%90, city:(i*31)%1000, "
      "email:'p'+toString(i)+'@x.io', name:'w'+toString(i%500)+' t'+toString(i%13)})", {"a": lo, "b": hi})
for lo, hi in batches(NR):
    Q("UNWIND range($a,$b) AS i CREATE (:Product {id:i, cat:(i*7)%200, price:i%1000})", {"a": lo, "b": hi})
for lo, hi in batches(NO):
    Q("UNWIND range($a,$b) AS i CREATE (:Order {id:i, status:i%5, amt:(i*13)%10000})", {"a": lo, "b": hi})
for lo, hi in batches(NA):
    Q("UNWIND range($a,$b) AS i CREATE (:Acct {id:i, email:'a'+toString(i)+'@x.io', ph:i%1000, grp:i%100, "
      "opt:CASE WHEN i%2=0 THEN i ELSE null END})", {"a": lo, "b": hi})
for lo, hi in batches(NL):
    Q("UNWIND range($a,$b) AS i CREATE (:Live {id:i, k:i%1000, cnt:0})", {"a": lo, "b": hi})
t_nodes = time.time() - t0
build_log = []   # index/constraint builds on the quiescent graph: (kind, what, entities, seconds)


def timed_index(label, prop, entities, etype="NODE", kind="range"):
    """create an index, poll to OPERATIONAL, return seconds (quiescent)."""
    pat = f"(n:{label})" if etype == "NODE" else f"()-[n:{label}]->()"
    ft = "FULLTEXT " if kind == "fulltext" else ""
    t = time.time()
    Q(f"CREATE {ft}INDEX FOR {pat} ON (n.{prop})")
    et = "NODE" if etype == "NODE" else "RELATIONSHIP"
    while True:
        st = idx_status(r, label, prop, et)
        if st == "OPERATIONAL":
            break
        if time.time() - t > TIMEOUT_BUILD:
            return None
        time.sleep(0.02)
    dt = time.time() - t
    build_log.append({"kind": f"{kind}-{etype.lower()}", "what": f"{label}.{prop}", "entities": entities,
                      "seconds": round(dt, 3)})
    return dt


for lbl, n in (("Person", NP), ("Product", NR), ("Order", NO), ("Acct", NA), ("Live", NL)):
    timed_index(lbl, "id", n)       # permanent id indexes, needed for edge ingest and point reads
for lo, hi in batches(NP):
    Q(f"UNWIND range($a,$b) AS i MATCH (u:Person {{id:i}}) UNWIND range(0,{BOUGHT_DEG - 1}) AS k "
      f"MATCH (p:Product {{id:(i*7919+k*104729)%$nr}}) CREATE (u)-[:BOUGHT {{qty:(i+k)%10+1}}]->(p)",
      {"a": lo, "b": hi, "nr": NR})
    Q(f"UNWIND range($a,$b) AS i MATCH (u:Person {{id:i}}) UNWIND range(0,{KNOWS_DEG - 1}) AS k "
      f"MATCH (v:Person {{id:(i*31+k*977+1)%$np}}) CREATE (u)-[:KNOWS {{since:2000+(i+k*3)%25}}]->(v)",
      {"a": lo, "b": hi, "np": NP})
for lo, hi in batches(NO):
    Q("UNWIND range($a,$b) AS o MATCH (u:Person {id:(o*13)%$np}) MATCH (x:Order {id:o}) "
      "CREATE (u)-[:PLACED]->(x)", {"a": lo, "b": hi, "np": NP})
    Q("UNWIND range($a,$b) AS o MATCH (x:Order {id:o}) MATCH (p:Product {id:(o*17)%$nr}) "
      "CREATE (x)-[:CONTAINS]->(p)", {"a": lo, "b": hi, "nr": NR})
for lo, hi in batches(NA):
    Q("UNWIND range($a,$b) AS i MATCH (x:Acct {id:i}) MATCH (y:Acct {id:(i+1)%$na}) "
      "CREATE (x)-[:LINK {lid:i, grp:i%100}]->(y)", {"a": lo, "b": hi, "na": NA})
ingest_s = time.time() - t0
nodes = sk.scalar(Q("MATCH (n) RETURN count(n)"))
edges = sk.scalar(Q("MATCH ()-[e]->() RETURN count(e)"))
checks.add("ingest node count", nodes == N_NODES, f"{nodes} != {N_NODES}", "critical")
checks.add("ingest edge count", edges == N_EDGES, f"{edges} != {N_EDGES}", "critical")
print(f"ingest {ingest_s:.1f}s nodes={nodes} edges={edges}", flush=True)


# ------------------------------------------------------------------ verification helpers
def verify_eq(rr, sp, v, ro=True):
    got = sk.scalar(sk.query(rr, G, eq_q(sp), {"v": v}, ro=ro))
    return got, sp["tab"].get(v, 0)


def verify_range(rr, sp, lo, hi):
    got = sk.scalar(sk.query(rr, G, range_q(sp), {"a": lo, "b": hi}, ro=True))
    return got, sum(sp["tab"].get(v, 0) for v in range(lo, hi + 1))


def verify_some(rr, sp, rng, n=4):
    """n random eq + range lookups vs ground truth; returns list of mismatches."""
    bad = []
    keys = sorted(sp["tab"])
    for j in range(n):
        v = keys[0] if j == 0 else rng.choice(keys)   # always probe the smallest value (0 / min key)
        g, e = verify_eq(rr, sp, v)
        if g != e:
            bad.append(f"eq {v}: got {g} want {e}")
    lo = rng.choice(keys)
    hi = min(max(keys), lo + rng.randrange(0, 20))
    g, e = verify_range(rr, sp, lo, hi)
    if g != e:
        bad.append(f"range [{lo},{hi}]: got {g} want {e}")
    return bad


def verify_all(rr, sp):
    bad = []
    for v in sorted(sp["tab"]):
        g, e = verify_eq(rr, sp, v)
        if g != e:
            bad.append(f"eq {v}: got {g} want {e}")
            if len(bad) > 5:
                break
    return bad


import random  # noqa: E402

# engine quirk probe: does EXPLAIN use an index scan on a label that has an UNRELATED index? (C does)
Q("CREATE (:Probe {id:1, x:1})")
timed_index("Probe", "id", 1)
QUIRK_INDEX_SCAN_ON_ANY_INDEX = "Index Scan" in explain(r, "MATCH (n:Probe) WHERE n.x = 1 RETURN count(n)")
Q("DROP INDEX FOR (n:Probe) ON (n.id)")
Q("MATCH (n:Probe) DELETE n")
print("quirk index-scan-on-any-index:", QUIRK_INDEX_SCAN_ON_ANY_INDEX, flush=True)


def no_index_plan(ex):
    return QUIRK_INDEX_SCAN_ON_ANY_INDEX or "Index Scan" not in ex


# ------------------------------------------------------------------ baseline: quiescent build timings
mon.mark("baseline")
base_info = {}
for name, sp in SPECS.items():
    ent = (NP if sp["label"] == "Person" else NR if sp["label"] == "Product" else NO) \
        if sp["kind"] == "node" else (NP * BOUGHT_DEG if sp["label"] == "BOUGHT" else NP * KNOWS_DEG)
    etype = "NODE" if sp["kind"] == "node" else "EDGE"
    rng = random.Random(7)
    bad_scan = verify_some(r, sp, rng, 2)
    ex0 = explain(r, eq_q(sp, True, sorted(sp["tab"])[0]))
    dt = timed_index(sp["label"], sp["prop"], ent, etype)
    base_info[name] = dt
    ex1 = explain(r, eq_q(sp, True, sorted(sp["tab"])[0]))
    bad_idx = verify_all(r, sp)
    checks.add(f"baseline {name}: scan result == ground truth", not bad_scan, bad_scan, "critical")
    checks.add(f"baseline {name}: index built", dt is not None, f"timeout {TIMEOUT_BUILD}s", "major")
    checks.add(f"baseline {name}: EXPLAIN scan before index", no_index_plan(ex0), ex0, "major")
    checks.add(f"baseline {name}: EXPLAIN uses index when operational", "Index Scan" in ex1, ex1, "major")
    checks.add(f"baseline {name}: indexed result == ground truth (all values)", not bad_idx, bad_idx, "critical")
    pat = f"(n:{sp['label']})" if sp["kind"] == "node" else f"()-[n:{sp['label']}]->()"
    Q(f"DROP INDEX FOR {pat} ON (n.{sp['prop']})")
    dl = wait_dropped(r, sp["label"], sp["prop"], "NODE" if sp["kind"] == "node" else "RELATIONSHIP")
    DROP_LAT.append(dl)
    checks.add(f"baseline {name}: index disappears from db.indexes after DROP", dl is not None, "timeout", "major")
    ex2 = explain(r, eq_q(sp, True, sorted(sp["tab"])[0]))
    checks.add(f"baseline {name}: EXPLAIN no index after drop", no_index_plan(ex2), ex2, "major")
# string index + unique constraint build timing on the largest label
timed_index("Person", "email", NP)
bad = []
rng = random.Random(3)
for _ in range(20):
    i = rng.randrange(NP)
    got = sk.rows(sk.query(r, G, "MATCH (n:Person {email:$e}) RETURN n.id", {"e": p_email(i)}, ro=True))
    if [sk.cval(x[0]) for x in got] != [i]:
        bad.append((i, got))
checks.add("baseline person.email string index lookups", not bad, bad[:3], "critical")
t = time.time()
try:
    sk.query(r, G, "CREATE (:Person {id:-1, email:'x'})", None)  # warm: no constraint yet; removed below
    Q("MATCH (n:Person {id:-1}) DELETE n")
    r.execute_command("GRAPH.CONSTRAINT", "CREATE", G, "UNIQUE", "NODE", "Person", "PROPERTIES", 1, "email")
    while True:
        st = con_status(r, "UNIQUE", "Person", "email", "NODE")
        if st != "UNDER CONSTRUCTION" and st is not None:
            break
        if time.time() - t > TIMEOUT_BUILD:
            break
        time.sleep(0.02)
    cdt = time.time() - t
    build_log.append({"kind": "unique-constraint-node", "what": "Person.email", "entities": NP,
                      "seconds": round(cdt, 3)})
    checks.add("baseline unique constraint on Person.email OPERATIONAL", st == "OPERATIONAL", st, "major")
    try:
        Q("CREATE (:Person {id:-2, email:'p5@x.io'})")
        checks.add("baseline unique constraint rejects duplicate", False, "accepted", "critical")
        Q("MATCH (n:Person {id:-2}) DELETE n")
    except Exception as e:
        checks.add("baseline unique constraint rejects duplicate", "constraint" in str(e).lower(), str(e), "critical")
    r.execute_command("GRAPH.CONSTRAINT", "DROP", G, "UNIQUE", "NODE", "Person", "PROPERTIES", 1, "email")
except Exception as e:
    checks.add("baseline unique constraint on Person.email", False, f"{type(e).__name__}: {e}", "major")
Q("DROP INDEX FOR (n:Person) ON (n.email)")
wait_dropped(r, "Person", "email", "NODE")
# fulltext build
dt = timed_index("Person", "name", NP, kind="fulltext")
bad = []
for w in (5, 77, 499):
    got = sk.scalar(Q(f"CALL db.idx.fulltext.queryNodes('Person','w{w}') YIELD node RETURN count(node)", ro=True))
    if got != T_NAME_W[w]:
        bad.append(f"w{w}: got {got} want {T_NAME_W[w]}")
got = sk.scalar(Q("CALL db.idx.fulltext.queryNodes('Person','t3') YIELD node RETURN count(node)", ro=True))
if got != T_NAME_T[3]:
    bad.append(f"t3: got {got} want {T_NAME_T[3]}")
checks.add("baseline fulltext Person.name results == ground truth", not bad and dt is not None, bad, "critical")
Q("DROP FULLTEXT INDEX FOR (n:Person) ON (n.name)")
wait_dropped(r, "Person", "name", "NODE")
print("baseline builds:", json.dumps(build_log), flush=True)


# ------------------------------------------------------------------ workers
def _counters_path(wid):
    return out / f"counters-{wid}.json"


def read_op(ctx):
    x = ctx.rng.random()
    if x < 0.7:
        i = ctx.rng.randrange(NP)
        res = sk.query(ctx.r, G, "MATCH (u:Person {id:$id})-[b:BOUGHT]->(p) RETURN p.id, b.qty",
                       {"id": i}, ro=True)
        got = sorted((sk.cval(c[0]), sk.cval(c[1])) for c in sk.rows(res))
        exp = sorted((b_tgt(i, k), b_qty(i, k)) for k in range(BOUGHT_DEG))
        if got != exp:
            finding("point 1-hop BOUGHT traversal == ground truth", False, f"id={i} got={got} want={exp}", "critical")
        return "point_1hop"
    if x < 0.85:
        i = ctx.rng.randrange(NP)
        res = sk.query(ctx.r, G, "MATCH (u:Person {id:$id})-[:KNOWS]->(v)-[:KNOWS]->(w) "
                                 "RETURN count(w)", {"id": i}, ro=True)
        exp = KNOWS_DEG * KNOWS_DEG
        got = sk.scalar(res)
        if got != exp:
            finding("point 2-hop KNOWS count == ground truth", False, f"id={i} got={got} want={exp}", "critical")
        return "point_2hop"
    i = ctx.rng.randrange(NO)
    res = sk.query(ctx.r, G, "MATCH (u:Person)-[:PLACED]->(o:Order {id:$id})-[:CONTAINS]->(p:Product) "
                             "RETURN u.id, p.id", {"id": i}, ro=True)
    got = [(sk.cval(c[0]), sk.cval(c[1])) for c in sk.rows(res)]
    exp = [((i * 13) % NP, (i * 17) % NR)]
    if got != exp:
        finding("order path == ground truth", False, f"id={i} got={got} want={exp}", "critical")
    return "order_path"


def verify_op(ctx):
    """Predicate lookups on the churned properties vs Python ground truth. Whatever the index state
    (none / building / operational / dropping) the answer must be identical."""
    names = ["person_age", "person_city", "product_cat", "order_status", "bought_qty", "knows_since"]
    name = ctx.rng.choice(names)
    sp = SPECS[name]
    keys = sorted(sp["tab"])
    if ctx.rng.random() < 0.7:
        v = ctx.rng.choice(keys)
        g, e = verify_eq(ctx.r, sp, v)
        what = f"eq {v}"
    else:
        lo = ctx.rng.choice(keys)
        hi = min(max(keys), lo + ctx.rng.randrange(0, 25))
        g, e = verify_range(ctx.r, sp, lo, hi)
        what = f"range [{lo},{hi}]"
    if g != e:
        finding(f"live verify {name}: result == ground truth", False, f"{what}: got {g} want {e}", "critical")
    return "verify_" + name


def verify_str_op(ctx):
    """String lookups by email (Person.email / Acct.email) vs ground truth: exact id, or none."""
    if ctx.rng.random() < 0.5:
        i = ctx.rng.randrange(NP)
        res = sk.query(ctx.r, G, "MATCH (n:Person {email:$e}) RETURN n.id", {"e": p_email(i)}, ro=True)
        got = [sk.cval(c[0]) for c in sk.rows(res)]
        exp = [i]
        lab = "Person"
    else:
        i = ctx.rng.randrange(NA)
        res = sk.query(ctx.r, G, "MATCH (n:Acct) WHERE n.email = $e AND n.bad IS NULL RETURN n.id",
                       {"e": f"a{i}@x.io"}, ro=True)
        got = [sk.cval(c[0]) for c in sk.rows(res)]
        exp = [i]
        lab = "Acct"
    if got != exp:
        finding(f"live verify {lab}.email lookup == ground truth", False, f"i={i} got={got}", "critical")
    return "verify_email"


def live_cmp_op(ctx):
    """index-eligible vs index-defeated predicate in ONE query (same snapshot) over the mutating Live label."""
    lo = ctx.rng.randrange(0, 1000)
    hi = min(999, lo + ctx.rng.choice([0, 0, 5, 50, 300]))
    res = sk.query(ctx.r, G,
                   "MATCH (n:Live) WHERE n.k >= $a AND n.k <= $b WITH count(n) AS x "
                   "OPTIONAL MATCH (m:Live) WHERE m.k + 0 >= $a AND m.k + 0 <= $b RETURN x, count(m)",
                   {"a": lo, "b": hi}, ro=True)
    row = sk.rows(res)[0]
    x, y = sk.cval(row[0]), sk.cval(row[1])
    # every comparison is recorded, so "k of n runs failed" has a real n (KNOWN_FLAKY, #3095)
    finding("Live: index-path count == scan-path count (same query)", x == y,
            "" if x == y else f"[{lo},{hi}] idx={x} scan={y}", "critical")
    return "live_cmp"


def w_init(ctx):
    ctx.state.update(inc=0, ins=0, dele=0, mine=[], seq=0)


def _save(ctx):
    s = ctx.state
    _counters_path(ctx.wid).write_text(json.dumps({"inc": s["inc"], "ins": s["ins"], "del": s["dele"]}))


def writer_op(ctx):
    s = ctx.state
    x = ctx.rng.random()
    if x < 0.7:
        res = sk.query(ctx.r, G, "MATCH (n:Live {id:$id}) SET n.k = $k, n.cnt = n.cnt + 1 RETURN n.cnt",
                       {"id": ctx.rng.randrange(NL), "k": ctx.rng.randrange(1000)})
        if len(sk.rows(res)) == 1:
            s["inc"] += 1
        _save(ctx)
        return "live_update"
    if x < 0.9 or not s["mine"]:
        s["seq"] += 1
        nid = 10_000_000 * (ctx.wid + 1) + s["seq"]
        sk.query(ctx.r, G, "CREATE (:Live {id:$id, k:$k, cnt:0})", {"id": nid, "k": ctx.rng.randrange(1000)})
        s["mine"].append(nid)
        s["ins"] += 1
        _save(ctx)
        return "live_insert"
    nid = s["mine"].pop(ctx.rng.randrange(len(s["mine"])))
    res = sk.query(ctx.r, G, "MATCH (n:Live {id:$id}) DELETE n", {"id": nid})
    s["dele"] += stat(res, "Nodes deleted")
    _save(ctx)
    return "live_delete"


# ------------------------------------------------------------------ churn worker
CHURN_ERRS = ("churn_step_error",)


def cstate_init(ctx):
    ctx.state.update(step=0, cycle=0)


def wait_index(ctx, label, prop, etype, check, timeout=TIMEOUT_BUILD):
    """poll until OPERATIONAL; run check() every poll while not. returns (seconds, n_checks_while_building)"""
    t = time.time()
    n_building = 0
    while time.time() - t < timeout:
        st = idx_status(ctx.r, label, prop, etype)
        if st == "OPERATIONAL":
            return time.time() - t, n_building
        if st is None:
            return None, n_building
        if "UNDER CONSTRUCTION" in st:
            n_building += 1
            check()
        time.sleep(0.01)
    return None, n_building


def range_step(ctx, name):
    sp = SPECS[name]
    etype = "NODE" if sp["kind"] == "node" else "RELATIONSHIP"
    pat = f"(n:{sp['label']})" if sp["kind"] == "node" else f"()-[n:{sp['label']}]->()"
    rng = ctx.rng

    def check():
        bad = verify_some(ctx.r, sp, rng, 2)
        if bad:
            finding(f"during build {name}: result == ground truth", False, bad, "critical")
    bad = verify_some(ctx.r, sp, rng, 2)
    finding(f"before index {name}: result == ground truth", not bad, bad, "critical")
    t = time.time()
    sk.query(ctx.r, G, f"CREATE INDEX FOR {pat} ON (n.{sp['prop']})")
    create_s = time.time() - t
    check()
    dt, nb = wait_index(ctx, sp["label"], sp["prop"], etype, check)
    finding(f"index {name} reaches OPERATIONAL", dt is not None, "timeout/absent", "major")
    event(who="churn", kind="range-" + sp["kind"], what=f"{sp['label']}.{sp['prop']}", build_s=(dt or -1) + create_s,
          checks_while_building=nb, t=time.time())
    if dt is not None:
        ex = explain(ctx.r, eq_q(sp, True, sorted(sp["tab"])[0]))
        finding(f"EXPLAIN uses index when {name} operational", "Index Scan" in ex, ex, "major")
        bad = verify_some(ctx.r, sp, rng, 6)
        if bad:
            v0 = sorted(sp["tab"])[0]
            dbg = {"retry_param": [verify_eq(ctx.r, sp, v0) for _ in range(2)],
                   "literal": sk.scalar(sk.query(ctx.r, G, eq_q(sp, True, v0), ro=True)),
                   "literal_rw": sk.scalar(sk.query(ctx.r, G, eq_q(sp, True, v0))),
                   "status": idx_status(ctx.r, sp["label"], sp["prop"], etype),
                   "explain": explain(ctx.r, eq_q(sp, True, v0))}
            bad = list(bad) + [json.dumps(dbg)]
        finding(f"after build {name}: indexed result == ground truth", not bad, bad, "critical")
    td = time.time()
    sk.query(ctx.r, G, f"DROP INDEX FOR {pat} ON (n.{sp['prop']})")
    bad = verify_some(ctx.r, sp, rng, 2)   # right after DROP, before the schema settles
    finding(f"immediately after drop {name}: result == ground truth", not bad, bad, "critical")
    dl = wait_dropped(ctx.r, sp["label"], sp["prop"], etype)
    event(who="churn", kind="drop-latency", what=name, build_s=None, drop_s=dl, t=time.time())
    finding(f"index {name} disappears from db.indexes after DROP", dl is not None, "timeout", "major")
    ex = explain(ctx.r, eq_q(sp, True, sorted(sp["tab"])[0]))
    finding(f"EXPLAIN no index after {name} dropped", no_index_plan(ex), ex, "major")
    bad = verify_some(ctx.r, sp, rng, 3)
    finding(f"after drop {name}: result == ground truth", not bad, bad, "critical")
    return "churn_range_" + name


def string_index_step(ctx):
    rng = ctx.rng
    q = "MATCH (n:Person {email:$e}) RETURN n.id"

    def check():
        for _ in range(2):
            i = rng.randrange(NP)
            got = [sk.cval(c[0]) for c in sk.rows(sk.query(ctx.r, G, q, {"e": p_email(i)}, ro=True))]
            if got != [i]:
                finding("string index Person.email: lookup == ground truth", False, f"i={i} got={got}", "critical")
    t = time.time()
    sk.query(ctx.r, G, "CREATE INDEX FOR (n:Person) ON (n.email)")
    dt, nb = wait_index(ctx, "Person", "email", "NODE", check)
    event(who="churn", kind="range-node-str", what="Person.email", build_s=(dt or -1) + (time.time() - t - (dt or 0)),
          checks_while_building=nb, t=time.time())
    finding("index Person.email reaches OPERATIONAL", dt is not None, "timeout", "major")
    ex = explain(ctx.r, "MATCH (n:Person {email:'p5@x.io'}) RETURN n.id")
    finding("EXPLAIN uses index when Person.email operational", "Index Scan" in ex, ex, "major")
    check()
    sk.query(ctx.r, G, "DROP INDEX FOR (n:Person) ON (n.email)")
    check()
    wait_dropped(ctx.r, "Person", "email", "NODE")
    ex = explain(ctx.r, "MATCH (n:Person {email:'p5@x.io'}) RETURN n.id")
    finding("EXPLAIN no index after Person.email dropped", no_index_plan(ex), ex, "major")
    check()
    return "churn_range_person_email"


def fulltext_step(ctx):
    rng = ctx.rng
    Wd = rng.randrange(500)
    Td = rng.randrange(13)
    qs = [(f"w{Wd}", T_NAME_W[Wd]), (f"t{Td}", T_NAME_T[Td]), (f"w{Wd} t{Td}", T_NAME_WT.get((Wd, Td), 0))]

    def run(q):
        return sk.scalar(sk.query(ctx.r, G, "CALL db.idx.fulltext.queryNodes('Person', $q) YIELD node "
                                            "RETURN count(node)", {"q": q}, ro=True))

    def check():  # while building: error or partial allowed, never MORE than truth
        for q, e in qs[:2]:
            try:
                g = run(q)
            except Exception:
                continue
            if g is not None and g > e:
                finding("fulltext during build: never more than ground truth", False, f"{q}: got {g} want <= {e}",
                        "critical")
    t = time.time()
    sk.query(ctx.r, G, "CREATE FULLTEXT INDEX FOR (n:Person) ON (n.name)")
    dt, nb = wait_index(ctx, "Person", "name", "NODE", check)
    event(who="churn", kind="fulltext-node", what="Person.name", build_s=(dt or -1) + (time.time() - t - (dt or 0)),
          checks_while_building=nb, t=time.time())
    finding("fulltext index reaches OPERATIONAL", dt is not None, "timeout", "major")
    for q, e in qs:
        try:
            g = run(q)
        except Exception as ex:
            g = f"ERR {ex}"
        finding("fulltext operational: result == ground truth", g == e, f"{q}: got {g} want {e}", "critical")
    sk.query(ctx.r, G, "DROP FULLTEXT INDEX FOR (n:Person) ON (n.name)")
    wait_dropped(ctx.r, "Person", "name", "NODE")
    try:
        g = run(qs[0][0])
        event(who="churn", kind="fulltext-after-drop", what=f"returned {g} (truth {qs[0][1]})", t=time.time())
    except Exception as ex:
        event(who="churn", kind="fulltext-after-drop", what=f"error {str(ex)[:80]}", t=time.time())
    return "churn_fulltext"


class Inserter(threading.Thread):
    """Hammer inserts of violating nodes while a constraint is being created."""

    def __init__(self, port, make_q, wid):
        super().__init__(daemon=True)
        self.port, self.make_q, self.stop_ev = port, make_q, threading.Event()
        self.ok = self.rej = self.other = 0
        self.wid = wid

    def run(self):
        rr = sk.connect(self.port)
        k = 0
        while not self.stop_ev.is_set():
            k += 1
            try:
                sk.query(rr, G, self.make_q(k))
                self.ok += 1
            except Exception as e:
                if "constraint" in str(e).lower():
                    self.rej += 1
                else:
                    self.other += 1
            time.sleep(0.002)

    def finish(self):
        self.stop_ev.set()
        self.join(30)


def wait_constraint(ctx, ctype, label, prop, etype):
    t = time.time()
    while time.time() - t < TIMEOUT_BUILD:
        st = con_status(ctx.r, ctype, label, prop, etype)
        if st is not None and "UNDER CONSTRUCTION" not in st:
            return st, time.time() - t
        time.sleep(0.01)
    return "TIMEOUT", time.time() - t


def con_create(ctx, ctype, etype, label, prop):
    ctx.r.execute_command("GRAPH.CONSTRAINT", "CREATE", G, ctype, etype, label, "PROPERTIES", 1, prop)


def con_drop(ctx, ctype, etype, label, prop):
    ctx.r.execute_command("GRAPH.CONSTRAINT", "DROP", G, ctype, etype, label, "PROPERTIES", 1, prop)


def ensure_index(ctx, label, prop, etype="NODE"):
    pat = f"(n:{label})" if etype == "NODE" else f"()-[n:{label}]->()"
    sk.query(ctx.r, G, f"CREATE INDEX FOR {pat} ON (n.{prop})")
    et = "NODE" if etype == "NODE" else "RELATIONSHIP"
    dt, _ = wait_index(ctx, label, prop, et, lambda: None)
    return dt


def drop_index(ctx, label, prop, etype="NODE"):
    pat = f"(n:{label})" if etype == "NODE" else f"()-[n:{label}]->()"
    sk.query(ctx.r, G, f"DROP INDEX FOR {pat} ON (n.{prop})")
    wait_dropped(ctx.r, label, prop, "NODE" if etype == "NODE" else "RELATIONSHIP")


def cleanup_bad(ctx):
    sk.query(ctx.r, G, "MATCH (n:Acct) WHERE n.bad = 1 DETACH DELETE n")


def dup_check(ctx, prop):
    return sk.scalar(sk.query(ctx.r, G, f"MATCH (n:Acct) WHERE n.{prop} IS NOT NULL WITH n.{prop} AS v, count(n) AS c "
                                        f"WHERE c > 1 RETURN count(v)", ro=True))


def unique_ok_step(ctx):
    """Acct.email unique, with dup inserters running during creation."""
    cleanup_bad(ctx)
    dt = ensure_index(ctx, "Acct", "email")
    finding("exact-match index for unique constraint builds", dt is not None, "timeout", "major")
    ins = Inserter(ctx.port, lambda k: f"CREATE (:Acct {{id:{900_000_000 + ctx.state['cycle'] * 100000 + k}, "
                                       f"email:'a{(k * 7919) % NA}@x.io', bad:1}})", ctx.wid)
    ins.start()
    time.sleep(0.05)
    t = time.time()
    try:
        con_create(ctx, "UNIQUE", "NODE", "Acct", "email")
    except Exception as e:
        ins.finish()
        finding("unique constraint create accepted", False, str(e), "major")
        cleanup_bad(ctx)
        drop_index(ctx, "Acct", "email")
        return "churn_unique_ok"
    st, _ = wait_constraint(ctx, "UNIQUE", "Acct", "email", "NODE")
    build_s = time.time() - t
    ins.finish()
    dups = dup_check(ctx, "email")
    event(who="churn", kind="unique-constraint", what="Acct.email", build_s=build_s, status=st, slipped=ins.ok,
          t=time.time())
    # invariant: OPERATIONAL => data is unique. FAILED is acceptable if a violation slipped in meanwhile.
    finding("unique: OPERATIONAL implies no duplicates exist (no slip-through)", not (st == "OPERATIONAL" and dups),
            f"status={st} dup_values={dups} inserted_ok={ins.ok}", "critical")
    finding("unique: constraint settles in OPERATIONAL or FAILED", st in ("OPERATIONAL", "FAILED"), st, "major")
    finding("unique: inserter saw no non-constraint errors", ins.other == 0, f"other={ins.other}", "major")
    if ins.ok == 0 and st == "OPERATIONAL":
        pass  # nothing slipped in: every attempt rejected (or none attempted before op)
    if st == "FAILED":
        finding("unique: FAILED only if a violation was actually inserted", ins.ok > 0 or dups > 0,
                f"FAILED with inserted_ok={ins.ok} dups={dups}", "major")
    cleanup_bad(ctx)
    if st == "OPERATIONAL":
        # now enforced
        try:
            sk.query(ctx.r, G, "CREATE (:Acct {id:-5, email:'a5@x.io', bad:1})")
            finding("unique OPERATIONAL rejects duplicate insert", False, "accepted", "critical")
            cleanup_bad(ctx)
        except Exception as e:
            finding("unique OPERATIONAL rejects duplicate insert", "unique constraint" in str(e).lower(), str(e),
                    "critical")
        try:
            sk.query(ctx.r, G, "MATCH (n:Acct {id:6}) SET n.email = 'a7@x.io'")
            finding("unique OPERATIONAL rejects duplicate via SET", False, "accepted", "critical")
            sk.query(ctx.r, G, "MATCH (n:Acct {id:6}) SET n.email = 'a6@x.io'")
        except Exception as e:
            finding("unique OPERATIONAL rejects duplicate via SET", "unique constraint" in str(e).lower(), str(e),
                    "critical")
        try:
            sk.query(ctx.r, G, "CREATE (:Acct {id:-6, email:'fresh@x.io', bad:1})")
            finding("unique OPERATIONAL accepts fresh value", True)
        except Exception as e:
            finding("unique OPERATIONAL accepts fresh value", False, str(e), "major")
        cleanup_bad(ctx)
        con_drop(ctx, "UNIQUE", "NODE", "Acct", "email")
        try:
            sk.query(ctx.r, G, "CREATE (:Acct {id:-7, email:'a5@x.io', bad:1})")
            finding("unique dropped: duplicates allowed again", True)
        except Exception as e:
            finding("unique dropped: duplicates allowed again", False, str(e), "major")
        cleanup_bad(ctx)
    else:
        try:
            con_drop(ctx, "UNIQUE", "NODE", "Acct", "email")
        except Exception:
            pass
    drop_index(ctx, "Acct", "email")
    return "churn_unique_ok"


def unique_fail_step(ctx):
    """Acct.ph has 1000 distinct values over NA nodes: creation must FAIL."""
    cleanup_bad(ctx)
    ensure_index(ctx, "Acct", "ph")
    con_create(ctx, "UNIQUE", "NODE", "Acct", "ph")
    st, dt = wait_constraint(ctx, "UNIQUE", "Acct", "ph", "NODE")
    event(who="churn", kind="unique-constraint-fail", what="Acct.ph", build_s=dt, status=st, t=time.time())
    finding("unique on violating data ends FAILED", st == "FAILED", f"status={st}", "critical")
    con_drop(ctx, "UNIQUE", "NODE", "Acct", "ph")
    drop_index(ctx, "Acct", "ph")
    return "churn_unique_fail"


def mandatory_step(ctx):
    """Acct.email mandatory with inserters creating nodes WITHOUT email during creation."""
    cleanup_bad(ctx)
    ins = Inserter(ctx.port, lambda k: f"CREATE (:Acct {{id:{800_000_000 + ctx.state['cycle'] * 100000 + k}, "
                                       f"ph:{k}, bad:1}})", ctx.wid)
    ins.start()
    time.sleep(0.05)
    t = time.time()
    try:
        con_create(ctx, "MANDATORY", "NODE", "Acct", "email")
    except Exception as e:
        ins.finish()
        finding("mandatory constraint create accepted", False, str(e), "major")
        cleanup_bad(ctx)
        return "churn_mandatory"
    st, _ = wait_constraint(ctx, "MANDATORY", "Acct", "email", "NODE")
    build_s = time.time() - t
    ins.finish()
    missing = sk.scalar(sk.query(ctx.r, G, "MATCH (n:Acct) WHERE n.email IS NULL RETURN count(n)", ro=True))
    event(who="churn", kind="mandatory-constraint", what="Acct.email", build_s=build_s, status=st, slipped=ins.ok,
          t=time.time())
    finding("mandatory: OPERATIONAL implies no node lacks the property (no slip-through)",
            not (st == "OPERATIONAL" and missing), f"status={st} missing={missing} inserted_ok={ins.ok}", "critical")
    finding("mandatory: constraint settles in OPERATIONAL or FAILED", st in ("OPERATIONAL", "FAILED"), st, "major")
    finding("mandatory: inserter saw no non-constraint errors", ins.other == 0, f"other={ins.other}", "major")
    if st == "FAILED":
        finding("mandatory: FAILED only if a violation was actually inserted", ins.ok > 0 or missing > 0,
                f"FAILED with inserted_ok={ins.ok} missing={missing}", "major")
    cleanup_bad(ctx)
    if st == "OPERATIONAL":
        try:
            sk.query(ctx.r, G, "CREATE (:Acct {id:-8, bad:1})")
            finding("mandatory OPERATIONAL rejects missing property", False, "accepted", "critical")
            cleanup_bad(ctx)
        except Exception as e:
            finding("mandatory OPERATIONAL rejects missing property", "mandatory constraint" in str(e).lower(),
                    str(e), "critical")
        try:
            sk.query(ctx.r, G, "MATCH (n:Acct {id:9}) SET n.email = null")
            finding("mandatory OPERATIONAL rejects SET null", False, "accepted", "critical")
            sk.query(ctx.r, G, "MATCH (n:Acct {id:9}) SET n.email = 'a9@x.io'")
        except Exception as e:
            finding("mandatory OPERATIONAL rejects SET null", "mandatory constraint" in str(e).lower(), str(e),
                    "critical")
    try:
        con_drop(ctx, "MANDATORY", "NODE", "Acct", "email")
    except Exception:
        pass
    return "churn_mandatory"


def mandatory_fail_step(ctx):
    """half of Acct lack `opt`: creation must FAIL."""
    cleanup_bad(ctx)
    con_create(ctx, "MANDATORY", "NODE", "Acct", "opt")
    st, dt = wait_constraint(ctx, "MANDATORY", "Acct", "opt", "NODE")
    event(who="churn", kind="mandatory-constraint-fail", what="Acct.opt", build_s=dt, status=st, t=time.time())
    finding("mandatory on violating data ends FAILED", st == "FAILED", f"status={st}", "critical")
    con_drop(ctx, "MANDATORY", "NODE", "Acct", "opt")
    return "churn_mandatory_fail"


def edge_constraint_step(ctx):
    cleanup_bad(ctx)
    ensure_index(ctx, "LINK", "lid", "EDGE")
    con_create(ctx, "UNIQUE", "RELATIONSHIP", "LINK", "lid")
    st, dt = wait_constraint(ctx, "UNIQUE", "LINK", "lid", "RELATIONSHIP")
    event(who="churn", kind="unique-constraint-edge", what="LINK.lid", build_s=dt, status=st, t=time.time())
    finding("edge unique constraint on unique data OPERATIONAL", st == "OPERATIONAL", f"status={st}", "major")
    con_create(ctx, "MANDATORY", "RELATIONSHIP", "LINK", "lid")
    st2, _ = wait_constraint(ctx, "MANDATORY", "LINK", "lid", "RELATIONSHIP")
    finding("edge mandatory constraint OPERATIONAL", st2 == "OPERATIONAL", f"status={st2}", "major")
    if st == "OPERATIONAL":
        try:
            sk.query(ctx.r, G, "MATCH (a:Acct {id:1}),(b:Acct {id:2}) CREATE (a)-[:LINK {lid:5}]->(b)")
            finding("edge unique rejects duplicate", False, "accepted", "critical")
            sk.query(ctx.r, G, "MATCH ()-[e:LINK {lid:5}]->() WITH e SKIP 1 DELETE e")
        except Exception as e:
            finding("edge unique rejects duplicate", "constraint" in str(e).lower(), str(e), "critical")
    if st2 == "OPERATIONAL":
        try:
            sk.query(ctx.r, G, "MATCH (a:Acct {id:1}),(b:Acct {id:2}) CREATE (a)-[:LINK]->(b)")
            finding("edge mandatory rejects missing property", False, "accepted", "critical")
            sk.query(ctx.r, G, "MATCH ()-[e:LINK]->() WHERE e.lid IS NULL DELETE e")
        except Exception as e:
            finding("edge mandatory rejects missing property", "constraint" in str(e).lower(), str(e), "critical")
    for ct in ("UNIQUE", "MANDATORY"):
        try:
            con_drop(ctx, ct, "RELATIONSHIP", "LINK", "lid")
        except Exception:
            pass
    # failing edge unique: grp = i%100 has duplicates
    ensure_index(ctx, "LINK", "grp", "EDGE")
    con_create(ctx, "UNIQUE", "RELATIONSHIP", "LINK", "grp")
    st3, _ = wait_constraint(ctx, "UNIQUE", "LINK", "grp", "RELATIONSHIP")
    finding("edge unique on violating data ends FAILED", st3 == "FAILED", f"status={st3}", "critical")
    con_drop(ctx, "UNIQUE", "RELATIONSHIP", "LINK", "grp")
    drop_index(ctx, "LINK", "grp", "EDGE")
    drop_index(ctx, "LINK", "lid", "EDGE")
    return "churn_edge_constraint"


def live_schema_step(ctx):
    """Index + constraints on the label that is being written right now."""
    rng = ctx.rng
    n_cmp = 0

    def cmp_once():
        nonlocal n_cmp
        lo = rng.randrange(0, 1000)
        hi = min(999, lo + rng.choice([0, 5, 100]))
        res = sk.query(ctx.r, G, "MATCH (n:Live) WHERE n.k >= $a AND n.k <= $b WITH count(n) AS x "
                                 "OPTIONAL MATCH (m:Live) WHERE m.k + 0 >= $a AND m.k + 0 <= $b RETURN x, count(m)",
                       {"a": lo, "b": hi}, ro=True)
        row = sk.rows(res)[0]
        n_cmp += 1
        # every comparison is recorded, so the check's "k of n runs failed" has a real n
        # (check.py tells #3095's intermittent misses from a total breakage by it)
        same = sk.cval(row[0]) == sk.cval(row[1])
        finding("Live (under writes) index == scan during build/after", same,
                "" if same else f"[{lo},{hi}] idx={sk.cval(row[0])} scan={sk.cval(row[1])}", "critical")
    cmp_once()
    sk.query(ctx.r, G, "CREATE INDEX FOR (n:Live) ON (n.k)")
    dt, nb = wait_index(ctx, "Live", "k", "NODE", cmp_once)
    event(who="churn", kind="range-node-live", what="Live.k", build_s=dt, checks_while_building=nb, t=time.time())
    finding("index Live.k reaches OPERATIONAL under writes", dt is not None, "timeout", "major")
    ex = explain(ctx.r, "MATCH (n:Live) WHERE n.k = 5 RETURN count(n)")
    finding("EXPLAIN uses index when Live.k operational", "Index Scan" in ex, ex, "major")
    for _ in range(10):
        cmp_once()
        time.sleep(0.05)
    # constraints under live writes: writers keep id unique and always set k, so both must go OPERATIONAL
    con_create(ctx, "UNIQUE", "NODE", "Live", "id")
    st, cdt = wait_constraint(ctx, "UNIQUE", "Live", "id", "NODE")
    finding("unique Live.id OPERATIONAL under writes", st == "OPERATIONAL", f"status={st}", "major")
    con_create(ctx, "MANDATORY", "NODE", "Live", "k")
    st2, cdt2 = wait_constraint(ctx, "MANDATORY", "Live", "k", "NODE")
    finding("mandatory Live.k OPERATIONAL under writes", st2 == "OPERATIONAL", f"status={st2}", "major")
    event(who="churn", kind="live-constraints", what="Live", build_s=cdt + cdt2, status=f"{st}/{st2}", t=time.time())
    for _ in range(10):
        cmp_once()
        time.sleep(0.05)
    if st2 == "OPERATIONAL":
        try:
            sk.query(ctx.r, G, "CREATE (:Live {id:-3, cnt:0})")
            finding("mandatory Live.k rejects node without k", False, "accepted", "critical")
            sk.query(ctx.r, G, "MATCH (n:Live {id:-3}) DELETE n")
        except Exception as e:
            finding("mandatory Live.k rejects node without k", "constraint" in str(e).lower(), str(e), "critical")
    for ct, p in (("UNIQUE", "id"), ("MANDATORY", "k")):
        try:
            con_drop(ctx, ct, "NODE", "Live", p)
        except Exception:
            pass
    sk.query(ctx.r, G, "DROP INDEX FOR (n:Live) ON (n.k)")
    wait_dropped(ctx.r, "Live", "k", "NODE")
    for _ in range(5):
        cmp_once()
    event(who="churn", kind="live-cmp-count", what="Live", n=n_cmp, t=time.time())
    return "churn_live_schema"


STEPS = [
    lambda c: range_step(c, "person_age"),
    unique_ok_step,
    lambda c: range_step(c, "bought_qty"),
    live_schema_step,
    fulltext_step,
    unique_fail_step,
    lambda c: range_step(c, "product_cat"),
    mandatory_step,
    string_index_step,
    mandatory_fail_step,
    lambda c: range_step(c, "knows_since"),
    edge_constraint_step,
    lambda c: range_step(c, "person_city"),
    lambda c: range_step(c, "order_status"),
]


class StepErr(Exception):
    pass


def churn_op(ctx):
    i = ctx.state["step"]
    ctx.state["step"] += 1
    if ctx.state["step"] % len(STEPS) == 0:
        ctx.state["cycle"] += 1
    try:
        return STEPS[i % len(STEPS)](ctx)
    except Exception as e:
        finding(f"churn step {i % len(STEPS)} raised", False, f"{type(e).__name__}: {e}", "major")
        # best effort cleanup so later steps start clean
        try:
            for ct, et, lb, p in (("UNIQUE", "NODE", "Acct", "email"), ("UNIQUE", "NODE", "Acct", "ph"),
                                  ("MANDATORY", "NODE", "Acct", "email"), ("MANDATORY", "NODE", "Acct", "opt")):
                try:
                    con_drop(ctx, ct, et, lb, p)
                except Exception:
                    pass
            cleanup_bad(ctx)
        except Exception:
            pass
        time.sleep(1)
        return "churn_step_error"


# ------------------------------------------------------------------ run
mon.mark("soak")
workers = [
    sk.WorkerSpec(read_op, rate=40, count=3, name="reader"),
    sk.WorkerSpec(verify_op, rate=2, count=2, name="verifier"),
    sk.WorkerSpec(verify_str_op, rate=10, count=1, name="verifier_str"),
    sk.WorkerSpec(live_cmp_op, rate=5, count=1, name="live_cmp"),
    sk.WorkerSpec(writer_op, rate=40, count=2, name="writer", init=w_init),
    sk.WorkerSpec(churn_op, rate=None, count=1, name="churn", init=cstate_init),
]
sk.run_load(srv, out, workers, a.duration)

# ------------------------------------------------------------------ post checks
mon.mark("verify")
r = srv.client()
agg = {}
for f in out.glob("findings-*.jsonl"):
    for line in f.read_text().splitlines():
        d = json.loads(line)
        e = agg.setdefault(d["name"], {"n": 0, "bad": 0, "detail": "", "sev": d["sev"]})
        e["n"] += 1
        if not d["ok"]:
            e["bad"] += 1
            if not e["detail"]:
                e["detail"] = d["detail"]
for name, e in sorted(agg.items()):
    checks.add(name, e["bad"] == 0, f"{e['bad']} of {e['n']} runs failed; first: {e['detail']}", e["sev"])
events = []
for f in out.glob("events-*.jsonl"):
    events += [json.loads(x) for x in f.read_text().splitlines()]
steps_done = Counter(e["kind"] for e in events)
checks.add("churn completed at least one pass over index and constraint kinds",
           all(steps_done.get(k, 0) > 0 for k in ("range-node", "range-edge", "fulltext-node", "unique-constraint",
                                                  "mandatory-constraint")), dict(steps_done), "major")

# Live ledger
inc = ins = dele = 0
for f in out.glob("counters-*.json"):
    d = json.loads(f.read_text())
    inc += d["inc"]; ins += d["ins"]; dele += d["del"]
live_n = sk.scalar(sk.query(r, G, "MATCH (n:Live) RETURN count(n)"))
live_cnt = sk.scalar(sk.query(r, G, "MATCH (n:Live) WHERE n.id < $nl RETURN sum(n.cnt)", {"nl": NL}))
checks.add("Live node count == initial + inserted - deleted", live_n == NL + ins - dele,
           f"{live_n} vs {NL}+{ins}-{dele}", "critical")
checks.add("Live acknowledged increments == sum(cnt)", live_cnt == inc, f"{live_cnt} vs {inc}", "critical")

# wait for indexes to settle, then no stray schema
t = time.time()
while time.time() - t < 120 and any("UNDER CONSTRUCTION" in str(x[3]) for x in idx_rows(r)):
    time.sleep(0.5)
left_idx = [(l, p) for l, p, et, st in idx_rows(r)]
left_con = con_rows(r)
checks.add("no index stuck UNDER CONSTRUCTION after load",
           not any("UNDER CONSTRUCTION" in str(x[3]) for x in idx_rows(r)), idx_rows(r), "major")
print("leftover indexes:", left_idx, "constraints:", left_con, flush=True)
for e_ in left_con:  # clean leftovers of an aborted step so the persistence phase is deterministic
    try:
        r.execute_command("GRAPH.CONSTRAINT", "DROP", G, e_[0], "NODE" if e_[3] == "NODE" else "RELATIONSHIP", e_[1],
                          "PROPERTIES", len(e_[2]), *e_[2])
    except Exception:
        pass
sk.query(r, G, "MATCH (n:Acct) WHERE n.bad = 1 DETACH DELETE n")
# Static data must be untouched by all of that
checks.add("Static label counts intact after churn",
           (sk.scalar(sk.query(r, G, "MATCH (n:Person) RETURN count(n)")),
            sk.scalar(sk.query(r, G, "MATCH (n:Acct) RETURN count(n)")),
            sk.scalar(sk.query(r, G, "MATCH ()-[e:BOUGHT]->() RETURN count(e)")),
            sk.scalar(sk.query(r, G, "MATCH ()-[e:LINK]->() RETURN count(e)"))) == (NP, NA, NP * BOUGHT_DEG, NA),
           "", "critical")

# quiescent: Live index vs scan over every k, then schema persistence across SAVE/restart
sk.query(r, G, "CREATE INDEX FOR (n:Live) ON (n.k)")
timed = None
t = time.time()
while time.time() - t < 300:
    if idx_status(r, "Live", "k", "NODE") == "OPERATIONAL":
        break
    time.sleep(0.05)
bad = []
for v in range(0, 1000, 7):
    res = sk.query(r, G, "MATCH (n:Live) WHERE n.k = $v WITH count(n) AS x OPTIONAL MATCH (m:Live) WHERE m.k + 0 = $v "
                         "RETURN x, count(m)", {"v": v}, ro=True)
    row = sk.rows(res)[0]
    if sk.cval(row[0]) != sk.cval(row[1]):
        bad.append((v, sk.cval(row[0]), sk.cval(row[1])))
checks.add("quiescent Live.k: index count == scan count for every sampled k", not bad, bad[:5], "critical")
spec_p = SPECS["person_age"]
sk.query(r, G, "CREATE INDEX FOR (n:Person) ON (n.age)")
sk.query(r, G, "CREATE INDEX FOR ()-[r:BOUGHT]-() ON (r.qty)")
sk.query(r, G, "CREATE FULLTEXT INDEX FOR (n:Person) ON (n.name)")
sk.query(r, G, "CREATE INDEX FOR (n:Acct) ON (n.email)")
t = time.time()
while time.time() - t < 600 and any(x[3] != "OPERATIONAL" for x in idx_rows(r)):
    time.sleep(0.2)
r.execute_command("GRAPH.CONSTRAINT", "CREATE", G, "UNIQUE", "NODE", "Acct", "PROPERTIES", 1, "email")
r.execute_command("GRAPH.CONSTRAINT", "CREATE", G, "MANDATORY", "NODE", "Acct", "PROPERTIES", 1, "id")
t = time.time()
while time.time() - t < 300 and any("UNDER" in str(c[4]) for c in con_rows(r)):
    time.sleep(0.2)
before_i, before_c = sorted(map(str, idx_rows(r))), sorted(map(str, con_rows(r)))
bad = verify_all(r, spec_p)
checks.add("before restart: Person.age indexed results == ground truth", not bad, bad, "critical")
bad = verify_all(r, SPECS["bought_qty"])
checks.add("before restart: BOUGHT.qty indexed results == ground truth", not bad, bad, "critical")
mon.mark("restart")
rs_s = srv.restart(save=True)
r = srv.client()
t = time.time()
while time.time() - t < 600 and any(x[3] != "OPERATIONAL" for x in idx_rows(r)):
    time.sleep(0.2)
after_i, after_c = sorted(map(str, idx_rows(r))), sorted(map(str, con_rows(r)))
checks.add("indexes survive SAVE/restart (same set, OPERATIONAL)", before_i == after_i, f"{before_i} != {after_i}",
           "critical")
checks.add("constraints survive SAVE/restart", before_c == after_c, f"{before_c} != {after_c}", "critical")
bad = verify_all(r, spec_p)
checks.add("after restart: Person.age indexed results == ground truth", not bad, bad, "critical")
bad = verify_some(r, SPECS["bought_qty"], random.Random(5), 5)
checks.add("after restart: BOUGHT.qty indexed results == ground truth", not bad, bad, "critical")
checks.add("after restart: EXPLAIN uses index", "Index Scan" in explain(r, eq_q(spec_p, True, 5)), "", "major")
try:
    sk.query(r, G, "CREATE (:Acct {id:-9, email:'a3@x.io'})")
    checks.add("after restart: unique constraint still enforced", False, "accepted dup", "critical")
    sk.query(r, G, "MATCH (n:Acct {id:-9}) DELETE n")
except Exception as e:
    checks.add("after restart: unique constraint still enforced", "constraint" in str(e).lower(), str(e), "critical")
try:
    sk.query(r, G, "CREATE (:Acct {email:'zzz@x.io'})")
    checks.add("after restart: mandatory constraint still enforced", False, "accepted", "critical")
    sk.query(r, G, "MATCH (n:Acct {email:'zzz@x.io'}) DELETE n")
except Exception as e:
    checks.add("after restart: mandatory constraint still enforced", "constraint" in str(e).lower(), str(e), "critical")
got = sk.scalar(sk.query(r, G, "CALL db.idx.fulltext.queryNodes('Person','w5') YIELD node RETURN count(node)", ro=True))
checks.add("after restart: fulltext results == ground truth", got == T_NAME_W[5], f"{got} vs {T_NAME_W[5]}", "critical")
checks.add("server alive at end", srv.alive(), "", "critical")
mon.stop()
s = sk.analyze(out, srv)
# unexpected errors on non-churn ops
for op, d in s["ops"].items():
    if d["errors"] and not op.startswith(("churn", "_")) and op not in CHURN_ERRS:
        checks.add(f"no errors on op {op}", False, f"{d['errors']} errors; {s['top_errors'][:3]}", "major")
checks.add("no server crash markers", not s["server_log_crash_markers"], s["server_log_crash_markers"][:3], "critical")


def summarize_builds():
    by = {}
    for e in events:
        if "build_s" in e and e["build_s"] is not None and e["build_s"] >= 0:
            by.setdefault(f"{e['kind']}:{e['what']}", []).append(round(e["build_s"], 2))
    return {k: {"n": len(v), "min": min(v), "median": sorted(v)[len(v) // 2], "max": max(v)} for k, v in by.items()}


s["checks"] = json.loads((out / "checks.json").read_text())
s["checks_failed"] = [c for c in s["checks"] if not c["ok"]]
s["meta"] = {"scenario": "s10_schema_under_load", "nodes": N_NODES, "edges": N_EDGES, "ingest_s": ingest_s,
             "nodes_load_s": t_nodes, "duration": a.duration, "scale": a.scale, "restart_s": rs_s,
             "labels": {"Person": NP, "Product": NR, "Order": NO, "Acct": NA, "Live": NL},
             "quiescent_build_log": build_log, "builds_under_load": summarize_builds(),
             "churn_event_counts": dict(steps_done),
             "checks_during_construction": sum(e.get("checks_while_building", 0) or 0 for e in events),
             "drop_latency_s": {"max": max([d for d in DROP_LAT + [e.get("drop_s") for e in events] if d is not None],
                                           default=None)},
             "quirk_index_scan_on_any_index": QUIRK_INDEX_SCAN_ON_ANY_INDEX,
             "fulltext_after_drop": sorted({e["what"] for e in events if e["kind"] == "fulltext-after-drop"})[:3],
             "live_ledger": {"inc": inc, "ins": ins, "del": dele},
             "constraint_outcomes": [(e["kind"], e["what"], e.get("status"), e.get("slipped"))
                                     for e in events if "status" in e][:40]}
(out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
srv.stop()
sk.print_summary(s)
print(json.dumps(s["meta"], indent=1, default=str))
