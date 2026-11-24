"""s02_fraud_realtime - real-time payment fraud detection.

Streaming MERGE ingest of card payments (Account/Device/IP/Merchant upserts + CREATE PAID edge
with amount/ts) and account transfers (XFER), concurrent with detection queries: shared-device /
shared-IP rings, 3-hop device->IP rings, velocity checks, merchant fan-in, money-movement cycles.
Fraud rings (device rings, XFER cycles, mule merchants) are planted in the base load and injected
mid-stream; every acknowledged write is logged and replayed in Python to get exact ground truth.

  python -I s02_fraud_realtime.py --module M --port P --out DIR --duration S --scale F
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
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
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for f in list(out.glob("ack-*.log")) + [out / "events.jsonl", out / "inject_fail.log"]:
    f.unlink(missing_ok=True)

G = "fraud"
SC = a.scale
N = max(2000, int(500_000 * SC))          # base accounts
N2 = 2 * N                                 # stream picks accounts from [0, 2N): half are new
ND = int(0.8 * N)                          # base devices
NI = int(0.5 * N)                          # base IPs
NM = max(500, int(5000 * SC))              # merchants
NBP = 2 * N                                # base PAID history edges
NBX = N // 3                               # base XFER edges
RINGS0 = max(4, int(40 * SC ** 0.5))       # planted base device rings
CYC0 = max(4, int(30 * SC ** 0.5))         # planted base XFER cycles
BATCH = 10                                 # tx per ingest query
XBATCH = 5
K_DEV = 7                                  # ring threshold: >= K accounts on one device
K_FAN = 60                                 # fan-in threshold (distinct payers)
MULE_F = 60
R_ACC, R_DEV, R_IP = 10_000_000, 5_000_000, 5_000_000      # base ring id spaces
S_ACC, S_DEV = 20_000_000, 6_000_000                       # stream ring spaces
C_ACC, SC_ACC = 30_000_000, 40_000_000                     # base / stream cycle accounts
MULE_M = 9_000_000

# rates (ops/s) - calibrated against smoke runs
ING_WORKERS, ING_RATE = 4, float(__import__("os").environ.get("S02_ING_RATE", 80))
XF_RATE = 15.0
INJ_RATE = 0.25
RD = {"velocity": (2, 40), "shared_device": (2, 25), "ring_3hop": (1, 10), "fanin": (1, 10),
      "cycle": (1, 25)}


# ------------------------------------------------------------ deterministic generators
def gen_tx(tid, ts):
    rng = random.Random(tid)
    acc = int(N2 * rng.random() ** 2)
    d = acc % ND if rng.random() > 0.02 else rng.randrange(ND)
    i = acc % NI if rng.random() > 0.02 else rng.randrange(NI)
    m = int(NM * rng.random() ** 3)
    return {"id": tid, "a": acc, "d": d, "i": i, "m": m, "amt": rng.randrange(1, 500), "ts": ts}


def gen_xfer(xid, ts):
    rng = random.Random(xid)
    src = min(N2 - 1002, int(N2 * rng.random() ** 2))
    dst = src + 1 + rng.randrange(1000)
    return {"id": xid, "a": src, "b": dst, "amt": rng.randrange(1, 2000), "ts": ts}


Q_TX = ("UNWIND $txs AS t MERGE (a:Account {id: t.a}) ON CREATE SET a.opened = t.ts "
        "MERGE (d:Device {id: t.d}) MERGE (i:IP {id: t.i}) MERGE (m:Merchant {id: t.m}) "
        "MERGE (a)-[:USES]->(d) MERGE (a)-[:FROM]->(i) "
        "CREATE (a)-[:PAID {id: t.id, amount: t.amt, ts: t.ts}]->(m)")
Q_XF = ("UNWIND $xs AS x MERGE (a:Account {id: x.a}) ON CREATE SET a.opened = x.ts "
        "MERGE (b:Account {id: x.b}) ON CREATE SET b.opened = x.ts "
        "CREATE (a)-[:XFER {id: x.id, amount: x.amt, ts: x.ts}]->(b)")
Q_CYC = "MATCH p = (a:Account {id: $id})-[:XFER*3..5]->(a) RETURN count(p)"
Q_CYC_ALT = "MATCH p = (a:Account {id: $id})-[:XFER*3..5]->(b:Account) WHERE b.id = $id RETURN count(p)"
Q_DEV = ("MATCH (a:Account {id: $id})-[:USES]->(:Device)<-[:USES]-(b:Account) WHERE b.id <> $id "
         "RETURN DISTINCT b.id")
Q_3HOP = ("MATCH (a:Account {id: $id})-[:USES]->(:Device)<-[:USES]-(b:Account)-[:FROM]->(:IP)"
          "<-[:FROM]-(c:Account) RETURN count(DISTINCT c)")
Q_VEL = ("MATCH (a:Account {id: $id})-[t:PAID]->() WHERE t.ts >= $since "
         "RETURN count(t), sum(t.amount)")
Q_FAN = ("MATCH (a:Account)-[t:PAID]->(m:Merchant {id: $id}) WHERE t.ts >= $since "
         "RETURN count(DISTINCT a)")


def cyc_query():
    return CYC_Q[0]


CYC_Q = [Q_CYC]


def skew_acc(rng):
    return int(N2 * rng.random() ** 2)


# ------------------------------------------------------------ workers
def ing_init(ctx):
    ctx.state["seq"] = 0
    ctx.state["f"] = open(out / f"ack-{ctx.wid}.log", "a")


def ingest_op(ctx):
    st = ctx.state
    base = ctx.wid * 10 ** 8 + st["seq"] * BATCH
    st["seq"] += 1
    ts = int(time.time())
    sk.query(ctx.r, G, Q_TX, {"txs": [gen_tx(base + j, ts) for j in range(BATCH)]})
    st["f"].write(f"t,{base},{ts}\n")
    st["f"].flush()
    return "ingest_tx"


def xfer_op(ctx):
    st = ctx.state
    base = 2 * 10 ** 12 + ctx.wid * 10 ** 8 + st["seq"] * XBATCH
    st["seq"] += 1
    ts = int(time.time())
    sk.query(ctx.r, G, Q_XF, {"xs": [gen_xfer(base + j, ts) for j in range(XBATCH)]})
    st["f"].write(f"x,{base},{ts}\n")
    st["f"].flush()
    return "ingest_xfer"


def _rows_ids(res):
    return {sk.cval(r[0]) for r in sk.rows(res)}


def inj_init(ctx):
    ctx.state["k"] = 0
    ctx.state["ev"] = open(out / "events.jsonl", "a")
    ctx.state["fail"] = open(out / "inject_fail.log", "a")


def inject_op(ctx):
    st = ctx.state
    k = st["k"]
    st["k"] += 1
    kind = k % 3
    ts = int(time.time())
    if kind == 0:                                      # device ring in the stream
        S = 8 + (k // 3) % 7
        items = []
        for j in range(S):
            t = gen_tx(10 ** 12 + k * 100 + j, ts)
            t.update(a=S_ACC + k * 100 + j, d=S_DEV + k, i=S_DEV + k)
            items.append(t)
        sk.query(ctx.r, G, Q_TX, {"txs": items})
        ev = {"k": k, "kind": "tx", "items": items}
        st["ev"].write(json.dumps(ev) + "\n"); st["ev"].flush()
        t0 = time.time()
        got = _rows_ids(sk.query(ctx.r, G, Q_DEV, {"id": items[0]["a"]}, ro=True))
        ctx.rec.record("detect_after_write_ring", (time.time() - t0) * 1e6)
        exp = {t["a"] for t in items[1:]}
        if got != exp:
            st["fail"].write(f"ring k={k} got={sorted(got)} exp={sorted(exp)}\n"); st["fail"].flush()
        return "inject_ring"
    if kind == 1:                                      # XFER cycle
        L = 3 + (k // 3) % 3
        accs = [SC_ACC + k * 10 + j for j in range(L)]
        items = [{"id": 3 * 10 ** 12 + k * 10 + j, "a": accs[j], "b": accs[(j + 1) % L],
                  "amt": 5000 + j, "ts": ts} for j in range(L)]
        sk.query(ctx.r, G, Q_XF, {"xs": items})
        st["ev"].write(json.dumps({"k": k, "kind": "xfer", "items": items}) + "\n"); st["ev"].flush()
        bad = []
        for acc in accs:
            t0 = time.time()
            c = sk.scalar(sk.query(ctx.r, G, CYC_Q[0], {"id": acc}, ro=True))
            ctx.rec.record("detect_after_write_cycle", (time.time() - t0) * 1e6)
            if c != 1:
                bad.append((acc, c))
        if bad:
            st["fail"].write(f"cycle k={k} bad={bad}\n"); st["fail"].flush()
        return "inject_cycle"
    # mule merchant: MULE_F distinct existing accounts pay one merchant in a burst
    payers = random.Random(10 ** 6 + k).sample(range(N), MULE_F)
    items = []
    for j, acc in enumerate(payers):
        t = gen_tx(10 ** 12 + k * 100 + j, ts)
        t.update(a=acc, d=acc % ND, i=acc % NI, m=MULE_M + k)
        items.append(t)
    sk.query(ctx.r, G, Q_TX, {"txs": items})
    st["ev"].write(json.dumps({"k": k, "kind": "tx", "items": items}) + "\n"); st["ev"].flush()
    t0 = time.time()
    c = sk.scalar(sk.query(ctx.r, G, Q_FAN, {"id": MULE_M + k, "since": ts - 5}, ro=True))
    ctx.rec.record("detect_after_write_fanin", (time.time() - t0) * 1e6)
    if c != MULE_F:
        st["fail"].write(f"mule k={k} got={c} exp={MULE_F}\n"); st["fail"].flush()
    return "inject_mule"


def rd_velocity(ctx):
    sk.query(ctx.r, G, Q_VEL, {"id": skew_acc(ctx.rng), "since": int(time.time()) - 300}, ro=True)
    return "velocity"


def rd_shared_device(ctx):
    sk.query(ctx.r, G, Q_DEV, {"id": skew_acc(ctx.rng)}, ro=True)
    return "shared_device"


def rd_3hop(ctx):
    sk.query(ctx.r, G, Q_3HOP, {"id": skew_acc(ctx.rng)}, ro=True)
    return "ring_3hop"


def rd_fanin(ctx):
    m = int(NM * ctx.rng.random() ** 3)
    sk.query(ctx.r, G, Q_FAN, {"id": m, "since": int(time.time()) - 120}, ro=True)
    return "fanin"


def rd_cycle(ctx):
    if ctx.rng.random() < 0.25:
        acc = C_ACC + ctx.rng.randrange(CYC0) * 10
    else:
        acc = skew_acc(ctx.rng)
    sk.query(ctx.r, G, CYC_Q[0], {"id": acc}, ro=True)
    return "cycle"


# ------------------------------------------------------------ main
srv = sk.Server(a.module, a.port, out)
mon = sk.Monitor(srv, out)
mon.start()
checks = sk.Checks(out)
r = srv.client()

# ---------------- base load
mon.mark("ingest")
t_ing = time.time()
T0 = int(time.time())
B = 5000


def run(q, p=None):
    return sk.query(r, G, q, p)


for lab, cnt in (("Account", N), ("Device", ND), ("IP", NI), ("Merchant", NM)):
    for i in range(0, cnt, B):
        if lab == "Account":
            run("UNWIND range($a, $b) AS i CREATE (:Account {id: i, opened: $t, country: i % 50})",
                {"a": i, "b": min(cnt, i + B) - 1, "t": T0 - 100000})
        else:
            run(f"UNWIND range($a, $b) AS i CREATE (:{lab} {{id: i}})", {"a": i, "b": min(cnt, i + B) - 1})
print(f"  nodes {time.time() - t_ing:.1f}s", flush=True)
for lab in ("Account", "Device", "IP", "Merchant"):
    run(f"CREATE INDEX FOR (n:{lab}) ON (n.id)")
for i in range(0, N, B):
    p = {"a": i, "b": min(N, i + B) - 1, "nd": ND, "ni": NI}
    run("UNWIND range($a, $b) AS i MATCH (x:Account {id: i}) MATCH (d:Device {id: i % $nd}) "
        "CREATE (x)-[:USES]->(d)", p)
    run("UNWIND range($a, $b) AS i MATCH (x:Account {id: i}) MATCH (d:IP {id: i % $ni}) "
        "CREATE (x)-[:FROM]->(d)", p)
print(f"  indexes+usesfrom {time.time() - t_ing:.1f}s", flush=True)
for i in range(0, NBP, B):
    run("UNWIND range($a, $b) AS i MATCH (x:Account {id: (i * 7919) % $n}) "
        "MATCH (m:Merchant {id: (i * 104729) % $nm}) "
        "CREATE (x)-[:PAID {id: -i - 1, amount: (i * 31) % 500 + 1, ts: $t - 3600 - ((i * 37) % 2000000)}]->(m)",
        {"a": i, "b": min(NBP, i + B) - 1, "n": N, "nm": NM, "t": T0})
print(f"  paid {time.time() - t_ing:.1f}s", flush=True)
XM = N - 1000
for i in range(0, NBX, B):
    run("UNWIND range($a, $b) AS i MATCH (x:Account {id: (i * 7919) % $xm}) "
        "MATCH (y:Account {id: ((i * 7919) % $xm) + 1 + (i * 13) % 1000}) "
        "CREATE (x)-[:XFER {id: -i - 1, amount: (i * 17) % 2000 + 1, ts: $t - 3600 - ((i * 41) % 2000000)}]->(y)",
        {"a": i, "b": min(NBX, i + B) - 1, "xm": XM, "t": T0})
print(f"  xfer {time.time() - t_ing:.1f}s", flush=True)
# planted base rings / cycles
ring_members = {}
for rr in range(RINGS0):
    S = 8 + rr % 7
    accs = [R_ACC + rr * 100 + j for j in range(S)]
    ring_members[rr] = accs
    run("UNWIND $ids AS i CREATE (:Account {id: i, opened: $t})", {"ids": accs, "t": T0 - 5000})
    run("CREATE (:Device {id: $d}) CREATE (:IP {id: $d})", {"d": R_DEV + rr})
    run("UNWIND $ids AS i MATCH (x:Account {id: i}) MATCH (d:Device {id: $d}) MATCH (p:IP {id: $d}) "
        "CREATE (x)-[:USES]->(d) CREATE (x)-[:FROM]->(p)", {"ids": accs, "d": R_DEV + rr})
cyc_members = {}
for c in range(CYC0):
    L = 3 + c % 3
    accs = [C_ACC + c * 10 + j for j in range(L)]
    cyc_members[c] = accs
    run("UNWIND $ids AS i CREATE (:Account {id: i, opened: $t})", {"ids": accs, "t": T0 - 5000})
    for j in range(L):
        run("MATCH (x:Account {id: $x}) MATCH (y:Account {id: $y}) "
            "CREATE (x)-[:XFER {id: $e, amount: $m, ts: $t}]->(y)",
            {"x": accs[j], "y": accs[(j + 1) % L], "e": -(10 ** 9 + c * 10 + j), "m": 1000 + j, "t": T0 - 4000})
ingest_s = time.time() - t_ing
print(f"base load {ingest_s:.1f}s", flush=True)

nodes = {lab: sk.scalar(run(f"MATCH (n:{lab}) RETURN count(n)")) for lab in ("Account", "Device", "IP", "Merchant")}
exp_nodes = {"Account": N + sum(len(v) for v in ring_members.values()) + sum(len(v) for v in cyc_members.values()),
             "Device": ND + RINGS0, "IP": NI + RINGS0, "Merchant": NM}
checks.add("base node counts", nodes == exp_nodes, f"{nodes} != {exp_nodes}", "critical")
npaid = sk.scalar(run("MATCH ()-[t:PAID]->() RETURN count(t)"))
checks.add("base PAID count", npaid == NBP, f"{npaid} != {NBP}", "critical")

# cycle query dialect (closed pattern reusing the bound variable vs WHERE-equality)
cyc_ok = None
for q in (Q_CYC, Q_CYC_ALT):
    try:
        v = sk.scalar(sk.query(r, G, q, {"id": cyc_members[0][0]}, ro=True))
        if v == 1:
            CYC_Q[0] = q
            cyc_ok = q
            break
    except Exception as e:  # noqa: BLE001
        print("cycle query variant failed:", str(e)[:120], flush=True)
checks.add("cycle query works on planted base cycle", cyc_ok is not None,
           f"neither variant returned 1; last={v if 'v' in dir() else None}", "major")
if cyc_ok == Q_CYC_ALT:
    checks.add("closed-pattern cycle form (a)-[*]->(a) supported", False, "used WHERE b.id fallback", "minor")

# ---------------- workload
specs = [sk.WorkerSpec(ingest_op, rate=ING_RATE, count=ING_WORKERS, name="ing", init=ing_init),
         sk.WorkerSpec(xfer_op, rate=XF_RATE, count=1, name="xfer", init=ing_init),
         sk.WorkerSpec(inject_op, rate=INJ_RATE, count=1, name="inj", init=inj_init)]
for nm, fn in (("velocity", rd_velocity), ("shared_device", rd_shared_device), ("ring_3hop", rd_3hop),
               ("fanin", rd_fanin), ("cycle", rd_cycle)):
    cnt, rate = RD[nm]
    specs.append(sk.WorkerSpec(fn, rate=rate, count=cnt, name="rd_" + nm))
start_ts = int(time.time())
mon.mark("soak")
t_soak = time.time()
sk.run_load(srv, out, specs, a.duration)
end_ts = int(time.time())
mon.mark("verify")
t_ver = time.time()
partial = sk.analyze(out, srv)
checks.add("server alive after load", srv.alive(), "", "critical")
for nm in ("ingest_tx", "ingest_xfer", "inject_ring", "inject_cycle", "inject_mule", "velocity",
           "shared_device", "ring_3hop", "fanin", "cycle"):
    o = partial["ops"].get(nm)
    if o is None:
        checks.add(f"op {nm} ran", False, "no samples", "major")
    elif o["errors"]:
        checks.add(f"op {nm} errors", False, f"{o['errors']}/{o['count']} errors: {partial['top_errors'][:2]}",
                   "major")
for nm in ("velocity", "shared_device", "cycle", "fanin"):
    o = partial["ops"].get(nm)
    if o:
        checks.add(f"SLO {nm} p99<=100ms", o["p99_ms"] <= 100, f"p99={o['p99_ms']:.1f}ms", "major")
lat = partial["ops"].get("_late_start")
if lat:
    checks.add("open-loop kept schedule (late starts <1% of writes)",
               lat["count"] < 0.01 * max(1, partial["ops"].get("ingest_tx", {}).get("count", 1)),
               f"late_start={lat['count']}", "major")
miss = partial["ops"].get("_missed_schedule")
if miss:
    checks.add("no missed schedule slots", False, f"missed={miss['count']} (server saturated)", "major")

# ---------------- ground truth replay
fwd_dev, rev_dev, fwd_ip, rev_ip = (defaultdict(set) for _ in range(4))
pc, sc, xc = {}, {}, {}      # per-account totals: PAID total, PAID stream-only, XFER out
payers = defaultdict(set)    # stream-only distinct payers per merchant
t_acc, t_dev, t_ip, t_mer = set(), set(), set(), set()
tot = {"paid_n": 0, "paid_sum": 0, "xf_n": 0, "xf_sum": 0}


def bump(dct, k, amt):
    e = dct.get(k)
    if e is None:
        dct[k] = [1, amt]
    else:
        e[0] += 1
        e[1] += amt


def add_tx(t, stream=True):
    acc, d, i, m = t["a"], t["d"], t["i"], t["m"]
    bump(pc, acc, t["amt"])
    tot["paid_n"] += 1
    tot["paid_sum"] += t["amt"]
    if stream:
        bump(sc, acc, t["amt"])
        payers[m].add(acc)
    if not (acc < N and d == acc % ND):
        fwd_dev[acc].add(d); rev_dev[d].add(acc)
    if not (acc < N and i == acc % NI):
        fwd_ip[acc].add(i); rev_ip[i].add(acc)
    if acc >= N:
        t_acc.add(acc)
    if d >= ND:
        t_dev.add(d)
    if i >= NI:
        t_ip.add(i)
    if m >= NM:
        t_mer.add(m)


def add_xf(x):
    bump(xc, x["a"], x["amt"])
    tot["xf_n"] += 1
    tot["xf_sum"] += x["amt"]
    for k in (x["a"], x["b"]):
        if k >= N:
            t_acc.add(k)


t = time.time()
for i in range(NBP):
    acc = (i * 7919) % N
    bump(pc, acc, (i * 31) % 500 + 1)
    tot["paid_n"] += 1
    tot["paid_sum"] += (i * 31) % 500 + 1
for i in range(NBX):
    src = (i * 7919) % XM
    amt = (i * 17) % 2000 + 1
    bump(xc, src, amt)
    tot["xf_n"] += 1
    tot["xf_sum"] += amt
for rr, accs in ring_members.items():
    for acc in accs:
        fwd_dev[acc].add(R_DEV + rr); rev_dev[R_DEV + rr].add(acc)
        fwd_ip[acc].add(R_DEV + rr); rev_ip[R_DEV + rr].add(acc)
        t_acc.add(acc)
    t_dev.add(R_DEV + rr); t_ip.add(R_DEV + rr)
for c, accs in cyc_members.items():
    L = len(accs)
    for j in range(L):
        bump(xc, accs[j], 1000 + j)
        tot["xf_n"] += 1
        tot["xf_sum"] += 1000 + j
        t_acc.add(accs[j])
n_ack_t = n_ack_x = 0
for f in sorted(out.glob("ack-*.log")):
    for line in f.read_text().splitlines():
        kind, base, ts = line.split(",")
        base, ts = int(base), int(ts)
        if kind == "t":
            n_ack_t += 1
            for j in range(BATCH):
                add_tx(gen_tx(base + j, ts))
        else:
            n_ack_x += 1
            for j in range(XBATCH):
                add_xf(gen_xfer(base + j, ts))
events = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()] \
    if (out / "events.jsonl").exists() else []
inj_ring_members, inj_cycles, inj_mules = [], [], []
for ev in events:
    if ev["kind"] == "tx":
        for t_ in ev["items"]:
            add_tx(t_)
        if ev["items"][0]["m"] >= MULE_M:
            inj_mules.append(MULE_M + ev["k"])
        else:
            inj_ring_members.append([t_["a"] for t_ in ev["items"]])
    else:
        for x in ev["items"]:
            add_xf(x)
        inj_cycles.append([x["a"] for x in ev["items"]])
print(f"replay {time.time() - t:.1f}s: {n_ack_t} tx batches, {n_ack_x} xfer batches, {len(events)} events",
      flush=True)
fails = (out / "inject_fail.log").read_text().splitlines() if (out / "inject_fail.log").exists() else []
checks.add("injected rings/cycles/mules detected immediately after write", not fails and len(events) > 0,
           f"{len(fails)} failures of {len(events)} events: {fails[:3]}", "critical")


def devs(acc):
    return ({acc % ND} if acc < N else set()) | fwd_dev.get(acc, set())


def ips(acc):
    return ({acc % NI} if acc < N else set()) | fwd_ip.get(acc, set())


def accs_dev(d):
    return (set(range(d, N, ND)) if d < ND else set()) | rev_dev.get(d, set())


def accs_ip(i):
    return (set(range(i, N, NI)) if i < NI else set()) | rev_ip.get(i, set())


def Q(q, p=None, ro=True, tmo=600000):
    return sk.query(r, G, q, p, ro=ro, timeout_ms=tmo)


try:
    # 1. totals
    c = sk.query(r, G, "MATCH ()-[t:PAID]->() RETURN count(t), sum(t.amount)", ro=True)
    got = [sk.cval(x) for x in sk.rows(c)[0]]
    checks.add("total PAID count/sum == Python", got == [tot["paid_n"], tot["paid_sum"]],
               f"{got} != {[tot['paid_n'], tot['paid_sum']]}", "critical")
    c = sk.query(r, G, "MATCH ()-[t:XFER]->() RETURN count(t), sum(t.amount)", ro=True)
    got = [sk.cval(x) for x in sk.rows(c)[0]]
    checks.add("total XFER count/sum == Python", got == [tot["xf_n"], tot["xf_sum"]],
               f"{got} != {[tot['xf_n'], tot['xf_sum']]}", "critical")
    # 2. no duplicate nodes + exact node counts
    exp = {"Account": N + len(t_acc), "Device": ND + len(t_dev), "IP": NI + len(t_ip),
           "Merchant": NM + len(t_mer)}
    # t_acc holds ids >= N only (base ring accts too); Device/IP: ring/stream-ring ids (>= ND/NI)
    got = {}
    dup = {}
    for lab in exp:
        got[lab] = sk.scalar(Q(f"MATCH (n:{lab}) RETURN count(n)"))
        dup[lab] = sk.scalar(Q(f"MATCH (n:{lab}) WITH n.id AS id, count(*) AS c WHERE c > 1 RETURN count(*)"))
    checks.add("MERGE created no duplicate Account/Device/IP/Merchant nodes", not any(dup.values()),
               f"duplicate keys per label: {dup}", "critical")
    checks.add("node counts == Python distinct keys", got == exp, f"{got} != {exp}", "critical")
    # 3. per-account PAID and XFER totals, all accounts
    for nm, rel, truth in (("PAID", "PAID", pc), ("XFER", "XFER", xc)):
        bad, checked = [], 0
        for lo, hi in [(i, min(N2, i + 250_000)) for i in range(0, N2, 250_000)] + [(R_ACC, SC_ACC + 10 ** 7)]:
            res = Q(f"MATCH (a:Account)-[t:{rel}]->() WHERE a.id >= $lo AND a.id < $hi "
                    f"RETURN a.id, count(t), sum(t.amount)", {"lo": lo, "hi": hi})
            gotd = {sk.cval(x[0]): [sk.cval(x[1]), sk.cval(x[2])] for x in sk.rows(res)}
            want = {k: v for k, v in truth.items() if lo <= k < hi}
            checked += len(want)
            for k in want.keys() | gotd.keys():
                gv = gotd.get(k)
                if gv is not None:
                    gv = [gv[0], int(gv[1])]
                if gv != want.get(k):
                    bad.append((k, gv, want.get(k)))
        checks.add(f"per-account {nm} count/sum == Python", not bad,
                   f"{checked} accounts, {len(bad)} mismatches, e.g. {bad[:3]}", "critical")
    # 4. velocity on stream-only window
    ids = sorted(sc, key=lambda k: -sc[k][0])[:100] + random.Random(5).sample(sorted(sc), min(200, len(sc)))
    bad = []
    for acc in ids:
        v = sk.query(r, G, Q_VEL, {"id": acc, "since": start_ts - 1}, ro=True)
        gv = [sk.cval(x) for x in sk.rows(v)[0]]
        if [gv[0], int(gv[1])] != sc[acc]:
            bad.append((acc, gv, sc[acc]))
    checks.add("velocity window count/sum == Python", not bad,
               f"{len(ids)} accounts, {len(bad)} mismatches, e.g. {bad[:3]}", "critical")
    # 5. shared-device ring lookup per member + sample of normal accounts
    sample = [x for v in ring_members.values() for x in v] + [x for v in inj_ring_members for x in v]
    sample += random.Random(6).sample(range(N), min(300, N)) + random.Random(7).sample(sorted(sc), min(100, len(sc)))
    bad = []
    for acc in sample:
        got = _rows_ids(sk.query(r, G, Q_DEV, {"id": acc}, ro=True))
        want = set()
        for d in devs(acc):
            want |= accs_dev(d)
        want.discard(acc)
        if got != want:
            bad.append((acc, len(got), len(want)))
    checks.add("shared-device neighbours == Python", not bad,
               f"{len(sample)} accounts, {len(bad)} mismatches, e.g. {bad[:3]}", "critical")
    # 6. 3-hop device->IP ring lookup
    bad = []
    sample3 = sample[:150] + sample[-150:]
    for acc in sample3:
        got = sk.scalar(sk.query(r, G, Q_3HOP, {"id": acc}, ro=True))
        want = set()
        for d in devs(acc):
            for b in accs_dev(d):
                for i in ips(b):
                    want |= accs_ip(i)
        if got != len(want):
            bad.append((acc, got, len(want)))
    checks.add("3-hop device->IP ring size == Python", not bad,
               f"{len(sample3)} accounts, {len(bad)} mismatches, e.g. {bad[:3]}", "critical")
    # 7. global ring scan finds exactly the devices with >= K accounts
    t = time.time()
    res = Q("MATCH (a:Account)-[:USES]->(d:Device) WITH d, count(a) AS c WHERE c >= $k RETURN d.id", {"k": K_DEV})
    scan_s = time.time() - t
    got = _rows_ids(res)
    want = {d for d in rev_dev if len(accs_dev(d)) >= K_DEV}
    planted = {R_DEV + rr for rr in ring_members} | {S_DEV + ev["k"] for ev in events
                                                     if ev["kind"] == "tx" and ev["items"][0]["m"] < MULE_M}
    checks.add("global ring scan == Python set and contains every planted ring",
               got == want and planted <= got,
               f"got={len(got)} want={len(want)} planted={len(planted)} missing={sorted(planted - got)[:5]} "
               f"extra={sorted(got - want)[:5]}", "critical")
    # 8. global fan-in scan since soak start
    t = time.time()
    res = Q("MATCH (a:Account)-[t:PAID]->(m:Merchant) WHERE t.ts >= $s "
            "WITH m, count(DISTINCT a) AS c WHERE c >= $k RETURN m.id", {"s": start_ts - 1, "k": K_FAN})
    fan_s = time.time() - t
    got = _rows_ids(res)
    want = {m for m, s in payers.items() if len(s) >= K_FAN}
    checks.add("global fan-in scan == Python set and contains every mule merchant",
               got == want and set(inj_mules) <= got,
               f"got={len(got)} want={len(want)} missing={sorted(set(inj_mules) - got)[:5]}", "critical")
    # 9. cycles: every planted member has exactly one cycle; sample of normal accounts none
    allc = [x for v in cyc_members.values() for x in v] + [x for v in inj_cycles for x in v]
    bad = []
    for acc in allc:
        v = sk.scalar(sk.query(r, G, CYC_Q[0], {"id": acc}, ro=True))
        if v != 1:
            bad.append((acc, v))
    norm = random.Random(8).sample(range(N2), 300)
    nbad = []
    for acc in norm:
        v = sk.scalar(sk.query(r, G, CYC_Q[0], {"id": acc}, ro=True))
        if v != 0:
            nbad.append((acc, v))
    checks.add("cycle detection: planted members find 1 cycle, 300 normal find 0",
               not bad and not nbad, f"{len(allc)} planted; planted bad={bad[:3]} normal bad={nbad[:3]}",
               "critical")
except Exception as e:  # noqa: BLE001
    scan_s = fan_s = None
    checks.add("verification phase completed", False, f"{type(e).__name__}: {str(e)[:500]}", "critical")

checks.add("server alive at end", srv.alive(), "", "critical")
mon.stop()
s = sk.analyze(out, srv)
s["meta"] = {"scenario": "s02_fraud_realtime", "scale": a.scale, "duration": a.duration,
             "accounts_base": N, "devices_base": ND, "ips_base": NI, "merchants": NM,
             "base_paid_edges": NBP, "base_xfer_edges": NBX, "planted_rings": RINGS0,
             "planted_cycles": CYC0, "ingest_s": ingest_s, "verify_s": time.time() - t_ver,
             "stream_tx": n_ack_t * BATCH + sum(len(e["items"]) for e in events if e["kind"] == "tx"),
             "stream_xfer": n_ack_x * XBATCH, "injected_events": len(events),
             "global_ring_scan_s": scan_s, "global_fanin_scan_s": fan_s,
             "cycle_query": "closed-pattern" if cyc_ok == Q_CYC else ("where-eq" if cyc_ok else "none"),
             "ing_rate_per_worker": ING_RATE}
(out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
srv.stop()
sk.print_summary(s)
