"""s07_mvcc_consistency — banking ledger: transactional isolation / atomicity under concurrency.

CLI: python -I s07_mvcc_consistency.py --module M --port P --out DIR --duration S --scale F

Model: :Account {id, balance, alive} (200k at scale 1), one :Sink, one :Counter {accounts,
transfers, amt}, :Guard {id:1} (unique-constraint bait). Writers do one-query transfers, account
create/delete (balance moved to/from the sink in the same query), edge delete, and deliberately
failing queries (div by zero / unique violation after partial writes). Readers check invariants
inside ONE query snapshot. Writers log every acknowledged write to a ledger file; at the end the
ledger is replayed in Python and compared with the final graph.
"""
import argparse
import json
import time
from pathlib import Path

import redis  # noqa: E402
from falkorbench.soak import soakkit as sk

ap = argparse.ArgumentParser()
sk.add_engine_args(ap)  # --module PATH | --image REF
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--duration", type=float, default=60)
ap.add_argument("--scale", type=float, default=0.05)
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)

G = "ledger"
N = max(2000, int(200_000 * a.scale))
INIT_BAL = 1000
ENDOW = 100
T_TOTAL = INIT_BAL * N          # conserved: sum(accounts) + sink
INIT_EDGE_AMT = 10
HOT = 1000                      # 30% of transfers hit the first HOT accounts (contention)
MAX_VIOL = 40

# worker rates (ops/s per worker); calibrated in smoke runs
R_TRANSFER, N_TRANSFER = 50, 4
R_CHURN, N_CHURN = 15, 1        # alternately create / delete own account
R_DELACC, N_DELACC = 3, 1
R_EDGEDEL, N_EDGEDEL = 20, 1
R_FAIL, N_FAIL = 20, 1
R_HEAVY, N_HEAVY = (3 if a.scale < 0.2 else 1), 2
R_LIGHT, N_LIGHT = 100, 2
PEAK_WRITERS = 6

Q_TRANSFER = ("MATCH (a:Account {id:$a}),(b:Account {id:$b}),(c:Counter) "
              "SET a.balance=a.balance-$x, b.balance=b.balance+$x, c.transfers=c.transfers+1, "
              "c.amt=c.amt+$x CREATE (a)-[:XFER {amt:$x,seq:$s}]->(b) RETURN 1")
Q_CREATE = ("MATCH (s:Sink),(c:Counter) SET s.balance=s.balance-%d, c.accounts=c.accounts+1 "
            "CREATE (:Account {id:$id,balance:%d,alive:true}) RETURN 1" % (ENDOW, ENDOW))
Q_DELACC = ("MATCH (a:Account {id:$id}),(s:Sink),(c:Counter) OPTIONAL MATCH (a)-[e:XFER]-() "
            "WITH a,s,c,count(e) AS k,sum(e.amt) AS ka "
            "SET s.balance=s.balance+a.balance, c.transfers=c.transfers-k, c.amt=c.amt-ka, "
            "c.accounts=c.accounts-1 WITH a,k,ka,a.balance AS bal DETACH DELETE a RETURN k,ka,bal")
Q_EDGEDEL = ("MATCH (c:Counter) MATCH (a:Account {id:$id})-[e:XFER]->() WITH c,e,e.amt AS amt LIMIT 1 "
             "SET c.transfers=c.transfers-1, c.amt=c.amt-amt DELETE e RETURN amt")
Q_FAIL = [
    ("MATCH (a:Account {id:$a}),(b:Account {id:$b}),(c:Counter) "
     "SET a.balance=a.balance-$x, b.balance=b.balance+$x, c.transfers=c.transfers+1 "
     "CREATE (a)-[:XFER {amt:$x,seq:-2}]->(b) WITH a,c RETURN 1/(c.transfers-c.transfers)", "division"),
    ("MATCH (a:Account {id:$a}),(s:Sink),(c:Counter) "
     "SET a.balance=a.balance-$x, s.balance=s.balance+$x, c.accounts=c.accounts+1 "
     "CREATE (:Account {id:$tmp,balance:50,alive:true}) CREATE (:Guard {id:1})", "constraint"),
    ("MATCH (a:Account {id:$a}),(s:Sink) SET s.balance=s.balance+a.balance WITH a,s "
     "DETACH DELETE a WITH s RETURN 1/(s.balance-s.balance)", "division"),
]
Q_HEAVY = ("MATCH (a:Account) WITH sum(a.balance) AS t, count(a) AS n, "
           "sum(CASE WHEN a.alive = true THEN 1 ELSE 0 END) AS na "
           "OPTIONAL MATCH ()-[e:XFER]->() WITH t,n,na,count(e) AS ne,sum(e.amt) AS amt "
           "OPTIONAL MATCH (:Account)-[f:XFER]->(:Account) WITH t,n,na,ne,amt,count(f) AS nf "
           "MATCH (x) WITH t,n,na,ne,amt,nf,count(x) AS tot "
           "MATCH (s:Sink),(c:Counter) "
           "RETURN t+s.balance, n, na, c.accounts, ne, nf, c.transfers, amt, c.amt, tot")
Q_LIGHT = ("MATCH (a:Account {id:$id}) OPTIONAL MATCH (a)-[e:XFER]->(b) "
           "RETURN a.id, a.balance, count(e), count(b), count(b.id)")


def ints(res):
    return [sk.cval(x) for x in sk.rows(res)[0]] if sk.rows(res) else None


def committed(res):
    """True if the reply's stats show any graph mutation."""
    st = [s.decode() if isinstance(s, bytes) else str(s) for s in res[-1]] if res else []
    return any(s.startswith(p) for s in st for p in
               ("Nodes created", "Nodes deleted", "Relationships", "Properties set", "Properties removed",
                "Labels added"))


# ---------------------------------------------------------------- workers
def winit(ctx):
    ctx.state["led"] = open(OUT / f"ledger-{ctx.wid}.jsonl", "a", buffering=1)
    ctx.state["viol"] = open(OUT / f"viol-{ctx.wid}.jsonl", "a", buffering=1)
    ctx.state["nviol"] = 0
    ctx.state["k"] = 0
    ctx.state["own"] = []


def led(ctx, **kw):
    ctx.state["led"].write(json.dumps(kw) + "\n")


def violation(ctx, kind, q, res, extra=""):
    ctx.state["nviol"] += 1
    if ctx.state["nviol"] <= MAX_VIOL:
        ctx.state["viol"].write(json.dumps({"kind": kind, "query": q, "result": str(res)[:1500],
                                            "extra": extra, "t": time.time()}) + "\n")


def wq(ctx, q, params):
    """Write query; on connection loss log an unknown outcome."""
    try:
        return sk.query(ctx.r, G, q, params)
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
        led(ctx, k="U", q=q[:40], p=params)
        raise


def pick_pair(ctx):
    rng = ctx.rng
    def one():
        return rng.randrange(HOT) if rng.random() < 0.3 else rng.randrange(N)
    x, y = one(), one()
    while y == x:
        y = rng.randrange(N)
    return x, y


def do_transfer(ctx, name):
    x, y = pick_pair(ctx)
    amt = ctx.rng.randint(1, 50)
    ctx.state["k"] += 1
    p = {"a": x, "b": y, "x": amt, "s": ctx.wid * 10**9 + ctx.state["k"]}
    res = wq(ctx, Q_TRANSFER, p)
    if sk.rows(res):
        led(ctx, k="T", a=x, b=y, x=amt)
    return name


def transfer_op(ctx):
    return do_transfer(ctx, "transfer")


def transfer_peak_op(ctx):
    return do_transfer(ctx, "transfer_peak")


def churn_op(ctx):
    st = ctx.state
    st["k"] += 1
    if len(st["own"]) < 5 or ctx.rng.random() < 0.5:
        nid = N + ctx.wid * 10**7 + st["k"]
        res = wq(ctx, Q_CREATE, {"id": nid})
        if sk.rows(res):
            led(ctx, k="C", id=nid)
            st["own"].append(nid)
        return "account_create"
    nid = st["own"].pop(ctx.rng.randrange(len(st["own"])))
    res = wq(ctx, Q_DELACC, {"id": nid})
    rs = sk.rows(res)
    if rs:
        k, ka, bal = (sk.cval(c) for c in rs[0])
        led(ctx, k="D", id=nid, ek=k, ea=ka, bal=bal)
    else:
        led(ctx, k="MISSING_OWN", id=nid)
    return "account_delete_own"


def delacc_op(ctx):
    nid = ctx.rng.randrange(N)
    res = wq(ctx, Q_DELACC, {"id": nid})
    rs = sk.rows(res)
    if rs:
        k, ka, bal = (sk.cval(c) for c in rs[0])
        led(ctx, k="D", id=nid, ek=k, ea=ka, bal=bal)
    return "account_delete"


def edgedel_op(ctx):
    res = wq(ctx, Q_EDGEDEL, {"id": ctx.rng.randrange(N)})
    rs = sk.rows(res)
    if rs:
        led(ctx, k="ED", amt=sk.cval(rs[0][0]))
    return "edge_delete"


def fail_op(ctx):
    i = ctx.rng.randrange(len(Q_FAIL))
    q, want = Q_FAIL[i]
    x, y = pick_pair(ctx)
    ctx.state["k"] += 1
    p = {"a": x, "b": y, "x": ctx.rng.randint(1, 50), "tmp": -(ctx.wid * 10**7 + ctx.state["k"])}
    try:
        res = sk.query(ctx.r, G, q, p)
    except redis.exceptions.ResponseError as e:
        msg = str(e).lower()
        if want in msg or ("zero" in msg and want == "division"):
            return f"fail_expected_{want}"
        raise  # unexpected error text -> counted as an error
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
        led(ctx, k="U", q="fail")
        raise
    if committed(res):
        violation(ctx, "failing query committed", q, res, json.dumps(p))
        led(ctx, k="FAILOP_COMMITTED", variant=i, p=p)
    return "fail_noop"


def heavy_op(ctx):
    res = sk.query(ctx.r, G, Q_HEAVY, ro=True)
    v = ints(res)
    if v is None:  # Sink and Counter are never deleted: a snapshot always has the row
        violation(ctx, "snapshot invariant", Q_HEAVY, res, "no row (Sink or Counter missing)")
        return "read_invariants_VIOLATION"
    tot_bal, n, na, cacc, ne, nf, ctr, amt, camt, tot = [int(x) if x is not None else 0 for x in v]
    bad = []
    if tot_bal != T_TOTAL:
        bad.append(f"total balance {tot_bal} != {T_TOTAL}")
    if n != cacc:
        bad.append(f"count(Account) {n} != counter.accounts {cacc}")
    if na != n:
        bad.append(f"alive-prop count {na} != label count {n}")
    if ne != ctr:
        bad.append(f"edge count {ne} != counter.transfers {ctr}")
    if nf != ne:
        bad.append(f"edges between live Accounts {nf} != all edges {ne} (dangling?)")
    if amt != camt:
        bad.append(f"sum(edge.amt) {amt} != counter.amt {camt}")
    if tot != n + 3:
        bad.append(f"count(all nodes) {tot} != accounts+3 ({n + 3})")
    if bad:
        violation(ctx, "snapshot invariant", Q_HEAVY, v, "; ".join(bad))
        return "read_invariants_VIOLATION"
    return "read_invariants"


def light_op(ctx):
    nid = ctx.rng.randrange(N)
    res = sk.query(ctx.r, G, Q_LIGHT, {"id": nid}, ro=True)
    rs = sk.rows(res)
    if rs:
        i, bal, ce, cb, cbi = (sk.cval(c) for c in rs[0])
        bad = []
        if i != nid:
            bad.append("id mismatch")
        if not isinstance(bal, int):
            bad.append(f"balance not int: {bal!r}")
        if not (ce == cb == cbi):
            bad.append(f"edge/endpoint mismatch e={ce} b={cb} b.id={cbi}")
        if bad:
            violation(ctx, "point invariant", Q_LIGHT, rs[0], f"id={nid} " + "; ".join(bad))
            return "read_point_VIOLATION"
    return "read_point"


# ---------------------------------------------------------------- main
srv = sk.Server(a.module, a.port, OUT)
mon = sk.Monitor(srv, OUT)
mon.start()
checks = sk.Checks(OUT)
r = srv.client()

mon.mark("ingest")
t = time.time()
B = 5000
for i in range(0, N, B):
    sk.query(r, G, "UNWIND range($a,$b) AS i CREATE (:Account {id:i,balance:%d,alive:true})" % INIT_BAL,
             {"a": i, "b": min(N, i + B) - 1})
sk.query(r, G, "CREATE INDEX FOR (a:Account) ON (a.id)")
sk.query(r, G, "CREATE INDEX FOR (g:Guard) ON (g.id)")
sk.query(r, G, "CREATE (:Sink {balance:0}), (:Guard {id:1}), "
               "(:Counter {accounts:%d, transfers:%d, amt:%d})" % (N, N, N * INIT_EDGE_AMT))
for i in range(0, N, B):
    sk.query(r, G, "UNWIND range($a,$b) AS i MATCH (x:Account {id:i}), "
                   "(y:Account {id:(i + 1 + (i * 7919) % ($n - 1)) % $n}) "
                   "CREATE (x)-[:XFER {amt:" + str(INIT_EDGE_AMT) + ",seq:-1}]->(y)",
             {"a": i, "b": min(N, i + B) - 1, "n": N})
constraints_ok = True
for lab in ("Account", "Guard"):
    try:
        r.execute_command("GRAPH.CONSTRAINT", "CREATE", G, "UNIQUE", "NODE", lab, "PROPERTIES", 1, "id")
    except Exception as e:
        constraints_ok = False
        checks.add(f"unique constraint on {lab} supported", False, str(e), "minor")
if constraints_ok:
    dl = time.time() + 120
    while time.time() < dl:
        st = [sk.cval(x[4]) for x in sk.rows(sk.query(r, G, "CALL db.constraints()"))]
        if st and all(s == "OPERATIONAL" for s in st) and len(st) == 2:
            break
        time.sleep(0.5)
    else:
        checks.add("constraints operational", False, str(st), "major")
ingest_s = time.time() - t

v0 = ints(sk.query(r, G, Q_HEAVY, ro=True))
checks.add("ingest invariants hold", v0 is not None and int(v0[0]) == T_TOTAL and int(v0[1]) == N
           and int(v0[4]) == N and int(v0[5]) == N and int(v0[6]) == N, str(v0), "critical")
if not constraints_ok:
    Q_FAIL[:] = [q for q in Q_FAIL if q[1] == "division"]


def specs(peak):
    s = []
    if peak:
        s.append(sk.WorkerSpec(transfer_peak_op, rate=None, count=PEAK_WRITERS, name="peakw", init=winit))
        s.append(sk.WorkerSpec(transfer_op, rate=R_TRANSFER, count=1, name="writer_xfer", init=winit))
    else:
        s.append(sk.WorkerSpec(transfer_op, rate=R_TRANSFER, count=N_TRANSFER, name="writer_xfer", init=winit))
    s += [sk.WorkerSpec(churn_op, rate=R_CHURN, count=N_CHURN, name="writer_churn", init=winit),
          sk.WorkerSpec(delacc_op, rate=R_DELACC, count=N_DELACC, name="writer_delacc", init=winit),
          sk.WorkerSpec(edgedel_op, rate=R_EDGEDEL, count=N_EDGEDEL, name="writer_edgedel", init=winit),
          sk.WorkerSpec(fail_op, rate=R_FAIL, count=N_FAIL, name="writer_fail", init=winit),
          sk.WorkerSpec(heavy_op, rate=R_HEAVY, count=N_HEAVY, name="reader_heavy", init=winit),
          sk.WorkerSpec(light_op, rate=R_LIGHT, count=N_LIGHT, name="reader_light", init=winit)]
    return s


d_soak, d_peak = a.duration * 0.7, a.duration * 0.3
mon.mark("soak")
sk.run_load(srv, OUT, specs(False), d_soak, seed=1, tag="")
mon.mark("load-peak")
sk.run_load(srv, OUT, specs(True), d_peak, seed=2, tag="pk")

# ---------------------------------------------------------------- verification
mon.mark("verify")
alive_end = srv.alive()
checks.add("server alive after load", alive_end, "", "critical")
viols = []
for f in sorted(OUT.glob("viol-*.jsonl")):
    for line in f.read_text().splitlines():
        viols.append(json.loads(line))
kinds = {}
for v in viols:
    kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
checks.add("no reader ever saw an invariant violation (snapshot isolation)",
           not [v for v in viols if v["kind"] in ("snapshot invariant", "point invariant")],
           json.dumps([v for v in viols if v["kind"] in ("snapshot invariant", "point invariant")][:3])[:1800],
           "critical")
checks.add("failing queries are fully rolled back (never committed)",
           not [v for v in viols if v["kind"] == "failing query committed"],
           json.dumps([v for v in viols if v["kind"] == "failing query committed"][:3])[:1800], "critical")
(OUT / "violations.json").write_text(json.dumps(viols[:200], indent=1))

# replay ledger
delta, deleted, created, unknown = {}, {}, set(), 0
sink_gain = n_T = n_ED = sum_k = sum_amt_T = sum_amt_ED = sum_ka = 0
missing_own = 0
for f in sorted(OUT.glob("ledger-*.jsonl")):
    for line in f.read_text().splitlines():
        e = json.loads(line)
        k = e["k"]
        if k == "T":
            delta[e["a"]] = delta.get(e["a"], 0) - e["x"]
            delta[e["b"]] = delta.get(e["b"], 0) + e["x"]
            n_T += 1
            sum_amt_T += e["x"]
        elif k == "C":
            created.add(e["id"])
        elif k == "D":
            if e["id"] in deleted:
                checks.add("account deleted twice", False, str(e), "critical")
            deleted[e["id"]] = e
            sink_gain += e["bal"]
            sum_k += e["ek"]
            sum_ka += e["ea"]
        elif k == "ED":
            n_ED += 1
            sum_amt_ED += e["amt"]
        elif k == "U":
            unknown += 1
        elif k == "MISSING_OWN":
            missing_own += 1
checks.add("no write with unknown outcome (connection lost)", unknown == 0, f"{unknown}", "critical")
checks.add("own-created accounts were all deletable", missing_own == 0, f"{missing_own}", "critical")

r = srv.client() if alive_end else r


def final_state():
    bal = {}
    step = 20000
    for lo in range(0, N, step):
        for row in sk.rows(sk.query(r, G, "MATCH (a:Account) WHERE a.id >= $lo AND a.id < $hi "
                                          "RETURN a.id, a.balance", {"lo": lo, "hi": min(N, lo + step)},
                                    ro=True)):
            bal[sk.cval(row[0])] = sk.cval(row[1])
    for row in sk.rows(sk.query(r, G, "MATCH (a:Account) WHERE a.id >= $lo RETURN a.id, a.balance",
                                {"lo": N}, ro=True)):
        bal[sk.cval(row[0])] = sk.cval(row[1])
    sink = sk.scalar(sk.query(r, G, "MATCH (s:Sink) RETURN s.balance", ro=True))
    cnt = ints(sk.query(r, G, "MATCH (c:Counter) RETURN c.accounts, c.transfers, c.amt", ro=True))
    ne, namt = ints(sk.query(r, G, "MATCH ()-[e:XFER]->() RETURN count(e), sum(e.amt)", ro=True))
    ng = sk.scalar(sk.query(r, G, "MATCH (g:Guard) RETURN count(g)", ro=True))
    nn = sk.scalar(sk.query(r, G, "MATCH (n) RETURN count(n)", ro=True))
    return bal, sink, cnt, ne, namt, ng, nn


def verify(tag, strict_ledger):
    bal, sink, cnt, ne, namt, ng, nn = final_state()
    exp_ids = (set(range(N)) - set(deleted)) | (created - set(deleted))
    checks.add(f"{tag}: surviving account id set equals ledger", set(bal) == exp_ids,
               f"missing={sorted(exp_ids - set(bal))[:10]} extra={sorted(set(bal) - exp_ids)[:10]} "
               f"n={len(bal)} expected={len(exp_ids)}", "critical")
    if strict_ledger:
        wrong = []
        for i in set(bal) & exp_ids:
            exp = (INIT_BAL + delta.get(i, 0)) if i < N else ENDOW
            if bal[i] != exp:
                wrong.append((i, bal[i], exp))
        checks.add(f"{tag}: every surviving balance == initial + acknowledged transfers", not wrong,
                   f"{len(wrong)} wrong, e.g. {wrong[:8]}", "critical")
        exp_sink = sink_gain - ENDOW * len(created)
        checks.add(f"{tag}: sink balance == moved balances of deleted accounts - endowments", sink == exp_sink,
                   f"sink={sink} expected={exp_sink}", "critical")
        exp_edges = N + n_T - n_ED - sum_k
        checks.add(f"{tag}: edge count == initial + acked transfers - deletions", ne == exp_edges,
                   f"edges={ne} expected={exp_edges}", "critical")
        exp_amt = N * INIT_EDGE_AMT + sum_amt_T - sum_amt_ED - sum_ka
        checks.add(f"{tag}: sum(edge.amt) matches ledger", namt == exp_amt, f"{namt} vs {exp_amt}", "critical")
        checks.add(f"{tag}: counter node matches ledger",
                   cnt == [N + len(created) - len(deleted), exp_edges, exp_amt], f"{cnt}", "critical")
    checks.add(f"{tag}: total balance conserved", sum(bal.values()) + sink == T_TOTAL,
               f"{sum(bal.values()) + sink} != {T_TOTAL}", "critical")
    checks.add(f"{tag}: counter == edge count == observed", cnt[1] == ne, f"{cnt} edges={ne}", "critical")
    checks.add(f"{tag}: guard node unique, node count consistent", ng == 1 and nn == len(bal) + 3,
               f"guards={ng} nodes={nn} accounts={len(bal)}", "critical")
    return bal, sink, cnt, ne


pre = None
if alive_end:
    pre = verify("final", True)
    # RDB round trip: same state after save+restart
    dt = srv.restart(save=True)
    r = srv.client()
    post = verify("after restart", False)
    checks.add("RDB round-trip preserves graph", pre == post, "state differs after restart", "critical")
else:
    dt = None
checks.add("server alive at end", srv.alive(), "", "critical")
mon.stop()
s = sk.analyze(OUT, srv)
heavy = s["ops"].get("read_invariants", {}).get("count", 0)
light = s["ops"].get("read_point", {}).get("count", 0)
checks.add("invariant readers actually ran", heavy > 10 and light > 100, f"heavy={heavy} light={light}", "major")
s = sk.analyze(OUT, srv)
s["meta"] = {"scenario": "s07_mvcc_consistency", "accounts": N, "initial_edges": N, "scale": a.scale,
             "ingest_s": ingest_s, "duration": a.duration, "restart_s": dt,
             "acked_transfers": n_T, "edge_deletes": n_ED, "accounts_created": len(created),
             "accounts_deleted": len(deleted), "violations_by_kind": kinds,
             "heavy_reads": heavy, "point_reads": light, "constraints": constraints_ok}
(OUT / "summary.json").write_text(json.dumps(s, indent=1, default=str))
srv.stop()
sk.print_summary(s)
print("meta:", json.dumps(s["meta"]))
