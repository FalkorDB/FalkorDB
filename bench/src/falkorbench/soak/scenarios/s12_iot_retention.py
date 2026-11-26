"""s12_iot_retention — IoT time series with a sliding-window retention worker (steady state).

Sites -> Devices -> Sensors; every sensor emits a Reading every PERIOD (deterministic lattice:
ts(s,k), v(s,k)), attached with :EMITTED and chained with :NEXT.  A retention worker deletes
readings older than the window, so live count is flat.  Ground truth is computed analytically
from the lattice + each writer's acknowledged progress, verified at pause-checkpoints.

CLI: --module --port --out --duration --scale
"""
import argparse
import json
import math
import os
import threading
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
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("pause*"):
    f.unlink()
for f in list(OUT.glob("ack-*")) + list(OUT.glob("viol*")) + list(OUT.glob("state-*")):
    f.unlink()
PAUSE = OUT / "pause"
VIOL = OUT / "viol.log"

G = "iot"
SPD = 20                                   # sensors per device
DEVICES = max(4, round(200 * a.scale))
SENSORS = DEVICES * SPD                    # full: 4000
SITES = max(2, DEVICES // 20)              # full: 10
P_MS = 1000                                # one reading / sensor / s -> full 4000 readings/s
W_S = min(500.0, 0.4 * a.duration)         # retention window (full: 480 s -> ~1.9M live)
W_MS = int(W_S * 1000)
K_PRE = W_MS // P_MS                       # preloaded slots per sensor
THRESH = 990                               # anomaly threshold (1% of readings)
M = 21                                     # partition modulus: s%21==0 -> single-event writer
NB = 3                                     # batch writers
T0P = int(time.time() * 1000) - W_MS       # lattice origin (preload spans [start-W, start])
K0 = None                                  # first live slot, set after ingest
INIT = {}                                  # sid -> last reading id after preload


def ts_of(s, k):
    return T0P + k * P_MS + (s * 37) % P_MS


def val_of(s, k):
    return (s * 7919 + k * 104729 + (s * k) % 977) % 1000


def now_ms():
    return int(time.time() * 1000)


def viol(msg):
    with open(VIOL, "a") as f:
        f.write(msg + "\n")


def check_reading(sid, ts, v):
    """Reading must lie on the lattice with the deterministic value."""
    off = (sid * 37) % P_MS
    d = ts - T0P - off
    if d % P_MS != 0:
        return False
    return val_of(sid, d // P_MS) == v


def stats_of(res, key):
    for line in (res[-1] if isinstance(res, list) and res else []):
        s = line.decode() if isinstance(line, bytes) else str(line)
        if s.startswith(key):
            return int(s.split(":")[1].split()[0])
    return 0


Q_FIRST = ("UNWIND $b AS e MATCH (s:Sensor {id: e.sid}) "
           "CREATE (r:Reading {sid: e.sid, ts: e.ts, v: e.v, st: e.st}), (s)-[:EMITTED]->(r) "
           "RETURN e.sid, r.ts")
Q_CHAIN = ("UNWIND $b AS e MATCH (s:Sensor {id: e.sid}) MATCH (p:Reading {ts: e.pts}) "
           "WHERE p.sid = e.sid "
           "CREATE (p)-[:NEXT]->(r:Reading {sid: e.sid, ts: e.ts, v: e.v, st: e.st}), "
           "(s)-[:EMITTED]->(r) RETURN e.sid, id(r)")


def emit(r, items):
    """items: [(sid, ts, v, pid|None)] -> {sid: new_id}. Batched UNWIND create + NEXT chain."""
    res = {}
    first = [{"sid": i[0], "ts": i[1], "v": i[2], "st": int(i[2] > 950)} for i in items if i[3] is None]
    rest = [{"sid": i[0], "ts": i[1], "v": i[2], "st": int(i[2] > 950), "pts": i[3]}
            for i in items if i[3] is not None]
    for q, b in ((Q_FIRST, first), (Q_CHAIN, rest)):
        if not b:
            continue
        got = sk.rows(sk.query(r, G, q, {"b": b}))
        for row in got:
            res[sk.cval(row[0])] = sk.cval(row[1])
        if len(got) != len(b):
            viol(f"emit: {len(b)} items but {len(got)} rows (chain guard failed?)")
            for e in b:
                if e["sid"] not in res:  # fall back: unchained create so progress is not lost
                    g2 = sk.rows(sk.query(r, G, Q_FIRST, {"b": [{k: e[k] for k in ("sid", "ts", "v", "st")}]}))
                    res[e["sid"]] = sk.cval(g2[0][1])
    return res


# ------------------------------------------------------------------ server + ingest
srv = sk.Server(a.module, a.port, OUT)
mon = sk.Monitor(srv, OUT)
mon.start()
checks = sk.Checks(OUT)
r = srv.client()

mon.mark("ingest")
t_ing = time.time()
sk.query(r, G, "CREATE INDEX FOR (r:Reading) ON (r.ts)")
sk.query(r, G, "CREATE INDEX FOR (s:Sensor) ON (s.id)")
sk.query(r, G, "CREATE INDEX FOR (d:Device) ON (d.id)")
sk.query(r, G, "CREATE INDEX FOR (s:Site) ON (s.id)")
sk.query(r, G, "UNWIND range(0, $n - 1) AS i CREATE (:Site {id: i, name: 'site-' + toString(i)})", {"n": SITES})
sk.query(r, G, "UNWIND range(0, $n - 1) AS i CREATE (:Device {id: i, site: i % $s})", {"n": DEVICES, "s": SITES})
sk.query(r, G, "UNWIND range(0, $n - 1) AS i CREATE (:Sensor {id: i, device: i / $k, kind: i % 4})",
         {"n": SENSORS, "k": SPD})
sk.query(r, G, "UNWIND range(0, $n - 1) AS i MATCH (d:Device {id: i}) MATCH (s:Site {id: i % $s}) "
         "CREATE (s)-[:HAS_DEVICE]->(d)", {"n": DEVICES, "s": SITES})
for i0 in range(0, SENSORS, 5000):
    sk.query(r, G, "UNWIND range($a, $b) AS i MATCH (s:Sensor {id: i}) MATCH (d:Device {id: i / $k}) "
             "CREATE (d)-[:HAS_SENSOR]->(s)", {"a": i0, "b": min(SENSORS, i0 + 5000) - 1, "k": SPD})

last = {}
CH = 2000
for k in range(K_PRE):
    items = [(s, ts_of(s, k), val_of(s, k), last.get(s)) for s in range(SENSORS)]
    for i in range(0, SENSORS, CH):
        emit(r, items[i:i + CH])
    for s in range(SENSORS):
        last[s] = ts_of(s, k)
INIT = last
ingest_s = time.time() - t_ing
n_pre = sk.scalar(sk.query(r, G, "MATCH (r:Reading) RETURN count(r)"))
checks.add("ingest reading count", n_pre == SENSORS * K_PRE, f"{n_pre} != {SENSORS * K_PRE}", "critical")
n_next = sk.scalar(sk.query(r, G, "MATCH ()-[e:NEXT]->() RETURN count(e)"))
checks.add("ingest NEXT count", n_next == SENSORS * (K_PRE - 1), f"{n_next}", "critical")
checks.add("ingest topology", sk.scalar(sk.query(
    r, G, "MATCH (:Site)-[:HAS_DEVICE]->(:Device)-[:HAS_SENSOR]->(:Sensor) RETURN count(*)")) == SENSORS,
    "", "critical")
print(f"ingest {ingest_s:.1f}s sensors={SENSORS} readings={n_pre} window={W_S:.0f}s", flush=True)
K0 = (now_ms() - T0P) // P_MS + 1


# ------------------------------------------------------------------ workers
def paused(ctx, kind):
    """Checkpoint barrier: ack (with state) once per generation, block while pause file exists."""
    try:
        gen = PAUSE.read_text().strip()
    except FileNotFoundError:
        return False
    if gen == "":
        return False
    if ctx.state.get("acked") != gen:
        st = ctx.state.get("nk")
        # checkpoint() reads an ack as soon as its name exists: write it whole under another
        # name, then rename it into place (atomic), so it never sees half the JSON
        ack = OUT / f"ack-{kind}-{ctx.wid}-{gen}"
        tmp = ack.with_name(ack.name + ".tmp")
        tmp.write_text(json.dumps({"nk": st, "last": ctx.state.get("last")}) if st is not None else "{}")
        os.replace(tmp, ack)
        ctx.state["acked"] = gen
    t = time.time()
    while PAUSE.exists() and time.time() - t < 300:
        time.sleep(0.05)
    return True


def writer_init(ctx):
    wi = ctx.wid  # writers are spawned first: wid 0..NB-1 batch, NB single
    single = wi == NB
    mine = [s for s in range(SENSORS) if (s % M == 0) == single and
            (single or ((s % M) - 1) % NB == wi)]
    ctx.state["mine"] = mine
    ctx.state["nk"] = {s: K0 for s in mine}
    ctx.state["last"] = {s: INIT[s] for s in mine}
    ctx.state["cur"] = 0


def writer(ctx):
    single = ctx.wid == NB
    if paused(ctx, "w"):
        return "_paused"
    mine, nk, lastid = ctx.state["mine"], ctx.state["nk"], ctx.state["last"]
    now = now_ms()
    cap = 1 if single else 300
    items = []
    n = len(mine)
    c = ctx.state["cur"]
    for j in range(n):
        s = mine[(c + j) % n]
        k = nk[s]
        t = ts_of(s, k)
        if t <= now:
            items.append((s, t, val_of(s, k), lastid[s]))
            if len(items) >= cap:
                c = (c + j + 1) % n
                break
    else:
        c = 0
    ctx.state["cur"] = c
    if not items:
        time.sleep(0.02)
        return "_idle"
    if single:
        s, t, v, pid = items[0]
        res = sk.rows(sk.query(
            ctx.r, G, "MATCH (s:Sensor {id: $sid}) MATCH (p:Reading {ts: $pts}) WHERE p.sid = $sid "
                      "CREATE (p)-[:NEXT]->(r:Reading {sid: $sid, ts: $ts, v: $v, st: $st}), "
                      "(s)-[:EMITTED]->(r) RETURN r.ts",
            {"sid": s, "pts": pid, "ts": t, "v": v, "st": int(v > 950)}))
        if not res:
            viol(f"single: chain guard failed sid={s}")
            res = sk.rows(sk.query(ctx.r, G, Q_FIRST, {"b": [{"sid": s, "ts": t, "v": v, "st": int(v > 950)}]}))
            newid = {s: sk.cval(res[0][1])}
        else:
            newid = {s: sk.cval(res[0][0])}
        op = "ingest_single"
    else:
        newid = emit(ctx.r, items)
        op = "ingest_batch"
    for s, t, v, pid in items:
        nk[s] += 1
        lastid[s] = t
    return op


def retention_init(ctx):
    pass


RET_BATCH = 5000


def retention(ctx):
    if paused(ctx, "r"):
        return "_paused"
    c = now_ms() - W_MS
    res = sk.query(ctx.r, G, f"MATCH (r:Reading) WHERE r.ts < $c WITH r LIMIT {RET_BATCH} DETACH DELETE r",
                   {"c": c})
    n = stats_of(res, "Nodes deleted")
    if n == 0:
        time.sleep(0.05)
        return "_idle"
    return "retention_delete"


def rnd_sensor(ctx):
    return ctx.rng.randrange(SENSORS)


def reader_recent(ctx):
    sid = rnd_sensor(ctx)
    lo = now_ms() - 60_000
    rs = sk.rows(sk.query(ctx.r, G, "MATCH (s:Sensor {id: $sid})-[:EMITTED]->(r:Reading) "
                          "WHERE r.ts >= $lo RETURN r.ts, r.v ORDER BY r.ts",
                          {"sid": sid, "lo": lo}, ro=True))
    prev = -1
    for row in rs:
        ts, v = sk.cval(row[0]), sk.cval(row[1])
        if ts < lo or ts <= prev or not check_reading(sid, ts, v):
            viol(f"recent sid={sid} bad row ts={ts} v={v} lo={lo} prev={prev}")
            break
        prev = ts
    if len(rs) > 60_000 // P_MS + 1:
        viol(f"recent sid={sid} too many rows {len(rs)}")
    return "q_recent_by_sensor"


def reader_latest(ctx):
    sid = rnd_sensor(ctx)
    rs = sk.rows(sk.query(ctx.r, G, "MATCH (s:Sensor {id: $sid})-[:EMITTED]->(r:Reading) "
                          "RETURN r.ts, r.v ORDER BY r.ts DESC LIMIT 1", {"sid": sid}, ro=True))
    if not rs:
        viol(f"latest sid={sid} no readings")
    else:
        ts, v = sk.cval(rs[0][0]), sk.cval(rs[0][1])
        if not check_reading(sid, ts, v) or ts < now_ms() - W_MS - 5000:
            viol(f"latest sid={sid} bad ts={ts} v={v}")
    return "q_latest_reading"


Q_AGG_DEV = ("MATCH (d:Device {id: $d})-[:HAS_SENSOR]->(s:Sensor)-[:EMITTED]->(r:Reading) "
             "WHERE r.ts >= $lo RETURN avg(r.v), min(r.v), max(r.v), count(r)")
Q_AGG_SITE = ("MATCH (st:Site {id: $d})-[:HAS_DEVICE]->(:Device)-[:HAS_SENSOR]->(s:Sensor)-[:EMITTED]->(r:Reading) "
              "WHERE r.ts >= $lo RETURN avg(r.v), min(r.v), max(r.v), count(r)")


def reader_agg_dev(ctx):
    d = ctx.rng.randrange(DEVICES)
    lo = now_ms() - 120_000
    rs = sk.rows(sk.query(ctx.r, G, Q_AGG_DEV, {"d": d, "lo": lo}, ro=True))
    mn, mx, cnt = sk.cval(rs[0][1]), sk.cval(rs[0][2]), sk.cval(rs[0][3])
    if cnt and (mn < 0 or mx > 999 or mn > mx):
        viol(f"agg_dev d={d} bad min/max {mn} {mx}")
    return "q_agg_device_2m"


def reader_agg_site(ctx):
    d = ctx.rng.randrange(SITES)
    lo = now_ms() - 120_000
    rs = sk.rows(sk.query(ctx.r, G, Q_AGG_SITE, {"d": d, "lo": lo}, ro=True))
    mn, mx, cnt = sk.cval(rs[0][1]), sk.cval(rs[0][2]), sk.cval(rs[0][3])
    if cnt and (mn < 0 or mx > 999 or mn > mx):
        viol(f"agg_site d={d} bad min/max {mn} {mx}")
    return "q_agg_site_2m"


def reader_anomaly(ctx):
    lo = now_ms() - 30_000
    rs = sk.rows(sk.query(
        ctx.r, G, "MATCH (r:Reading) WHERE r.ts >= $lo AND r.v > $th "
                  "MATCH (r)<-[:EMITTED]-(s:Sensor)<-[:HAS_SENSOR]-(d:Device)<-[:HAS_DEVICE]-(st:Site) "
                  "RETURN st.id, d.id, s.id, r.v, r.ts ORDER BY r.v DESC LIMIT 20",
        {"lo": lo, "th": THRESH}, ro=True))
    for row in rs:
        site, dev, sid, v, ts = [sk.cval(x) for x in row]
        if v <= THRESH or ts < lo or dev != sid // SPD or site != dev % SITES or not check_reading(sid, ts, v):
            viol(f"anomaly bad row {site} {dev} {sid} {v} {ts}")
            break
    return "q_anomaly_join_30s"


# ------------------------------------------------------------------ ground truth + checkpoints
def gt_sensor(s, cutoff, nk):
    """(count, sum, min, max) of live readings of sensor s with ts >= cutoff."""
    base = T0P + (s * 37) % P_MS
    kmin = max(0, -((-(cutoff - base)) // P_MS))
    n = sm = 0
    mn, mx = 10 ** 9, -1
    for lo, hi in ((0, K_PRE - 1), (K0, nk - 1)):
        for k in range(max(lo, kmin), hi + 1):
            v = val_of(s, k)
            n += 1
            sm += v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
    return n, sm, mn, mx


samples = []        # [t, count, max_id]
ck_results = []
sup_stop = threading.Event()


def run_retention_final(rc, cutoff):
    t = time.time()
    tot = 0
    while time.time() - t < 180:
        res = sk.query(rc, G, f"MATCH (r:Reading) WHERE r.ts < $c WITH r LIMIT 5000 DETACH DELETE r", {"c": cutoff})
        n = stats_of(res, "Nodes deleted")
        tot += n
        if n == 0:
            return tot, True
    return tot, False


def checkpoint(idx, rc, t_run0):
    gen = str(idx + 1)
    t_start = time.time()
    PAUSE.write_text(gen)
    need = [f"ack-w-{i}-{gen}" for i in range(NB + 1)] + [f"ack-r-{NB + 1}-{gen}"]
    deadline = time.time() + 120
    while time.time() < deadline and not all((OUT / n).exists() for n in need):
        time.sleep(0.1)
    missing = [n for n in need if not (OUT / n).exists()]
    if missing:
        checks.add(f"checkpoint{idx} writers quiesced", False, f"missing acks {missing}", "major")
        PAUSE.unlink()
        return
    nk = {}
    for i in range(NB + 1):
        d = json.loads((OUT / f"ack-w-{i}-{gen}").read_text())
        nk.update({int(k): v for k, v in d["nk"].items()})
    C = now_ms() - W_MS
    deleted, done = run_retention_final(rc, C)
    # ---- verification
    t_v = time.time()
    problems = []
    exp = {s: gt_sensor(s, C, nk[s]) for s in range(SENSORS)}
    got = {}
    for row in sk.rows(sk.query(rc, G, "MATCH (s:Sensor)-[:EMITTED]->(r:Reading) "
                                "RETURN s.id, count(r), sum(r.v), min(r.v), max(r.v)")):
        c = [sk.cval(x) for x in row]
        got[c[0]] = tuple(c[1:])
    bad = [s for s in range(SENSORS) if got.get(s, (0, 0, None, None))[:2] != exp[s][:2] and
           not (exp[s][0] == 0 and s not in got)]
    badmm = [s for s in range(SENSORS) if exp[s][0] and got.get(s) and got[s][2:] != exp[s][2:]]
    if bad:
        problems.append(f"{len(bad)} sensors count/sum mismatch e.g. s={bad[0]} got={got.get(bad[0])} exp={exp[bad[0]]}")
    if badmm:
        problems.append(f"{len(badmm)} sensors min/max mismatch")
    tot_exp = sum(e[0] for e in exp.values())
    tot_got = sk.scalar(sk.query(rc, G, "MATCH (r:Reading) RETURN count(r)"))
    if tot_got != tot_exp:
        problems.append(f"total readings {tot_got} != expected {tot_exp} (orphans/leak?)")
    old = sk.scalar(sk.query(rc, G, "MATCH (r:Reading) WHERE r.ts < $c RETURN count(r)", {"c": C}))
    if old:
        problems.append(f"{old} readings older than window survived retention")
    nonempty = sum(1 for e in exp.values() if e[0])
    nn = sk.scalar(sk.query(rc, G, "MATCH ()-[e:NEXT]->() RETURN count(e)"))
    if nn != tot_exp - nonempty:
        problems.append(f"NEXT edges {nn} != {tot_exp - nonempty}")
    ne = sk.scalar(sk.query(rc, G, "MATCH (:Sensor)-[e:EMITTED]->(:Reading) RETURN count(e)"))
    if ne != tot_exp:
        problems.append(f"EMITTED edges {ne} != {tot_exp}")
    # NEXT chain integrity on sampled sensors
    import random
    rng = random.Random(idx)
    for s in rng.sample(range(SENSORS), min(15, SENSORS)):
        rs = [[sk.cval(x) for x in row] for row in sk.rows(sk.query(
            rc, G, "MATCH (s:Sensor {id: $sid})-[:EMITTED]->(r:Reading) OPTIONAL MATCH (r)-[:NEXT]->(n:Reading) "
                   "RETURN r.ts, n.ts ORDER BY r.ts", {"sid": s}))]
        ok = len(rs) == exp[s][0]
        for i in range(len(rs) - 1):
            if rs[i][1] != rs[i + 1][0]:
                ok = False
                break
        if rs and rs[-1][1] is not None:
            ok = False
        if not ok:
            problems.append(f"NEXT chain broken for sensor {s} (rows={len(rs)} exp={exp[s][0]})")
    # window aggregations per device / site over the second half of the window
    lo2 = C + W_MS // 2
    for d in rng.sample(range(DEVICES), min(6, DEVICES)):
        parts = [gt_sensor(s, lo2, nk[s]) for s in range(d * SPD, (d + 1) * SPD)]
        for qn, q, key, ss in (("device", Q_AGG_DEV, d, parts),):
            rs = sk.rows(sk.query(rc, G, q, {"d": key, "lo": lo2}))[0]
            c = sum(p[0] for p in ss)
            avg = sk.cval(rs[0])
            ok = sk.cval(rs[3]) == c and (c == 0 or (
                sk.cval(rs[1]) == min(p[2] for p in ss) and sk.cval(rs[2]) == max(p[3] for p in ss) and
                abs(avg - sum(p[1] for p in ss) / c) < 1e-6))
            if not ok:
                problems.append(f"agg {qn} {key} mismatch got={[sk.cval(x) for x in rs]} exp_cnt={c}")
    for st in rng.sample(range(SITES), min(2, SITES)):
        ss = [gt_sensor(s, lo2, nk[s]) for s in range(SENSORS) if (s // SPD) % SITES == st]
        rs = sk.rows(sk.query(rc, G, Q_AGG_SITE, {"d": st, "lo": lo2}))[0]
        c = sum(p[0] for p in ss)
        ok = sk.cval(rs[3]) == c and (c == 0 or abs(sk.cval(rs[0]) - sum(p[1] for p in ss) / c) < 1e-6)
        if not ok:
            problems.append(f"agg site {st} mismatch got={[sk.cval(x) for x in rs]} exp_cnt={c}")
    # index range scan vs full scan vs truth over the second half of the window
    exp_win = sum(gt_sensor(s, lo2, nk[s])[0] for s in range(SENSORS))
    ix = sk.scalar(sk.query(rc, G, "MATCH (r:Reading) WHERE r.ts >= $lo RETURN count(r)", {"lo": lo2}))
    fs = sk.scalar(sk.query(rc, G, "MATCH (r:Reading) WHERE r.ts + 0 >= $lo RETURN count(r)", {"lo": lo2}))
    if not (ix == fs == exp_win):
        problems.append(f"window count: index-scan={ix} full-scan={fs} expected={exp_win}")
    # anomaly query vs truth (last 30 s of window cutoff region)
    # Live readings are preloaded slots [0, K_PRE) and written slots [K0, nk) not older than the
    # retention cutoff C. With a window under 30 s (short runs) the 30 s lookback reaches past
    # C and into the preload, so both bounds matter.
    lo3 = now_ms() - 30_000
    exp_an = sorted(((val_of(s, k), s) for s in range(SENSORS)
                     for k in (*range(K_PRE), *range(max(K0, 0), nk[s]))
                     if ts_of(s, k) >= max(lo3, C) and val_of(s, k) > THRESH), reverse=True)
    rs = sk.rows(sk.query(rc, G, "MATCH (r:Reading) WHERE r.ts >= $lo AND r.v > $th RETURN count(r)",
                          {"lo": lo3, "th": THRESH}))
    if sk.cval(rs[0][0]) != len(exp_an):
        problems.append(f"anomaly count {sk.cval(rs[0][0])} != {len(exp_an)}")
    mxid = sk.rows(sk.query(rc, G, "MATCH (r:Reading) RETURN max(id(r))"))
    ck_results.append({"idx": idx, "t": round(time.time() - t_run0, 1), "live": tot_got, "deleted_final": deleted,
                       "retention_converged": done, "max_id": sk.cval(mxid[0][0]), "problems": problems,
                       "pause_s": round(time.time() - t_start, 1), "verify_s": round(time.time() - t_v, 1)})
    checks.add(f"checkpoint{idx + 1} ground truth (counts/sums/min/max, window, NEXT chain, aggregations)",
               not problems and done, "; ".join(problems) + ("" if done else " retention did not converge"),
               "critical")
    PAUSE.unlink()
    print(f"checkpoint {idx + 1}: live={tot_got} problems={len(problems)} pause={time.time() - t_start:.0f}s",
          flush=True)


def supervisor(t_run0, duration):
    rc = srv.client()
    fracs = [0.30, 0.55, 0.80, 0.93]
    nxt_ck = 0
    next_sample = 0.0
    while not sup_stop.is_set():
        el = time.time() - t_run0
        if el >= duration:
            break
        if nxt_ck < len(fracs) and el >= fracs[nxt_ck] * duration:
            try:
                checkpoint(nxt_ck, rc, t_run0)
            except Exception as e:
                checks.add(f"checkpoint{nxt_ck + 1} executed", False, f"{type(e).__name__}: {e}", "major")
                try:
                    PAUSE.unlink()
                except FileNotFoundError:
                    pass
            nxt_ck += 1
            continue
        if el >= next_sample:
            try:
                row = sk.rows(sk.query(rc, G, "MATCH (r:Reading) RETURN count(r), max(id(r))"))[0]
                samples.append([round(el, 1), sk.cval(row[0]), sk.cval(row[1])])
            except Exception:
                pass
            next_sample = el + max(15.0, duration / 40)
        sup_stop.wait(1.0)


# ------------------------------------------------------------------ run
single_rate = max(1.0, (SENSORS // M + 1) / (P_MS / 1000) * 1.5)
n_batch_sens = SENSORS - SENSORS // M - 1
batch_rate = max(2.0, n_batch_sens / NB / (P_MS / 1000) / 40)
specs = [sk.WorkerSpec(writer, rate=batch_rate, count=NB, name="wbatch", init=writer_init),
         sk.WorkerSpec(writer, rate=single_rate, count=1, name="wsingle", init=writer_init),
         sk.WorkerSpec(retention, rate=4, count=1, name="retention"),
         sk.WorkerSpec(reader_recent, rate=25, count=2, name="rd_recent"),
         sk.WorkerSpec(reader_latest, rate=25, count=2, name="rd_latest"),
         sk.WorkerSpec(reader_agg_dev, rate=8, count=1, name="rd_aggdev"),
         sk.WorkerSpec(reader_agg_site, rate=1.5, count=1, name="rd_aggsite"),
         sk.WorkerSpec(reader_anomaly, rate=2, count=1, name="rd_anomaly")]
mon.mark("soak")
t_run0 = time.time()
sup = threading.Thread(target=supervisor, args=(t_run0, a.duration), daemon=True)
sup.start()
sk.run_load(srv, OUT, specs, a.duration)
sup_stop.set()
sup.join(600)

# ------------------------------------------------------------------ post
mon.mark("verify")
nviol = len(VIOL.read_text().splitlines()) if VIOL.exists() else 0
checks.add("inline read checks (lattice values, ordering, window bounds, anomaly rows, chain guard)",
           nviol == 0, (VIOL.read_text()[:800] if nviol else ""), "critical")
checks.add("all 4 checkpoints ran", len(ck_results) == 4, f"{len(ck_results)}", "major")
# id reuse
maxids = [x[2] for x in samples if x[2] is not None] + [c["max_id"] for c in ck_results]
lives = [x[1] for x in samples if x[1]]
peak_live = max(lives) if lives else 0
max_id_end = max(maxids[-3:]) if maxids else 0
id_ratio = max_id_end / peak_live if peak_live else None
nsamp_total = sum(1 for _ in samples)
total_inserted = None
checks.add("deleted ids are reused (max(id) stays bounded)", id_ratio is not None and id_ratio < 1.5,
           f"max_id_end={max_id_end} peak_live={peak_live} ratio={id_ratio}", "major")
rr = srv.client()
mem_info = rr.info("memory")
t_del = time.time()
used_before_del = mem_info.get("used_memory", 0) / 2**20
deleted_total = 0
while time.time() - t_del < 400:
    res = sk.query(rr, G, "MATCH (r:Reading) WITH r LIMIT 20000 DETACH DELETE r")
    n = stats_of(res, "Nodes deleted")
    deleted_total += n
    if n == 0:
        break
import psutil  # noqa: E402
rss_after_del = psutil.Process(srv.pid).memory_info().rss / 2**20
used_after_del = rr.info("memory").get("used_memory", 0) / 2**20
checks.add("graph empty of readings after final delete",
           sk.scalar(sk.query(rr, G, "MATCH (r:Reading) RETURN count(r)")) == 0, "", "major")
checks.add("server alive at end", srv.alive(), "", "critical")
mon.stop()
s = sk.analyze(OUT, srv)
# second-half memory slopes
rows_ = [l.split(",") for l in (OUT / "monitor.csv").read_text().splitlines()[1:]]
soak = [(float(x[0]), float(x[2]), float(x[3])) for x in rows_ if x[7] == "soak" and x[2] != "nan" and x[3] != "nan"]
h2 = soak[len(soak) // 2:]
xs = [x[0] / 3600 for x in h2]
rss_slope = sk._linreg(xs, [x[1] for x in h2]) if len(h2) > 3 else None
um_slope = sk._linreg(xs, [x[2] for x in h2]) if len(h2) > 3 else None
mean_rss = sum(x[1] for x in h2) / len(h2) if h2 else 0
proj = (rss_slope or 0) * (a.duration / 3600)
LONG = a.duration >= 600     # short smoke runs are still ramping up / too noisy to judge
checks.add("RSS plateaus in second half (slope projected over run length <= 10% of mean RSS)",
           mean_rss == 0 or proj <= 0.10 * mean_rss or not LONG,
           f"rss_slope={rss_slope:.0f} MB/h used_mem_slope={um_slope:.0f} MB/h mean_rss={mean_rss:.0f}"
           if rss_slope is not None else "no samples", "major" if LONG else "minor")
for op in ("ingest_batch", "ingest_single", "retention_delete", "q_recent_by_sensor", "q_latest_reading",
           "q_agg_device_2m", "q_agg_site_2m", "q_anomaly_join_30s"):
    d = s["ops"].get(op)
    if d and d["drift"] and d["drift"]["ratio"]:
        checks.add(f"p99 does not drift up: {op}", d["drift"]["ratio"] <= (1.5 if LONG else 3.0),
                   f"early={d['drift']['p99_early_ms']:.2f}ms late={d['drift']['p99_late_ms']:.2f}ms",
                   "major" if LONG else "minor")
sk.check_no_operation_errors(checks, s)
s = sk.analyze(OUT, srv)
s["meta"] = {"scenario": "s12_iot_retention", "sites": SITES, "devices": DEVICES, "sensors": SENSORS,
             "window_s": W_S, "live_readings_target": SENSORS * K_PRE, "ingest_s": ingest_s,
             "duration": a.duration, "scale": a.scale,
             "rss_slope_mb_per_h_2nd_half": rss_slope, "used_mem_slope_mb_per_h_2nd_half": um_slope,
             "mean_rss_2nd_half_mb": mean_rss, "max_id_end": max_id_end, "peak_live": peak_live,
             "max_id_over_live": id_ratio, "id_samples": samples, "checkpoints": ck_results,
             "used_mem_before_final_delete_mb": used_before_del, "used_mem_after_final_delete_mb": used_after_del,
             "rss_after_final_delete_mb": rss_after_del, "final_delete_nodes": deleted_total,
             "viol_lines": nviol}
(OUT / "summary.json").write_text(json.dumps(s, indent=1, default=str))
srv.stop()
sk.print_summary(s)
print(json.dumps({k: v for k, v in s["meta"].items() if k not in ("id_samples",)}, default=str)[:3000])
