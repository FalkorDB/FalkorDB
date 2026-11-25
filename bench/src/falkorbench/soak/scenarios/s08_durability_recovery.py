"""s08_durability_recovery — persistence + crash recovery as in production.

Models: a graph service with AOF (RDB preamble) and/or RDB persistence, continuously written
(typed-property create/update/delete of nodes+edges), periodically BGSAVE / BGREWRITEAOF'd, and
killed (SIGKILL / SIGTERM / SHUTDOWN SAVE / DEBUG RELOAD) every few minutes.

Full scale (--scale 1.0): "big" graph 1.5M nodes / 7.5M edges (indexed), ledger graph "led"
(4 writers x 5000 ids = 20k nodes + edges, 14 typed property kinds), 3 static multi-label graphs,
"special" (edge-case values) and an empty graph.

Ground truth:
  * big graph: every property is a function of the node id; counts/sums/sampled neighbour lists are
    computed in Python independent of the server. A counter property (cnt) is bumped by writers
    and verified against an ack ledger.
  * led graph: per-writer append-only ledger (pending line before send, ack line after) with ack
    timestamps. After each kill the observed graph must equal base + acked ops (+ at most the one
    in-flight op per writer). appendfsync always => exact; everysec => only ops acked within the
    last LOSS_WINDOW seconds before the kill may be missing.
  * RDB phases: restored graph digest == digest at the last successful SAVE (writes after it vanish).
  * static graphs: digest of a full dump before == after every restart.
Segments end with the kill while workers are still running (in-flight ops), then the server is
restarted, recovery time is measured and everything is verified; the verified observed state
becomes the base for the next segment.

CLI: --module --port --out --duration --scale   (ports used: --port only)
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from falkorbench.soak import soakkit as sk
import redis  # noqa: E402

ap = argparse.ArgumentParser()
sk.add_engine_args(ap)  # --module PATH | --image REF
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--duration", type=float, default=60)
ap.add_argument("--scale", type=float, default=0.05)
a = ap.parse_args()
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)

BIG, LED = "big", "led"
STATIC = ["multi0", "multi1", "multi2", "special", "emptyg"]
N = max(5000, int(1_500_000 * a.scale))  # 7.5M edges; 2M/10M ingests in ~7.5 min on Rust (too close to the 8 min cap)
DEG = 5
W = 4                                   # ledger writers
R = max(500, int(5000 * a.scale))       # ids per writer
M = max(500, int(20000 * a.scale))      # static multi-graph size
VKEY = 100000 if a.scale >= 1.0 else 10000
LOSS_WINDOW = 3.5                       # appendfsync everysec: seconds before kill that may be lost
SEG = max(12.0, a.duration / 10.0)      # seconds per load segment (8 segments)
RATE_W = float(os.environ.get("S08_RATE_W", 25))     # per ledger writer
RATE_B = float(os.environ.get("S08_RATE_B", 30))     # per bump worker
RATE_R = float(os.environ.get("S08_RATE_R", 60))     # per reader
RATE_P = 20.0                                         # latency probe

LEDGER_FIELDS = ["s", "i", "f", "b", "li", "ls", "lm", "p", "v", "d", "ldt", "du", "lt", "x"]
STATIC_FIELDS = ["s", "i", "f", "b", "li", "ls", "p", "v", "d", "du", "e1", "e2", "e3", "e4"]

AOF_COMMON = ["--appendonly", "yes", "--aof-use-rdb-preamble", "yes"]
EXTRA_BASE = ["--enable-debug-command", "yes", "--appenddirname", "aof"]
EXTRA_STRICT = EXTRA_BASE + AOF_COMMON + ["--appendfsync", "always"]
EXTRA_EVERYSEC = EXTRA_BASE + AOF_COMMON + ["--appendfsync", "everysec"]
EXTRA_RDB = EXTRA_BASE + ["--appendonly", "no"]

srv = sk.Server(a.module, a.port, out, module_args=["VKEY_MAX_ENTITY_COUNT", str(VKEY)], extra=EXTRA_BASE)
import atexit  # noqa: E402


def _cleanup():
    try:
        srv.kill9()
    except Exception:
        pass


atexit.register(_cleanup)
mon = sk.Monitor(srv, out)
mon.start()
checks = sk.Checks(out)
r = srv.client()
meta: dict = {"scenario": "s08_durability_recovery", "scale": a.scale, "duration": a.duration,
              "big_nodes": N, "big_edges": N * DEG, "led_ids": W * R, "static_m": M, "vkey_max": VKEY,
              "seg_s": SEG, "recoveries": [], "windows": [], "notes": []}


def q(g, text, params=None, ro=False):
    return sk.query(r, g, text, params, ro=ro)


def rd():
    return srv.client()


# ----------------------------------------------------------------- canonical values
def canon(cell):
    """compact cell -> JSON-able canonical [tag, value]; floats parsed, bytes decoded."""
    if not (isinstance(cell, list) and len(cell) == 2 and isinstance(cell[0], int)):
        return cell
    t, v = cell
    if t == 1:
        return None
    if t == 2:
        return [2, v.decode() if isinstance(v, bytes) else v]
    if t == 3:
        return [3, int(v)]
    if t == 4:
        return [4, v in (b"true", "true", 1, True)]
    if t == 5:
        f = float(v)
        return [5, f if math.isfinite(f) else str(f)]
    if t == 6:
        return [6, [canon(c) for c in v]]
    if t == 11:
        return [11, [float(x) for x in v]]
    if t == 12:
        return [12, [float(x) for x in v]]
    if t in (13, 14, 15, 16):
        return [t, int(v)]
    return [t, repr(v)]


def eq(x, y):
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
    if isinstance(x, list):
        return isinstance(y, list) and len(x) == len(y) and all(eq(p, q_) for p, q_ in zip(x, y))
    if isinstance(x, float) and isinstance(y, float):
        return x == y or abs(x - y) <= 1e-14 * max(1.0, abs(x), abs(y))
    return type(x) == type(y) and x == y


def sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------- value generators (ledger)
import datetime as _dt  # noqa: E402

BIGINTS = [2**53 + 1, -(2**53) - 1, 2**62, 9223372036854775807]  # trimmed by the startup probe
LT_OFF = None  # calibrated offset of localtime compact value (differs between modules)
USTR = ["é", "ß", "漢字", "😀", "q\"uote", "back\\slash", " sp ace ", "", "x"]


def g_str(rng):
    n = rng.choice([0, 1, 5, 20, 60, 200, 200, 3000]) if rng.random() < 0.97 else 6000
    s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789 ") for _ in range(min(n, 40)))
    if n > 40:
        s = (s or "k") * (n // max(1, len(s)))
    if rng.random() < 0.3:
        s += rng.choice(USTR)
    return sk.lit(s), [2, s]


def g_int(rng):
    c = rng.random()
    v = rng.randint(-100, 100) if c < 0.3 else rng.randint(-10**9, 10**9) if c < 0.8 else \
        rng.choice(BIGINTS)
    return repr(v), [3, v]


def _fl(rng):
    c = rng.random()
    if c < 0.25:
        return rng.randint(-1000, 1000) / 1024.0
    if c < 0.4:
        return float(rng.randint(-1000, 1000))
    x = round(rng.uniform(1, 1e6), 6) * rng.choice([-1, 1])
    return x


def g_f(rng):
    x = _fl(rng)
    return repr(x), [5, x]


def g_b(rng):
    v = rng.random() < 0.5
    return ("true" if v else "false"), [4, v]


def g_li(rng):
    xs = [rng.randint(-1000, 1000) for _ in range(rng.choice([0, 1, 3, 10, 50]))]
    return "[" + ",".join(map(repr, xs)) + "]", [6, [[3, v] for v in xs]]


def g_ls(rng):
    xs = [rng.choice(["a", "bb", "é", "漢", "x y", ""]) + str(rng.randint(0, 99)) for _ in range(rng.choice([0, 1, 3, 8]))]
    return "[" + ",".join(sk.lit(x) for x in xs) + "]", [6, [[2, v] for v in xs]]


def g_lm(rng):
    items, cs = [], []
    for _ in range(rng.choice([1, 2, 4, 6])):
        k = rng.choice("ifsbn")
        if k == "i":
            e, c = g_int(rng)
        elif k == "f":
            e, c = g_f(rng)
        elif k == "s":
            e, c = g_str(rng)
            if len(c[1]) > 200:
                e, c = sk.lit("m"), [2, "m"]
        elif k == "b":
            e, c = g_b(rng)
        else:
            xs = [rng.randint(0, 9) for _ in range(3)]
            e, c = repr(xs), [6, [[3, v] for v in xs]]
        items.append(e)
        cs.append(c)
    return "[" + ",".join(items) + "]", [6, cs]


def g_p(rng):
    la, lo = rng.randint(-90 * 1024, 90 * 1024) / 1024.0, rng.randint(-180 * 1024, 180 * 1024) / 1024.0
    return f"point({{latitude:{la!r},longitude:{lo!r}}})", [11, [la, lo]]


def g_v(rng):
    xs = [rng.randint(-800, 800) / 8.0 for _ in range(rng.choice([1, 2, 4, 16]))]
    return "vecf32([" + ",".join(map(repr, xs)) + "])", [12, xs]


def _epoch(dt):
    return int((dt - _dt.datetime(1970, 1, 1)).total_seconds())


def g_d(rng):
    d = _dt.date.fromordinal(rng.randint(_dt.date(1900, 1, 1).toordinal(), _dt.date(2100, 12, 28).toordinal()))
    return f"date('{d.isoformat()}')", [14, _epoch(_dt.datetime(d.year, d.month, d.day))]


def g_ldt(rng):
    d = _dt.datetime(rng.randint(1900, 2100), rng.randint(1, 12), rng.randint(1, 28),
                     rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59))
    return f"localdatetime('{d.isoformat()}')", [13, _epoch(d)]


def g_du(rng):
    d, h, m, s = rng.randint(0, 400), rng.randint(0, 23), rng.randint(0, 59), rng.randint(1, 59)
    return f"duration('P{d}DT{h}H{m}M{s}S')", [16, d * 86400 + h * 3600 + m * 60 + s]


def g_lt(rng):
    h, m, s = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
    return f"localtime('{h:02d}:{m:02d}:{s:02d}')", [15, LT_OFF + h * 3600 + m * 60 + s]


GEN = {"s": g_str, "i": g_int, "f": g_f, "b": g_b, "li": g_li, "ls": g_ls, "lm": g_lm, "p": g_p,
       "v": g_v, "d": g_d, "ldt": g_ldt, "du": g_du, "lt": g_lt}


def gen_field(name, rng):
    if name == "x":  # property whose type changes between writes
        return GEN[rng.choice(["s", "i", "f", "b", "li"])](rng)
    return GEN[name](rng)


# ----------------------------------------------------------------- server helpers
def start_server(timeout=900):
    """Server.start() with a long deadline (large AOF/RDB loads)."""
    srv.start(timeout=timeout)


def wait_info(key, want=0, timeout=900, section="persistence"):
    c = rd()
    t = time.time()
    while time.time() - t < timeout:
        if int(c.info(section).get(key, 0)) == want:
            return time.time() - t
        time.sleep(0.1)
    return None


def disk_mb():
    tot = 0
    for p, _, fs in os.walk(out):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(p, f))
            except OSError:
                pass
    return tot / 2**20


def wait_indexes(graph, timeout=900):
    t = time.time()
    while time.time() - t < timeout:
        res = sk.query(rd(), graph, "CALL db.indexes()", ro=True)
        st = [canon(row[7]) for row in sk.rows(res)]
        if st and all(s and s[1] == "OPERATIONAL" for s in st):
            return True
        if not st:
            return False
        time.sleep(0.2)
    return False


# ----------------------------------------------------------------- ground truth + observation
def big_truth():
    si = sl = sw = 0
    for i in range(N):
        si += (i * 7919) % 1000003
        sl += len("node-%d" % i)
        sw += sum((i * 31 + k) % 1000 for k in range(DEG))
    return {"n": N, "e": N * DEG, "sum_i": si, "sum_len": sl, "sum_w": sw}


def observe_big(rc, truth, rng_seed):
    """returns (ok, detail, cnt_sum). Verifies full aggregates + sampled nodes/neighbour lists."""
    bad = []
    g = lambda t: sk.query(rc, BIG, t, ro=True)  # noqa: E731
    row = sk.rows(g("MATCH (n:N) RETURN count(n), sum(n.i), sum(size(n.s)), sum(n.cnt)"))[0]
    n, si, sl, cs = (canon(c)[1] for c in row)
    row = sk.rows(g("MATCH ()-[e:E]->() RETURN count(e), sum(e.w)"))[0]
    e, sw = (canon(c)[1] for c in row)
    for k, got in (("n", n), ("sum_i", si), ("sum_len", sl), ("e", e), ("sum_w", sw)):
        if got != truth[k]:
            bad.append(f"{k}: got {got} want {truth[k]}")
    import random
    rr = random.Random(rng_seed)
    ids = list({rr.randrange(N) for _ in range(300)})
    res = sk.rows(sk.query(rc, BIG, "UNWIND $ids AS x MATCH (n:N {id:x}) RETURN n.id, n.i, n.s, n.f", {"ids": ids}, ro=True))
    seen = {}
    for rw in res:
        cid, ci, cs_, cf = (canon(c) for c in rw)
        seen[cid[1]] = (ci[1], cs_[1], cf[1])
    for i in set(ids):
        if seen.get(i) != ((i * 7919) % 1000003, "node-%d" % i, i / 8.0):
            bad.append(f"node {i}: {seen.get(i)}")
            break
    got = {}
    # one query per id: "UNWIND $ids + MATCH (n {id:x})-[]->(m)" is quadratic on Rust (see report)
    for i in ids[:100]:
        for rw in sk.rows(sk.query(rc, BIG, "MATCH (n:N {id:$id})-[e:E]->(m) RETURN e.w, m.id", {"id": i}, ro=True)):
            w_, m_ = (canon(c)[1] for c in rw)
            got.setdefault(i, []).append((w_, m_))
    for i in ids[:100]:
        want = sorted(((i * 31 + k) % 1000, (i * 7919 + k * 104729 + 1) % N) for k in range(DEG))
        if sorted(got.get(i, [])) != want:
            bad.append(f"neighbours of {i}: {sorted(got.get(i, []))} != {want}")
            break
    return (not bad), "; ".join(bad[:5]), cs


def observe_led(rc):
    cols = ["n.id"] + [f"n.{f}" for f in LEDGER_FIELDS]
    res = sk.rows(sk.query(rc, LED, "MATCH (n:L) RETURN " + ",".join(cols), ro=True))
    nodes, dups = {}, 0
    for rw in res:
        cs = [canon(c) for c in rw]
        nid = cs[0][1]
        if nid in nodes:
            dups += 1
        nodes[nid] = {f: c for f, c in zip(LEDGER_FIELDS, cs[1:]) if c is not None}
    res = sk.rows(sk.query(rc, LED, "MATCH (a:L)-[e:R]->(b:L) RETURN a.id, b.id, e.w", ro=True))
    edges, edup = {}, 0
    for rw in res:
        x, y, w = (canon(c)[1] for c in rw)
        if (x, y) in edges:
            edup += 1
        edges[(x, y)] = w
    tot_e = sk.scalar(sk.query(rc, LED, "MATCH ()-[e]->() RETURN count(e)", ro=True))
    tot_n = sk.scalar(sk.query(rc, LED, "MATCH (n) RETURN count(n)", ro=True))
    return nodes, edges, {"dup_nodes": dups, "dup_edges": edup, "total_edges": tot_e, "total_nodes": tot_n}


def static_digest(rc):
    d = {}
    cols = ",".join(f"n.{f}" for f in STATIC_FIELDS)
    for g in STATIC:
        nodes = sk.rows(sk.query(rc, g, f"MATCH (n) RETURN labels(n), n.id, {cols}", ro=True))
        edges = sk.rows(sk.query(rc, g, "MATCH (a)-[e]->(b) RETURN a.id, type(e), b.id, e.w", ro=True))
        nr = sorted(json.dumps([canon(c) for c in rw], sort_keys=True) for rw in nodes)
        er = sorted(json.dumps([canon(c) for c in rw], sort_keys=True) for rw in edges)
        d[g] = {"n": len(nr), "e": len(er), "sha": sha([nr, er])}
    return d


def led_state_dump(nodes, edges):
    return {"nodes": {str(k): v for k, v in nodes.items()}, "edges": [[x, y, w] for (x, y), w in edges.items()]}


def write_base(ep, nodes, edges, bump_sum):
    d = led_state_dump(nodes, edges)
    d["bump_sum"] = bump_sum
    (out / f"base-{ep}.json").write_text(json.dumps(d))


# ----------------------------------------------------------------- ledger replay
def replay(ep, base):
    """expected state from base + ledger files of epoch ep.
    returns (nodes, edges, tlast, pend, bump_acked_times, bump_pending)"""
    nodes = {int(k): v for k, v in base["nodes"].items()}
    edges = {(x, y): w for x, y, w in base["edges"]}
    tlast, pend = {}, {}
    for f in sorted(out.glob(f"led-{ep}-*.jsonl")):
        waiting = {}
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue  # torn last line (writer killed) — cannot happen, writers are not killed
            if "p" in d:
                waiting[d["p"]] = d
            elif "a" in d:
                p = waiting.pop(d["a"], None)
                if p:
                    for rec in p["r"]:
                        if rec[0] == "n":
                            k = ("n", rec[1])
                            if rec[2] is None:
                                nodes.pop(rec[1], None)
                            else:
                                nodes[rec[1]] = rec[2]
                        else:
                            k = ("e", rec[1], rec[2])
                            if rec[3] is None:
                                edges.pop((rec[1], rec[2]), None)
                            else:
                                edges[(rec[1], rec[2])] = rec[3]
                        tlast[k] = d["t"]
            elif "x" in d:
                waiting.pop(d["x"], None)
        for p in waiting.values():
            for rec in p["r"]:
                k = ("n", rec[1]) if rec[0] == "n" else ("e", rec[1], rec[2])
                pend.setdefault(k, []).append(rec[2] if rec[0] == "n" else rec[3])
    bump_t, bump_pend = [], 0
    for f in sorted(out.glob(f"bump-{ep}-*.jsonl")):
        waiting = set()
        for line in f.read_text().splitlines():
            d = json.loads(line)
            if "p" in d:
                waiting.add(d["p"])
            elif "a" in d:
                waiting.discard(d["a"])
                bump_t.append(d["t"])
            elif "x" in d:
                waiting.discard(d["x"])
        bump_pend += len(waiting)
    return nodes, edges, tlast, pend, bump_t, bump_pend


def compare_led(exp_n, exp_e, tlast, pend, onodes, oedges, info, tkill, window):
    """returns (violations, tolerated, detail list)"""
    viol, tol, det = 0, 0, []
    if info["dup_nodes"] or info["dup_edges"]:
        viol += info["dup_nodes"] + info["dup_edges"]
        det.append(f"duplicates nodes={info['dup_nodes']} edges={info['dup_edges']}")
    if info["total_nodes"] != len(onodes) or info["total_edges"] != len(oedges):
        viol += 1
        det.append(f"stray entities: total_nodes={info['total_nodes']} L={len(onodes)} total_edges={info['total_edges']} R={len(oedges)}")
    keys = set(("n", i) for i in exp_n) | set(("n", i) for i in onodes) | \
        set(("e", x, y) for x, y in exp_e) | set(("e", x, y) for x, y in oedges) | set(pend)
    for k in keys:
        if k[0] == "n":
            e, o = exp_n.get(k[1]), onodes.get(k[1])
        else:
            e, o = exp_e.get((k[1], k[2])), oedges.get((k[1], k[2]))
        if eq(e, o) or any(eq(o, p) for p in pend.get(k, [])):
            continue
        if window and tlast.get(k, 0) > tkill - window:
            tol += 1
            continue
        viol += 1
        if len(det) < 6:
            if isinstance(e, dict) and isinstance(o, dict):
                fd = [f"{f}: exp {json.dumps(e.get(f))[:70]} obs {json.dumps(o.get(f))[:70]}"
                      for f in sorted(set(e) | set(o)) if not eq(e.get(f), o.get(f))]
                det.append(f"{k} (pending={json.dumps(pend.get(k))[:60]}, tlast={tlast.get(k)}): " + " | ".join(fd[:4]))
            else:
                det.append(f"{k}: expected {json.dumps(e)[:160]} observed {json.dumps(o)[:160]}")
    return viol, tol, det


# ----------------------------------------------------------------- workers
BASEF = ""
SEGDIR = ""
EP = 0


def ledger_init(ctx):
    base = json.loads(Path(BASEF).read_text())
    lo, hi = ctx.wid * R, (ctx.wid + 1) * R
    ctx.state["nodes"] = {int(k): v for k, v in base["nodes"].items() if lo <= int(k) < hi}
    ctx.state["edges"] = {(x, y): w for x, y, w in base["edges"] if lo <= x < hi}
    ctx.state["seq"] = 0
    ctx.state["led"] = open(Path(SEGDIR) / f"led-{EP}-{ctx.wid}.jsonl", "a")


def _log(f, d):
    f.write(json.dumps(d) + "\n")
    f.flush()


def ledger_op(ctx):
    st, rng = ctx.state, ctx.rng
    nodes, edges = st["nodes"], st["edges"]
    lo = ctx.wid * R
    nid = lo + rng.randrange(R)
    recs = []
    name = None
    if nid not in nodes:
        props = []
        state = {}
        for f in LEDGER_FIELDS:
            if rng.random() < 0.6:
                e, c = gen_field(f, rng)
                props.append(f"{f}:{e}")
                state[f] = c
        query = f"CREATE (:L {{id:{nid}" + "".join("," + p for p in props) + "})"
        recs = [["n", nid, state]]
        name = "create_node"
    else:
        c = rng.random()
        if c < 0.10:
            inc = [(x, y) for (x, y) in edges if x == nid or y == nid]
            query = f"MATCH (n:L {{id:{nid}}}) DETACH DELETE n"
            recs = [["n", nid, None]] + [["e", x, y, None] for x, y in inc]
            name = "delete_node"
        elif c < 0.55:
            new = dict(nodes[nid])
            sets = []
            for f in rng.sample(LEDGER_FIELDS, rng.randint(1, 3)):
                if rng.random() < 0.2:
                    sets.append(f"n.{f}=null")
                    new.pop(f, None)
                else:
                    e, cc = gen_field(f, rng)
                    sets.append(f"n.{f}={e}")
                    new[f] = cc
            query = f"MATCH (n:L {{id:{nid}}}) SET " + ",".join(sets)
            recs = [["n", nid, new]]
            name = "update_node"
        elif c < 0.75 and len(edges) < 3 * R:
            b = None
            for _ in range(8):
                cand = lo + rng.randrange(R)
                if cand in nodes and (nid, cand) not in edges:
                    b = cand
                    break
            if b is None:
                return ledger_op(ctx)
            w = rng.randint(-1000, 1000)
            query = f"MATCH (a:L {{id:{nid}}}),(b:L {{id:{b}}}) CREATE (a)-[:R {{w:{w}}}]->(b)"
            recs = [["e", nid, b, w]]
            name = "create_edge"
        elif edges:
            (x, y) = rng.choice(list(edges))
            if rng.random() < 0.7 or len(edges) >= 3 * R:
                query = f"MATCH (:L {{id:{x}}})-[e:R]->(:L {{id:{y}}}) DELETE e"
                recs = [["e", x, y, None]]
                name = "delete_edge"
            else:
                w = rng.randint(-1000, 1000)
                query = f"MATCH (:L {{id:{x}}})-[e:R]->(:L {{id:{y}}}) SET e.w={w}"
                recs = [["e", x, y, w]]
                name = "update_edge"
        else:
            return ledger_op(ctx)
    st["seq"] += 1
    seq = st["seq"]
    _log(st["led"], {"p": seq, "t": time.time(), "r": recs})
    try:
        sk.query(ctx.r, LED, query)
    except redis.exceptions.ConnectionError:
        raise  # outcome unknown: pending line stays
    except Exception as e:
        _log(st["led"], {"x": seq})
        e.op = name
        raise
    for rec in recs:
        if rec[0] == "n":
            if rec[2] is None:
                nodes.pop(rec[1], None)
            else:
                nodes[rec[1]] = rec[2]
        else:
            if rec[3] is None:
                edges.pop((rec[1], rec[2]), None)
            else:
                edges[(rec[1], rec[2])] = rec[3]
    _log(st["led"], {"a": seq, "t": time.time()})
    return name


def bump_init(ctx):
    ctx.state["f"] = open(Path(SEGDIR) / f"bump-{EP}-{ctx.wid}.jsonl", "a")
    ctx.state["seq"] = 0


def bump_op(ctx):
    st = ctx.state
    st["seq"] += 1
    s = st["seq"]
    _log(st["f"], {"p": s})
    try:
        sk.query(ctx.r, BIG, "MATCH (n:N {id:$id}) SET n.cnt = n.cnt + 1", {"id": ctx.rng.randrange(N)})
    except redis.exceptions.ConnectionError:
        raise
    except Exception as e:
        _log(st["f"], {"x": s})
        raise
    _log(st["f"], {"a": s, "t": time.time()})
    return "bump_cnt"


def reader_init(ctx):
    ctx.state["v"] = open(Path(SEGDIR) / f"viol-{EP}-{ctx.wid}.txt", "a")


def reader_op(ctx):
    rng = ctx.rng
    c = rng.random()
    if c < 0.5:
        i = rng.randrange(N)
        res = sk.rows(sk.query(ctx.r, BIG, "MATCH (n:N {id:$id})-[e:E]->(m) RETURN m.id, e.w", {"id": i}, ro=True))
        want = sorted(((i * 7919 + k * 104729 + 1) % N, (i * 31 + k) % 1000) for k in range(DEG))
        got = sorted((canon(x[0])[1], canon(x[1])[1]) for x in res)
        if got != want:
            ctx.state["v"].write(f"1hop {i}: {got} != {want}\n")
            ctx.state["v"].flush()
        return "read_1hop"
    if c < 0.8:
        i = rng.randrange(N)
        res = sk.rows(sk.query(ctx.r, BIG, "MATCH (n:N {id:$id}) RETURN n.i, n.s, n.f", {"id": i}, ro=True))
        if not res or [canon(x) for x in res[0]] != [[3, (i * 7919) % 1000003], [2, "node-%d" % i], [5, i / 8.0]]:
            ctx.state["v"].write(f"lookup {i}: {res}\n")
            ctx.state["v"].flush()
        return "read_lookup"
    i = ctx.wid  # ledger read
    sk.query(ctx.r, LED, "MATCH (n:L {id:$id}) RETURN n.id, n.s, n.p, n.v", {"id": rng.randrange(W * R)}, ro=True)
    return "read_led"


def probe_op(ctx):
    sk.query(ctx.r, BIG, "MATCH (n:N {id:$id}) RETURN n.i", {"id": ctx.rng.randrange(N)}, ro=True)
    return "probe_lookup"


# ----------------------------------------------------------------- latency windows
def window_stats(tag, t_start, t_end, label):
    """p99 per op inside vs outside [t_start, t_end] (wall clock) for the run_load with this tag."""
    runs = json.loads((out / "load_runs.json").read_text())
    run = [x for x in runs if x["tag"] == tag][-1]
    t0 = run["start"]
    lo, hi = int(t_start - t0), int(math.ceil(t_end - t0))
    ins, outs = {}, {}
    for f in out.glob(f"lat-{tag}*.jsonl"):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            if "op" not in d or d["op"].startswith("_"):
                continue
            tgt = ins if lo <= d["s"] <= hi else outs
            m = tgt.setdefault(d["op"], {"h": {}, "n": 0, "err": 0})
            m["n"] += d["n"]
            m["err"] += d["err"]
            for b, c in d["h"].items():
                m["h"][int(b)] = m["h"].get(int(b), 0) + c
    res = {}
    for op in set(ins) | set(outs):
        i_, o_ = ins.get(op), outs.get(op)
        res[op] = {"in_n": i_["n"] if i_ else 0, "in_err": i_["err"] if i_ else 0,
                   "in_p99_ms": sk.hpct(i_["h"], .99) / 1e3 if i_ and i_["h"] else None,
                   "in_max_ms": sk.bucket_val(max(i_["h"])) / 1e3 if i_ and i_["h"] else None,
                   "out_p99_ms": sk.hpct(o_["h"], .99) / 1e3 if o_ and o_["h"] else None}
    return {"label": label, "window_s": round(t_end - t_start, 2), "ops": res}


def bg_thread(cmd, delay, rec, flag_key):
    """after `delay` seconds issue BGSAVE/BGREWRITEAOF and time it until finished."""
    def run():
        time.sleep(delay)
        c = rd()
        t = time.time()
        try:
            c.execute_command(cmd)
        except Exception as e:
            rec["err"] = str(e)[:200]
            return
        rec["cmd_s"] = time.time() - t
        rec["start"] = t
        try:
            w = wait_info(flag_key, 0, timeout=900)
        except Exception:
            rec["interrupted"] = True  # server killed mid-save: that is a deliberate crash test
            rec["end"] = time.time()
            return
        rec["end"] = time.time()
        rec["total_s"] = rec["end"] - t
        rec["status_ok"] = w is not None
        try:
            ip = c.info("persistence")
            rec["cow_mb"] = max(ip.get("rdb_last_cow_size", 0), ip.get("aof_last_cow_size", 0)) / 2**20
            rec["status"] = (ip.get("rdb_last_bgsave_status"), ip.get("aof_last_bgrewrite_status"))
        except Exception:
            pass
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


# ----------------------------------------------------------------- phase 0: ingest
mon.mark("ingest")
t_ing = time.time()
# calibrate localtime representation (C and Rust use different epochs)
q("calib", "CREATE (:T {a:localtime('00:00:00'), b:localtime('12:00:00')})")
cr = sk.rows(q("calib", "MATCH (n:T) RETURN n.a, n.b", ro=True))[0]
ca, cb = canon(cr[0])[1], canon(cr[1])[1]
if cb - ca == 43200:
    LT_OFF = ca
else:
    LT_OFF = None
    LEDGER_FIELDS.remove("lt")
    meta["notes"].append(f"localtime compact repr not linear ({ca},{cb}); lt excluded")
    checks.add("localtime compact value is linear in seconds", False, f"{ca},{cb}", "minor")
# probe: does SET after an INT64_MAX value apply? (the C reference loses such updates)
q("calib", "CREATE (:P {id:1, i:9223372036854775807})")
q("calib", "MATCH (n:P) SET n.i = -27")
if sk.scalar(q("calib", "MATCH (n:P) RETURN n.i", ro=True)) != -27:
    BIGINTS.remove(9223372036854775807)
    meta["notes"].append("module loses SET n.i=-27 after n.i=INT64_MAX (C reference defect); INT64_MAX excluded from ledger")
    checks.add("SET after INT64_MAX property value is applied", False, "SET n.i=-27 left INT64_MAX", "minor")
q("calib", "MATCH (n:P) SET n.i = -4611686018427387904")
if sk.scalar(q("calib", "MATCH (n:P) RETURN n.i", ro=True)) != -4611686018427387904:
    meta["notes"].append("module mangles negative int literal -2^62 (becomes positive); not used in ledger")
    checks.add("negative int literal -2^62 stored correctly", False, "", "minor")
r.execute_command("GRAPH.DELETE", "calib")

q(BIG, "CREATE INDEX FOR (n:N) ON (n.id)")
B = 20000
for i in range(0, N, B):
    q(BIG, "UNWIND range($a,$b) AS i CREATE (:N {id:i, i:(i*7919)%1000003, s:'node-'+toString(i), f:i/8.0, cnt:0})",
      {"a": i, "b": min(N, i + B) - 1})
wait_indexes(BIG)
EB = 5000
for i in range(0, N, EB):
    q(BIG, f"UNWIND range($a,$b) AS i UNWIND range(0,{DEG - 1}) AS k MATCH (u:N {{id:i}}),(v:N {{id:(i*7919+k*104729+1)%$n}}) "
           f"CREATE (u)-[:E {{w:(i*31+k)%1000}}]->(v)", {"a": i, "b": min(N, i + EB) - 1, "n": N})
q(LED, "CREATE INDEX FOR (n:L) ON (n.id)")
q(LED, "CREATE (:L {id:-1})")
q(LED, "MATCH (n:L {id:-1}) DELETE n")
# static graphs
for k, (lab, rel) in enumerate([("Aa", "R1"), ("Bb:Cc", "R2"), ("Dd", "R3")]):
    g = f"multi{k}"
    first = lab.split(":")[0]
    q(g, f"CREATE INDEX FOR (n:{first}) ON (n.id)")
    q(g, f"UNWIND range(1,$m) AS i CREATE (:{lab} {{id:i, s:'a'+toString(i)+'é', i:i*{k + 3}, f:i/16.0, b:(i%2=0), li:[i,i+1], "
         f"ls:['x'+toString(i)], p:point({{latitude:(i%90)*1.0, longitude:(i%180)*1.0}}), v:vecf32([i/8.0, 1.5]), "
         f"d:date('2000-01-01'), du:duration('P1DT2H')}})", {"m": M})
    wait_indexes(g)
    q(g, f"UNWIND range(1,$m) AS i MATCH (a:{first} {{id:i}}),(b:{first} {{id:(i*7)%$m+1}}) CREATE (a)-[:{rel} {{w:i}}]->(b)", {"m": M})
q("special", "CREATE (:S {id:1, s:'', i:-9223372036854775807, f:0.0, e4:-0.0, li:[], ls:[''], "
             "p:point({latitude:-90.0, longitude:180.0}), v:vecf32([0.0]), d:date('1900-01-01'), du:duration('P0DT0H0M1S')})")
q("special", "MATCH (n:S {id:1}) SET n.e1 = 0.0/0.0, n.e2 = 1.0/0.0, n.e3 = -1.0/0.0")
q("special", "CREATE (:S {id:2, s:'" + "é漢😀" * 2000 + "', i:9223372036854775807, f:1000000000000000.0, e4:0.000001, li:[1,2,3], "
             "ls:['a','b'], b:false, v:vecf32([1.5,2.5,3.5])})")
q("special", "MATCH (a:S {id:1}),(b:S {id:2}) CREATE (a)-[:SP {w:-1}]->(b)")
q("emptyg", "CREATE (:Tmp {a:1})")
q("emptyg", "MATCH (n:Tmp) DELETE n")
ingest_s = time.time() - t_ing

mon.mark("verify-ingest")
TRUTH = big_truth()
ok, det, cs0 = observe_big(r, TRUTH, 1)
checks.add("ingest: big graph matches python ground truth", ok, det, "critical")
SD0 = static_digest(r)
checks.add("ingest: static graphs non-empty", all(SD0[g]["n"] > 0 for g in STATIC[:4]) and SD0["emptyg"]["n"] == 0,
           json.dumps(SD0)[:300], "critical")
GL0 = sorted(x.decode() if isinstance(x, bytes) else x for x in r.execute_command("GRAPH.LIST"))
meta["ingest_s"] = round(ingest_s, 1)
meta["graph_list"] = GL0
write_base(0, {}, {}, 0)
BUMP_BASE = 0
EPOCH = 0

# ----------------------------------------------------------------- enable AOF (RDB preamble base)
mon.mark("aof-rewrite")
t = time.time()
r.execute_command("CONFIG", "SET", "appendonly", "yes")
w = wait_info("aof_rewrite_in_progress", 0, 900)
ip = r.info("persistence")
meta["initial_aof_rewrite_s"] = round(time.time() - t, 2)
checks.add("initial AOF rewrite (RDB preamble) succeeded", ip.get("aof_last_bgrewrite_status") == "ok", str(ip.get("aof_last_bgrewrite_status")), "critical")
r.execute_command("CONFIG", "SET", "appendfsync", "always")
srv.extra = EXTRA_STRICT
meta["disk_mb_after_aof_base"] = round(disk_mb())

# ----------------------------------------------------------------- segment machinery
specs_seq = []


def make_specs(rate_scale=1.0):
    return [sk.WorkerSpec(ledger_op, rate=RATE_W * rate_scale, count=W, name="ledw", init=ledger_init),
            sk.WorkerSpec(bump_op, rate=RATE_B, count=2, name="bump", init=bump_init),
            sk.WorkerSpec(reader_op, rate=RATE_R, count=3, name="rdr", init=reader_init),
            sk.WorkerSpec(probe_op, rate=RATE_P, count=1, name="probe")]


def recover(rec, kind):
    """start server after it died; measure recovery; returns client."""
    if kind == "term":
        try:
            srv.wait(300)
        except subprocess.TimeoutExpired:
            srv.kill9()
    t_down = time.time()
    start_server()
    t_ping = time.time()
    c = rd()
    res = sk.query(c, LED, "MATCH (n:L) RETURN count(n)", ro=True)
    t_q = time.time()
    rec["ping_s"] = round(t_ping - t_down, 3)
    rec["first_query_s"] = round(t_q - t_down, 3)
    rec["kill_to_query_s"] = round(t_q - rec["tkill"], 3)
    meta["recoveries"].append(dict(rec))
    return c


def do_end(kind, rec):
    rec["tkill"] = time.time()
    if kind == "kill9":
        srv.kill9()
    elif kind == "term":
        srv.terminate()


def check_after_restart(ep, label, mode, tkill, rc):
    """ledger / counters / big / static verification against the epoch's ledger; rebases."""
    global BUMP_BASE
    window = LOSS_WINDOW if mode == "everysec" else 0
    base = json.loads(Path(f"{out}/base-{ep - 1}.json").read_text())
    en, ee, tlast, pend, bump_t, bump_pend = replay(ep, base)
    T = [time.time()]
    onodes, oedges, info = observe_led(rc)
    T.append(time.time())
    viol, tol, det = compare_led(en, ee, tlast, pend, onodes, oedges, info, tkill, window)
    checks.add(f"{label}: led graph == ledger", viol == 0,
               f"nodes={len(onodes)} edges={len(oedges)} tolerated-in-window={tol}; " + "; ".join(det),
               "critical")
    T.append(time.time())
    okb, detb, cs = observe_big(rc, TRUTH, 100 + ep)
    T.append(time.time())
    checks.add(f"{label}: big graph aggregates/samples == ground truth", okb, detb, "critical")
    acked = len(bump_t)
    lost_allow = sum(1 for x in bump_t if x > tkill - window) if window else 0
    delta = cs - BUMP_BASE
    cnt_ok = (acked - lost_allow) <= delta <= acked + bump_pend
    checks.add(f"{label}: cnt counter in [acked-window_loss, acked+inflight]",
               cnt_ok, f"delta={delta} acked={acked} pend={bump_pend} lost_allow={lost_allow}", "critical")
    sd = static_digest(rc)
    T.append(time.time())
    meta["notes"].append(f"{label} verify breakdown s: led={T[1]-T[0]:.1f} big={T[3]-T[2]:.1f} static={T[4]-T[3]:.1f}")
    checks.add(f"{label}: static graphs digest unchanged", sd == SD0,
               "; ".join(g for g in STATIC if sd[g] != SD0[g]), "critical")
    gl = sorted(x.decode() if isinstance(x, bytes) else x for x in rc.execute_command("GRAPH.LIST"))
    checks.add(f"{label}: GRAPH.LIST unchanged", gl == GL0, f"{gl} != {GL0}", "critical")
    ix = all(wait_indexes(g, 600) for g in (BIG, LED, "multi0"))
    plan = " ".join(str(x) for x in rc.execute_command("GRAPH.EXPLAIN", BIG, "MATCH (n:N {id:5}) RETURN n"))
    checks.add(f"{label}: indexes OPERATIONAL and used", ix and "Index" in plan, f"ix={ix} plan={plan[:100]}", "major")
    tolerated_total = tol
    # rebase
    write_base(ep, onodes, oedges, cs)
    BUMP_BASE = cs
    return viol, tol, delta - acked


def run_segment(label, mode, endkind, bgsave=None, bgrewrite=None):
    """one load segment ending in `endkind` action while the workers are running."""
    global BASEF, SEGDIR, EP, EPOCH
    EPOCH += 1
    EP = EPOCH
    BASEF = f"{out}/base-{EP - 1}.json"
    SEGDIR = str(out)
    tag = f"e{EP}_"
    rec = {"label": label, "mode": mode, "kind": endkind}
    bgrec = {}
    thr = None
    if bgsave is not None:
        thr = bg_thread("BGSAVE", bgsave * SEG, bgrec, "rdb_bgsave_in_progress")
    if bgrewrite is not None:
        thr = bg_thread("BGREWRITEAOF", bgrewrite * SEG, bgrec, "aof_rewrite_in_progress")
    timer = None
    if endkind in ("kill9", "term"):
        timer = threading.Timer(SEG - 0.3, do_end, args=(endkind, rec))
        timer.start()
    mon.mark(f"soak-{label}")
    sk.run_load(srv, out, make_specs(), SEG, seed=EP, tag=tag)
    if timer:
        timer.join(30)
    if thr:
        thr.join(1200)
    if bgrec.get("start"):
        ws = window_stats(tag, bgrec["start"], bgrec.get("end", bgrec["start"] + 1), label)
        ws["bg"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in bgrec.items() if k not in ("start", "end")}
        meta["windows"].append(ws)
        judge_window(ws, label, bgrec)
    elif bgsave is not None or bgrewrite is not None:
        checks.add(f"{label}: background persistence command accepted", False, str(bgrec), "major")
    return rec, bgrec


def judge_window(ws, label, bgrec):
    if bgrec.get("interrupted"):
        meta["notes"].append(f"{label}: server killed while background save/rewrite was running")
        return
    worst = None
    for op, d in ws["ops"].items():
        if d["in_p99_ms"] and d["out_p99_ms"]:
            ratio = d["in_p99_ms"] / max(d["out_p99_ms"], 0.05)
            if worst is None or d["in_p99_ms"] > worst[1]:
                worst = (op, d["in_p99_ms"], d["out_p99_ms"], ratio)
    bad = worst and worst[3] > 10 and worst[1] > 250
    checks.add(f"{label}: p99 during background save/rewrite not >10x and >250ms worse", not bad,
               f"worst {worst[0] + ' %.1f vs %.1f ms' % (worst[1], worst[2]) if worst else 'n/a'}, "
               f"cmd stall {bgrec.get('cmd_s', 0) * 1000:.0f} ms, total {bgrec.get('total_s', 0):.1f}s; {worst}",
               "major")
    checks.add(f"{label}: background save/rewrite finished ok", bool(bgrec.get("status_ok")), str(bgrec.get("status")), "critical")


# ----------------------------------------------------------------- AOF phases
PLAN = [("A1-strict-kill9", "strict", "kill9", {}),
        ("A2-strict-kill9+bgsave", "strict", "kill9", {"bgsave": 0.15}),
        ("A3-strict-sigterm", "strict", "term", {}),
        ("A4-strict-kill9+bgrewriteaof", "strict", "kill9", {"bgrewrite": 0.15}),
        ("B1-everysec-kill9", "everysec", "kill9", {}),
        ("B2-everysec-kill9+bgsave", "everysec", "kill9", {"bgsave": 0.15})]
cur_mode = "strict"
deadline_all = time.time() + 3 * a.duration + 1800
for label, mode, endkind, kw in PLAN:
    if time.time() > deadline_all:
        checks.add("segments completed within time budget", False, label, "major")
        break
    if mode != cur_mode:
        c0 = rd()
        c0.execute_command("CONFIG", "SET", "appendfsync", mode)
        srv.extra = EXTRA_EVERYSEC if mode == "everysec" else EXTRA_STRICT
        cur_mode = mode
    rec, bgrec = run_segment(label, mode, endkind, **kw)
    mon.mark(f"recover-{label}")
    rc = recover(rec, endkind)
    mon.mark(f"verify-{label}")
    tv = time.time()
    viol, tol, d = check_after_restart(EP, label, mode, rec["tkill"], rc)
    rec["verify_s"] = round(time.time() - tv, 1)
    rec["lost_in_window"] = tol
    rec["disk_mb"] = round(disk_mb())
    meta["recoveries"][-1].update({"verify_s": rec["verify_s"], "lost_in_window": tol, "disk_mb": rec["disk_mb"]})
    checks.add(f"{label}: recovery (kill->first query) < 180s", rec["kill_to_query_s"] < 180,
               f"{rec['kill_to_query_s']}s", "major")
    # no stale ledger files kept when all is well
    if viol == 0:
        for f in list(out.glob(f"led-{EP}-*.jsonl")) + list(out.glob(f"bump-{EP}-*.jsonl")):
            f.unlink()
        old = out / f"base-{EP - 2}.json"
        if old.exists():
            old.unlink()

# ----------------------------------------------------------------- RDB phases
# transition: write a fresh RDB from the AOF-mode state, restart in RDB-only mode, state must match
mon.mark("rdb-transition")
rc = rd()
n0, e0, i0 = observe_led(rc)
sd_pre = static_digest(rc)
_, _, cs_pre = observe_big(rc, TRUTH, 7)
t = time.time()
rc.execute_command("SAVE")
meta["save_s_full_graph"] = round(time.time() - t, 2)
rdb_path = out / f"{srv.name}-{srv.port}.rdb"
meta["rdb_mb"] = round(rdb_path.stat().st_size / 2**20, 1)
srv.stop()
srv.extra = EXTRA_RDB
t = time.time()
start_server()
meta["rdb_only_restart_s"] = round(time.time() - t, 2)
rc = rd()
n1, e1, i1 = observe_led(rc)
checks.add("RDB-only restart after SAVE: led graph identical", eq(n0, n1) and eq(e0, e1), f"{len(n0)} vs {len(n1)} nodes", "critical")
ok, det, cs1 = observe_big(rc, TRUTH, 7)
checks.add("RDB-only restart after SAVE: big graph == ground truth and cnt identical", ok and cs1 == cs_pre, f"{det} cnt {cs1} vs {cs_pre}", "critical")
checks.add("RDB-only restart after SAVE: static digest identical", static_digest(rc) == SD0, "", "critical")
write_base(EPOCH, n1, e1, cs1)
BUMP_BASE = cs1
EPOCH += 0


def rdb_snapshot_state():
    c = rd()
    nn, ee_, ii = observe_led(c)
    _, _, cs_ = observe_big(c, TRUTH, 9)
    return {"nodes": nn, "edges": ee_, "cnt": cs_, "static": static_digest(c)}


def rdb_equal(s1, s2):
    return eq(s1["nodes"], s2["nodes"]) and eq(s1["edges"], s2["edges"]) and s1["cnt"] == s2["cnt"] and s1["static"] == s2["static"]


# R1: BGSAVE under write load, quiescent SAVE at end, kill9, restored == pre-kill
label = "R1-rdb-bgsave-under-load"
rec, bgrec = run_segment(label, "rdb", "none", bgsave=0.4)
mon.mark(f"verify-{label}")
# ledger check live (no kill): observed must equal ledger exactly
rcl = rd()
base = json.loads(Path(f"{out}/base-{EP - 1}.json").read_text())
en, ee, tlast, pend, bump_t, bump_pend = replay(EP, base)
on, oe, info = observe_led(rcl)
viol, tol, det = compare_led(en, ee, tlast, pend, on, oe, info, time.time(), 0)
checks.add(f"{label}: live led graph == ledger after load", viol == 0, "; ".join(det), "critical")
rcl.execute_command("SAVE")
snap = rdb_snapshot_state()
rdb_mtime = rdb_path.stat().st_mtime
rec["tkill"] = time.time()
srv.kill9()
rec["label"] = label
recover(rec, "kill9")
after = rdb_snapshot_state()
checks.add(f"{label}: restored graph == state at last SAVE", rdb_equal(snap, after), "", "critical")
checks.add(f"{label}: kill->first query < 180s", rec["kill_to_query_s"] < 180, f"{rec['kill_to_query_s']}s", "major")
write_base(EP, after["nodes"], after["edges"], after["cnt"])
BUMP_BASE = after["cnt"]

# R2: SAVE, then keep writing under load (+ BGSAVE-free), kill9 -> restored == state at SAVE
label = "R2-rdb-save-then-writes-lost"
rd().execute_command("SAVE")
snap2 = rdb_snapshot_state()
rdb_mtime2 = rdb_path.stat().st_mtime
rec, bgrec = run_segment(label, "rdb", "kill9")
recover(rec, "kill9")
after2 = rdb_snapshot_state()
checks.add(f"{label}: restored graph == state at last SAVE (post-SAVE writes gone)", rdb_equal(snap2, after2),
           f"nodes {len(snap2['nodes'])} vs {len(after2['nodes'])}, cnt {snap2['cnt']} vs {after2['cnt']}", "critical")
checks.add(f"{label}: kill->first query < 180s", rec["kill_to_query_s"] < 180, f"{rec['kill_to_query_s']}s", "major")
write_base(EP, after2["nodes"], after2["edges"], after2["cnt"])
BUMP_BASE = after2["cnt"]
meta["recoveries"][-1]["label"] = label

# ----------------------------------------------------------------- DEBUG RELOAD, SHUTDOWN SAVE
mon.mark("debug-reload")
rc = rd()
pre = rdb_snapshot_state()
t = time.time()
try:
    rc.execute_command("DEBUG", "RELOAD")
    meta["debug_reload_s"] = round(time.time() - t, 2)
    post = rdb_snapshot_state()
    checks.add("DEBUG RELOAD: graph state identical", rdb_equal(pre, post), "", "critical")
except Exception as e:
    meta["debug_reload_s"] = None
    checks.add("DEBUG RELOAD supported", False, str(e)[:200], "minor")
mon.mark("shutdown-save")
t = time.time()
try:
    rc.execute_command("SHUTDOWN", "SAVE")
except Exception:
    pass
try:
    srv.wait(600)
except subprocess.TimeoutExpired:
    srv.kill9()
meta["shutdown_save_s"] = round(time.time() - t, 2)
t = time.time()
start_server()
meta["restart_after_shutdown_save_s"] = round(time.time() - t, 2)
post = rdb_snapshot_state()
checks.add("SHUTDOWN SAVE + restart: graph state identical", rdb_equal(pre, post), "", "critical")

# ----------------------------------------------------------------- end
checks.add("server alive at end", srv.alive(), "", "critical")
crash = srv.crashed()
checks.add("no crash/panic markers in server log", not crash, str(crash[:3]), "critical")
viols = []
for f in out.glob("viol-*.txt"):
    viols += f.read_text().splitlines()
checks.add("readers saw no wrong results (degree/props of big graph)", not viols, "; ".join(viols[:3]), "critical")
meta["disk_mb_end"] = round(disk_mb())
mon.stop()
s = sk.analyze(out, srv)
meta["rdb_mb_end"] = round(rdb_path.stat().st_size / 2**20, 1) if rdb_path.exists() else None
s["meta"] = meta
(out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
srv.stop()
sk.print_summary(s)
print("meta:", json.dumps({k: v for k, v in meta.items() if k not in ("windows",)}, default=str, indent=1))
for w_ in meta["windows"]:
    print("window", w_["label"], w_["window_s"], "s", w_["bg"])
    for op, d in w_["ops"].items():
        print("   ", op, d)
