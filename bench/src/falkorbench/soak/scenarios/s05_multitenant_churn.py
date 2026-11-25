"""s05_multitenant_churn — multi-tenant SaaS: thousands of small per-tenant graphs with churn.

Tenants = graphs (100..20k nodes, skewed), content fully determined by seed. Workers each own a
partition of tenants (registry hash reg:<wid> in plain Redis = Python-side ground truth) and run a
mix: tenant reads/writes, checksum verification, create+delete churn, GRAPH.COPY, RENAME, DEL /
GRAPH.DELETE, GRAPH.LIST, GRAPH.MEMORY, EXPLAIN/PROFILE, CONFIG GET, EXISTS/TYPE/DBSIZE.
Memory is sampled after every churn cycle; at the end everything is deleted and residual is
compared to a fresh baseline.
"""
import argparse
import multiprocessing as mp
import random
import time
from pathlib import Path

import psutil  # noqa: E402
import redis  # noqa: E402
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

NT = max(30, int(3000 * a.scale))      # live tenants
W = 6                                   # tenant-manager workers (each owns NT/W tenants)
CYCLES = 6                              # churn cycles (memory sampled after each)
RATE = 60                               # ops/s per worker (open loop)
DEG = 3


# ---------------------------------------------------------------- tenant model
def t_size(seed):
    u = random.Random(seed).random()
    return int(100 * 200 ** (u ** 3))          # 100 .. 20000, heavily skewed to small


def t_params(seed):
    return t_size(seed), seed % 997 + 1, 1 + seed % 13   # n, value multiplier, edge step


def t_truth(seed):
    n, m, step = t_params(seed)
    vs = [(i * m) % 1000 for i in range(n)]
    esum = 0
    for i in range(n):
        for k in range(1, DEG + 1):
            esum += vs[i] + (i + k * step) % n
    return n, n * DEG, sum(vs), esum


Q_NODES = "UNWIND range(0, $n - 1) AS i CREATE (:T {id: i, v: (i * $m) % 1000, s: 'x' + toString(i)})"
Q_EDGES = ("MATCH (a:T) WITH a ORDER BY a.id WITH collect(a) AS ns UNWIND range(0, $n - 1) AS i "
           "UNWIND range(1, $d) AS k WITH ns[i] AS a, ns[(i + k * $step) % $n] AS b "
           "CREATE (a)-[:E]->(b)")
Q_SUM = "MATCH (a:T)-[:E]->(b:T) RETURN count(*), sum(a.v + b.id)"
Q_NSUM = "MATCH (a:T) RETURN count(a), sum(a.v)"


def create_tenant(r, name, seed):
    n, m, step = t_params(seed)
    sk.query(r, name, Q_NODES, {"n": n, "m": m})
    sk.query(r, name, Q_EDGES, {"d": DEG, "step": step, "n": n})


def check_tenant(r, name, seed):
    """None if the graph matches its seed, else a description."""
    n, e, vsum, esum = t_truth(seed)
    try:
        c, s = [sk.cval(x) for x in sk.rows(sk.query(r, name, Q_NSUM, ro=True))[0]]
        c2, s2 = [sk.cval(x) for x in sk.rows(sk.query(r, name, Q_SUM, ro=True))[0]]
    except Exception as ex:
        return f"{name} seed={seed}: query failed {ex}"
    if (c, int(s or 0), c2, int(s2 or 0)) != (n, vsum, e, esum):
        return f"{name} seed={seed}: got {(c, s, c2, s2)} want {(n, vsum, e, esum)}"
    return None


# ---------------------------------------------------------------- ingest (parallel, own processes)
def _ingest(args):
    port, owner, count = args
    r = sk.connect(port)
    ctr = int(r.get(f"ctr:{owner}") or 0)
    for _ in range(count):
        ctr += 1
        seed = owner * 1_000_000 + ctr
        name = f"t{seed}"
        create_tenant(r, name, seed)
        r.hset(f"reg:{owner}", name, seed)
        r.incr("cnt:create")
    r.set(f"ctr:{owner}", ctr)
    return owner


# ---------------------------------------------------------------- workers
def w_init(ctx):
    ctx.state["own"] = {k.decode(): int(v) for k, v in ctx.r.hgetall(f"reg:{ctx.wid}").items()}
    ctx.state["ctr"] = int(ctx.r.get(f"ctr:{ctx.wid}") or 0)
    ctx.state["copy_ok"] = ctx.r.get("cfg:copy_ok") == b"1"


def bad(ctx, msg):
    ctx.r.rpush("soak:bad", msg[:500])


def new_name(ctx, prefix):
    ctx.state["ctr"] += 1
    ctx.r.set(f"ctr:{ctx.wid}", ctx.state["ctr"])
    return ctx.wid * 1_000_000 + ctx.state["ctr"]


def drop(ctx, name):
    st = ctx.state
    st["own"].pop(name, None)
    ctx.r.hdel(f"reg:{ctx.wid}", name)
    ctx.r.sadd("soak:gone", name)
    ctx.r.incr("cnt:delete")


def pick(ctx):
    own = ctx.state["own"]
    if not own:
        return None, None
    name = ctx.rng.choice(list(own))
    return name, own[name]


def op(ctx):
    rng, r, st = ctx.rng, ctx.r, ctx.state
    x = rng.random()
    name, seed = pick(ctx)
    if name is None:
        return "idle"
    if x < 0.50:
        n = t_size(seed)
        i = rng.randrange(n)
        if rng.random() < 0.7:
            res = sk.query(r, name, "MATCH (a:T {id: $i})-[:E]->(b) RETURN b.id ORDER BY b.id", {"i": i}, ro=True)
            m, step = t_params(seed)[1:]
            want = sorted((i + k * step) % n for k in range(1, DEG + 1))
            got = sorted(sk.cval(c[0]) for c in sk.rows(res))
            if got != want:
                bad(ctx, f"read {name} id={i} got {got} want {want}")
            return "tenant_read_1hop"
        sk.query(r, name, "MATCH (a:T {id: $i})-[:E*2]->(b) RETURN count(DISTINCT b)", {"i": i}, ro=True)
        return "tenant_read_2hop"
    if x < 0.62:
        i = rng.randrange(t_size(seed))
        sk.query(r, name, "MATCH (a:T {id: $i}) SET a.w = coalesce(a.w, 0) + 1", {"i": i})
        return "tenant_write_set"
    if x < 0.66:
        sk.query(r, name, "CREATE (:Evt {k: $k})", {"k": rng.randrange(1000)})
        return "tenant_write_create"
    if x < 0.70:
        e = check_tenant(r, name, seed)
        if e:
            bad(ctx, "checksum " + e)
        return "verify_checksum"
    if x < 0.74:
        r.execute_command("GRAPH.EXPLAIN", name, "MATCH (a:T {id: 1})-[:E]->(b) RETURN b")
        return "explain"
    if x < 0.77:
        r.execute_command("GRAPH.PROFILE", name, "MATCH (a:T)-[:E]->(b) RETURN count(b)")
        return "profile"
    if x < 0.79:
        r.execute_command("GRAPH.CONFIG", "GET", "*")
        return "config_get"
    if x < 0.82:
        res = r.execute_command("GRAPH.MEMORY", "USAGE", name)
        d = res if isinstance(res, dict) else dict(zip(res[::2], res[1::2]))
        if int(d.get(b"total_graph_sz_mb", 0)) < 0:
            bad(ctx, f"memory usage weird {name}")
        return "memory_usage"
    if x < 0.83:
        got = r.execute_command("GRAPH.LIST")
        names = {g.decode() for g in got}
        if not set(st["own"]) <= names:
            bad(ctx, f"LIST missing own tenants {sorted(set(st['own']) - names)[:3]}")
        return "graph_list"
    if x < 0.87:
        t = r.execute_command("TYPE", name)
        ex = r.exists(name)
        if ex != 1 or t != b"graphdata" and t != b"graphdata\x00":
            bad(ctx, f"TYPE/EXISTS {name}: {t} {ex}")
        r.dbsize()
        return "keyspace_info"
    if x < 0.93:  # churn: create a tenant, delete another (live count constant)
        s2 = ctx.wid * 1_000_000 + new_name(ctx, "t") % 1_000_000
        nm = f"t{s2}"
        create_tenant(r, nm, s2)
        st["own"][nm] = s2
        r.hset(f"reg:{ctx.wid}", nm, s2)
        r.incr("cnt:create")
        victim, vseed = pick(ctx)
        if victim == nm and len(st["own"]) > 1:
            victim, vseed = name, seed
        if rng.random() < 0.7:
            r.execute_command("GRAPH.DELETE", victim)
        else:
            r.delete(victim)
        drop(ctx, victim)
        if r.exists(victim):
            bad(ctx, f"deleted graph still exists {victim}")
        try:
            sk.query(r, victim, "MATCH (n) RETURN count(n)", ro=True)
            bad(ctx, f"deleted graph still queryable {victim}")
        except redis.exceptions.ResponseError:
            pass
        return "churn_create_delete"
    if x < 0.96 and st["copy_ok"]:
        s2 = new_name(ctx, "c")
        dst = f"c{ctx.wid}x{s2}"
        try:
            r.execute_command("GRAPH.COPY", name, dst)
        except redis.exceptions.ResponseError as ex:
            if "could not fork" in str(ex):   # C forks per COPY; transient fork failure
                return "graph_copy_fork_fail"
            raise
        e = check_tenant(r, dst, seed)
        if e:
            bad(ctx, "copy " + e)
        st["own"][dst] = seed
        r.hset(f"reg:{ctx.wid}", dst, seed)
        r.incr("cnt:create")
        # keep count constant: drop the copy's oldest sibling-or-random victim
        victim, _ = pick(ctx)
        r.execute_command("GRAPH.DELETE", victim)
        drop(ctx, victim)
        return "graph_copy"
    # rename
    s2 = new_name(ctx, "r")
    dst = f"r{ctx.wid}x{s2}"
    r.rename(name, dst)
    st["own"].pop(name)
    st["own"][dst] = seed
    r.hdel(f"reg:{ctx.wid}", name)
    r.hset(f"reg:{ctx.wid}", dst, seed)
    r.sadd("soak:gone", name)
    if r.exists(name):
        bad(ctx, f"renamed-from still exists {name}")
    e = check_tenant(r, dst, seed)
    if e:
        bad(ctx, "rename " + e)
    return "rename"


# ---------------------------------------------------------------- main
srv = sk.Server(a.module, a.port, out)
mon = sk.Monitor(srv, out)
mon.start()
checks = sk.Checks(out)
r = srv.client()
proc = psutil.Process(srv.pid)


def mem_sample(label, settle=3.0):
    time.sleep(settle)
    um = r.info("memory")["used_memory"] / 2**20
    rss = proc.memory_info().rss / 2**20
    return {"label": label, "rss_mb": round(rss, 1), "used_mb": round(um, 1),
            "live": sum(r.hlen(f"reg:{o}") for o in range(W)),
            "creates": int(r.get("cnt:create") or 0), "deletes": int(r.get("cnt:delete") or 0)}


def live_truth():
    t = {}
    for o in range(W):
        for k, v in r.hgetall(f"reg:{o}").items():
            t[k.decode()] = int(v)
    return t


mon.mark("baseline")
base = mem_sample("fresh-baseline", 2)

# feature detection (C may lack GRAPH.COPY)
copy_ok = True
try:
    create_tenant(r, "probe_src", 1)
    r.execute_command("GRAPH.COPY", "probe_src", "probe_dst")
    e = check_tenant(r, "probe_dst", 1)
    checks.add("GRAPH.COPY copies contents", e is None, str(e), "critical")
except redis.exceptions.ResponseError as ex:
    copy_ok = False
    checks.add("GRAPH.COPY supported", False, str(ex), "minor")
r.delete("probe_src", "probe_dst")
r.set("cfg:copy_ok", b"1" if copy_ok else b"0")

# ---------------- ingest
mon.mark("ingest")
t0 = time.time()
per = [NT // W + (1 if i < NT % W else 0) for i in range(W)]
with mp.get_context("fork").Pool(W) as pool:
    pool.map(_ingest, [(a.port, o, per[o]) for o in range(W)])
ingest_s = time.time() - t0
sizes = sorted(t_size(s) for s in live_truth().values())
tot_nodes = sum(sizes)
after_ingest = mem_sample("after-ingest")
live = live_truth()
checks.add("ingest: GRAPH.LIST == live set", {g.decode() for g in r.execute_command("GRAPH.LIST")} == set(live),
           f"{len(live)} expected", "critical")

# ---------------- soak: churn cycles with memory sampled after each
specs = lambda: [sk.WorkerSpec(op, rate=RATE, count=W, name="tenant", init=w_init)]
cyc = []
per_cycle = a.duration / CYCLES
for c in range(CYCLES):
    mon.mark("soak")
    sk.run_load(srv, out, specs(), per_cycle, seed=c + 1, tag=f"c{c}_")
    mon.mark("quiesce")
    s = mem_sample(f"cycle-{c}")
    cyc.append(s)
    print("cycle", s, flush=True)
    if not srv.alive():
        break

# ---------------- verify
mon.mark("verify")
live = live_truth()
listed = {g.decode() for g in r.execute_command("GRAPH.LIST")}
checks.add("GRAPH.LIST == Python live set", listed == set(live),
           f"extra={sorted(listed - set(live))[:5]} missing={sorted(set(live) - listed)[:5]}", "critical")
gone = {g.decode() for g in r.smembers("soak:gone")}
still = [g for g in gone if r.exists(g) and g not in live]
checks.add("deleted/renamed-away graphs are gone", not still, str(still[:5]), "critical")
checks.add("deleted names absent from GRAPH.LIST", not (gone - set(live)) & listed, "", "critical")
nkeys = r.dbsize()
other = len(r.keys("*")) - len(listed)
checks.add("DBSIZE == live graphs + registry keys", nkeys == len(live) + other, f"{nkeys} vs {len(live)}+{other}",
           "major")
badl = [x.decode() for x in r.lrange("soak:bad", 0, 20)]
checks.add("no in-flight correctness errors", not badl, "; ".join(badl[:5]), "critical")
names = list(live)
random.Random(7).shuffle(names)
t0 = time.time()
vbad, vn = [], 0
for nm in names:
    if time.time() - t0 > 240:
        break
    e = check_tenant(r, nm, live[nm])
    vn += 1
    if e:
        vbad.append(e)
checks.add("final checksum on every live tenant", not vbad,
           f"{vn}/{len(names)} tenants; " + "; ".join(vbad[:3]), "critical")

# plateau analysis (cycle samples)
half = cyc[len(cyc) // 2:]
if len(cyc) >= 4:
    d_um = half[-1]["used_mb"] - half[0]["used_mb"]
    d_rss = half[-1]["rss_mb"] - half[0]["rss_mb"]
    ref_um, ref_rss = half[0]["used_mb"], half[0]["rss_mb"]
    checks.add("steady state: used_memory plateau (2nd half growth <10%+20MB)", d_um < 0.1 * ref_um + 20,
               f"+{d_um:.1f} MB over 2nd half from {ref_um:.0f}", "major")
    checks.add("steady state: RSS plateau (2nd half growth <10%+50MB)", d_rss < 0.1 * ref_rss + 50,
               f"+{d_rss:.1f} MB over 2nd half from {ref_rss:.0f}", "major")
checks.add("live graph count constant through soak", all(c["live"] == NT for c in cyc), str([c["live"] for c in cyc]),
           "major")

# ---------------- delete everything, residual vs baseline
mon.mark("teardown")
mem_before_del = mem_sample("before-delete", 1)
t0 = time.time()
for nm in list(live):
    r.execute_command("GRAPH.DELETE", nm)
delete_all_s = time.time() - t0
checks.add("GRAPH.LIST empty after deleting all", r.execute_command("GRAPH.LIST") == [], "", "critical")
res = []
for i in range(6):   # allow async frees to land
    res.append(mem_sample(f"residual-{i}", 5 if i else 2))
fin = res[-1]
creates, deletes = fin["creates"], fin["deletes"]
soak_creates = cyc[-1]["creates"] - after_ingest["creates"] if cyc else 0
grow_um = (cyc[-1]["used_mb"] - cyc[0]["used_mb"]) if cyc else 0
grow_rss = (cyc[-1]["rss_mb"] - cyc[0]["rss_mb"]) if cyc else 0
soak_cycles_after_first = (cyc[-1]["creates"] - cyc[0]["creates"]) if cyc else 0
res_um = fin["used_mb"] - base["used_mb"]
res_rss = fin["rss_mb"] - base["rss_mb"]
checks.add("used_memory returns to baseline after deleting all (residual <50MB)", res_um < 50,
           f"residual used_memory +{res_um:.1f} MB (base {base['used_mb']})", "major")
checks.add("RSS returns near baseline after deleting all (residual <10% of peak+100MB)",
           res_rss < 0.1 * max(c["rss_mb"] for c in cyc + [after_ingest]) + 100,
           f"residual RSS +{res_rss:.1f} MB (base {base['rss_mb']}, peak {max(c['rss_mb'] for c in cyc + [after_ingest])})",
           "minor")
peak_rss = max(c["rss_mb"] for c in cyc + [after_ingest])
# allocator-retention vs leak: refill the same number of graphs and see whether RSS exceeds the old peak
for o in range(W):
    r.delete(f"reg:{o}")
r.delete("soak:gone")
mon.mark("refill")
t0 = time.time()
with mp.get_context("fork").Pool(W) as pool:
    pool.map(_ingest, [(a.port, o, per[o]) for o in range(W)])
refill_s = time.time() - t0
refill = mem_sample("after-refill")
checks.add("refill after delete-all: RSS not above 1.15x previous peak (memory reused, no leak)",
           refill["rss_mb"] < 1.15 * peak_rss + 30,
           f"refill RSS {refill['rss_mb']} vs old peak {peak_rss}; baseline {base['rss_mb']}", "major")
refill_bad = [e for nm, sd in list(live_truth().items())[:200] if (e := check_tenant(r, nm, sd))]
checks.add("refilled tenants verify (200 sampled)", not refill_bad, "; ".join(refill_bad[:3]), "critical")
for nm in list(live_truth()):
    r.execute_command("GRAPH.DELETE", nm)
res2 = [mem_sample(f"residual2-{i}", 3) for i in range(2)]
checks.add("server alive at end", srv.alive(), "", "critical")
mon.stop()
s = sk.analyze(out, srv)
s["meta"] = {"scenario": "s05_multitenant_churn", "tenants": NT, "total_nodes": tot_nodes,
             "total_edges": tot_nodes * DEG, "size_p50": sizes[len(sizes) // 2], "size_max": sizes[-1],
             "ingest_s": ingest_s, "copy_supported": copy_ok, "duration": a.duration, "scale": a.scale,
             "mem_baseline": base, "mem_after_ingest": after_ingest, "mem_cycles": cyc,
             "mem_before_delete": mem_before_del, "mem_residual": res, "mem_refill": refill, "mem_residual2": res2, "refill_s": refill_s, "delete_all_s": delete_all_s,
             "total_creates": creates, "total_deletes": deletes,
             "used_growth_mb_per_1000_cycles": (grow_um / soak_cycles_after_first * 1000) if soak_cycles_after_first else None,
             "rss_growth_mb_per_1000_cycles": (grow_rss / soak_cycles_after_first * 1000) if soak_cycles_after_first else None,
             "residual_used_mb": res_um, "residual_rss_mb": res_rss}
(out / "summary.json").write_text(sk.json.dumps(s, indent=1, default=str))
srv.stop()
sk.print_summary(s)
print(sk.json.dumps(s["meta"], indent=1, default=str))
