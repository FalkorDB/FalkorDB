"""Independent check 4: seeds (D1), relax (D2), every published metric against my own full
customization, scoped-recompute pops/work against my own implementation, S2's new arcs against
the closure of (old arcs ∪ inserted pairs), ids, P, coverage, closure, Down mirror, and CCH
distances against my own Dijkstra on my own rebuilt graphs. Does not import walkthrough.py."""
import heapq
import json
import os
import sys
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "..", "example.json")))
STEP = {s["id"]: s for s in J["steps"]}
GI = json.load(open(os.path.join(HERE, "graphs_independent.json")))
INF = float("inf")
problems = []


def bad(msg):
    problems.append(msg)
    print("MISMATCH:", msg)


def W(v):
    out = {}
    for t, s, d, w in GI["V"][str(v)]:
        out[(s, d)] = min(out.get((s, d), INF), w)
    return out


def nz(x):
    return INF if x is None else x


def fill(n, pairs):
    adj = {r: set() for r in range(1, n + 1)}
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)
    up = {}
    for r in range(1, n + 1):
        hi = sorted(x for x in adj[r] if x > r)
        up[r] = hi
        for a, b in combinations(hi, 2):
            adj[a].add(b)
            adj[b].add(a)
    return {(r, h) for r in up for h in up[r]}


class H:
    def __init__(self, arcs, n, node_of):
        self.arcs = dict(arcs)          # (lo, hi) -> id
        self.n = n
        self.node_of = node_of
        self.up = {r: [] for r in range(1, n + 1)}
        self.dn = {r: [] for r in range(1, n + 1)}
        for (lo, hi), i in sorted(self.arcs.items()):
            self.up[lo].append((hi, i))
            self.dn[hi].append((lo, i))


def from_json(v):
    h = J["hierarchies"][v]
    arcs = {(lo, hi): i for lo, hi, i, _, _ in h["arcs"]}
    node_of = {r: x for r, x in h["maps"]["node_of"]}
    hh = H(arcs, h["maps"]["n"], node_of)
    m = h["metric"]
    hh.metric = {i: (nz(m["up"][i]), nz(m["down"][i]), m["up_middle"][i], m["down_middle"][i])
                 for i in range(len(m["up"]))}
    hh.maps = h["maps"]
    return hh


def seeds(h, Wv):
    out = {}
    for (lo, hi), i in h.arcs.items():
        a, b = h.node_of.get(lo), h.node_of.get(hi)
        out[i] = (Wv.get((a, b), INF), Wv.get((b, a), INF))
    return out


def full(h, Wv, count=None):
    s = seeds(h, Wv)
    up = {i: s[i][0] for i in s}
    dn = {i: s[i][1] for i in s}
    mu, md = {i: None for i in s}, {i: None for i in s}
    tri = imp = 0
    for x in range(1, h.n + 1):
        for (y, iy), (z, iz) in combinations(h.up[x], 2):
            t = h.arcs[(y, z)]
            tri += 1
            cu, cd = dn[iy] + up[iz], dn[iz] + up[iy]
            hit = False
            if cu < up[t]:
                up[t], mu[t], hit = cu, x, True
            if cd < dn[t]:
                dn[t], md[t], hit = cd, x, True
            imp += hit
    if count is not None:
        count.update(tri=tri, improved=imp)
    return {i: (up[i], dn[i], mu[i], md[i]) for i in up}


def scoped(h, Wv, metric, front):
    """My recustomize_seeded: pop by (lo, hi), seed + apex merge, write on change, push the
    tops of the triangles this arc is a bottom of. Returns (metric, pops, changed, work)."""
    M = dict(metric)
    heap = sorted({(lo, hi, h.arcs[(lo, hi)]) for lo, hi in front})
    heapq.heapify(heap)
    queued = {i for _, _, i in heap}
    pops = changed = merge = 0
    pushes = len(heap)
    while heap:
        lo, hi, i = heapq.heappop(heap)
        pops += 1
        a, b = h.node_of.get(lo), h.node_of.get(hi)
        bu, bd = Wv.get((a, b), INF), Wv.get((b, a), INF)
        mu = md = None
        p = q = 0
        L, R = h.dn[lo], h.dn[hi]
        while p < len(L) and q < len(R):
            merge += 1
            if L[p][0] < R[q][0]:
                p += 1
            elif L[p][0] > R[q][0]:
                q += 1
            else:
                x, xl, xh = L[p][0], L[p][1], R[q][1]
                cu, cd = M[xl][1] + M[xh][0], M[xh][1] + M[xl][0]
                if cu < bu:
                    bu, mu = cu, x
                if cd < bd:
                    bd, md = cd, x
                p += 1
                q += 1
        old = M[i]
        M[i] = (bu, bd, mu, md)
        if (bu, bd) != old[:2]:
            changed += 1
            for w_, _ in h.up[lo]:
                if w_ != hi:
                    d = (min(hi, w_), max(hi, w_))
                    j = h.arcs[d]
                    if j not in queued:
                        queued.add(j)
                        heapq.heappush(heap, (d[0], d[1], j))
                        pushes += 1
    return M, pops, changed, merge + pushes + 2 * pops


def dijkstra(Wv, nodes):
    adj = {}
    for (u, v), w in Wv.items():
        adj.setdefault(u, []).append((v, w))
    out = {}
    for s in nodes:
        dist = {s: 0}
        pq = [(0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj.get(u, []):
                if d + w < dist.get(v, INF):
                    dist[v] = d + w
                    heapq.heappush(pq, (d + w, v))
        out[s] = dist
    return out


def cch(h, rank_of, s, t):
    if s == t:
        return 0
    if s not in rank_of or t not in rank_of:
        return INF

    def search(r, k):
        dist = {r: 0}
        pq = [(0, r)]
        while pq:
            d, x = heapq.heappop(pq)
            if d > dist[x]:
                continue
            for y, i in h.up[x]:
                nd = d + h.metric[i][k]
                if nd < dist.get(y, INF):
                    dist[y] = nd
                    heapq.heappush(pq, (nd, y))
        return dist
    f, b = search(rank_of[s], 0), search(rank_of[t], 1)
    return min((f[m] + b[m] for m in f if m in b), default=INF)


# ---------------- D1 seeds at v7 on the C5 topology
base = STEP["C5"]["data"]["base"]
Ap, Aj, Ax = base["Up"]["Ap"], base["Up"]["Aj"], base["Up"]["Ax"]
arcs7 = {}
for r in range(1, len(Ap)):
    for k in range(Ap[r - 1], Ap[r]):
        arcs7[(r, Aj[k])] = Ax[k]
node_of7 = {r: x for r, x in STEP["C5"]["data"]["node_of"]}
h7 = H(arcs7, len(Ap) - 1, node_of7)
s7 = seeds(h7, W(7))
d1 = STEP["D1"]["data"]
fin = sum(x != INF for p in s7.values() for x in p)
mismatch = [i for i in s7 if (nz(d1["seeds_up"][i]), nz(d1["seeds_down"][i])) != s7[i]]
print("D1: finite seeds", fin, "mismatching ids", mismatch)
if mismatch or fin != 56:
    bad("D1 seeds")
cnt = {}
m7 = full(h7, W(7), cnt)
mv7 = d1 = STEP["D2"]["data"]["metric_v7"]
j7 = {i: (nz(mv7["up"][i]), nz(mv7["down"][i]), mv7["up_middle"][i], mv7["down_middle"][i]) for i in m7}
print("D2: triangles", cnt["tri"], "improved", cnt["improved"], "json", len(STEP["D2"]["data"]["triangles"]),
      sum(1 for t in STEP["D2"]["data"]["triangles"] if t["up_won"] or t["down_won"]))
if j7 != m7:
    bad("D2 metric at v7")
print("D2 batches", STEP["D2"]["data"]["batches"])

# ---------------- D3 Seal at v8: scoped from the v7 metric, front = arc of pair (5,6)
ro7 = {x: r for r, x in node_of7.items() if x is not None}
fr = [tuple(sorted((ro7[5], ro7[6])))]
M8, pops, ch, work = scoped(h7, W(8), m7, fr)
f8 = full(h7, W(8))
print("D3: pops", pops, "changed", ch, "work", work, "equals full:", M8 == f8)
if M8 != f8 or pops != 6 or work != STEP["D3"]["data"]["flush"]["work"]["total"]:
    bad("D3 seal")
# adoption at v9: scoped from M8, front = arc of (9,10)
fr = [tuple(sorted((ro7[9], ro7[10])))]
M9, pops, ch, work = scoped(h7, W(9), M8, fr)
print("D3 adoption: pops", pops, "changed", ch, "work", work, "equals full:", M9 == full(h7, W(9)))
if work != STEP["D3"]["data"]["adoption"]["flush"]["work"]["total"]:
    bad("adoption work")

# ---------------- every published version
pubs = ["9", "10", "11", "12", "13", "14", "15", "16", "16 fold"]
gv = {"9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "16 fold": 16}
id_of = {}
prev = None
for v in pubs:
    h = from_json(v)
    Wv = W(gv[v])
    nodes = GI["nodes"][str(gv[v])]
    rank_of = {x: r for x, r in h.maps["rank_of"]}
    # metric = full customization (values and middles)
    f = full(h, Wv)
    diff = [i for i in f if f[i] != h.metric[i]]
    extra = [i for i in h.metric if i not in f and h.metric[i] != (INF, INF, None, None)]
    if diff or extra:
        bad(f"v{v}: metric differs from my full customization at ids {diff[:10]} extra {extra}")
    # closure, coverage, P, ids
    clo = [(r, a, b) for r in h.up for (a, _), (b, _) in combinations(h.up[r], 2) if (a, b) not in h.arcs]
    cov = [(u, w_) for (u, w_) in Wv if u not in rank_of or w_ not in rank_of or
           tuple(sorted((rank_of[u], rank_of[w_]))) not in h.arcs]
    free = set(h.maps["free_ranks"])
    P = [r for r in range(1, h.n + 1) if (r in free) != (h.node_of.get(r) is None) or
         (h.node_of.get(r) is None and (h.up[r] or h.dn[r]))]
    for (lo, hi), i in h.arcs.items():
        if id_of.setdefault(i, (lo, hi)) != (lo, hi):
            bad(f"v{v}: id {i} reused")
    if len(set(h.arcs.values())) != len(h.arcs):
        bad(f"v{v}: duplicate ids")
    if clo or cov or P:
        bad(f"v{v}: closure {clo[:3]} coverage {cov[:3]} P {P}")
    # queries
    ref = dijkstra(Wv, nodes)
    qbad = [(s, t) for s in nodes for t in nodes if cch(h, rank_of, s, t) != ref[s].get(t, INF)]
    if qbad:
        bad(f"v{v}: {len(qbad)} query mismatches, e.g. {qbad[:5]}")
    # JSON distance tables
    if "distances" in J["hierarchies"][v]:
        dd = {(s, t): nz(x) for s, t, x in J["hierarchies"][v]["distances"]}
        if any(dd[(s, t)] != ref[s].get(t, INF) for s in nodes for t in nodes if s != t):
            bad(f"v{v}: distance table differs from Dijkstra")
    # Down mirror in the deltas
    dl = J["hierarchies"][v]["delta"]
    upd = {(e[0], e[1], e[2]) for p in dl["Up"]["pages"] if p["kind"] == "leaf" for e in p["entries"]} if dl["Up"] else set()
    dnd = {(e[1], e[0], e[2]) for p in dl["Down"]["pages"] if p["kind"] == "leaf" for e in p["entries"]} if dl["Down"] else set()
    if upd != dnd:
        bad(f"v{v}: Down.delta is not Up.delta transposed")
    print(f"v{v}: n={h.n} arcs={len(h.arcs)} cap={h.maps['arc_capacity']} delta={len(upd)} "
          f"metric ok={not diff} closure={len(clo)} cov={len(cov)} P={len(P)} queries={len(nodes)**2} bad={len(qbad)}")
    prev = h

# ---------------- S2 = closure of (old arcs ∪ inserted pairs) - old arcs, ids appended in (lo,hi) order
for v_old, v_new, sid in (("10", "11", "F2"), ("11", "12", "F3"), ("13", "14", "F5"), ("15", "16", "F6")):
    ho, hn = from_json(v_old), from_json(v_new)
    rec = STEP[sid]["data"].get("flush") or STEP[sid]["data"].get("repair") or STEP[sid]["data"]["v16"]["flush"]
    ins = [tuple(p) for p in rec["classify"]["s2_insert"]]
    closed = fill(hn.n, set(ho.arcs) | set(ins))
    newarcs = sorted(closed - set(ho.arcs))
    got = sorted(set(hn.arcs) - set(ho.arcs))
    cap = ho.maps["arc_capacity"]
    ids_ok = all(hn.arcs[a] == cap + k for k, a in enumerate(newarcs))
    print(f"{sid}: S2 new arcs mine {len(newarcs)} json {len(got)} equal={newarcs == got} ids appended from {cap}: {ids_ok}")
    if newarcs != got or not ids_ok or set(ho.arcs) - set(hn.arcs):
        bad(f"{sid}: S2 arcs")

# ---------------- F1..F5 scoped recompute: pops/changed/work with my implementation
for v_old, v_new, sid in (("9", "10", "F1"), ("10", "11", "F2"), ("11", "12", "F3"), ("12", "13", "F4")):
    ho, hn = from_json(v_old), from_json(v_new)
    rec = STEP[sid]["data"]["flush"]
    base_metric = dict(ho.metric)
    for i in hn.arcs.values():
        base_metric.setdefault(i, (INF, INF, None, None))
    front = [tuple(p[:2]) for p in rec["front"]]
    M, pops, ch, work = scoped(hn, W(int(v_new)), base_metric, front)
    s2w = rec["work"]["s2"]
    eq = all(M[i] == hn.metric[i] for i in hn.arcs.values())
    print(f"{sid}: pops {pops} changed {ch} work {work}+{s2w}={work + s2w} json {rec['work']['total']} eq {eq}")
    if not eq or work + s2w != rec["work"]["total"] or pops != rec["work"]["pops"]:
        bad(f"{sid} scoped")
# F5's writer attempt (scoped over v13 + S2), to check the 506
ho, hn = from_json("13"), from_json("14")
base_metric = dict(ho.metric)
for i in hn.arcs.values():
    base_metric.setdefault(i, (INF, INF, None, None))
rec = STEP["F5"]["data"]["repair"]
front = [tuple(p[:2]) for p in rec["front"]]
M, pops, ch, work = scoped(hn, W(14), base_metric, front)
print(f"F5 attempt: pops {pops} work {work}+{rec['s2']['work']}={work + rec['s2']['work']} json "
      f"{STEP['F5']['data']['attempt']['work']['total']}; scoped result equals full: "
      f"{all(M[i] == hn.metric[i] for i in hn.arcs.values())}; front {len(front)} vs n/8 = {hn.n / 8}")
# v15 commit and v16 commit
for v_old, v_new, rec in (("14", "15", None), ("15", "16", STEP["F6"]["data"]["v16"]["flush"])):
    ho, hn = from_json(v_old), from_json(v_new)
    print(f"v{v_new}: arcs {len(hn.arcs)} (old {len(ho.arcs)})")
print("problems:", len(problems))
sys.exit(1 if problems else 0)
