"""Independent check 3: B8 split, B11 postorder relabel, C1 counts, C2 Supergraph (allocation,
B' = (B \\ D) ∪ A), C3 etree and counts over B', C4 game and fill, C5 CSR ids, and the
noncontiguous-subtree claim. The elimination tree here comes from symbolic elimination
(filled graph, parent = lowest higher neighbour), not Liu's algorithm. Does not import
walkthrough.py."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "..", "example.json")))
STEP = {s["id"]: s for s in J["steps"]}
G = json.load(open(os.path.join(HERE, "graphs_independent.json")))
problems = []


def bad(msg):
    problems.append(msg)
    print("MISMATCH:", msg)


def fill(n, pairs):
    """Symbolic elimination in order 1..n. Returns up rows (sorted) of the filled graph."""
    adj = {r: set() for r in range(1, n + 1)}
    for a, b in pairs:
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    up = {}
    for r in range(1, n + 1):
        hi = sorted(x for x in adj[r] if x > r)
        up[r] = hi
        for i in range(len(hi)):
            for j in range(i + 1, len(hi)):
                adj[hi[i]].add(hi[j])
                adj[hi[j]].add(hi[i])
    return up


S = {tuple(x) for x in STEP["B1"]["data"]["S"]}

# ---- B8: separator ranks and the dropped entries
sp = STEP["B8"]["data"]
sep = sp["separators"][0]["separator"]
A0 = {k for k in S if k[0] < 16 and k[1] < 16}
dropped = sorted(k for k in A0 if k[0] in sep or k[1] in sep)
print("B8: separator", sep, "ranks", sp["separators"][0]["separator_ranks"], "dropped", len(dropped))
if len(dropped) != len(sp["removed"]):
    bad("B8 dropped count")

# ---- B11: postorder relabel of the ND order
rank_nd = dict(map(tuple, STEP["B11"]["data"]["rank_nd"]))
n_r = 20
Bnd = {(rank_nd[a], rank_nd[b]) for a, b in S}
up_nd = fill(n_r, Bnd)
par_nd = {r: (up_nd[r][0] if up_nd[r] else None) for r in range(1, n_r + 1)}
kids = {r: [] for r in range(1, n_r + 1)}
roots = []
for r in range(1, n_r + 1):
    (kids[par_nd[r]] if par_nd[r] else roots).append(r)
order = []


def post(r):
    for c in sorted(kids[r]):
        post(c)
    order.append(r)


for r in sorted(roots):
    post(r)
q = {old: new for new, old in enumerate(order, 1)}
moved = sum(1 for r in q if q[r] != r)
jq = dict(map(tuple, STEP["B11"]["data"]["q"]))
print("B11: postorder moves", moved, "ranks; roots", roots)
if jq != q:
    bad(f"B11 q differs: mine {q}")
rank_of = {x: q[r] for x, r in rank_nd.items()}
if {x: r for x, r in STEP["B11"]["data"]["rank_of"] if r is not None} != rank_of:
    bad("B11 rank_of differs")
# the relabel is a valid postorder: every subtree is a consecutive range
B = {(rank_of[a], rank_of[b]) for a, b in S}
upB = fill(n_r, B)
parB = {r: (upB[r][0] if upB[r] else None) for r in range(1, n_r + 1)}


def subtree(par, r):
    out = {r}
    changed = True
    while changed:
        changed = False
        for x, p in par.items():
            if p in out and x not in out:
                out.add(x)
                changed = True
    return out


nc_B = [r for r in range(1, n_r + 1) if (lambda s: max(s) - min(s) + 1 != len(s))(subtree(parB, r))]
print("B11: noncontiguous subtrees after relabel (expected none):", nc_B)
if nc_B:
    bad("B11 relabel not a postorder")
# also: does the relabelled order keep the elimination fill unchanged (postorder is fill-preserving)?
arcs_nd = sum(len(v) for v in up_nd.values())
arcs_B = sum(len(v) for v in upB.values())
print("C1: arcs under ND order", arcs_nd, "after relabel", arcs_B, "json", STEP["C1"]["data"]["arcs"])
if arcs_B != STEP["C1"]["data"]["arcs"] or len(B) != 54:
    bad("C1 arcs")
cc = {r: len(upB[r]) for r in range(1, n_r + 1)}
if cc != dict(map(tuple, STEP["C1"]["data"]["col_counts"])):
    bad("C1 column counts differ from the filled graph's row sizes")
if {r: p for r, p in STEP["C1"]["data"]["etree"]} != parB:
    bad("C1 etree differs")

# ---- C2: Supergraph at v7
W7 = {}
for t, s, d, w in G["V"]["7"]:
    W7[(s, d)] = min(W7.get((s, d), 1e18), w)
live7 = {(min(a, b), max(a, b)) for a, b in W7}
logpairs = [(1, 2), (0, 20), (3, 20), (12, 19)]
nbrs = lambda x: {b for a, b in W7 if a == x} | {a for a, b in W7 if b == x}
free = [1, 2]
n1 = n_r
ro = dict(rank_of)
alloc = []
E = sorted({x for p in logpairs if p in live7 for x in p})
for x in E:
    if x in ro:
        continue
    d = len(nbrs(x))
    if d <= 1 and free:
        r = free.pop(0)
        kind = "pool"
    else:
        n1 += 1
        r = n1
        kind = "top"
    ro[x] = r
    alloc.append({"node": x, "degree": d, "rank": r, "kind": kind})
print("C2: allocation", alloc)
if alloc != STEP["C2"]["data"]["allocation"]:
    bad("C2 allocation")
Bp = {(ro[a], ro[b]) for a, b in live7} | {(ro[b], ro[a]) for a, b in live7}
if Bp != {tuple(x) for x in STEP["C2"]["data"]["B_prime"]}:
    bad("C2 B' is not the live pattern at v7")
print("C2: B' entries", len(Bp), "dropped pairs", len((B - Bp)) // 2, "added", len(Bp - B) // 2)

# ---- C3/C4: game over B'
upP = fill(n1, Bp)
parP = {r: (upP[r][0] if upP[r] else None) for r in range(1, n1 + 1)}
arcs = sum(len(v) for v in upP.values())
U0 = {(a, b) for a, b in Bp if a < b}
print("C3/C4: arcs", arcs, "fill", arcs - len(U0), "json", STEP["C3"]["data"]["arcs"], STEP["C4"]["data"]["fill"])
if arcs != STEP["C3"]["data"]["arcs"] or arcs - len(U0) != STEP["C4"]["data"]["fill"]:
    bad("C3/C4 counts")
if {r: p for r, p in STEP["C3"]["data"]["etree"]} != parP:
    bad("C3 etree")
if {r: len(upP[r]) for r in upP} != dict(map(tuple, STEP["C3"]["data"]["col_counts"])):
    bad("C3 col counts")
for s_ in STEP["C4"]["data"]["steps"]:
    if s_["up"] != upP[s_["rank"]]:
        bad(f"C4 row {s_['rank']}")
nc = [r for r in range(1, n_r + 1) if (lambda s: max(s) - min(s) + 1 != len(s))(subtree(parP, r))]
nc_all = [r for r in range(1, n1 + 1) if (lambda s: max(s) - min(s) + 1 != len(s))(subtree(parP, r))]
print("C3: noncontiguous subtrees (ranks 1..20)", nc, " (ranks 1..21)", nc_all, "json",
      STEP["C3"]["data"]["noncontiguous_subtrees"])
print("C3: parent of pool rank 1 =", parP[1], "; old root(s)", [r for r in parB if parB[r] is None],
      "; new root(s)", [r for r in parP if parP[r] is None], "; parent of old root 20 =", parP[20])
print("C3: subtree of 20 =", sorted(subtree(parP, 20)))
for r in nc:
    print("   ", r, sorted(subtree(parP, r)))
# why does each subtree break? r containing rank 1 vs containing 21
for r in nc:
    s = subtree(parP, r)
    print("    rank", r, "contains rank 1:", 1 in s, " contains 21:", 21 in s)
# ---- C5: CSR ids row-major
base = STEP["C5"]["data"]["base"]
Ap, Aj, Ax = base["Up"]["Ap"], base["Up"]["Aj"], base["Up"]["Ax"]
k = 0
for r in range(1, n1 + 1):
    row = list(zip(Aj[Ap[r - 1]:Ap[r]], Ax[Ap[r - 1]:Ap[r]]))
    if [h for h, _ in row] != upP[r] or [i for _, i in row] != list(range(k, k + len(row))):
        bad(f"C5 row {r}")
    k += len(row)
print("C5: rows", len(Ap) - 1, "ids 0 ..", k - 1)
print("problems:", len(problems))
sys.exit(1 if problems else 0)
