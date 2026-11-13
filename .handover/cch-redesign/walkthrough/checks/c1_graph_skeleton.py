"""Independent check 1: rebuild every version's live graph from the brief's description,
then recompute S (B1), degrees (B2), peeling (B3) and components (B4), and compare with
example.json. Does not import walkthrough.py."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "..", "example.json")))
STEP = {s["id"]: s for s in J["steps"]}
problems = []


def bad(msg):
    problems.append(msg)
    print("MISMATCH:", msg)


def rw(a, b):
    lo, hi = min(a, b), max(a, b)
    return 1 + (5 * lo + 3 * hi) % 9


# ---- live directed edges per version, as a dict (type, src, dst) -> weight
grid = []
for r in range(4):
    for c in range(4):
        x = 4 * r + c
        if c < 3:
            grid.append((x, x + 1))
        if r < 3:
            grid.append((x, x + 4))
assert len(grid) == 24


def v5_live():
    E = {}
    for a, b in grid + [(15, 16), (16, 17)]:
        E[("ROAD", a, b)] = rw(a, b)
        if (a, b) != (6, 7):
            E[("ROAD", b, a)] = rw(a, b)
    # 4-5 pending add (live), 2-7 pending delete (dead), 14->13 replaced: live weight 6
    E[("ROAD", 14, 13)] = 6
    E[("FERRY", 10, 9)] = 2
    E[("FERRY", 5, 10)] = 3
    return E


V = {5: v5_live()}


def nxt(v, f):
    E = dict(V[v - 1])
    f(E)
    V[v] = E


def road(E, a, b, w):
    E[("ROAD", a, b)] = w
    E[("ROAD", b, a)] = w


def unroad(E, a, b):
    del E[("ROAD", a, b)]
    del E[("ROAD", b, a)]


nxt(6, lambda E: unroad(E, 1, 2))
nxt(7, lambda E: (road(E, 12, 19, 4), road(E, 0, 20, 3), road(E, 3, 20, 6)))
nxt(8, lambda E: E.__setitem__(("ROAD", 5, 6), 1))
nxt(9, lambda E: unroad(E, 9, 10))
nxt(10, lambda E: E.__setitem__(("ROAD", 8, 9), 1))
nxt(11, lambda E: road(E, 1, 2, 2))
nxt(12, lambda E: (road(E, 4, 21, 5), road(E, 12, 22, 2), road(E, 15, 22, 7)))


def delnode(E, x):
    for k in [k for k in E if x in k[1:]]:
        del E[k]


nxt(13, lambda E: delnode(E, 7))
nxt(14, lambda E: road(E, 1, 9, 3))
nxt(15, lambda E: E.__setitem__(("ROAD", 12, 13), 2))
nxt(16, lambda E: road(E, 14, 17, 4))

NODES = {5: set(range(19))}
for v in range(6, 17):
    NODES[v] = set(NODES[v - 1])
NODES_ADD = {7: [19, 20], 12: [21, 22]}
for v, xs in NODES_ADD.items():
    for w in range(v, 17):
        NODES[w] |= set(xs)
for w in range(13, 17):
    NODES[w].discard(7)

# compare with versions[*].graph (live relationships and node sets)
for v in range(5, 17):
    gj = J["versions"][str(v)]["graph"]
    got = {(r["type"], r["src"], r["dst"]): r["weight"] for r in gj["relationships"]}
    if got != V[v]:
        bad(f"v{v} live relationships differ: extra {set(got.items()) - set(V[v].items())}, "
            f"missing {set(V[v].items()) - set(got.items())}")
    if set(gj["nodes"]) != NODES[v]:
        bad(f"v{v} nodes differ: {sorted(set(gj['nodes']) ^ NODES[v])}")


def W(v):
    out = {}
    for (t, s, d), w in V[v].items():
        out[(s, d)] = min(out.get((s, d), float("inf")), w)
    return out


# ---- tensor layers at v5 (A)
A = STEP["A"]["data"]["graph"]["tensors"]
print("A: ROAD m/dp/dm =", len(A["ROAD"]["m"]), len(A["ROAD"]["dp"]), len(A["ROAD"]["dm"]),
      " FERRY m =", len(A["FERRY"]["m"]))
# m at v5 = every relationship flushed before v5: grid+chain two-way except 6->7 one-way,
# road 2-7 both directions, 14->13 (old id). 4-5 not in m.
m_expect = 24 * 2 - 1 - 2 + 2 * 2 + 2    # grid both ways minus 6->7's reverse minus 4-5 both, +chain, +2-7
if len(A["ROAD"]["m"]) != m_expect:
    bad(f"A: ROAD m has {len(A['ROAD']['m'])}, expected {m_expect}")
dp_pairs = sorted((s, d) for s, d, _ in A["ROAD"]["dp"])
if dp_pairs != [(4, 5), (5, 4), (14, 13)]:
    bad(f"A: ROAD dp = {dp_pairs}")
if sorted(map(tuple, A["ROAD"]["dm"])) != [(2, 7), (7, 2)]:
    bad(f"A: ROAD dm = {A['ROAD']['dm']}")

# ---- B1: S from slabs. Slab 0 rows 0-9 at v5, slab 1 rows 10-19 at v6, live pattern.
P = {(s, d) for (s, d) in W(5) if s < 10} | {(s, d) for (s, d) in W(6) if 10 <= s < 20}
S = {(a, b) for (a, b) in P | {(b, a) for (a, b) in P} if a != b}
print("B1: |S| =", len(S))
if len(S) != STEP["B1"]["data"]["nnz"] or {tuple(x) for x in STEP["B1"]["data"]["S"]} != S:
    bad("B1: S differs")
S_nomask = S | {(2, 7), (7, 2)}
if {tuple(x) for x in STEP["B1"]["data"]["S_without_dm_mask"]} != S_nomask:
    bad("B1: S without dm mask differs")
print("B1: |S without dm| =", len(S_nomask), " (1,2) in S:", (1, 2) in S)

# ---- B2 degrees
deg = {}
for a, b in S:
    deg[a] = deg.get(a, 0) + 1
ranked = sorted(deg)
print("B2: ranked", len(ranked), "no rank", sorted(set(range(20)) - set(ranked)))
if STEP["B2"]["data"]["ranked"] != len(ranked) or STEP["B2"]["data"]["n_r"] != len(ranked) + 2:
    bad("B2 counts")
if [[i, deg[i]] for i in ranked] != STEP["B2"]["data"]["d"]:
    bad("B2 degrees")

# ---- B3 peeling: repeatedly remove degree-1 nodes, ranks in (round, id) order from 3
alive = set(ranked)
adj = {x: {b for a, b in S if a == x} for x in ranked}
r = 3
peel = {}
rounds = 0
while True:
    f = sorted(x for x in alive if len(adj[x] & alive) == 1)
    if not f:
        break
    rounds += 1
    for x in f:
        peel[x] = r
        r += 1
    alive -= set(f)
print("B3: peeled", peel, "in", rounds, "rounds")
if [[i, rr] for i, rr in sorted(peel.items())] != STEP["B3"]["data"]["peeled_ranks"]:
    bad("B3 peel ranks")

# ---- B4 components
core = sorted(alive)
seen, comps = set(), []
for s in core:
    if s in seen:
        continue
    st, c = [s], []
    seen.add(s)
    while st:
        x = st.pop()
        c.append(x)
        for y in adj[x] & alive:
            if y not in seen:
                seen.add(y)
                st.append(y)
    comps.append(sorted(c))
print("B4:", [len(c) for c in comps], "ranks", r, "..", r + len(core) - 1)
cells = STEP["B4"]["data"]["cells"]
if len(cells) != len(comps) or cells[0]["verts"] != comps[0] or (cells[0]["lo"], cells[0]["hi"]) != (r, r + 15):
    bad("B4 component/ranks")

json.dump({"V": {str(v): [[t, s, d, w] for (t, s, d), w in sorted(E.items())] for v, E in V.items()},
           "nodes": {str(v): sorted(n) for v, n in NODES.items()}},
          open(os.path.join(HERE, "graphs_independent.json"), "w"))
print("problems:", len(problems))
sys.exit(1 if problems else 0)
