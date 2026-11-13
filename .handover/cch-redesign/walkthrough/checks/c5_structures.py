"""Independent check 5: F5's full-seed record (by id, base and delta) against my per-pair seeds;
bases (Down = Upᵀ with the same ids); the fold's new base = old base ⊕ delta at F; F7's page
sharing (path copy) and chunk copies; the fold trigger history; F4's arcs at rank 12; the E log.
Does not import walkthrough.py."""
import json
import os
import sys

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


def arcs_of(v):
    return {(lo, hi): (i, a, b) for lo, hi, i, a, b in J["hierarchies"][v]["arcs"]}


# ---- F5: the Repair's full seed by id, base and delta
rep = STEP["F5"]["data"]["repair"]
fs = rep["full_seed"]
W14 = W(14)
A14 = arcs_of("14")
wrong = []
for (lo, hi), (i, a, b) in A14.items():
    want = (W14.get((a, b), INF), W14.get((b, a), INF))
    got = (nz(fs["seeds_up"][i]), nz(fs["seeds_down"][i]))
    if want != got:
        wrong.append((lo, hi, i, got, want))
delta_ids = {e[2] for e in fs["delta_entries"]}
print("F5 full seed: arcs", len(A14), "delta entries", len(delta_ids), "wrong seeds", wrong[:5])
if wrong:
    bad("F5 full seed by id")
print("F5 path:", rep["path"], "| sweep triangles", rep.get("sweep_triangles"), "| work", rep["work"])

# ---- bases: Down = Upᵀ, same ids
for name, b in J["bases"].items():
    up = {}
    Ap, Aj, Ax = b["Up"]["Ap"], b["Up"]["Aj"], b["Up"]["Ax"]
    for r in range(1, len(Ap)):
        for k in range(Ap[r - 1], Ap[r]):
            up[(r, Aj[k])] = Ax[k]
    dn = {}
    Dp, Dj, Dx = b["Down"]["Ap"], b["Down"]["Aj"], b["Down"]["Ax"]
    for r in range(1, len(Dp)):
        row = Dj[Dp[r - 1]:Dp[r]]
        if row != sorted(row):
            bad(f"{name}: Down row {r} not sorted")
        for k in range(Dp[r - 1], Dp[r]):
            dn[(Dj[k], r)] = Dx[k]
    print(f"{name}: rows {len(Ap) - 1}, arcs {len(up)}, Down mirrors Up with same ids: {up == dn}")
    if up != dn:
        bad(f"{name}: Down is not Upᵀ")

# ---- fold: base 2 = base 1 ⊕ delta at v15 (ids as values)
b2 = J["bases"]["base 2"]
b2arcs = {}
Ap, Aj, Ax = b2["Up"]["Ap"], b2["Up"]["Aj"], b2["Up"]["Ax"]
for r in range(1, len(Ap)):
    for k in range(Ap[r - 1], Ap[r]):
        b2arcs[(r, Aj[k])] = Ax[k]
A15 = {k: v[0] for k, v in arcs_of("15").items()}
print("fold: base 2 == v15 arcs (base 1 ⊕ delta):", b2arcs == A15, "| rows", len(Ap) - 1)
if b2arcs != A15:
    bad("fold base")

# ---- fold trigger history: delta entries per version against sqrt(|base 1|)
import math
thr = math.sqrt(len(J["bases"]["base 1"]["Up"]["Aj"]))
for v in ["9", "10", "11", "12", "13", "14", "15", "16"]:
    d = J["hierarchies"][v]["delta"]["Up"]
    n = sum(len(p["entries"]) for p in d["pages"] if p["kind"] == "leaf") if d else 0
    print(f"  v{v}: Up.delta entries {n:2d}  > sqrt(54) = {thr:.2f}? {n > thr}")

# ---- F4: arcs at node 7's rank
h13 = J["hierarchies"]["13"]
r7 = dict(map(tuple, h13["maps"]["rank_of"]))[7]
at = [(lo, hi, i) for lo, hi, i, _, _ in h13["arcs"] if r7 in (lo, hi)]
inf = all(h13["metric"]["up"][i] is None and h13["metric"]["down"][i] is None for _, _, i in at)
print("F4: rank of 7 =", r7, "arcs at it:", len(at), "all +inf:", inf)

# ---- F7: page sharing. v16 pages - v15 pages must be exactly the root-to-leaf paths of v16's trees
#      that lead to the two inserted keys, in each of Up and Down.
def pages(v, side):
    d = J["hierarchies"][v]["delta"][side]
    return {p["page"]: p for p in d["pages"]}, d["root"]


def path(pg, root, key):
    out, pid = [], root
    while True:
        out.append(pid)
        p = pg[pid]
        if p["kind"] == "leaf":
            assert any(e[:2] == list(key) for e in p["entries"]), (key, p)
            return out
        seps = [tuple(s) for s in p["separators"]]
        i = sum(1 for s in seps if s <= key)
        pid = p["children"][i]


tot15, tot16, copied_expected = set(), set(), set()
split_bad = []
for side in ("Up", "Down"):
    p15, _ = pages("15", side)
    p16, root16 = pages("16", side)
    tot15 |= set(p15)
    tot16 |= set(p16)
    keys15 = {tuple(e[:2]) for p in p15.values() if p["kind"] == "leaf" for e in p["entries"]}
    keys16 = {tuple(e[:2]) for p in p16.values() if p["kind"] == "leaf" for e in p["entries"]}
    new = sorted(keys16 - keys15)            # the keys v16's writer inserted
    on_path = set()
    for k in new:
        on_path |= set(path(p16, root16, k))
    copied_expected |= on_path
    # a leaf that split leaves a new half off every insert's path: a new leaf whose parent is on a
    # path, holding only keys of one v15 leaf that an insert reached (a copy of a leaf no insert reached
    # does not count)
    parent = {c: pid for pid, p in p16.items() if p["kind"] == "branch" for c in p["children"]}
    _, root15 = pages("15", side)
    reached_keys = []                         # the v15 leaves an inserted key lands in
    for k in new:
        pid = root15
        while p15[pid]["kind"] == "branch":
            pid = p15[pid]["children"][sum(1 for sp in p15[pid]["separators"] if tuple(sp) <= k)]
        reached_keys.append({tuple(e[:2]) for e in p15[pid]["entries"]})
    for pid in set(p16) - set(p15) - on_path:
        p = p16[pid]
        ks = {tuple(e[:2]) for e in p.get("entries", [])}
        if p["kind"] == "leaf" and parent.get(pid) in on_path and any(ks <= o for o in reached_keys):
            copied_expected.add(pid)
        else:
            split_bad.append(pid)
    if not on_path <= set(p16) - set(p15):
        split_bad.append(("path page not copied", side))
    # validity of the v16 tree: in-order keys sorted, separators bound subtrees
    def walk(pid, lo=None, hi=None):
        p = p16[pid]
        if p["kind"] == "leaf":
            ks = [tuple(e[:2]) for e in p["entries"]]
            assert ks == sorted(ks) and all((lo is None or k >= lo) and (hi is None or k < hi) for k in ks)
            return ks
        seps = [tuple(s) for s in p["separators"]]
        out = []
        bounds = [lo] + seps + [hi]
        for i, c in enumerate(p["children"]):
            out += walk(c, bounds[i], bounds[i + 1])
        return out
    keys = walk(root16)
    assert keys == sorted(keys)
copied = tot16 - tot15
shared = tot16 & tot15
print(f"F7: v16 pages {len(tot16)}, copied {len(copied)}, shared {len(shared)}; copied == paths of the "
      f"inserts plus split-off leaves: {copied == copied_expected}")
f7 = STEP["F7"]["data"]
if len(copied) != len(f7["pages_copied_by_v16"]) or len(shared) != len(f7["pages_shared"]) or copied != copied_expected \
        or split_bad:
    bad(f"F7 pages (unexplained new pages: {split_bad})")
print("F7 chunks copied by v16:", f7["chunks_copied_by_v16"])
# which ids did v16 change?
m15, m16 = J["hierarchies"]["15"]["metric"], J["hierarchies"]["16"]["metric"]
chg = [i for i in range(len(m15["up"])) if (m15["up"][i], m15["down"][i], m15["up_middle"][i], m15["down_middle"][i]) !=
       (m16["up"][i], m16["down"][i], m16["up_middle"][i], m16["down_middle"][i])]
print("F7: ids whose metric changed v15 -> v16 (besides new ids 85, 86):", chg, "chunks", sorted({i // 16 for i in chg}))

# ---- E timeline: each batch fit and log rule by seq
ev = STEP["E"]["data"]["events"]
for e in ev:
    print("  E:", e.get("actor"), e.get("name", e.get("version")), "fork", e.get("fork"), e.get("fit"),
          "consumed", e.get("consumed"), "relinked", e.get("relinked"), "log",
          [(n["seq"], n["node"]) for n in e["published_log"]], e["entry"])
print("F events:")
for e in STEP["F6"]["data"]["events"] + STEP["F5"]["data"]["events"]:
    print("  ", e.get("actor"), e.get("version", e.get("name")), e.get("entry"), [n["seq"] for n in e.get("published_log", [])])
print("problems:", len(problems))
sys.exit(1 if problems else 0)
