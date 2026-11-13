"""Independent check 9: the MVCC view (example.json "mvcc") and the matrix track ("track").
Reads example.json only; does not import walkthrough.py.

1. Every instance the page shows for a state matches that state's own dump in "hierarchies"
   (maps, pool, delta entries, metric values, base arrays), so a shared instance id cannot hide
   different contents in two states.
2. A state's new delta pages are the root-to-leaf paths of its new keys, plus split halves: new
   leaves off every path, under a path page, holding only keys of one old leaf an insert reached.
3. A state's new metric chunks are exactly the chunks holding a value it changed or appended; a
   full-path Repair (its card's path is "full") builds every chunk new, sharing none with its base.
4. The topology instance changes exactly when a topology field does; the Cch when the topology or a
   metric array does.
5. "held only by this state" equals what that state holds and no other published state holds.
6. A Built entry keeps a log only while upkeep is set, and only nodes with seq > since.
7. The track: its order is B11's π, its supergraph is C4's arcs and fill, its ids are C5's Ax."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "..", "example.json")))
STEP = {s["id"]: s for s in J["steps"]}
MV, TR = J["mvcc"], J["track"]
COLS = MV["columns"]
COL = {c["key"]: c for c in COLS}
TOPO, METRIC = MV["fields"]["topology"], MV["fields"]["metric"]
CHUNK = MV["chunk"]
HKEY = {"14w": "13", "16f": "16 fold"}
problems = []


def bad(msg):
    problems.append(msg)
    print("MISMATCH:", msg)


def inst(kind, c):
    iid = c["objects"][kind]["iid"]
    return MV["instances"][kind]["null" if iid is None else str(iid)]


def hier(c):
    return J["hierarchies"][HKEY.get(c["key"], c["key"])]


def metric_values(kind, c):
    x = inst(kind, c)
    out = []
    for cid in x["chunks"]:
        out.extend(MV["metric_chunks"][str(cid)])
    return out[:x["length"]]


ARR = {"metric up": "up", "metric down": "down", "metric up_middle": "up_middle", "metric down_middle": "down_middle"}

# ---- 1. instances against each state's own dump
for c in COLS:
    H = hier(c)
    if inst("rank_of", c)["contents"] != H["maps"]["rank_of"]:
        bad(f"{c['label']} rank_of")
    no = {r: n for r, n in H["maps"]["node_of"]}
    if dict((r, n) for r, n in inst("node_of", c)["contents"]) != no:
        bad(f"{c['label']} node_of")
    if inst("free_ranks", c)["contents"] != H["maps"]["free_ranks"]:
        bad(f"{c['label']} free_ranks")
    for kind, side in (("Up.delta", "Up"), ("Down.delta", "Down")):
        x = inst(kind, c)
        want = sorted(e for p in H["delta"][side]["pages"] if p["kind"] == "leaf" for e in p["entries"])
        if sorted(x["entries"]) != want or x["root"] != H["delta"][side]["root"]:
            bad(f"{c['label']} {kind}")
        for p in H["delta"][side]["pages"]:
            if MV["pages"][str(p["page"])] != p:
                bad(f"{c['label']} {kind} page {p['page']}")
    for kind, arr in ARR.items():
        if metric_values(kind, c) != H["metric"][arr]:
            bad(f"{c['label']} {kind} values")
    for kind, side in (("Up.base", "Up"), ("Down.base", "Down")):
        if inst(kind, c)["contents"] != J["bases"][H["base"]][side] or c["objects"][kind]["iid"] != H["base"]:
            bad(f"{c['label']} {kind}")
    if [[q, a] for q, a in inst("added_since_fold", c)["contents"]] != H["added_since_fold"]:
        bad(f"{c['label']} added_since_fold")
print("1: instances match each state's dump for", len(COLS), "states")


# ---- 2. delta pages
def leaf_of(pg, root, key):
    pid = root
    while pg[pid]["kind"] == "branch":
        pid = pg[pid]["children"][sum(1 for sp in pg[pid]["separators"] if tuple(sp) <= key)]
    return pid


npage = 0
for c in COLS:
    p = COL.get(c["pred"]) if c["pred"] else None
    if p is None or c["key"] == "16f":
        continue
    for kind in ("Up.delta", "Down.delta"):
        a, b = inst(kind, p), inst(kind, c)
        if a["root"] == b["root"]:
            continue
        pg = {pid: MV["pages"][str(pid)] for pid in b["pages"] + a["pages"]}
        new_keys = sorted({tuple(e[:2]) for e in b["entries"]} - {tuple(e[:2]) for e in a["entries"]})
        path = set()
        for k in new_keys:
            pid = b["root"]
            while True:
                path.add(pid)
                if pg[pid]["kind"] == "leaf":
                    break
                pid = pg[pid]["children"][sum(1 for sp in pg[pid]["separators"] if tuple(sp) <= k)]
        made = set(b["pages"]) - set(a["pages"])
        reached = [{tuple(e[:2]) for e in pg[leaf_of(pg, a["root"], k)]["entries"]} for k in new_keys] \
            if a["root"] is not None else []
        parent = {ch: pid for pid in b["pages"] if pg[pid]["kind"] == "branch" for ch in pg[pid]["children"]}
        split = {q for q in made - path if pg[q]["kind"] == "leaf" and parent.get(q) in path
                 and any({tuple(e[:2]) for e in pg[q]["entries"]} <= r for r in reached)}
        if not path <= made or made != path | split:
            bad(f"{c['label']} {kind}: made {sorted(made)} paths {sorted(path)} split {sorted(split)}")
        npage += 1
print("2: delta page copies checked in", npage, "state/delta pairs")

# ---- 3. metric chunks
full_path = {cd["col"] for cd in MV["cards"] if cd.get("path") == "full" and cd.get("col")}
nch = 0
for c in COLS:
    p = COL.get(c["pred"]) if c["pred"] else None
    if p is None:
        continue
    if (c["key"] in full_path) != c["metric_rebuilt"]:
        bad(f"{c['label']} metric_rebuilt flag")
    for kind in METRIC:
        a, b = metric_values(kind, p), metric_values(kind, c)
        ca, cb = inst(kind, p)["chunks"], inst(kind, c)["chunks"]
        new = {i for i, cid in enumerate(cb) if i >= len(ca) or ca[i] != cid}
        if c["key"] in full_path:
            want = set(range(len(cb)))
            if set(ca) & set(cb):
                bad(f"{c['label']} {kind}: a rebuilt metric shares chunks with its base")
        else:
            want = {i // CHUNK for i in range(len(b)) if i >= len(a) or a[i] != b[i]}
        if new != want:
            bad(f"{c['label']} {kind}: copied {sorted(new)} expected {sorted(want)}")
        nch += 1
print("3: metric chunk copies checked in", nch, "state/array pairs")

# ---- 4. topology and Cch identity follow their fields
for c in COLS:
    p = COL.get(c["pred"]) if c["pred"] else None
    if p is None:
        continue
    tf = any(c["objects"][k]["iid"] != p["objects"][k]["iid"] for k in TOPO) or c["scalars"] != p["scalars"]
    mf = any(c["objects"][k]["iid"] != p["objects"][k]["iid"] for k in METRIC)
    tn = c["objects"]["TopologyVersion"]["iid"] != p["objects"]["TopologyVersion"]["iid"]
    cn = c["objects"]["Cch"]["iid"] != p["objects"]["Cch"]["iid"]
    if tf != tn or (tn or mf) != cn:
        bad(f"{c['label']} topology/Cch identity: fields {tf}/{mf}, new {tn}/{cn}")
print("4: topology and Cch identities checked")


# ---- 5. freed when a state drops
def held(c):
    out = {("Cch", c["objects"]["Cch"]["iid"]), ("TopologyVersion", c["objects"]["TopologyVersion"]["iid"])}
    for k in TOPO + METRIC:
        if c["objects"][k]["iid"] is not None:        # an empty delta is no instance
            out.add((k, c["objects"][k]["iid"]))
    for k in ("Up.delta", "Down.delta"):
        out |= {("delta page", q) for q in inst(k, c)["pages"]}
    for k in METRIC:
        out |= {("metric chunk", q) for q in inst(k, c)["chunks"]}
    for k in ("rank_of", "node_of"):
        out |= {(k + " chunk", q) for _, q in inst(k, c)["chunks"]}
    out |= {("log node", f"seq {n['seq']}") for n in c["log"]}
    return out


for i, c in enumerate(COLS):
    others = set().union(*(held(x) for x in COLS if x is not c))
    want = {}
    for kind, iid in held(c) - others:
        label = iid if kind in ("delta page", "metric chunk", "log node", "rank_of chunk", "node_of chunk") \
            else c["objects"][kind]["label"]
        want.setdefault(kind, set()).add(label)
    got = {k: set(v) for k, v in c["held_only"].items()}
    if got != want:
        bad(f"{c['label']} held only: {got} != {want}")
print("5: held-only sets checked")

# ---- 6. retention
for c in COLS:
    e = c["entry"]
    if e["state"] != "Built":
        continue
    seqs = [n["seq"] for n in c["log"]]
    if (e.get("upkeep") is None and seqs) or (e.get("upkeep") and any(q <= e["upkeep"]["since"] for q in seqs)):
        bad(f"{c['label']} log {seqs} with upkeep {e.get('upkeep')}")
print("6: retention checked")

# ---- 7. the track
pi = {x: r for x, r in TR["order"]["pi"]}
if pi != {x: r for x, r in STEP["B11"]["data"]["rank_of"] if r is not None}:
    bad("track order != B11 rank_of")
c4 = STEP["C4"]["data"]["steps"]
arcs = {(s["rank"], hi) for s in c4 for hi in s["up"]}
fill = {(s["rank"], f) for s in c4 for f in s["fill"]}
if {tuple(a) for a in TR["Up"]["arcs"]} != arcs or {tuple(a) for a in TR["Up"]["fill"]} != fill:
    bad("track supergraph != C4")
b = STEP["C5"]["data"]["base"]["Up"]
ids = {}
for r in range(len(b["Ap"]) - 1):
    for k in range(b["Ap"][r], b["Ap"][r + 1]):
        ids[(r + 1, b["Aj"][k])] = b["Ax"][k]
if {(lo, hi): i for lo, hi, i in TR["base"]["ids"]} != ids:
    bad("track ids != C5 Ax")
# the one extract: B = S(π,π)
if {tuple(e) for e in TR["B"]["entries"]} != {(pi[a], pi[b_]) for a, b_ in TR["S"]["entries"]}:
    bad("track B != S(π,π)")
moved = {r for r, _ in TR["order"]["moved_by_relabel"]}
print("7: track order, supergraph, ids and the extract checked;", len(arcs), "arcs,", len(fill), "fill; relabel moves",
      sorted(moved))

print("problems:", len(problems))
sys.exit(1 if problems else 0)
