#!/usr/bin/env python3
"""Tables for m1-reuse-quality.md and straddlers.json.

usage: parse_reuse.py main.txt strad.txt diag.txt cmetis.txt straddlers.json
"""
import json
import re
import statistics as st
import sys
from collections import defaultdict

main_p, strad_p, diag_p, cmetis_p, json_p = sys.argv[1:6]
KV = re.compile(r"(\w+)=(\[[^\]]*\]|\S+)")


def kv(line):
    return {k: v for k, v in KV.findall(line)}


def gname(d):
    return d["graph"].split(".")[0].replace("roadNet-", "")


# ---------- main sweep ----------
results = []  # dicts, each with 'depths' list
loads = {}
rss = {}
cur = None
for line in open(main_p):
    if line.startswith("LOAD"):
        d = kv(line)
        loads[(d["graph"], d["variant"], d["seed"])] = line.split()[-3]
    elif line.startswith("RSS"):
        parts = line.split()
        d = kv(line)
        rss[(d["graph"], d["variant"], d["seed"])] = int(parts[-1]) if parts[-1].isdigit() else None
    elif line.startswith("RESULT"):
        cur = kv(line)
        cur["depths"] = []
        results.append(cur)
    elif line.strip().startswith("DEPTH") and cur is not None:
        cur["depths"].append(kv(line))

by = defaultdict(dict)  # (graph, variant) -> seed -> result
for r in results:
    by[(gname(r), r["variant"])][int(r["seed"])] = r

GRAPHS = ["CA", "PA", "TX"]
VARS = [("shem", "feral SHEM"), ("mm", "mutual-max, re-coarsen (R)"), ("mm-reuse", "mutual-max, reuse (Reuse)")]


def med(xs):
    return st.median(xs)


cm = {}
for line in open(cmetis_p):
    if line.startswith("RESULT"):
        d = kv(line)
        cm[gname(d)] = d

print("## Checks over the main sweep")
allres = [r for r in results]
print(
    f"runs={len(allres)} closure_violations={sum(int(r['closure_violations']) for r in allres)} "
    f"cover_all_ok={all(r['cover_ok'] == 'true' for r in allres)} tree_mismatch={sum(int(r['tree_mismatch']) for r in allres)}"
)
reuse_runs = [r for r in allres if r["variant"] == "mm-reuse"]
print(
    "reuse runs: b2_checked=%d b2_mismatch=%d ident_fail=%d b1_invalid=%d coarsen_levels_after_d0=%d reuse_fresh_levels=%d roots_after_d0=%d"
    % (
        sum(int(r["b2_checked"]) for r in reuse_runs),
        sum(int(r["b2_mismatch"]) for r in reuse_runs),
        sum(int(r["ident_fail"]) for r in reuse_runs),
        sum(int(r["b1_invalid"]) for r in reuse_runs),
        sum(int(r["coarsen_levels_after_d0"]) for r in reuse_runs),
        sum(int(r["reuse_fresh_levels"]) for r in reuse_runs),
        sum(int(r["roots_after_d0"]) for r in reuse_runs),
    )
)
# depth-0 identical between R and Reuse
same_root = []
for g in GRAPHS:
    for s in range(1, 6):
        a = by[(g, "mm")].get(s)
        b = by[(g, "mm-reuse")].get(s)
        if a and b:
            same_root.append(a["root_hash"] == b["root_hash"])
print(f"depth-0 hash equal R vs Reuse: {sum(same_root)}/{len(same_root)}")
import os
rp = os.path.join(os.path.dirname(main_p), "roots.txt")
if os.path.exists(rp):
    rr = defaultdict(dict)
    for line in open(rp):
        if line.startswith("RESULT"):
            d = kv(line)
            rr[(gname(d), int(d["seed"]))][d["variant"]] = d
    first_eq = 0
    tot = 0
    nroots = {}
    tie = 0
    for (g, s), vv in sorted(rr.items()):
        if "mm" in vv and "mm-reuse" in vv:
            a = vv["mm"]["roots"].split(",")
            b = vv["mm-reuse"]["roots"].split(",")
            tot += 1
            first_eq += a[0] == b[0]
            nroots[g] = (len(a), [x.split(":")[0] for x in a])
            for v in ["mm", "mm-reuse"]:
                m = by[(g, v)].get(s)
                tie += bool(m) and all(m[k] == vv[v][k] for k in ["arcs", "triangles", "etree_height"])
            print(f"  roots {g} seed {s}: R {[x[:20] for x in a]} Reuse {[x[:20] for x in b]} first_equal={a[0] == b[0]} all_equal={a == b}")
    print(f"first top-level component identical at depth 0 (R vs Reuse): {first_eq}/{tot}; components bisected at depth 0: {nroots}; roots runs tied to main sweep: {tie}/{2*tot}")
print("check rep seconds (reuse):", {f"{gname(r)}{r['seed']}": r["check_s"] for r in reuse_runs})
print("loads:", sorted(set(loads.values())))
print("rss MB max per graph/variant:", {
    f"{k[0]}/{k[1]}": max((v or 0) for kk, v in rss.items() if kk[0] == k[0] and kk[1] == k[1]) // 2**20
    for k in rss})

print("\n## Main table (5 seeds; medians, range in brackets)\n")
print("| Graph | Ordering | Arcs | Triangles | Tree height | Order time (s) |")
print("|---|---|---|---|---|---|")
for g in GRAPHS:
    for v, label in VARS:
        rs = list(by[(g, v)].values())
        if not rs:
            continue
        a = [int(r["arcs"]) for r in rs]
        t = [int(r["triangles"]) / 1e6 for r in rs]
        h = [int(r["etree_height"]) for r in rs]
        tm = [float(r["order_s_median"]) for r in rs]
        print(
            f"| {g} | {label} | {med(a):,.0f} [{min(a)/1e6:.2f}–{max(a)/1e6:.2f}M] | {med(t):.1f}M [{min(t):.1f}–{max(t):.1f}] | "
            f"{med(h):.0f} [{min(h)}–{max(h)}] | {med(tm):.2f} [{min(tm):.2f}–{max(tm):.2f}] |"
        )
    if g in cm:
        d = cm[g]
        print(f"| {g} | C METIS (1 run, earlier report) | {int(d['arcs']):,} | {int(d['triangles'])/1e6:.1f}M | {d['etree_height']} | — |")

print("\n## Paired by seed\n")
print("| Graph | Seed | Reuse arcs | vs R | vs SHEM | Reuse triangles | vs R | vs SHEM | height R → Reuse | time R → Reuse (s) |")
print("|---|---|---|---|---|---|---|---|---|---|")
wins = defaultdict(lambda: [0, 0])
for g in GRAPHS:
    for s in range(1, 6):
        u = by[(g, "mm-reuse")].get(s)
        r = by[(g, "mm")].get(s)
        h = by[(g, "shem")].get(s)
        if not (u and r and h):
            continue
        ua, ra, ha = int(u["arcs"]), int(r["arcs"]), int(h["arcs"])
        ut, rt, ht = int(u["triangles"]), int(r["triangles"]), int(h["triangles"])
        for key, cond in [("arcs<=R", ua <= ra), ("tri<=R", ut <= rt), ("arcs<=SHEM", ua <= ha), ("tri<=SHEM", ut <= ht)]:
            wins[(g, key)][0] += cond
            wins[(g, key)][1] += 1
        print(
            f"| {g} | {s} | {ua:,} | {ua/ra-1:+.2%} | {ua/ha-1:+.2%} | {ut/1e6:.1f}M | {ut/rt-1:+.2%} | {ut/ht-1:+.2%} | "
            f"{r['etree_height']} → {u['etree_height']} | {r['order_s_median']} → {u['order_s_median']} |"
        )
print()
for g in GRAPHS:
    print(g, {k: f"{w[0]}/{w[1]}" for (gg, k), w in wins.items() if gg == g})
print()
for g in GRAPHS:
    if g not in cm:
        continue
    ca, ct = int(cm[g]["arcs"]), int(cm[g]["triangles"])
    row = []
    for v, _ in VARS:
        rs = list(by[(g, v)].values())
        if rs:
            row.append(f"{v}: arcs {med([int(r['arcs']) for r in rs])/ca-1:+.1%} tri {med([int(r['triangles']) for r in rs])/ct-1:+.1%}")
    print(f"gap to C METIS {g}: " + "; ".join(row))

# ---------- per depth: levels reused / fresh, R's coarsening ----------
print("\n## Per depth (sums over 5 seeds; wratio = heaviest coarsest vertex / R's cap, median over seeds of the per-run p50)\n")
for g in GRAPHS:
    print(f"\n### {g}\n")
    print("| Depth | cells (Reuse) | levels reused | per cell | fresh | over R's cap | wratio p50 / p90 / max | R: levels coarsened | R: vertices coarsened (n_fine sum) | Reuse: vertices coarsened |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    maxd = 0
    agg = defaultdict(lambda: defaultdict(list))
    for v in ["mm", "mm-reuse"]:
        for s, r in by[(g, v)].items():
            for d in r["depths"]:
                dd = int(d["d"])
                maxd = max(maxd, dd)
                agg[(v, dd)]["x"].append(d)
    for dd in range(maxd + 1):
        U = agg[("mm-reuse", dd)]["x"]
        R = agg[("mm", dd)]["x"]
        if not U:
            continue
        cells = sum(int(x["reuse_cells"]) for x in U)
        reused = sum(int(x["reused"]) for x in U)
        fresh = sum(int(x["fresh"]) for x in U)
        over = sum(int(x["overcap"]) for x in U)
        p50 = med([float(x["wratio_p50"]) for x in U])
        p90 = med([float(x["wratio_p90"]) for x in U])
        mx = max(float(x["wratio_max"]) for x in U)
        rl = sum(int(x["coarsen_levels"]) for x in R)
        rn = sum(int(x["coarsen_nfine"]) for x in R)
        un = sum(int(x["coarsen_nfine"]) for x in U)
        print(
            f"| {dd} | {cells} | {reused} | {reused/cells if cells else 0:.1f} | {fresh} | "
            f"{over}/{cells} ({over/cells if cells else 0:.0%}) | {p50:.2f} / {p90:.2f} / {mx:.2f} | {rl} | {rn:,} | {un:,} |"
        )

# ---------- straddlers ----------
strad = defaultdict(dict)  # (graph, seed) -> (d, l) -> rec
stradrun = {}
cur = None
for line in open(strad_p):
    if line.startswith("STRADRUN"):
        cur = kv(line)
        stradrun[(gname(cur), int(cur["seed"]))] = cur
    elif line.strip().startswith("STRAD ") and cur is not None:
        d = kv(line)
        strad[(gname(cur), int(cur["seed"]))][(int(d["d"]), int(d["l"]))] = {k: int(v) for k, v in d.items()}

print("\n## Strad runs (check rep alone) tied to the main sweep\n")
for (g, s), d in sorted(stradrun.items()):
    m = by[(g, "mm-reuse")].get(s)
    tie = m and all(d[k] == m[k] for k in ["arcs", "triangles", "etree_height"])
    print(f"{g} seed {s}: arcs={d['arcs']} tri={d['triangles']} same_as_main={tie} b2_checked={d['b2_checked']} b2_mismatch={d['b2_mismatch']} ident_fail={d['ident_fail']} b1_invalid={d['b1_invalid']} closure={d['closure_violations']} cover={d['cover_ok']} tree={d['tree_mismatch']} check_s={d['check_s']}")

FIELDS = ["splits_all", "n_all", "nnz_all", "strad_side_all", "splits", "n", "nnz", "sep_only", "strad_side",
          "strad_child", "strad_rows_nnz", "used", "x_n", "x_nnz", "xprev_nnz", "rin", "rout", "x_strad",
          "b2_checked", "b2_mismatch", "b1_invalid"]

print("\n## Straddlers per split depth (sums over 5 seeds, needed levels only)\n")
print("share = side straddlers / parent coarse vertices; entries = their rows' entries / parent entries; "
      "B2 input = straddler-member entries re-contracted / entries a full re-contraction reads\n")
out_json = {
    "definition": {
        "depth": "ND depth d of the cell being split; its children (depth d+1) take these levels",
        "level": "root (depth-0) hierarchy level, 1 = first contraction of the component",
        "needed": "levels at or above the deepest one some child with > 200 vertices (nd_to_amd_switch) coarsens with; "
                  "fields without _all count only splits and levels that are needed",
        "n / nnz": "parent coarse vertices / stored entries (both directions) at this level, summed over splits",
        "strad_side": "parent coarse vertices with fine members in >= 2 of {A, B, separator}: the design's cells",
        "strad_child": "same, with A and B further split into connected components (feral's children)",
        "sep_only": "parent coarse vertices wholly in the separator (dropped by the masked split, no re-contraction)",
        "share": "strad_side / n",
        "entry_share": "strad_rows_nnz / nnz: parent entries in straddler rows",
        "x_n / x_nnz": "children's vertices / entries at this level (children > 200 vertices only)",
        "xprev_nnz": "entries of the children's previous level: what B1 (full re-contraction) reads",
        "rin": "of xprev_nnz, the entries in rows of straddler members: what B2 re-contracts",
        "rout": "entries in the children's straddler-part rows at this level (B2's re-contracted output)",
        "x_strad": "children's coarse vertices that are straddler parts",
        "source": "m1bench strad mode (check rep of mm-reuse), roadNet CA/PA/TX, seeds 1-5",
    },
    "graphs": {},
}
for g in GRAPHS:
    seeds = sorted(s for (gg, s) in strad if gg == g)
    if not seeds:
        continue
    keys = sorted({k for s in seeds for k in strad[(g, s)]})
    gj = {"per_seed": {}, "summed": []}
    for s in seeds:
        gj["per_seed"][str(s)] = [
            {"depth": d, "level": l, **{f: strad[(g, s)][(d, l)].get(f, 0) for f in FIELDS}}
            for (d, l) in sorted(strad[(g, s)])
        ]
    for (d, l) in keys:
        tot = {f: sum(strad[(g, s)].get((d, l), {}).get(f, 0) for s in seeds) for f in FIELDS}
        shares = [strad[(g, s)][(d, l)]["strad_side"] / strad[(g, s)][(d, l)]["n"]
                  for s in seeds if (d, l) in strad[(g, s)] and strad[(g, s)][(d, l)]["n"]]
        gj["summed"].append({
            "depth": d, "level": l, **tot,
            "share": tot["strad_side"] / tot["n"] if tot["n"] else None,
            "share_seed_median": med(shares) if shares else None,
            "share_all_levels": tot["strad_side_all"] / tot["n_all"] if tot["n_all"] else None,
            "entry_share": tot["strad_rows_nnz"] / tot["nnz"] if tot["nnz"] else None,
            "b2_input_share": tot["rin"] / tot["xprev_nnz"] if tot["xprev_nnz"] else None,
        })
    out_json["graphs"][g] = gj

    print(f"\n### {g}\n")
    print("| Split depth | needed levels | share at level 1 | share, all needed levels | max share (level) | entry share | B2 input / B1 input | child straddler share |")
    print("|---|---|---|---|---|---|---|---|")
    depths = sorted({d for (d, l) in keys})
    for d in depths:
        rows = [x for x in gj["summed"] if x["depth"] == d and x["n"] > 0]
        if not rows:
            continue
        n = sum(x["n"] for x in rows)
        sside = sum(x["strad_side"] for x in rows)
        nnz = sum(x["nnz"] for x in rows)
        snz = sum(x["strad_rows_nnz"] for x in rows)
        rin = sum(x["rin"] for x in rows)
        xprev = sum(x["xprev_nnz"] for x in rows)
        schild = sum(x["strad_child"] for x in rows)
        l1 = next((x for x in rows if x["level"] == 1), None)
        mx = max(rows, key=lambda x: x["share"])
        print(
            f"| {d} | {len(rows)} | {l1['share']:.2%} | {sside/n:.2%} | {mx['share']:.1%} (l{mx['level']}) | {snz/nnz:.2%} | "
            f"{rin/xprev if xprev else 0:.2%} | {schild/n:.2%} |"
            if l1 else f"| {d} | {len(rows)} | — | {sside/n:.2%} | {mx['share']:.1%} (l{mx['level']}) | {snz/nnz:.2%} | {rin/xprev if xprev else 0:.2%} | {schild/n:.2%} |"
        )

json.dump(out_json, open(json_p, "w"), indent=1)
print(f"\nwrote {json_p}")

# ---------- diagnostics ----------
dres = defaultdict(dict)
for line in open(diag_p) if diag_p != "-" else []:
    if line.startswith("RESULT"):
        d = kv(line)
        d["depths"] = []
        dres[(gname(d), d["variant"])][int(d["seed"])] = d
        last = d
    elif line.strip().startswith("DEPTH"):
        last["depths"].append(kv(line))
if dres:
    print("\n## Diagnostics (1 repetition per seed)\n")
    print("| Graph | Variant | arcs median | tri median | paired vs R (arcs, tri ≤) | paired vs SHEM (arcs, tri ≤) | vertices coarsened after depth 0, share of R's |")
    print("|---|---|---|---|---|---|---|")
    for (g, v), ss in sorted(dres.items()):
        a = [int(x["arcs"]) for x in ss.values()]
        t = [int(x["triangles"]) / 1e6 for x in ss.values()]
        pr = [0, 0, 0]
        ph = [0, 0, 0]
        fr = []
        for s, x in ss.items():
            r = by[(g, "mm")].get(s)
            h = by[(g, "shem")].get(s)
            if r:
                pr[0] += int(x["arcs"]) <= int(r["arcs"])
                pr[1] += int(x["triangles"]) <= int(r["triangles"])
                pr[2] += 1
                rn = sum(int(dd["coarsen_nfine"]) for dd in r["depths"] if int(dd["d"]) >= 1)
                un = sum(int(dd["coarsen_nfine"]) for dd in x["depths"] if int(dd["d"]) >= 1)
                fr.append(un / rn if rn else 0)
            if h:
                ph[0] += int(x["arcs"]) <= int(h["arcs"])
                ph[1] += int(x["triangles"]) <= int(h["triangles"])
                ph[2] += 1
        print(f"| {g} | {v} | {med(a):,.0f} | {med(t):.1f}M | {pr[0]}/{pr[2]}, {pr[1]}/{pr[2]} | {ph[0]}/{ph[2]}, {ph[1]}/{ph[2]} | {med(fr) if fr else 0:.1%} |")
    for (g, v), ss in sorted(dres.items()):
        for s, x in sorted(ss.items()):
            r = by[(g, "mm")].get(s)
            if r:
                print(f"  {g} {v} seed {s}: arcs {int(x['arcs'])/int(r['arcs'])-1:+.2%} tri {int(x['triangles'])/int(r['triangles'])-1:+.2%} vs R; time {x['order_s_median']} (R {r['order_s_median']})")
