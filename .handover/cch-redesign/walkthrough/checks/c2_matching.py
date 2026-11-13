"""Independent check 2: the mutual-max matching rounds, the weight cap, the contraction
C·A·Cᵀ and C·w, at every level of every depth, plus the B5.h star. Proposal targets are
found by a direct argmax over neighbours (no un-scrambling). Does not import walkthrough.py."""
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


def fmix32(h):   # MurmurHash3 fmix32, written from the reference
    h &= 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def check_level(tag, lv, depth, level, nxt=None):
    verts = lv["verts"]
    A = {(i, j): a for i, j, a in lv["A"]}
    w = dict((v, x) for v, x in lv["w"])
    cell = dict((v, c) for v, c in lv["cell"]) if "cell" in lv else {v: 0 for v in verts}
    salt = fmix32(1 + 64 * depth + level)
    if salt != lv["salt"]:
        bad(f"{tag}: salt {lv['salt']} != fmix32(1+64d+l) = {salt}")
    tvw = {}
    for v in verts:   # level-0 vertex count of the cell = total weight
        tvw[cell[v]] = tvw.get(cell[v], 0) + w[v]
    cap = {c: 1.5 * t / 4 for c, t in tvw.items()}
    jcap = {c: x for c, x in lv["cap"]}
    for c in jcap:
        if abs(jcap[c] - cap[c]) > 1e-9:
            bad(f"{tag}: cap of cell {c} {jcap[c]} != {cap[c]}")
    capped = sorted({tuple(sorted(k)) for k in A if w[k[0]] + w[k[1]] > cap[cell[k[0]]]})
    if capped != sorted(map(tuple, lv["capped_out"])):
        bad(f"{tag}: capped {capped} != {lv['capped_out']}")
    un = set(verts) if "rounds" not in lv else set(lv["rounds"][0]["unmatched"])
    mate = {}
    nrounds = 0
    for rnd in range(1, 9):
        prop = {}
        for i in sorted(un):
            best = None
            for (a, j), x in A.items():
                if a != i or j not in un or w[i] + w[j] > cap[cell[i]]:
                    continue
                k = (x << 32) | fmix32(j ^ salt)
                if best is None or k > best[0]:
                    best = (k, j)
            if best:
                prop[i] = best[1]
        mutual = sorted((i, j) for i, j in prop.items() if prop.get(j) == i and i < j)
        jr = lv["rounds"][rnd - 1] if rnd - 1 < len(lv["rounds"]) else None
        if jr is None or sorted(map(tuple, jr["proposal"])) != sorted(prop.items()) or \
                sorted(map(tuple, jr["mutual"])) != mutual:
            bad(f"{tag} round {rnd}: mine prop {sorted(prop.items())} mutual {mutual}; json {jr and jr['proposal']} {jr and jr['mutual']}")
        if not mutual:
            break
        nrounds += 1
        for i, j in mutual:
            mate[i], mate[j] = j, i
            un -= {i, j}
    if nrounds != lv["matched_rounds"]:
        bad(f"{tag}: matched rounds {nrounds} != {lv['matched_rounds']}")
    pairs = sorted(sorted([i, j]) for i, j in mate.items() if i < j)
    if pairs != lv["pairs"]:
        bad(f"{tag}: pairs {pairs} != {lv['pairs']}")
    # shrink per cell and the 2-hop trigger
    for th in lv["two_hop"]:
        c = th["cell"]
        nb = sum(1 for v in verts if cell[v] == c)
        pc = sum(1 for v in verts if cell[v] == c and v in mate) // 2
        if (nb, nb - pc) != (th["before"], th["after_matching"]) or ((pc / nb) < 0.15) != th["fires"]:
            bad(f"{tag}: 2-hop record for cell {c}")
    # contraction
    if nxt is not None:
        group = {}
        for v in verts:
            group[v] = min(v, mate.get(v, v))
        leaders = sorted(set(group.values()))
        cid = {l: k for k, l in enumerate(leaders)}
        Ac = {}
        for (i, j), a in A.items():
            ci, cj = cid[group[i]], cid[group[j]]
            if ci != cj:
                Ac[(ci, cj)] = Ac.get((ci, cj), 0) + a
        wc = {}
        for v in verts:
            wc[cid[group[v]]] = wc.get(cid[group[v]], 0) + w[v]
        if Ac != {(i, j): a for i, j, a in nxt["A"]}:
            bad(f"{tag}: contraction A differs")
        if wc != dict((v, x) for v, x in nxt["w"]):
            bad(f"{tag}: contraction w differs")
        # symmetry of the coarse matrix and conservation of weight
        if any(Ac.get((j, i)) != a for (i, j), a in Ac.items()):
            bad(f"{tag}: coarse A not symmetric")
    print(f"{tag}: {len(verts)} -> {len(verts) - len(pairs)} in {nrounds} matched rounds; capped {len(capped)}; ok")


for li in range(3):
    s = STEP[f"B5.{li}"]["data"]
    check_level(f"B5.{li}", s, 0, li, s["next"])
d1 = STEP["B9.1"]["data"]
for lv in d1["levels"]:
    nxt = d1["levels"][lv["level"] + 1] if lv["level"] + 1 < len(d1["levels"]) else d1["coarsest"]
    check_level(f"B9.1 level {lv['level']}", lv, 1, lv["level"], nxt)

# ---- the star (B5.h): match, then anchor grouping in scramble order under the cap
st = STEP["B5.h"]["data"]
salt = fmix32(1)
A = {(i, j): a for i, j, a in st["A"]}
jp = st["two_hop"][0]["pairs"]
two_hop_pairs = {tuple(sorted(p)) for p in jp}
# the level's pairs are the matched pairs plus the 2-hop pairs; check_level checks the matched part
check_level("B5.h star matching", dict(st, cell=[[v, 0] for v in st["verts"]], matched_rounds=1,
                                     capped_out=[], two_hop=[],
                                     pairs=[p for p in st["pairs"] if tuple(sorted(p)) not in two_hop_pairs]), 0, 0)
mate = {}
for i, j in st["rounds"][0]["mutual"]:
    mate[i], mate[j] = j, i
un = [v for v in st["verts"] if v not in mate]
anchors = {}
for v in un:
    best = max(((a << 32) | fmix32(j ^ salt), j) for (i, j), a in A.items() if i == v)
    anchors[v] = best[1]
groups = {}
for v, a in anchors.items():
    groups.setdefault(a, []).append(v)
pairs = []
cap = 1.5 * 7 / 4
for a, vs in groups.items():
    vs = sorted(vs, key=lambda v: fmix32(v ^ salt))
    i = 0
    while i + 1 < len(vs):
        if 2 <= cap:
            pairs.append(sorted([vs[i], vs[i + 1]]))
            i += 2
        else:
            i += 1
print("B5.h: anchor pairs", sorted(pairs), "json", sorted(jp), "next verts", len(st["next"]["verts"]))
if sorted(pairs) != sorted(jp) or len(st["next"]["verts"]) != 7 - 1 - len(pairs):
    bad("B5.h anchor pairs")
# and the level's pair list is exactly the mutual pairs plus the anchor pairs
whole = sorted([sorted([i, j]) for i, j in st["rounds"][0]["mutual"]] + [sorted(p) for p in pairs])
if sorted(sorted(p) for p in st["pairs"]) != whole:
    bad(f"B5.h pairs {st['pairs']} != mutual + anchor {whole}")

# ---- forced 2-hop on the running example at depth 0 level 2 (the claimed shared anchor)
lv = STEP["B5.2"]["data"]
A = {(i, j): a for i, j, a in lv["A"]}
w = dict(map(tuple, lv["w"]))
salt = lv["salt"]
mate = {x for p in lv["pairs"] for x in p}
un = [v for v in lv["verts"] if v not in mate]
anc = {v: max(((a << 32) | fmix32(j ^ salt), j) for (i, j), a in A.items() if i == v)[1] for v in un}
print("B5.2 forced 2-hop: unmatched", un, "anchors", anc)
print("problems:", len(problems))
sys.exit(1 if problems else 0)
