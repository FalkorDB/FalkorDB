"""Independent check 7: G4's break under a faithful model of 02's rule (02-final-design.md:262:
'Take the lowest rank in free_ranks; the holder becomes rank-less'), checked with 03's P; and the
S7 decay signals of 03 §6 against the published arc counts."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "..", "example.json")))
STEP = {s["id"]: s for s in J["steps"]}
node_of = {r: x for r, x in STEP["B11"]["data"]["rank_to_node"]}   # Order's ranks, before Supergraph
rank_of = {x: r for r, x in node_of.items() if x is not None}
free = [1, 2]
rank_of[18], node_of[1] = 1, 18                             # 02: isolated node 18 on pool rank 1

def P(free, node_of, rank_of):
    v = [r for r in node_of if (r in free) != (node_of[r] is None)]
    v += [x for x, r in rank_of.items() if node_of.get(r) != x]
    return v

print("02 state before the pop, P violations at ranks:", P(free, node_of, rank_of))
r = free.pop(0)                                             # node 19 (degree 1) takes rank 1
holder = node_of[r]
del rank_of[holder]                                          # 02: the holder becomes rank-less
node_of[r], rank_of[19] = 19, r
node_of[21] = 20; rank_of[20] = 21                          # node 20: top rank
print("02 state after the pop (faithful), P violations:", P(free, node_of, rank_of))

arcs = {v: len(J["hierarchies"][v]["arcs"]) for v in J["hierarchies"]}
print("arcs per version:", arcs)
build = 54
for v, prev in (("11", "10"), ("12", "11"), ("14", "13"), ("16", "15")):
    added = arcs[v] - arcs[prev]
    print(f"v{v}: arcs {arcs[v]} >= 1.25*{build}={1.25*build}? {arcs[v] >= 1.25*build}; "
          f"commit adds {added} > arcs/8 = {arcs[prev]/8:.3f} (before) / {arcs[v]/8:.3f} (after)? "
          f"{added > arcs[prev]/8} / {added > arcs[v]/8}")
