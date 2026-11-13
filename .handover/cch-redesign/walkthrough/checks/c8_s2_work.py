"""Independent check 8: S2 (Davis–Hager with an overlay) re-run from each predecessor's arcs;
its row reads + delta inserts (Up and Down) against the JSON's s2 work."""
import heapq, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "..", "example.json")))
STEP = {s["id"]: s for s in J["steps"]}
def arcs(v):
    up = {}
    for lo, hi, i, _, _ in J["hierarchies"][v]["arcs"]:
        up.setdefault(lo, set()).add(hi)
    return up
for old, sid, rec in (("10", "F2", STEP["F2"]["data"]["flush"]), ("11", "F3", STEP["F3"]["data"]["flush"]),
                      ("13", "F5", STEP["F5"]["data"]["repair"]), ("15", "v16", STEP["F6"]["data"]["v16"]["flush"])):
    up = arcs(old)
    over = {}
    heap = []
    for lo, hi in rec["classify"]["s2_insert"]:
        over.setdefault(lo, set()).add(hi)
        heap.append(lo)
    heapq.heapify(heap)
    done, reads = set(), 0
    while heap:
        j = heapq.heappop(heap)
        if j in done:
            continue
        done.add(j)
        row = sorted(up.get(j, set()) | over.get(j, set()))
        reads += len(row)
        p = row[0]
        prow = up.get(p, set()) | over.get(p, set())
        reads += len(prow)
        need = set(row[1:]) - prow
        if need:
            over.setdefault(p, set()).update(need)
            heapq.heappush(heap, p)
    new = sum(len(s) for s in over.values())
    print(f"{sid}: new arcs {new}, rows read {reads}, work {reads + 2 * new}, json {rec['s2']['work'] if 's2' in rec else rec['work']['s2']}")
