"""SNAP edge list -> nd.c input (n, nnz, xadj, adjncy as int64), with the bench's load() rules:
symmetric, self-loops dropped, duplicates removed, isolated nodes removed (ids remapped in order)."""
import sys, array
src, dst = sys.argv[1], sys.argv[2]
adj = {}
for line in open(src):
    if line.startswith('#'): continue
    a, b = line.split()[:2]
    u, v = int(a), int(b)
    if u == v: continue
    adj.setdefault(u, set()).add(v)
    adj.setdefault(v, set()).add(u)
ids = sorted(adj)
remap = {u: k for k, u in enumerate(ids)}
xadj = array.array('q', [0])
adjncy = array.array('q')
for u in ids:
    adjncy.extend(sorted(remap[v] for v in adj[u]))
    xadj.append(len(adjncy))
with open(dst, 'wb') as f:
    array.array('q', [len(ids), len(adjncy)]).tofile(f)
    xadj.tofile(f); adjncy.tofile(f)
print(dst, 'n', len(ids), 'nnz', len(adjncy))
