# Checks derived claims about fill under a fixed order (rank space: vertex i has rank i).
import random, sys
from itertools import combinations

def fill(n, edges):
    """GNP child absorption: up[r] = adj_up(r) U union(up[c] - {r}) over etree children c."""
    adj = [set() for _ in range(n)]
    for a, b in edges:
        if a != b: adj[min(a,b)].add(max(a,b))
    up = [None]*n; parent = [None]*n; children = [[] for _ in range(n)]
    for r in range(n):
        row = set(adj[r])
        for c in children[r]: row |= (up[c] - {r})
        up[r] = row
        if row:
            p = min(row); parent[r] = p; children[p].append(r)
    return up, parent

def ancestors(parent, x):
    out = []
    while x is not None:
        out.append(x); x = parent[x]
    return out

def liu_etree(n, edges):
    lower = [[] for _ in range(n)]
    for a, b in edges:
        if a != b: lower[max(a,b)].append(min(a,b))
    parent = [None]*n; anc = [None]*n
    for j in range(n):
        for i in lower[j]:
            while i is not None and i < j:
                nxt = anc[i]; anc[i] = j
                if nxt is None: parent[i] = j
                i = nxt
    return parent

def msf_etree(n, edges):
    # Kruskal with weight max(i,j); any MSF gives the same component tree
    E = sorted({(max(a,b), min(a,b)) for a,b in edges if a!=b})
    uf = list(range(n)); top = list(range(n)); parent=[None]*n
    def find(x):
        while uf[x]!=x: uf[x]=uf[uf[x]]; x=uf[x]
        return x
    for j,i in E:
        ri, rj = find(i), find(j)
        if ri==rj: continue
        # component of i (all ranks < j) gets parent j at its top
        parent[top[ri]] = j
        uf[ri] = rj  # j's component keeps top j
        top[rj] = j
    return parent

def frontier_fill(n, edges, parent):
    # row-subtree propagation: X(i,j) for i<j; frontier step: i -> parent[i] while < j
    X = set(); F = set()
    for a,b in edges:
        if a!=b: F.add((min(a,b), max(a,b)))
    X |= F; iters = 0
    while F:
        iters += 1
        G = set()
        for i,j in F:
            p = parent[i]
            if p is not None and p < j and (p,j) not in X: G.add((p,j))
        X |= G; F = G
    return X, iters

def height(parent, n):
    h = [0]*n
    for r in range(n):
        if parent[r] is not None: h[parent[r]] = max(h[parent[r]], h[r]+1)
    return max(h) if n else 0

def postorder_ranges(parent, n):
    # in a postorder, subtree of a = contiguous range [a - size(a) + 1, a]
    size=[1]*n
    for r in range(n):
        if parent[r] is not None: size[parent[r]] += size[r]
    return size

random.seed(int(sys.argv[1]) if len(sys.argv)>1 else 1)
stats = dict(trials=0, anc_case=0, cross_case=0, early_stop_ok=0)
for trial in range(3000):
    n = random.randint(4, 40)
    m = random.randint(n//2, 3*n)
    edges = {(random.randrange(n), random.randrange(n)) for _ in range(m)}
    edges = {(a,b) for a,b in edges if a!=b}
    up, parent = fill(n, edges)
    # (A) Liu etree == GNP parent == MSF component tree
    assert liu_etree(n, edges) == parent, "liu"
    assert msf_etree(n, edges) == parent, "msf"
    # (B) frontier (row-subtree) fill == GNP fill; iterations <= height
    X, iters = frontier_fill(n, edges, parent)
    assert X == {(r,w) for r in range(n) for w in up[r]}, "frontier"
    assert iters <= height(parent, n) + 1
    # (C) insert one edge u<v
    u, v = sorted(random.sample(range(n), 2))
    if v in up[u]: continue
    stats['trials'] += 1
    up2, parent2 = fill(n, edges | {(u,v)})
    changed = {r for r in range(n) if up2[r] != up[r]}
    newpath = set(ancestors(parent2, u))
    assert changed <= newpath, "Davis-Hager: only new path rows change"
    assert newpath == set(ancestors(parent, u)) | set(ancestors(parent, v)), "new path = P(u) U P(v)"
    Pu = ancestors(parent, u)
    if v in Pu:
        stats['anc_case'] += 1
        assert parent2 == parent, "ancestor case keeps etree"
        expect = set()
        for j in Pu:
            if j == v or v in up[j]: break
            expect.add(j)
        assert changed == expect, "ancestor case: walk until v present"
        for j in changed: assert up2[j] == up[j] | {v}
    else:
        stats['cross_case'] += 1
        Pv = set(ancestors(parent, v)); a = next((x for x in Pu if x in Pv), None)
        if a is not None:
            # rows at the old LCA and above are unchanged
            assert all(r < a for r in changed), "rows >= old LCA unchanged"
            stats['early_stop_ok'] += 1
print(stats)
