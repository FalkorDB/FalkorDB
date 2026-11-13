# Claim (Ribizel & Anzt path decomposition, as a mask-free frontier loop):
# with ranks in an etree postorder, tag each entry (r, s) of U0 with the next larger r' in
# column s (or s itself if none). Repeatedly move (r, s, tag) -> (parent[r], s, tag) and keep it
# while parent[r] < tag. Every generated entry is new (frontiers are pairwise disjoint and
# disjoint from U0), and U0 plus all frontiers == U (the fill).
import random, sys
sys.path.insert(0, sys.argv[1])
from fill_check_lib import fill
from subtree_reorder_check_lib import postorder
random.seed(int(sys.argv[2])); worst = 0
for t in range(3000):
    n = random.randint(4, 50)
    E = {(random.randrange(n), random.randrange(n)) for _ in range(random.randint(n//2, 3*n))}
    E = {(a,b) for a,b in E if a!=b}
    _, parent = fill(n, E); po = postorder(parent, n); new = {o:i for i,o in enumerate(po)}
    E = {(new[a], new[b]) for a,b in E}
    up, parent = fill(n, E)
    U = {(r, w) for r in range(n) for w in up[r]}
    cols = {}
    for a,b in E: cols.setdefault(max(a,b), set()).add(min(a,b))
    F = []
    for s, rows in cols.items():
        rs = sorted(rows)
        for i, r in enumerate(rs): F.append((r, s, rs[i+1] if i+1 < len(rs) else s))
    seen = {(r, s) for r, s, _ in F}; iters = 0
    while F:
        iters += 1; G = []
        for r, s, tag in F:
            p = parent[r]
            if p is not None and p < tag:
                assert (p, s) not in seen, "frontier entry generated twice"
                seen.add((p, s)); G.append((p, s, tag))
        F = G
    assert seen == U
    worst = max(worst, iters)
print("ok, max iterations", worst)
