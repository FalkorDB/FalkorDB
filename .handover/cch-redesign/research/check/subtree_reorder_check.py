# Claim: with ranks in an etree postorder, any re-ordering of the ranks inside one subtree's
# contiguous range [a-size(a)+1, a] changes only the rows of ranks in that range; rows outside
# are identical (same rank numbers, same entries). Also: rows inside depend only on the edges
# touching the subtree.
import random, sys
sys.path.insert(0, sys.argv[1])
from fill_check_lib import fill

def postorder(parent, n):
    children = [[] for _ in range(n)]; roots=[]
    for r in range(n):
        (children[parent[r]] if parent[r] is not None else roots).append(r)
    out=[]
    def visit(x):
        stack=[(x,0)]
        while stack:
            y,i = stack.pop()
            if i < len(children[y]): stack.append((y,i+1)); stack.append((children[y][i],0))
            else: out.append(y)
    for r in roots: visit(r)
    return out  # out[new_rank] = old_rank

random.seed(int(sys.argv[2]))
checked = 0
for t in range(3000):
    n = random.randint(5, 40)
    edges = {(random.randrange(n), random.randrange(n)) for _ in range(random.randint(n//2, 3*n))}
    edges = {(a,b) for a,b in edges if a!=b}
    up, parent = fill(n, edges)
    po = postorder(parent, n); new = {old:i for i,old in enumerate(po)}
    E = {(new[a], new[b]) for a,b in edges}
    up, parent = fill(n, E)
    assert all(parent[r] is None or parent[r] > r for r in range(n))
    size=[1]*n
    for r in range(n):
        if parent[r] is not None: size[parent[r]] += size[r]
    a = random.randrange(n); lo = a - size[a] + 1
    rng = list(range(lo, a+1)); perm = rng[:]; random.shuffle(perm)
    mp = {x:x for x in range(n)}; mp.update(dict(zip(rng, perm)))
    E2 = {(mp[x], mp[y]) for x,y in E}
    up2, _ = fill(n, E2)
    for r in range(n):
        if lo <= r <= a: continue
        assert up2[r] == up[r], (r, lo, a)
    checked += 1
print("checked", checked)
