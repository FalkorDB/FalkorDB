import sys, random, statistics
sys.path.insert(0, sys.argv[1])
from fill_check_lib import fill
from grid_nd import nd
random.seed(5)
for k in map(int, sys.argv[2:]):
    cells=[(x,y) for x in range(k) for y in range(k)]
    order = nd(cells); rank={c:i for i,c in enumerate(order)}; n=len(order)
    E=set()
    for (x,y) in cells:
        for dx,dy in ((1,0),(0,1)):
            if x+dx<k and y+dy<k: E.add((rank[(x,y)], rank[(x+dx,y+dy)]))
    up, parent = fill(n, E); base = sum(len(r) for r in up)
    def anc(p):
        out=[]
        while p is not None: out.append(p); p=parent[p]
        return out
    cross=[]; ancs=[]
    for _ in range(40):
        # cross: leaf-ish cells on opposite sides of the top separator (x<k/2 vs x>k/2)
        u = rank[(random.randrange(0,k//2-1), random.randrange(k))]
        v = rank[(random.randrange(k//2+1,k), random.randrange(k))]
        up2,_ = fill(n, E|{(u,v)}); cross.append(sum(len(r) for r in up2)-base)
        # ancestor case: a leaf-ish node and a random ancestor not already adjacent
        w = rank[(random.randrange(k), random.randrange(k))]
        cand=[a for a in anc(parent[w])[1:] if a not in up[w]] if parent[w] is not None else []
        if cand:
            a=random.choice(cand); up3,_=fill(n, E|{(w,a)}); ancs.append(sum(len(r) for r in up3)-base)
    print(f"k={k} n={n} arcs={base} | cross-top-separator edge: median {statistics.median(cross)} max {max(cross)} | ancestor edge: median {statistics.median(ancs)} max {max(ancs)}")
