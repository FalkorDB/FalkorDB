# 2D k x k grid, geometric nested dissection; insert one edge and count new fill arcs.
import sys, random
sys.path.insert(0, sys.argv[1])
from fill_check_lib import fill
def nd(cells):
    # cells: list of (x,y); returns order (eliminated first .. last)
    if len(cells) <= 4: return list(cells)
    xs = sorted({c[0] for c in cells}); ys = sorted({c[1] for c in cells})
    if len(xs) >= len(ys):
        mid = xs[len(xs)//2]; sep=[c for c in cells if c[0]==mid]
        A=[c for c in cells if c[0]<mid]; B=[c for c in cells if c[0]>mid]
    else:
        mid = ys[len(ys)//2]; sep=[c for c in cells if c[1]==mid]
        A=[c for c in cells if c[1]<mid]; B=[c for c in cells if c[1]>mid]
    return nd(A)+nd(B)+sep
for k in map(int, sys.argv[2:]):
    cells=[(x,y) for x in range(k) for y in range(k)]
    order = nd(cells); rank={c:i for i,c in enumerate(order)}; n=len(order)
    E=set()
    for (x,y) in cells:
        for dx,dy in ((1,0),(0,1)):
            if x+dx<k and y+dy<k: E.add((rank[(x,y)], rank[(x+dx,y+dy)]))
    up, parent = fill(n, E); base = sum(len(r) for r in up)
    def h(p):
        d=0
        while p is not None: d+=1; p=parent[p]
        return d
    H = max(h(r) for r in range(n))
    # cross edge: two cells in opposite halves (left/right of top separator), near the middle of each half
    u = rank[(k//4, k//2)]; v = rank[(3*k//4, k//2)]
    up2,_ = fill(n, E|{(u,v)}); cross = sum(len(r) for r in up2)-base
    # local edge: a cell to its etree grandparent-ish ancestor (ancestor case)
    w = rank[(1,1)]; anc = parent[parent[parent[w]]]
    up3,_ = fill(n, E|{(w,anc)}); local = sum(len(r) for r in up3)-base
    # brand-new node placed at the top (rank n) joined to 3 grid nodes in one leaf cell
    nbrs=[rank[(0,0)], rank[(0,1)], rank[(1,0)]]
    up4,_ = fill(n+1, E|{(x,n) for x in nbrs}); top = sum(len(r) for r in up4)-base
    print(f"k={k} n={n} |E|={len(E)} arcs={base} etree_height={H} new_arcs: cross_edge={cross} ancestor_edge={local} new_top_node_deg3={top}")
