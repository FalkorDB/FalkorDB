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
