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
