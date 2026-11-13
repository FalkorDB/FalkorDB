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
