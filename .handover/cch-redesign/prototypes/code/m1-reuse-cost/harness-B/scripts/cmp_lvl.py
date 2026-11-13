"""Compare R's per-level (n, nnz, rounds, twohop) between a new log and the earlier bench's log (threads=1 run=0)."""
import sys
def lv(path):
    out = {}
    for line in open(path):
        w = line.split()
        if not w or w[0] != 'LVL': continue
        kv = {}
        for x in w[1:]:
            k, v = x.split('=', 1)
            kv.setdefault(k, v)
        if kv['threads'] != '1' or kv['run'] != '0': continue
        out[(int(kv['depth']), int(kv['level']))] = (kv['n'], kv['nnz'], kv['rounds'], kv['twohop'])
    return out
a, b = lv(sys.argv[1]), lv(sys.argv[2])
keys = sorted(set(a) | set(b))
diff = [k for k in keys if a.get(k) != b.get(k)]
print(f'levels new={len(a)} old={len(b)} differing={len(diff)}', diff[:5], [(a.get(k), b.get(k)) for k in diff[:3]])
