"""Per-call totals per (depth, method), median over runs, from RLVL/BLVL/DEPTHX lines.
Usage: percall.py LOG [LOG2] [threads]  -> prints table; with two logs, side by side + ratio,
and checks that the hierarchies (per-level n/nnz/rounds, B dims/nnz) are identical."""
import sys, re, statistics as st, collections as co
def parse(path, th):
    calls = co.defaultdict(lambda: co.defaultdict(float))  # (run) -> {(d,meth,call): s}
    shape = {}
    for line in open(path, errors='replace'):
        w = line.split()
        if not w or w[0] not in ('RLVL', 'BLVL'): continue
        kv = {}
        for x in w[1:]:
            if '=' in x:
                k, v = x.split('=', 1); kv[k] = v
        if int(kv['threads']) != th: continue
        run = int(kv['run']); d = int(kv['depth'])
        meth = 'R' if w[0] == 'RLVL' else kv['method']
        skip = {'threads','run','depth','level','n','nnz','rounds','twohop','method','dim','live','par','mixed','seponly','pieces','extras','rin','xprev'}
        for k, v in kv.items():
            if k in skip: continue
            calls[run][(d, meth, k)] += float(v)
        if run == 0:
            if w[0] == 'RLVL': shape[('R', d, int(kv['level']))] = (kv['n'], kv['nnz'], kv['rounds'])
            else: shape[(meth, d, int(kv['level']))] = (kv['dim'], kv['live'], kv['nnz'])
    keys = set(k for r in calls.values() for k in r)
    med = {k: st.median([calls[r].get(k, 0.0) for r in calls]) for k in keys}
    spread = {k: (min(calls[r].get(k, 0.0) for r in calls), max(calls[r].get(k, 0.0) for r in calls)) for k in keys}
    return med, spread, shape, len(calls)
th = int(sys.argv[3]) if len(sys.argv) > 3 else 1
a, sa, sha, ra = parse(sys.argv[1], th)
if len(sys.argv) > 2 and sys.argv[2] != '-':
    b, sb, shb, rb = parse(sys.argv[2], th)
    print(f'# runs: {ra} vs {rb}; hierarchy identical: {sha == shb} ({len(sha)} level shapes)')
    tot = co.defaultdict(lambda: [0.0, 0.0])
    for k in sorted(set(a) | set(b), key=lambda k: (k[0], k[1], -a.get(k, 0))):
        x, y = a.get(k, 0.0), b.get(k, 0.0)
        tot[(k[0], k[1])][0] += x; tot[(k[0], k[1])][1] += y
        if max(x, y) < 0.0005: continue
        print(f'd={k[0]} {k[1]:3s} {k[2]:15s} A={x*1e3:8.2f} [{sa.get(k,(0,0))[0]*1e3:.1f}-{sa.get(k,(0,0))[1]*1e3:.1f}]  B={y*1e3:8.2f} [{sb.get(k,(0,0))[0]*1e3:.1f}-{sb.get(k,(0,0))[1]*1e3:.1f}]  B/A={y/x if x else float("nan"):.2f}')
    for k, (x, y) in sorted(tot.items()):
        print(f'TOTAL d={k[0]} {k[1]:3s} A={x*1e3:.1f} B={y*1e3:.1f} B/A={y/x if x else float("nan"):.3f}')
else:
    for k in sorted(a, key=lambda k: (k[0], k[1], -a[k])):
        print(f'd={k[0]} {k[1]:3s} {k[2]:15s} {a[k]*1e3:8.2f} [{sa[k][0]*1e3:.1f}-{sa[k][1]*1e3:.1f}]')
