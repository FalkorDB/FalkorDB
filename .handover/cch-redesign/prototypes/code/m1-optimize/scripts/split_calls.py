#!/usr/bin/env python3
"""split_calls.py LOG: capstop (D0 + ROOTMAP + SPLIT + CS) per call, median over timed runs, grouped
into host passes, GraphBLAS calls and the uncoarsening stand-ins (project, bnd)."""
import sys, re, collections, statistics
log = sys.argv[1]
opts = re.search(r'^M1O (.*?) M1O_BREAK', open(log).read(), re.M).group(1)
on = set(re.findall(r'\w+', opts)) if opts != 'None' else set()
host_names = {'b2_splice', 'number', 'hcon', 'c_host', 'split_host', 'rootmap', 'gfresh' if 'gath' in on else None, 'key' if 'hkey' in on else None, 'ct_build' if 'pk' in on else None}
stand = {'project', 'bnd'}
per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
for line in open(log):
    if not line.startswith('PHASE '): continue
    m = re.match(r'PHASE threads=(\d+) run=(\d+) depth=(\d+) path=(\w+)', line)
    th, run, d, p = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    if run == 0 or p not in ('D0', 'ROOTMAP', 'SPLIT', 'CS'): continue
    for n, w, i in re.findall(r' c:([^=]+)=([\d.]+)/(\d+)', line):
        e = per[(th, run)][n]; e[0] += float(w); e[1] += int(i)
for th in sorted({k[0] for k in per}):
    runs = [r for (t, r) in per if t == th]
    names = set().union(*[per[(th, r)].keys() for r in runs])
    med = {n: (statistics.median(per[(th, r)][n][0] for r in runs), statistics.median(per[(th, r)][n][1] for r in runs)) for n in names}
    g = collections.defaultdict(lambda: [0.0, 0])
    for n, (w, i) in med.items():
        k = 'stand-in' if n in stand else 'host' if n in host_names else 'graphblas'
        g[k][0] += w; g[k][1] += i
    tot = sum(v[0] for v in g.values())
    print(f'threads={th} total={tot:.3f}s ' + ' '.join(f'{k}={v[0]:.3f}s/{v[1]/1e9:.2f}G' for k, v in sorted(g.items())))
    print('   ' + ' '.join(f'{n}={w:.3f}' for n, (w, i) in sorted(med.items(), key=lambda x: -x[1][0]) if w >= 0.005))
