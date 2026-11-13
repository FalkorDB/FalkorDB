#!/usr/bin/env python3
"""tot.py LOG...: per (log, threads, path) the totals over timed runs (run >= 1):
R = D0 + sum_d(SPLIT + R); B2 = D0 + ROOTMAP + sum_d(SPLIT + B2); CS = D0 + ROOTMAP + sum_d(SPLIT + CS).
Columns: wall (sum of timed calls), pwall (phase wall clock), instr, cycles; median [min-max] of the runs,
plus the 1-minute load at each run start and the gap pwall - wall."""
import sys, re, statistics, collections

PATHS = {'R': (['D0'], ['SPLIT', 'R']), 'B2': (['D0', 'ROOTMAP'], ['SPLIT', 'B2']), 'CS': (['D0', 'ROOTMAP'], ['SPLIT', 'CS'])}
for log in sys.argv[1:]:
    data = collections.defaultdict(lambda: collections.defaultdict(dict))
    load = {}
    feral = []
    for line in open(log):
        m = re.match(r'LOAD threads=(\d+) run=(\d+) .*load averages: ([\d.]+)', line)
        if m:
            load[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
        m = re.match(r'FERAL run=(\d+) sec=([\d.]+) instr=(\d+)', line)
        if m:
            feral.append((float(m.group(2)), int(m.group(3))))
        if not line.startswith('PHASE '):
            continue
        kv = dict(re.findall(r'(\w+)=([^\s]+)', line.split(' c:')[0]))
        th, run, d, p = int(kv['threads']), int(kv['run']), int(kv['depth']), kv['path']
        data[(th, run)][(d, p)] = (float(kv['wall']), float(kv['pwall']), int(kv['instr']), int(kv['cyc']))
    print(f'== {log.split("/")[-1]}')
    if feral:
        print(f'   feral runs: ' + ' '.join(f'{s:.3f}s/{i/1e9:.2f}G' for s, i in feral))
    for th in sorted({k[0] for k in data}):
        runs = sorted(r for (t, r) in data if t == th and r >= 1)
        for path, (p0, p1) in PATHS.items():
            vals = []
            for r in runs:
                dd = data[(th, r)]
                w = pw = i = c = 0
                for (d, p), v in dd.items():
                    if (d == 0 and p in p0) or (d >= 1 and p in p1):
                        w += v[0]; pw += v[1]; i += v[2]; c += v[3]
                vals.append((w, pw, i / 1e9, c / 1e9))
            col = lambda k: [v[k] for v in vals]
            fmt = lambda xs, f: f'{statistics.median(xs):{f}} [{min(xs):{f}}-{max(xs):{f}}]'
            print(f'   th={th} {path:3} wall {fmt(col(0), ".3f")}  pwall {fmt(col(1), ".3f")}  instr {fmt(col(2), ".2f")}G  cyc {fmt(col(3), ".2f")}G  load1 ' + ','.join(f'{load.get((th, r), float("nan")):.2f}' for r in runs))
