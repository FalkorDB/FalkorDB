#!/usr/bin/env python3
"""Per-phase rusage from PHASE lines: median and spread over runs, per thread count.
usage: ru_table.py LOG [--depths]"""
import sys, collections, statistics
log = sys.argv[1]
per_depth = '--depths' in sys.argv
F = ('wall', 'instr', 'cycles', 'user', 'sys', 'runnable')
acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
dep = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
for line in open(log, errors='replace'):
    if not line.startswith('PHASE '):
        continue
    kv = dict(x.split('=', 1) for x in line.split()[1:])
    name, th, run, d = kv['name'], kv['threads'], int(kv['run']), int(kv['depth'])
    g = name if name in ('r0', 'keep0', 'warm0', 'labels', 'feral') else name + '_d>=1'
    for f in F:
        acc[(g, th)][run][f] += float(kv[f])
        if name in ('r', 'b2', 'split', 'b1'):
            dep[(name, th, d)][run][f] += float(kv[f])
def med(xs):
    return statistics.median(xs)
print(f"{'phase':12} {'thr':>3} {'wall s':>16} {'instr G':>18} {'cycles G':>18} {'IPC':>5} {'cpu s':>7} {'sys s':>6} {'runnable s':>10} {'cpu/wall':>8}")
for (g, th), runs in sorted(acc.items(), key=lambda x: (x[0][1], x[0][0])):
    rs = sorted(runs)
    v = {f: [runs[r][f] for r in rs] for f in F}
    w, i, c = v['wall'], v['instr'], v['cycles']
    cpu = [runs[r]['user'] + runs[r]['sys'] for r in rs]
    sp = lambda xs, s=1.0: f"{med(xs)/s:.3f} [{min(xs)/s:.3f}-{max(xs)/s:.3f}]"
    print(f"{g:12} {th:>3} {sp(w):>16} {sp(i, 1e9):>18} {sp(c, 1e9):>18} {med(i)/med(c):5.2f} {med(cpu):7.3f} {med(v['sys']):6.3f} {med(v['runnable']):10.3f} {med(cpu)/med(w):8.2f}")
if per_depth:
    print()
    print('per depth (median of runs): name thr depth wall instrG cyclesG cpu')
    for (name, th, d), runs in sorted(dep.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        rs = sorted(runs)
        w = med([runs[r]['wall'] for r in rs]); i = med([runs[r]['instr'] for r in rs]); c = med([runs[r]['cycles'] for r in rs])
        cpu = med([runs[r]['user'] + runs[r]['sys'] for r in rs])
        print(f"{name:6} {th:>2} {d:3} {w:8.4f} {i/1e9:8.4f} {c/1e9:8.4f} {cpu:8.4f}")
