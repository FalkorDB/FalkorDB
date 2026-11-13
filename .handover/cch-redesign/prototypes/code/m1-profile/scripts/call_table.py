#!/usr/bin/env python3
"""Per timed call, per phase group: median over runs of calls, wall, instructions, cycles, CPU.
usage: call_table.py LOG"""
import sys, collections, statistics
log = sys.argv[1]
acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
for line in open(log, errors='replace'):
    if not line.startswith('CALLRU '):
        continue
    kv = dict(x.split('=', 1) for x in line.split()[1:])
    name, th, run = kv['name'], kv['threads'], int(kv['run'])
    g = name if name in ('r0', 'keep0', 'warm0') else name + '_d>=1'
    c = kv['call']
    for f in ('n', 'wall', 'instr', 'cycles', 'cpu'):
        acc[(g, th, c)][run][f] += float(kv[f])
med = statistics.median
groups = collections.defaultdict(list)
for (g, th, c), runs in acc.items():
    rs = sorted(runs)
    m = {f: med([runs[r][f] for r in rs]) for f in ('n', 'wall', 'instr', 'cycles', 'cpu')}
    m['iw'] = (min(runs[r]['instr'] for r in rs), max(runs[r]['instr'] for r in rs))
    groups[(g, th)].append((c, m))
for (g, th), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
    if g in ('warm0',):
        continue
    tw = sum(m['wall'] for _, m in rows); ti = sum(m['instr'] for _, m in rows)
    print(f'== {g} threads={th}: timed calls wall={tw:.3f}s instr={ti/1e9:.2f}G')
    print(f"   {'call':14} {'calls':>7} {'wall s':>8} {'wall%':>6} {'instr G':>8} {'instr%':>6} {'IPC':>5} {'cpu s':>7} {'cpu/wall':>8} {'instr/call K':>12}")
    for c, m in sorted(rows, key=lambda x: -x[1]['wall']):
        ipc = m['instr'] / m['cycles'] if m['cycles'] else 0
        print(f"   {c:14} {m['n']:7.0f} {m['wall']:8.3f} {100*m['wall']/tw:6.1f} {m['instr']/1e9:8.3f} {100*m['instr']/ti:6.1f} {ipc:5.2f} {m['cpu']:7.3f} {m['cpu']/m['wall'] if m['wall'] else 0:8.2f} {m['instr']/m['n']/1e3:12.1f}")
