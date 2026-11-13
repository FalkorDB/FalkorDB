#!/usr/bin/env python3
"""DERIVED: instructions by partition = per-call measured instructions (CALLRU median) x that call's
sample share per partition (compute-thread wall at 1 thread, all-thread CPU at 4 threads).
usage: derive_instr.py CALLRU_LOG SUMMARY_T1.json SUMMARY_T4.json"""
import sys, json, collections, statistics
log, s1, s4 = sys.argv[1:4]
acc = collections.defaultdict(lambda: collections.defaultdict(float))
for line in open(log, errors='replace'):
    if not line.startswith('CALLRU '):
        continue
    kv = dict(x.split('=', 1) for x in line.split()[1:])
    g = kv['name'] if kv['name'] in ('r0', 'keep0', 'warm0') else kv['name'] + '_d>=1'
    acc[(g, kv['threads'], kv['call'])][int(kv['run'])] += float(kv['instr'])
instr = {k: statistics.median(v.values()) for k, v in acc.items()}
PARTS = ['kernel', 'gb_wait', 'convert', 'builder', 'mem', 'alloc', 'sort', 'host', 'omp', 'other']
for th, path, key in (('1', s1, 'wall_call_part'), ('4', s4, 'cpu_call_part')):
    S = json.load(open(path))
    for g in ('r0', 'r_d>=1', 'b2_d>=1', 'split_d>=1'):
        cp = S[g][key]
        per_call = collections.defaultdict(dict)
        for k, v in cp.items():
            c, p = k.split('|')
            per_call[c][p] = v
        tot = collections.defaultdict(float)
        covered = 0.0
        for (gg, t, c), ins in instr.items():
            if gg != g or t != th:
                continue
            d = per_call.get(c)
            if not d:
                tot['(no samples)'] += ins
                continue
            s = sum(d.values())
            for p, v in d.items():
                tot[p] += ins * v / s
            covered += ins
        T = sum(tot.values())
        print(f'{g:11} thr={th} timed-call instr {T/1e9:7.2f}G  ' + '  '.join(f'{p}={100*tot.get(p,0)/T:.1f}%' for p in PARTS + ['(no samples)'] if tot.get(p, 0)))
