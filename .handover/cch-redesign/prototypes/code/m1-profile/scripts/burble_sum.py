#!/usr/bin/env python3
"""Summarize GxB_BURBLE output per phase window and timed call: normalized message -> count, seconds.
usage: burble_sum.py LOG"""
import sys, re, collections
log = sys.argv[1]
cur_phase = None; cur_call = None
agg = collections.defaultdict(lambda: [0, 0.0])
buf = None
for line in open(log, errors='replace'):
    if line.startswith('@@BURBLE_BEGIN'):
        kv = dict(x.split('=', 1) for x in line.split()[1:])
        cur_phase = f"{kv['phase']}_d{kv['depth']}" if kv['phase'] != 'r0' else 'r0'
        cur_call = None; continue
    if line.startswith('@@BURBLE_END'):
        cur_phase = None; continue
    if line.startswith('@@CALL'):
        cur_call = line.split()[1]; continue
    if cur_phase is None:
        continue
    s = line.strip()
    if s.startswith('[ '):
        buf = s[2:]
        if s.endswith(']'):
            pass
        continue
    if buf is not None and s.endswith('sec ]'):
        t = float(s.split()[0])
        msg = re.sub(r'\b\d+(\.\d+)?(e-?\d+)?\b', '#', buf)
        msg = re.sub(r'nthreads # ntasks #', 'nthreads # ntasks #', msg)
        agg[(cur_phase, cur_call or '(untimed)', msg)][0] += 1
        agg[(cur_phase, cur_call or '(untimed)', msg)][1] += t
        buf = None
        continue
    if buf is not None:
        buf += ' ' + s
phases = sorted({k[0] for k in agg})
for ph in phases:
    tot = sum(v[1] for k, v in agg.items() if k[0] == ph)
    print(f'=== {ph}: burble-timed {tot:.4f}s')
    rows = sorted(((k[1], k[2], v) for k, v in agg.items() if k[0] == ph), key=lambda x: -x[2][1])
    for call, msg, (n, t) in rows:
        if t < 0.0005 * max(tot, 1e-9) and n < 50:
            continue
        print(f'  {call:13} n={n:4d} {t:8.4f}s {100*t/tot:5.1f}%  {msg[:200]}')
