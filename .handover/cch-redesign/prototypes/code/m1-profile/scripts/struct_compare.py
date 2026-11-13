#!/usr/bin/env python3
"""Compare the structural (non-timing) output of two m1_reuse logs, for the given threads/run.
usage: struct_compare.py A.log B.log [threads...]   (run 0 of each thread count; B may lack B1)"""
import sys, re
TIMING = re.compile(r'^[A-Za-z_\']+=[0-9.]+$')
STRUCT_KEYS = {'RLVL': ['depth', 'level', 'n', 'nnz', 'rounds', 'twohop'],
               'BLVL': ['depth', 'method', 'level', 'dim', 'live', 'nnz', 'par', 'mixed', 'seponly', 'pieces', 'extras', 'rin', 'xprev'],
               'DEPTHX': ['depth', 'method', 'levels', 'cells', 'live', 'cap', 'coarsest']}
def rows(path, th, skip_b1):
    out = []
    for line in open(path, errors='replace'):
        tag = line.split(' ', 1)[0]
        if tag in ('DEPTH0', 'LABELS', 'SHARE', 'RANOUT'):
            if tag == 'RANOUT' and skip_b1:
                continue
            out.append(line.strip())
            continue
        if tag in STRUCT_KEYS:
            kv = {}
            for x in line.split()[1:]:
                if '=' in x:
                    k, v = x.split('=', 1)
                    kv.setdefault(k, v)
            if kv.get('threads') != th or kv.get('run') != '0':
                continue
            if skip_b1 and kv.get('method') == 'B1':
                continue
            if tag == 'DEPTHX' and kv.get('method') in ('B1', 'B2') and kv.get('depth') == '0':
                continue
            out.append(tag + ' ' + ' '.join(f'{k}={kv.get(k)}' for k in STRUCT_KEYS[tag]))
    return out
a, b = sys.argv[1], sys.argv[2]
ths = sys.argv[3:] or ['1', '4']
skip_b1 = not any(l.startswith('BLVL') and ' method=B1 ' in l for l in open(b, errors='replace'))
bad = 0
for th in ths:
    ra, rb = rows(a, th, skip_b1), rows(b, th, skip_b1)
    if ra != rb:
        bad += 1
        print(f'threads={th}: DIFFER ({len(ra)} vs {len(rb)} rows)')
        for x, y in zip(ra, rb):
            if x != y:
                print('  A:', x); print('  B:', y); break
    else:
        print(f'threads={th}: identical ({len(ra)} structural rows)')
sys.exit(1 if bad else 0)
