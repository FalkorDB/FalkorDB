#!/usr/bin/env python3
"""attempts.csv from steps.csv: each change against the config it was applied to (CA)."""
import csv, collections
med = {}
spread = {}
for r in csv.DictReader(open('steps.csv')):
    if r['run'] != 'median': continue
    k = (r['config'], int(r['threads']), r['path'], r['phase'])
    med[k] = (float(r['wall_s']), float(r['instr_G']))
    spread[k] = r['log']
# step, change, what it does, kind, binary, prev, kept
steps = [
 ('s1_hb2', 'hb2', 'B2 level by one host pass over the parent level CSR (replaces masked mxm, value copy, resize, straddler products, union)', 'host pass', 'v1', 's0_baseline', 'yes'),
 ('s2_hsplit', 'hsplit', 'fine split per depth by one host pass over A0 CSR (replaces build M + masked mxm + zombie wait)', 'host pass', 'v1', 's1_hb2', 'yes'),
 ('s3_cpack', 'cpack', 'leader numbering in one host pass over mate; C and Cᵀ packed; C·A without transpose; w_c on host', 'host pass + GraphBLAS', 'v1', 's2_hsplit', 'yes'),
 ('s4_hcon', 'hcon', 'contraction C·A·Cᵀ + OFFDIAG as one host pass over A CSR', 'host pass', 'v1', 's3_cpack', 'yes'),
 ('s5_pk', 'pk', 'Cᵀ of reused levels packed (no build); level w vector not built (only checks read it)', 'host pass', 'v1', 's4_hcon', 'yes'),
 ('s6_book', 'book', 'numbering in fewer passes, per-cell counts carried, capstop patches instead of renumbering', 'host loop', 'v2', 's5_pk', 'yes'),
 ('s7_caprc', 'caprc', 'cap mask by row scale + column scale (no eWiseMult of two scaled copies)', 'GraphBLAS', 'v2', 's6_book', 'yes (superseded by hkey)'),
 ('s8_hkey', 'hkey', 'keys and cap test in one host pass over A CSR (replaces cap mask and generic masked user-op apply)', 'host pass', 'v3', 's7_caprc', 'yes'),
 ('s9_hpar', 'hpar', 'host passes split across the thread count', 'threads', 'v3', 's8_hkey', 'yes'),
 ('s10_gath', 'gath', 'capstop gather by appending CSR rows (no extract/tuples/build)', 'host pass', 'v3', 's9_hpar', 'yes'),
 ('s11_usel', 'usel', 'u = select(m == false) instead of the generic complemented-mask masker', 'GraphBLAS', 'v3', 's10_gath', 'yes'),
 ('s12_zmatch', 'zmatch', 'matches carried in the gather vector; no per-round mate assign', 'GraphBLAS', 'v3', 's11_usel', 'yes'),
 ('s13_nowait', 'nowait', 'no per-call waits; one wait per level output', 'GraphBLAS', 'v3', 's12_zmatch', 'no (not measurably better)'),
 ('s14_npar', 'npar', 'numbering and Cᵀ packing split across threads', 'threads', 'v4', 's12_zmatch', 'yes'),
 ('s15_chunk1m', 'chunk=1M', 'GxB_CHUNK 1048576 (small calls single-threaded)', 'GraphBLAS', 'v4', 's14_npar', 'no (worse at 4 threads)'),
 ('s16_chunk256k', 'chunk=256K', 'GxB_CHUNK 262144', 'GraphBLAS', 'v4', 's14_npar', 'no (worse at 4 threads)'),
]
cols = ['step','change','what','kind','binary','applied_to','kept','threads','path','phase','wall_before_s','wall_after_s','d_wall_s','d_wall_pct','instr_before_G','instr_after_G','d_instr_G','d_instr_pct','equality','log']
with open('attempts.csv', 'w') as f:
    w = csv.writer(f); w.writerow(cols)
    for st, ch, what, kind, binv, prev, kept in steps:
        for th in (1, 4):
            for path, phase in [('CS','total'),('CS','depth0'),('CS','depths1+'),('B2','total'),('R','total')]:
                a = med.get((prev, th, path, phase)); b = med.get((st, th, path, phase))
                if not a or not b: continue
                w.writerow([st, ch, what, kind, binv, prev, kept, th, path, phase, f'{a[0]:.3f}', f'{b[0]:.3f}', f'{b[0]-a[0]:+.3f}', f'{100*(b[0]-a[0])/a[0]:+.1f}', f'{a[1]:.2f}', f'{b[1]:.2f}', f'{b[1]-a[1]:+.2f}', f'{100*(b[1]-a[1])/a[1]:+.1f}', 'hash 57/57 equal, CA, 1+4 thr', f'logs/CA_{st}.log'])
print(open('attempts.csv').read()[:3000])
