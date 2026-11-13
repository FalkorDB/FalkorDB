"""Sum the AUDIT probe lines per (threads, run), median over runs. Usage: probes.py LOG"""
import sys, statistics as st, collections as co
th = run = None; depth = None; lcur = 0
b2 = co.defaultdict(lambda: co.defaultdict(float)); sp = co.defaultdict(lambda: co.defaultdict(float)); lv = co.defaultdict(lambda: co.defaultdict(float))
b2d = co.defaultdict(lambda: co.defaultdict(float))
for line in open(sys.argv[1], errors='replace'):
    w = line.split()
    if not w: continue
    kv = dict(x.split('=', 1) for x in w[1:] if '=' in x)
    if w[0] == 'LOAD' and 'threads' in kv:
        th, run = int(kv['threads']), int(kv['run']); depth = 0; lcur = 0; continue
    if w[0] == 'DEPTHX' and kv.get('method') == 'R':
        depth = int(kv['depth']); lcur = depth + 1; continue
    if th is None: continue
    if w[0] == 'AUDITB2':
        k = (th, run)
        for f in ('official', 'probe', 'host', 'del_t', 'ins_t', 'wait_t'): b2[k][f] += float(kv[f])
        for f in ('nnzP', 'del', 'ins'): b2[k][f] += int(kv[f])
        b2[k]['levels'] += 1
        b2d[(th, run, depth)]['official'] += float(kv['official']); b2d[(th, run, depth)]['probe'] += float(kv['probe'])
    elif w[0] == 'AUDITSPLIT':
        k = (th, run)
        for f in ('official', 'probe', 'host', 'del_t', 'wait_t'): sp[k][f] += float(kv[f])
        for f in ('nnz_prev', 'del'): sp[k][f] += int(kv[f])
        sp[k]['depths'] += 1
    elif w[0] == 'AUDITL':
        k = (th, run, 'd0' if lcur == 0 else 'R>=1')
        for f, v in kv.items():
            if f[:2] in ('of', 'a_', 'b_', 'c_'): lv[k][f] += float(v)
        lv[k]['levels'] += 1
def med(d, keys, f):
    xs = [d[k][f] for k in keys]
    return st.median(xs), min(xs), max(xs)
for th in sorted({k[0] for k in b2}):
    ks = [k for k in b2 if k[0] == th]
    print(f'\n## threads={th}: B2 masked split (m_build+mask_mxm+mask_vals+resize+s_merge) vs in-place delete+insert, all depths >= 1, {len(ks)} runs')
    for f in ('official', 'probe', 'host', 'del_t', 'ins_t', 'wait_t'):
        m, lo, hi = med(b2, ks, f); print(f'  {f:9s} {m*1e3:9.1f} ms  [{lo*1e3:.1f}-{hi*1e3:.1f}]')
    print(f'  levels={b2[ks[0]]["levels"]:.0f} nnzP_sum={b2[ks[0]]["nnzP"]:.0f} deleted={b2[ks[0]]["del"]:.0f} inserted={b2[ks[0]]["ins"]:.0f}')
    ks2 = [k for k in sp if k[0] == th]
    print(f'## threads={th}: fine split (buildM+mxm) vs in-place delete, all depths, {len(ks2)} runs')
    for f in ('official', 'probe', 'host', 'del_t', 'wait_t'):
        m, lo, hi = med(sp, ks2, f); print(f'  {f:9s} {m*1e3:9.1f} ms  [{lo*1e3:.1f}-{hi*1e3:.1f}]')
    print(f'  depths={sp[ks2[0]]["depths"]:.0f} nnz_prev_sum={sp[ks2[0]]["nnz_prev"]:.0f} deleted={sp[ks2[0]]["del"]:.0f}')
    for part in ('d0', 'R>=1'):
        ks3 = [k for k in lv if k[0] == th and k[2] == part]
        if not ks3: continue
        print(f'## threads={th}: coarsen_level probes, {part}, {len(ks3)} runs, {lv[ks3[0]]["levels"]:.0f} levels')
        g = lambda f: med(lv, ks3, f)[0] * 1e3
        print(f'  match: official assign+unext = {g("off_assign")+g("off_unext"):.1f} (assign {g("off_assign"):.1f}, unext {g("off_unext"):.1f}); '
              f'select-true {g("a_sel_true"):.1f}, accum-assign {g("a_assign_accum"):.1f}, emult+add {g("a_emult_add"):.1f}, select-false (u pattern) {g("a_sel_false"):.1f}')
        off = g("off_CA") + g("off_CAC") + g("off_offdiag")
        print(f'  contract: official CA+CAC\'+offdiag = {off:.1f} (CA {g("off_CA"):.1f}, CAC\' {g("off_CAC"):.1f}, offdiag {g("off_offdiag"):.1f}); c_build {g("off_cbuild"):.1f}; Cw {g("off_Cw"):.1f}')
        print(f'    (a) CA no wait {g("a_CA_nowait"):.1f} + C<!I> {g("a_CAC_maskNotI"):.1f} = {g("a_CA_nowait")+g("a_CAC_maskNotI"):.1f}')
        print(f'    (b) CA no wait {g("b_CA_nowait"):.1f} + CAC\' {g("b_CAC"):.1f} + offdiag {g("b_offdiag"):.1f} = {g("b_CA_nowait")+g("b_CAC")+g("b_offdiag"):.1f}')
        hc = g("c_hostC_pack")+g("c_hostC_CA")+g("c_hostC_CAC")+g("c_hostC_offdiag")+g("c_hostC_Cw")
        print(f'    (c) host C pack {g("c_hostC_pack"):.1f} + CA {g("c_hostC_CA"):.1f} + CAC\' {g("c_hostC_CAC"):.1f} + offdiag {g("c_hostC_offdiag"):.1f} + Cw {g("c_hostC_Cw"):.1f} = {hc:.1f}  vs official incl. c_build+Cw = {off+g("off_cbuild")+g("off_Cw"):.1f}')
# per depth B2 (threads 1)
print('\n## B2 per depth, threads=1, median over runs (ms): official / in-place')
ds = sorted({k[2] for k in b2d if k[0] == 1})
for d in ds:
    ks = [k for k in b2d if k[0] == 1 and k[2] == d]
    o = st.median(b2d[k]['official'] for k in ks); p = st.median(b2d[k]['probe'] for k in ks)
    print(f'  depth {d:2d}: {o*1e3:7.1f} / {p*1e3:7.1f}')
