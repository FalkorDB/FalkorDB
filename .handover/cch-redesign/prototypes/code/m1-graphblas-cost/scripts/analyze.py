import sys, statistics as st, collections as co
lv = co.defaultdict(list)      # (th,d,l,call) -> [sec]
info = {}                      # (d,l) -> (n,nnz,rounds,twohop)
dep = co.defaultdict(list)     # (th,d) -> [(levels,wall,split)]
spl = co.defaultdict(list)
fer = []
calls_seen = []
for line in open(sys.argv[1]):
    w = line.split()
    if not w: continue
    if w[0] == 'FERAL': fer.append(float(w[2].split('=')[1]))
    if w[0] not in ('LVL','DEPTH','SPLIT'): continue
    kv = {}
    for x in w[1:]:
        k,v = x.split('=',1)
        if k in kv: k = k + '_t'
        kv[k] = v
    th = int(kv['threads']); run = int(kv['run'])
    if w[0] == 'LVL':
        d, l = int(kv['depth']), int(kv['level'])
        info[(d,l)] = (int(kv['n']), int(kv['nnz']), int(kv['rounds']), int(kv['twohop']))
        tot = 0
        for k,v in kv.items():
            if k in ('threads','run','depth','level','n','nnz','rounds','twohop'): continue
            lv[(th,d,l,k)].append(float(v)); tot += float(v)
            if k not in calls_seen: calls_seen.append(k)
        lv[(th,d,l,'TOTAL')].append(tot)
    elif w[0] == 'DEPTH':
        dep[(th,int(kv['depth']))].append((int(kv['levels']), float(kv['wall']), float(kv['split'])))
    else:
        spl[(th,kv['label'])].append((int(kv['kept']), float(kv['buildM']), float(kv['mxm'])))
med = lambda xs: st.median(xs) if xs else 0.0
ths = sorted({k[0] for k in lv})
order = ['cap','key','idx','mxv','m_unkey','m_full','m_extract','m_eq','m_assign','m_unext','twohop_t','c_leader','c_number_host','c_extract','c_build','CA',"CAC'",'offdiag','Cw','project','bnd']
calls = [c for c in order if c in calls_seen] + [c for c in calls_seen if c not in order]
print('feral runs', fer, 'median', med(fer))
for th in ths:
    print(f'\n## depth 0, threads={th}: per-level median ms (runs={len(dep[(th,0)])})')
    print('| lvl | n | nnz | rnd | 2hop | ' + ' | '.join(calls) + ' | total |')
    L = sorted(l for (d,l) in info if d == 0)
    colsum = co.defaultdict(float)
    for l in L:
        n,nnz,r,t = info[(0,l)]
        cells = []
        for c in calls + ['TOTAL']:
            m = med(lv.get((th,0,l,c), [])) * 1e3
            colsum[c] += m
            cells.append(f'{m:.2f}' if (th,0,l,c) in lv else '-')
        print(f'| {l} | {n} | {nnz} | {r} | {t} | ' + ' | '.join(cells) + ' |')
    print('| sum | | | | | ' + ' | '.join(f'{colsum[c]:.1f}' for c in calls + ['TOTAL']) + ' |')
for th in ths:
    print(f'\n## per depth, threads={th} (median over runs; seconds)')
    print('| depth | levels | coarsen+project (sum of timed calls) | wall | split |')
    tot_calls = tot_wall = tot_split = 0
    for d in sorted({k[1] for k in dep if k[0]==th}):
        L = [l for (dd,l) in info if dd == d]
        sums = []
        runs = len(dep[(th,d)])
        for r in range(runs):
            sums.append(sum(lv[(th,d,l,'TOTAL')][r] for l in L if len(lv[(th,d,l,'TOTAL')])>r))
        x = dep[(th,d)]
        s = med(sums); wl = med([y[1] for y in x]); sp = med([y[2] for y in x])
        tot_calls += s; tot_wall += wl; tot_split += sp
        print(f'| {d} | {x[0][0]} | {s:.3f} | {wl:.3f} | {sp:.3f} |')
    print(f'| all | | {tot_calls:.3f} | {tot_wall:.3f} | {tot_split:.3f} |')
    print(f'TOTAL threads={th}: calls+split={tot_calls+tot_split:.3f}s wall+split={tot_wall+tot_split:.3f}s vs feral {med(fer):.3f}s')
for k,v in sorted(spl.items()):
    print('SPLIT', k, 'kept', v[0][0], 'buildM ms', round(med([x[1] for x in v])*1e3,2), 'mxm ms', round(med([x[2] for x in v])*1e3,2))
