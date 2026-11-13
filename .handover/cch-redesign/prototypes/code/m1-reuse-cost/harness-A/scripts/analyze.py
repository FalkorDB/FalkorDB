"""Parse m1_reuse logs: per-depth medians for R, B1, B2, totals vs feral, B2 call breakdown, shares.

usage: python3 -I analyze.py LOG [GRAPH_KEY_FOR_STRADDLERS_JSON] [--depths]
"""
import sys, statistics as st, collections as co, json

log = sys.argv[1]
gkey = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
show_depths = '--depths' in sys.argv
SJ = '/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1reuse/straddlers.json'

med = lambda xs: st.median(xs) if xs else float('nan')
dep = co.defaultdict(list)          # (th, d, method) -> [total]
calls = co.defaultdict(list)        # (th, d, method, call) -> [sec]
split = co.defaultdict(list)        # (th, d) -> [split]
extra = co.defaultdict(list)        # (th, d, method) -> [extra]
rlvl = co.defaultdict(lambda: co.defaultdict(list))   # (th,d) -> call -> per-run sums
rlvl_run = co.defaultdict(float)    # (th,run,d,call) -> sum
fer = []
loads = []
shares = {}
blvl = {}
ranout = []
labels = []
info = {}
for line in open(log):
    w = line.split()
    if not w:
        continue
    if w[0] == 'FERAL':
        fer.append(float(w[2].split('=')[1]))
    if w[0] == 'LOAD':
        loads.append(float(w[-1].split('=')[1]))
    if w[0] == 'LABELS':
        labels.append(line.strip())
    if w[0] == 'RANOUT':
        ranout.append(line.strip())
    if w[0] == 'SHARE':
        kv = dict(x.split('=') for x in w[1:])
        shares[(int(kv['depth']), int(kv['level']))] = (int(kv['n']), int(kv['strad']))
    if w[0] not in ('DEPTHX', 'RLVL', 'BLVL'):
        continue
    kv = {}
    for x in w[1:]:
        k, v = x.split('=', 1)
        kv[k] = v
    th, run, d = int(kv['threads']), int(kv['run']), int(kv['depth'])
    if w[0] == 'DEPTHX':
        m = kv['method']
        dep[(th, d, m)].append(float(kv['total']))
        split[(th, d)].append(float(kv['split']))
        extra[(th, d, m)].append(float(kv['extra']))
        if m == 'R':
            info[d] = kv
        for k, v in kv.items():
            if k not in ('threads', 'run', 'depth', 'method', 'total', 'split', 'extra', 'levels', 'wall', 'cells', 'live', 'cap', 'coarsest'):
                calls[(th, d, m, k)].append(float(v))
    elif w[0] == 'RLVL':
        for k, v in kv.items():
            if k in ('threads', 'run', 'depth', 'level', 'n', 'nnz', 'rounds', 'twohop'):
                continue
            rlvl_run[(th, run, d, k)] += float(v)
    elif w[0] == 'BLVL' and run == 0:
        blvl[(th, d, kv['method'], int(kv['level']))] = kv

ths = sorted({k[0] for k in dep})
depths = sorted({k[1] for k in dep})
print(f'log: {log}')
print(f'feral runs {fer} median {med(fer):.3f}s; LOAD lines min {min(loads) if loads else 0:.2f} max {max(loads) if loads else 0:.2f}')
for th in ths:
    tot = co.defaultdict(float)
    rows = []
    for d in depths:
        r = med(dep[(th, d, 'R')]); b1 = med(dep[(th, d, 'B1')]); b2 = med(dep[(th, d, 'B2')]); sp = med(split[(th, d)])
        nr = len(dep[(th, d, 'R')])
        tot['R'] += r; tot['B1'] += b1; tot['B2'] += b2; tot['split'] += sp
        rows.append((d, r, b1, b2, sp, nr))
    print(f'\n## threads={th}: totals over {len(depths)} depths (median of runs per depth, summed)')
    print(f"R {tot['R']:.3f}s  B1 {tot['B1']:.3f}s  B2 {tot['B2']:.3f}s  feral {med(fer):.3f}s  (split, in every path: {tot['split']:.3f}s)")
    print(f"ratios to feral: R {tot['R']/med(fer):.2f}  B1 {tot['B1']/med(fer):.2f}  B2 {tot['B2']/med(fer):.2f}; B1/R {tot['B1']/tot['R']:.2f} B2/R {tot['B2']/tot['R']:.2f}")
    # per-run totals, to show the spread
    for m in ('R', 'B1', 'B2'):
        per = [sum(dep[(th, d, m)][k] for d in depths) for k in range(min(len(dep[(th, d, m)]) for d in depths))]
        print(f'  {m} per-run totals: ' + ' '.join(f'{x:.3f}' for x in per))
    if show_depths:
        print('| depth | cells | live | R | B1 | B2 | split | runs |')
        for d, r, b1, b2, sp, nr in rows:
            i = info.get(d, {})
            print(f"| {d} | {i.get('cells', 1)} | {i.get('live', '')} | {r:.3f} | {b1:.3f} | {b2:.3f} | {sp:.3f} | {nr} |")
    # call breakdown over all depths >= 1
    for m in ('B1', 'B2'):
        ks = sorted({k[3] for k in calls if k[0] == th and k[2] == m})
        br = {k: sum(med(calls[(th, d, m, k)]) for d in depths if d >= 1 and calls[(th, d, m, k)]) for k in ks}
        print(f'  {m} calls, depths>=1 (s): ' + ' '.join(f'{k}={v:.3f}' for k, v in sorted(br.items(), key=lambda x: -x[1])))
    rk = sorted({k[3] for k in rlvl_run if k[0] == th})
    runs = sorted({k[1] for k in rlvl_run if k[0] == th})
    rb = {k: sum(med([rlvl_run[(th, r, d, k)] for r in runs]) for d in depths if d >= 1) for k in rk}
    rx = sum(med(extra[(th, d, 'R')]) for d in depths if d >= 1)
    print(f'  R calls, depths>=1 (s): compact={rx:.3f} ' + ' '.join(f'{k}={v:.3f}' for k, v in sorted(rb.items(), key=lambda x: -x[1])))

# shares: ours vs the quality stage
def ours_by_depth():
    byd = co.defaultdict(list)
    for (d, l), (n, s) in shares.items():
        byd[d].append((l, n, s))
    out = {}
    for d, rs in byd.items():
        rs.sort()
        n = sum(r[1] for r in rs); s = sum(r[2] for r in rs)
        l1 = [r for r in rs if r[0] == 1]
        out[d] = (100 * l1[0][2] / l1[0][1] if l1 else float('nan'), 100 * s / n if n else float('nan'))
    return out

o = ours_by_depth()
q = {}
if gkey:
    J = json.load(open(SJ))
    byd = co.defaultdict(list)
    for r in J['graphs'][gkey]['summed']:
        byd[r['depth']].append(r)
    for d, rs in byd.items():
        rs.sort(key=lambda r: r['level'])
        n = sum(r['n'] for r in rs); s = sum(r['strad_side'] for r in rs)
        l1 = [r for r in rs if r['level'] == 1 and r['n']]
        q[d] = (100 * l1[0]['strad_side'] / l1[0]['n'] if l1 else float('nan'), 100 * s / n if n else float('nan'))
print('\n## straddler share (split depth: level 1 / over needed levels), ours vs quality stage' + (f' ({gkey})' if gkey else ''))
print('| split depth | ours l1 | quality l1 | ours needed | quality needed | ratio needed |')
for d in sorted(set(o) | set(q)):
    a = o.get(d, (float('nan'),) * 2); b = q.get(d, (float('nan'),) * 2)
    print(f'| {d} | {a[0]:.4f} % | {b[0]:.4f} % | {a[1]:.4f} % | {b[1]:.4f} % | {a[1]/b[1] if b[1] else float("nan"):.2f} |')
for x in ranout:
    print(x)
