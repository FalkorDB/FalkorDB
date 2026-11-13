"""Parse m1rcB m1_reuse logs. usage: python3 -I analyze.py LOG [CA|PA|TX] [--depths] [--calls]"""
import sys, statistics as st, collections as co, json
log = sys.argv[1]
gkey = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
SJ = '/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1reuse/straddlers.json'
med = lambda xs: st.median(xs) if xs else float('nan')
tot = co.defaultdict(list)    # (th, d, var) -> [calls+split] per run
calls = co.defaultdict(list)  # (th, var, call) -> per-run sums over d>=1
callrun = co.defaultdict(float)
info = {}                     # (d, var) -> fields of run 0 / threads first
fer, loads, strad = [], [], []
runs_seen = set()
for line in open(log):
    w = line.split()
    if not w: continue
    if w[0] == 'FERAL': fer.append(float(w[2].split('=')[1]))
    if w[0] == 'LOAD' and w[1].startswith('threads'):
        loads.append(float(line.split('averages:')[1].split()[0]))
    if w[0] == 'STRAD':
        strad.append({k: int(v) for k, v in (x.split('=') for x in w[1:])})
    if w[0] != 'VAR': continue
    kv = dict(x.split('=', 1) for x in w[1:])
    th, run, d, var = int(kv['threads']), int(kv['run']), int(kv['depth']), kv['var']
    runs_seen.add((th, run))
    t = float(kv['calls']) + float(kv['split'])
    tot[(th, d, var, run)].append(t)
    if d >= 1:
        for k, v in kv.items():
            if k.startswith('c:'):
                callrun[(th, var, k[2:], run)] += float(v)
    info.setdefault((th, d, var), kv)
ths = sorted({t for t, _ in runs_seen})
runs = sorted({r for _, r in runs_seen})
depths = sorted({k[1] for k in tot})
F = med(fer)
print(f'feral runs {fer} median {F:.3f}s; LOAD at run starts min {min(loads):.2f} max {max(loads):.2f}')
def depth_med(th, d, var):
    if d == 0:
        r = [tot[(th, 0, 'R', run)][0] + (tot[(th, 0, 'ROOTMAP', run)][0] if var != 'R' else 0) for run in runs]
    else:
        r = [tot[(th, d, var, run)][0] for run in runs if (th, d, var, run) in tot]
    return med(r)
VARS = ['R', 'B1', 'B2', 'CS']
for th in ths:
    print(f'\n## threads={th}: totals (sum over depths of the per-depth median of {len(runs)} runs), seconds')
    sums = {v: sum(depth_med(th, d, v) for d in depths) for v in VARS}
    # per-run totals for spread
    spread = {v: [sum((tot[(th, d, 'R', run)][0] + (tot[(th, 0, 'ROOTMAP', run)][0] if v != 'R' else 0)) if d == 0 else tot[(th, d, v, run)][0] for d in depths) for run in runs] for v in VARS}
    print('| var | total | / feral | / R | per-run totals |')
    for v in VARS:
        print(f'| {v} | {sums[v]:.2f} | {sums[v]/F:.2f} | {sums[v]/sums["R"]:.2f} | {", ".join(f"{x:.2f}" for x in spread[v])} |')
    print(f'TOTALS threads={th} ' + ' '.join(f'{v}={sums[v]:.3f}' for v in VARS) + f' feral={F:.3f}')
if '--depths' in sys.argv:
    for th in ths:
        print(f'\n## per depth, threads={th} (median s, incl. split)')
        print('| d | live | R lv | R | B1 | B2 | CS | CS reused/fresh | CS coarsened | R coarsened | split |')
        for d in depths:
            r = info.get((th, d, 'R'), {}); c = info.get((th, d, 'CS'), {})
            sp = med([float(info.get((th, d, 'R'), {}).get('split', 0))])
            print(f"| {d} | {r.get('coarsest','')} | {r.get('levels','')} | {depth_med(th,d,'R'):.3f} | {depth_med(th,d,'B1'):.3f} | {depth_med(th,d,'B2'):.3f} | {depth_med(th,d,'CS'):.3f} | {c.get('reused','-')}/{c.get('fresh','-')} | {c.get('coarsened','-')} | {r.get('coarsened','')} | {sp:.3f} |")
# capstop fresh share (vertices coarsened after depth 0), from run 0 of the first thread count
th0 = ths[0]
rc = sum(int(info[(th0, d, 'R')]['coarsened']) for d in depths if d >= 1)
cc = sum(int(info[(th0, d, 'CS')]['coarsened']) for d in depths if d >= 1)
print(f'\nCAPSTOP fresh vertices coarsened after depth 0: {cc} of R {rc} = {100*cc/rc:.1f} %')
for d in depths:
    if d >= 1:
        rr = int(info[(th0, d, 'R')]['coarsened']); c = int(info[(th0, d, 'CS')]['coarsened'])
        print(f'  d={d}: CS {c} / R {rr} = {100*c/max(rr,1):.1f} %  cells_fresh={info[(th0,d,"CS")]["cells_fresh"]} plim={info[(th0,d,"CS")]["cells_plim"]} reused={info[(th0,d,"CS")]["reused"]} fresh_levels={info[(th0,d,"CS")]["fresh"]}')
if '--calls' in sys.argv:
    for th in ths:
        for v in VARS:
            ks = sorted({k[2] for k in callrun if k[0] == th and k[1] == v})
            row = sorted(((med([callrun[(th, v, k, run)] for run in runs]), k) for k in ks), reverse=True)
            print(f'\nCALLS threads={th} var={v} (depths>=1, median over runs): ' + ', '.join(f'{k}={x:.3f}' for x, k in row if x >= 0.005))
# straddler shares
byd = co.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
for r in strad:
    b = byd[r['depth'] - 1]   # split depth = the depth of the cell being split
    b[0] += r['strad']; b[1] += r['parent_live']; b[4] += r['rin']; b[5] += r['prev_nnz']
    if r['level'] == 1: b[2] += r['strad']; b[3] += r['parent_live']
q = {}
if gkey:
    J = json.load(open(SJ))['graphs'][gkey]['summed']
    for r in J:
        qq = q.setdefault(r['depth'], [0, 0, 0, 0])
        qq[0] += r['strad_side']; qq[1] += r['n']
        if r['level'] == 1: qq[2] += r['strad_side']; qq[3] += r['n']
print('\n## straddler share by split depth (this run: reused levels; quality: needed levels, 5 seeds)')
print('| split depth | level 1 | quality level 1 | all reused levels | quality needed | ratio | B2/B1 input |')
for d in sorted(byd):
    b = byd[d]; qq = q.get(d)
    s1 = 100 * b[2] / max(b[3], 1); sa = 100 * b[0] / max(b[1], 1)
    if qq:
        q1 = 100 * qq[2] / max(qq[3], 1); qa = 100 * qq[0] / max(qq[1], 1)
        print(f'| {d} | {s1:.3f} % | {q1:.3f} % | {sa:.3f} % | {qa:.3f} % | {sa/qa if qa else float("nan"):.1f} | {100*b[4]/max(b[5],1):.3f} % |')
    else:
        print(f'| {d} | {s1:.3f} % | - | {sa:.3f} % | - | - | {100*b[4]/max(b[5],1):.3f} % |')
