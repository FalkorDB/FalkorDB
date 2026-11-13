#!/usr/bin/env python3
"""m1opt/analyze.py LOG [LOG...] [--csv OUT --config NAME --append] [--calls PATH]
Per (threads, path, phase): median [min-max] over the timed runs (run >= 1) of wall and
instructions, from PHASE lines. Paths: R = D0 + sum_d(SPLIT + R); B2 = D0 + ROOTMAP + sum_d(SPLIT + B2);
CS = D0 + ROOTMAP + sum_d(SPLIT + CS). Phases: depth0, depths1+ (incl. the split), total."""
import sys, re, statistics, collections, os
args = sys.argv[1:]
def popopt(k, default=None):
    if k in args:
        i = args.index(k); v = args[i+1]; del args[i:i+2]; return v
    return default
csv = popopt('--csv'); config = popopt('--config', 'x'); calls_path = popopt('--calls')
append = '--append' in args
if append: args.remove('--append')
graph_of = lambda log: re.search(r'roadNet-(\w+)\.txt', open(log).read(4000)).group(1)
rows = []
for log in args:
    g = graph_of(log)
    load = {}
    data = collections.defaultdict(lambda: collections.defaultdict(dict))  # (th,run) -> depth -> path -> rec
    calls = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    for line in open(log):
        if line.startswith('LOAD threads='):
            m = re.match(r'LOAD threads=(\d+) run=(\d+) .*load averages: ([\d.]+)', line)
            if m: load[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
        if not line.startswith('PHASE '): continue
        kv = dict(re.findall(r'(\w+)=([^\s]+)', line.split(' c:')[0]))
        th, run, d, p = int(kv['threads']), int(kv['run']), int(kv['depth']), kv['path']
        data[(th, run)][d][p] = (float(kv['wall']), int(kv['instr']), int(kv['cyc']), float(kv['pwall']), int(kv['pinstr']))
        for name, w, i in re.findall(r' c:([^=]+)=([\d.]+)/(\d+)', line):
            calls[(th, run)][(p, 0 if d == 0 else 1)][name] += float(w)
    for (th, run), dd in sorted(data.items()):
        if run == 0: continue
        def tot(paths, depths):
            w = i = c = 0
            for d in depths:
                for p in paths:
                    if p in dd.get(d, {}):
                        r = dd[d][p]; w += r[0]; i += r[1]; c += r[2]
            return w, i, c
        deep = [d for d in dd if d >= 1]
        for path, p0, p1 in [('R', ['D0'], ['SPLIT', 'R']), ('B2', ['D0', 'ROOTMAP'], ['SPLIT', 'B2']), ('CS', ['D0', 'ROOTMAP'], ['SPLIT', 'CS'])]:
            a = tot(p0, [0]); b = tot(p1, deep)
            for phase, v in [('depth0', a), ('depths1+', b), ('total', tuple(x + y for x, y in zip(a, b)))]:
                rows.append(dict(graph=g, config=config, threads=th, path=path, phase=phase, run=run, wall_s=v[0], instr_G=v[1] / 1e9, cycles_G=v[2] / 1e9, load1=load.get((th, run), float('nan')), log=os.path.basename(log)))
        # split separately
        b = tot(['SPLIT'], deep)
        rows.append(dict(graph=g, config=config, threads=th, path='SPLIT', phase='depths1+', run=run, wall_s=b[0], instr_G=b[1] / 1e9, cycles_G=b[2] / 1e9, load1=load.get((th, run), float('nan')), log=os.path.basename(log)))
    if calls_path:
        p, ph = calls_path.split(':') if ':' in calls_path else (calls_path, '1')
        for th in sorted({k[0] for k in calls}):
            names = collections.defaultdict(list)
            for (t2, run), dd in calls.items():
                if t2 != th or run == 0: continue
                for n, w in dd[(p, int(ph))].items(): names[n].append(w)
            print(f'CALLS {g} threads={th} path={p} phase={"depths1+" if ph=="1" else "depth0"}: ' + ' '.join(f'{n}={statistics.median(v):.3f}' for n, v in sorted(names.items(), key=lambda x: -statistics.median(x[1]))))
# summary
grp = collections.defaultdict(list)
for r in rows: grp[(r['graph'], r['config'], r['threads'], r['path'], r['phase'])].append(r)
print(f"{'graph':5} {'config':14} {'th':>2} {'path':5} {'phase':9} {'wall_s med [min-max]':26} {'instr_G med [min-max]':28} runs load1")
for k, rs in sorted(grp.items()):
    w = [r['wall_s'] for r in rs]; i = [r['instr_G'] for r in rs]; l = [r['load1'] for r in rs]
    print(f"{k[0]:5} {k[1]:14} {k[2]:>2} {k[3]:5} {k[4]:9} {statistics.median(w):7.3f} [{min(w):.3f}-{max(w):.3f}]     {statistics.median(i):8.2f} [{min(i):.2f}-{max(i):.2f}]     {len(rs)} {min(l):.1f}-{max(l):.1f}")
if csv:
    new = not (append and os.path.exists(csv))
    with open(csv, 'a' if append else 'w') as f:
        if new: f.write('graph,config,threads,path,phase,run,wall_s,instr_G,cycles_G,load1,log\n')
        for r in rows:
            f.write(f"{r['graph']},{r['config']},{r['threads']},{r['path']},{r['phase']},{r['run']},{r['wall_s']:.6f},{r['instr_G']:.4f},{r['cycles_G']:.4f},{r['load1']:.2f},{r['log']}\n")
        for k, rs in sorted(grp.items()):
            w = sorted(r['wall_s'] for r in rs); i = sorted(r['instr_G'] for r in rs); c = sorted(r['cycles_G'] for r in rs)
            f.write(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},median,{statistics.median(w):.6f},{statistics.median(i):.4f},{statistics.median(c):.4f},,spread_wall={min(w):.3f}-{max(w):.3f};spread_instr={min(i):.2f}-{max(i):.2f}\n")
