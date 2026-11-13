import sys, re, collections, statistics
def load(log):
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    for line in open(log):
        if not line.startswith('PHASE '): continue
        m = re.match(r'PHASE threads=(\d+) run=(\d+) depth=(\d+) path=(\w+)', line)
        th, run, d, p = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        if run == 0 or p not in ('D0', 'ROOTMAP', 'SPLIT', 'CS'): continue
        grp = 'D0' if p in ('D0', 'ROOTMAP') else 'SPLIT' if p == 'SPLIT' else 'CS'
        for n, w, i in re.findall(r' c:([^=]+)=([\d.]+)/(\d+)', line):
            e = per[(th, run)][(grp, n)]; e[0] += float(w); e[1] += int(i)
    out = {}
    for th in (1, 4):
        runs = [r for (t, r) in per if t == th]
        keys = set().union(*[per[(th, r)].keys() for r in runs])
        out[th] = {k: (statistics.median(per[(th, r)][k][0] for r in runs), statistics.median(per[(th, r)][k][1] for r in runs) / 1e9) for k in keys}
    return out
b, f = load(sys.argv[1]), load(sys.argv[2])
for th in (1, 4):
    print(f'--- threads={th}')
    for name, src in (('baseline', b), ('final', f)):
        items = sorted(src[th].items(), key=lambda x: (x[0][0], -x[1][0]))
        print(name, ' '.join(f'{g}:{n}={w:.3f}s/{i:.2f}G' for (g, n), (w, i) in items if w >= 0.004))
