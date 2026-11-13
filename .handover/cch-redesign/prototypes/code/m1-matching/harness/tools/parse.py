import sys, re, statistics as st
from collections import defaultdict
rows=[]
for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith("RESULT"): continue
        d=dict(kv.split("=",1) for kv in re.findall(r"(\w+=\S+)", line) if not kv.startswith("order_s_all"))
        rows.append(d)
by=defaultdict(list)
for d in rows: by[(d['graph'].split('.')[0], d['variant'])].append(d)
def med(xs): return st.median(xs)
print(f"{'graph':11} {'variant':11} {'n':>2} {'arcs med':>11} {'arcs min..max':>23} {'tri med (M)':>11} {'tri min..max (M)':>17} {'height med':>10} {'h min..max':>10} {'depth':>6} {'order s med':>11} {'2hop':>5} {'levels':>6} viol cover")
for (g,v),ds in sorted(by.items()):
    a=[int(d['arcs']) for d in ds]; t=[int(d['triangles'])/1e6 for d in ds]; h=[int(d['etree_height']) for d in ds]
    tm=[float(d['order_s_median']) for d in ds]; md=[float(d['mean_depth']) for d in ds]
    print(f"{g:11} {v:11} {len(ds):>2} {med(a):>11,.0f} {min(a):>11,}..{max(a):<11,} {med(t):>11.1f} {min(t):>8.1f}..{max(t):<8.1f} {med(h):>10.0f} {min(h):>4}..{max(h):<5} {med(md):>6.0f} {med(tm):>11.2f} {med([int(d['two_hop_fallbacks']) for d in ds]):>5.0f} {med([int(d['nd_levels_total']) for d in ds]):>6.0f} {sum(int(d['closure_violations']) for d in ds)} {all(d['cover_ok']=='true' for d in ds)}")
# per-seed paired comparison shem vs mm
print()
for g in sorted({k[0] for k in by}):
    s={d['seed']:d for d in by.get((g,'shem'),[])}; m={d['seed']:d for d in by.get((g,'mm'),[])}
    for seed in sorted(set(s)&set(m)):
        a,b=s[seed],m[seed]
        print(f"{g} seed {seed}: arcs shem {int(a['arcs']):,} mm {int(b['arcs']):,} ({int(b['arcs'])/int(a['arcs'])-1:+.2%}); tri shem {int(a['triangles'])/1e6:.1f}M mm {int(b['triangles'])/1e6:.1f}M ({int(b['triangles'])/int(a['triangles'])-1:+.2%}); height {a['etree_height']}->{b['etree_height']}; order s {a['order_s_median']}->{b['order_s_median']}")
