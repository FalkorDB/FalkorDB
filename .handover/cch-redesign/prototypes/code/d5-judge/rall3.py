import sys
rows={}
for fn in sys.argv[1:]:
    for line in open(fn):
        if not line.startswith('RESULT'): continue
        d=dict(kv.split('=',1) for kv in line.split()[1:] if '=' in kv)
        g=d['graph'].split('.')[0].replace('roadNet-','')
        rows[(g,d['variant'],int(d['seed']))]=(int(d['arcs']),int(d['triangles']),int(d['etree_height']))
tot={}
for v in ['mm','mm-reuse','mm-reuse-capstop','mm-reuse-ge']:
    out=[]
    for g in ['CA','PA','TX']:
        k=sum(all(rows[(g,v,s)][i]<=rows[(g,'shem',s)][i] for i in range(3)) for s in range(1,6))
        out.append(f'{g} {k}/5')
    print(v,out)
