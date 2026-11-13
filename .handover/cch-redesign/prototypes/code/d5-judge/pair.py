import re,sys,statistics as st
rows={}
for fn in sys.argv[1:]:
    for line in open(fn):
        if not line.startswith('RESULT'): continue
        d=dict(kv.split('=',1) for kv in line.split()[1:] if '=' in kv)
        g=d['graph'].split('.')[0].replace('roadNet-','')
        rows[(g,d['variant'],int(d['seed']))]=(int(d['arcs']),int(d['triangles']),int(d['etree_height']),float(d['order_s_median']))
for g in ['CA','PA','TX']:
    for v in ['mm-reuse','mm-reuse-capstop','mm-reuse-ge']:
        c={'a<=S':0,'t<=S':0,'h<=S':0,'all3<=S':0,'a<=R':0,'t<=R':0,'h<=R':0}
        hs=[];ts=[]
        for s in range(1,6):
            x=rows.get((g,v,s)); S_=rows[(g,'shem',s)]; R=rows[(g,'mm',s)]
            if not x: continue
            c['a<=S']+=x[0]<=S_[0]; c['t<=S']+=x[1]<=S_[1]; c['h<=S']+=x[2]<=S_[2]
            c['all3<=S']+=(x[0]<=S_[0] and x[1]<=S_[1] and x[2]<=S_[2])
            c['a<=R']+=x[0]<=R[0]; c['t<=R']+=x[1]<=R[1]; c['h<=R']+=x[2]<=R[2]
            hs.append(x[2]); ts.append(x[3])
        med=lambda i,vv: st.median([rows[(g,vv,s)][i] for s in range(1,6) if (g,vv,s) in rows])
        print(g,v,c,'med arcs',med(0,v),'tri',med(1,v)/1e6,'h',med(2,v),'t',med(3,v),'| SHEM h',med(2,'shem'),'R h',med(2,'mm'))
    # R vs SHEM height
    c=sum(rows[(g,'mm',s)][2]<=rows[(g,'shem',s)][2] for s in range(1,6))
    print(g,'R h<=SHEM',c)
