#!/bin/bash
# Final session: baseline and final config on CA, PA, TX at 1 and 4 threads, feral and C METIS beside them.
S=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad
W=$S/m1opt; cd $W
F=hb2,hsplit,cpack,hcon,pk,book,hkey,hpar,gath,usel,zmatch,npar
L=$W/logs/session; mkdir -p $L
for G in CA PA TX; do
  echo "SESSION $G baseline $(date) $(uptime)"
  ./measure.sh bin/graph-test-v4 fin_base "" $G M1_FERAL=3 2>&1 | tail -1
  mv logs/${G}_fin_base.log $L/${G}_base.log
  echo "SESSION $G final $(date) $(uptime)"
  ./measure.sh bin/graph-test-v4 fin_opt $F $G 2>&1 | tail -1
  mv logs/${G}_fin_opt.log $L/${G}_final.log
  for r in 1 2 3; do
    echo "CMETIS $G run=$r $(uptime)" >> $L/cmetis.log
    /usr/bin/time -l $S/cmetis-driver/nd $S/m1audit/cmetis/$G.bin $W/cache/cmetis_perm_$G.bin >> $L/cmetis.log 2>&1
  done
done
echo "SESSION done $(date)"
