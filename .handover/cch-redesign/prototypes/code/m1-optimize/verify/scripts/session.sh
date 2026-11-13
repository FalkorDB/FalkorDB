#!/bin/bash
# Verifier session on CA: base, final, C METIS x3, final, base (ABBA), all with the verifier's own build.
S=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad
V=$S/m1verify-work; BIN=$V/bin/graph-test-v; L=$V/logs
F=hb2,hsplit,cpack,hcon,pk,book,hkey,hpar,gath,usel,zmatch,npar
COM="M1_GRAPH=$S/m1gbdata/roadNet-CA.txt M1_THREADS=1,4 M1_RUNS=3 M1_DEPTHS=14 M1_PERM_CACHE=$V/cache/CA.perm"
HB=$V/cache/ca_base_mine.hash
echo "S base1 $(date) $(uptime)"; $V/run.sh $BIN $L/CA_base1.log $COM M1O= M1_FERAL=3 M1_HASH=write:$HB; echo "rc=$?"
echo "S final1 $(date) $(uptime)"; $V/run.sh $BIN $L/CA_final1.log $COM M1O=$F M1_FERAL=0 M1_HASH=check:$HB M1_RULECHECK=1; echo "rc=$?"
for r in 1 2 3; do
  echo "CMETIS CA run=$r $(uptime)" >> $L/cmetis.log
  /usr/bin/time -l $S/cmetis-driver/nd $S/m1audit/cmetis/CA.bin $V/cache/cmetis_perm_CA.bin >> $L/cmetis.log 2>&1
done
echo "S final2 $(date) $(uptime)"; $V/run.sh $BIN $L/CA_final2.log $COM M1O=$F M1_FERAL=0 M1_HASH=check:$HB; echo "rc=$?"
echo "S base2 $(date) $(uptime)"; $V/run.sh $BIN $L/CA_base2.log $COM M1O= M1_FERAL=3 M1_HASH=check:$HB; echo "rc=$?"
echo "S done $(date) $(uptime)"
