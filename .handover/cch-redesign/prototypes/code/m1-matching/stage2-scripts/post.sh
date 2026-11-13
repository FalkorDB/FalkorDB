#!/bin/bash
S=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad
B=$S/m1reuse/bin/m1bench-diag
D=$S/roaddata
O=$S/m1reuse/out
run() { # tag file cmd...
  local f=$1; shift
  echo "LOAD $(uptime | sed 's/.*load averages*: //') :: $*" >> $f
  "$@" >> $f 2>> $O/post.err
  echo "EXIT $? :: $*" >> $f
}
: > $O/strad.txt; : > $O/breaks.txt; : > $O/same.txt; : > $O/diag.txt; : > $O/post.err
# 1. straddler statistics: the check rep alone, all graphs and seeds
for g in PA TX CA; do for seed in 1 2 3 4 5; do
  run $O/strad.txt $B strad $D/roadNet-$g.txt.gz mm-reuse $seed
done; done
# 2. deliberate breaks (PA, seed 1)
run $O/breaks.txt $B checkbreak $D/roadNet-PA.txt.gz mm-reuse
for v in mm-reuse-b1 mm-reuse-b2 mm-reuse-b3 mm-reuse-b4 mm-reuse-b5; do
  run $O/breaks.txt $B order $D/roadNet-PA.txt.gz $v 1 1
done
M1_REP_BREAK=1 run $O/breaks.txt $B order $D/roadNet-PA.txt.gz mm-reuse 1 2
# 3. stock and mutual-max paths unchanged
for g in PA TX CA; do run $O/same.txt $B same $D/roadNet-$g.txt.gz; done
M1_SAME_BREAK=1 run $O/same.txt $B same $D/roadNet-PA.txt.gz
for g in PA TX CA; do for v in shem mm; do for seed in 1 2; do
  run $O/same.txt $B samebase $D/roadNet-$g.txt.gz $v $seed
done; done; done
M1_SAMEBASE_BREAK=1 run $O/same.txt $B samebase $D/roadNet-PA.txt.gz mm 1
# 3b. depth 0 per top-level component, R against Reuse (1 rep each, no check rep)
: > $O/roots.txt
for g in PA TX CA; do for seed in 1 2 3 4 5; do for v in mm mm-reuse; do
  M1_NOCHECK=1 run $O/roots.txt $B order $D/roadNet-$g.txt.gz $v $seed 1
done; done; done
# 4. diagnostics, 1 rep, no check rep
for g in PA TX CA; do for seed in 1 2 3 4 5; do for v in mm-reuse-capstop mm-reuse-ge; do
  M1_NOCHECK=1 run $O/diag.txt $B order $D/roadNet-$g.txt.gz $v $seed 1
done; done; done
echo DONE >> $O/diag.txt
