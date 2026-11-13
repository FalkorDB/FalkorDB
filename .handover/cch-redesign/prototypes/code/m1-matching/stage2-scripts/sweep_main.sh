#!/bin/bash
# Main comparison: stock SHEM, mutual-max re-coarsen (R = mm), mutual-max with reuse (mm-reuse).
# 3 timed reps per (graph, variant, seed); reuse also gets one untimed check rep first.
S=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad
B=$S/m1reuse/bin/m1bench-main
D=$S/roaddata
O=$S/m1reuse/out/main.txt
: > $O
for g in PA TX CA; do for seed in 1 2 3 4 5; do for v in shem mm mm-reuse; do
  echo "LOAD graph=$g variant=$v seed=$seed $(uptime | sed 's/.*load averages*: //')" >> $O
  /usr/bin/time -l $B order $D/roadNet-$g.txt.gz $v $seed 3 >> $O 2> $S/m1reuse/out/last.err
  echo "RSS graph=$g variant=$v seed=$seed exit=$? $(grep 'maximum resident' $S/m1reuse/out/last.err | awk '{print $1}')" >> $O
done; done; done
echo DONE >> $O
