#!/bin/bash
W=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1match
D=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/roaddata
B=$W/target/release/m1bench
O=$W/out
# main comparison: 3 timed reps per (graph, variant, seed)
for g in CA PA TX; do for seed in 1 2 3 4 5; do for v in shem mm; do
  $B order $D/roadNet-$g.txt.gz $v $seed 3 >> $O/main.txt 2>/dev/null
done; done; done
# diagnostics: 1 rep
for g in CA PA TX; do for seed in 1 2 3 4 5; do for v in shem-cap mm-nocap; do
  $B order $D/roadNet-$g.txt.gz $v $seed 1 >> $O/diag.txt 2>/dev/null
done; done; done
# round-cap sensitivity, CA
for seed in 1 2 3 4 5; do for v in mm-r1 mm-r2 mm-r3 mm-r4 mm-r6 mm-seq2hop; do
  $B order $D/roadNet-CA.txt.gz $v $seed 1 >> $O/rounds.txt 2>/dev/null
done; done
# breaks: min-weight proposals; feral order with 1000 random swaps
for seed in 1 2 3; do
  $B order $D/roadNet-CA.txt.gz mm-inv $seed 1 >> $O/breaks.txt 2>/dev/null
done
$B order $D/roadNet-CA.txt.gz swap1000 1 1 >> $O/breaks.txt 2>/dev/null
# standalone coarsening of the largest component to <= 100 (the design's cell floor)
for g in CA PA TX; do for v in shem shem-cap mm mm-nocap; do
  $B coarsen $D/roadNet-$g.txt.gz $v 1 >> $O/coarsen.txt 2>/dev/null
done; done
echo DONE >> $O/main.txt
