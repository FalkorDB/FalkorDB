#!/bin/bash
A=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1rcA
D=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1gbdata
cd $A
for R in fine all; do
  for G in CA PA TX; do
    ./run.sh logs/main_${G}_${R}.log M1_GRAPH=$D/roadNet-$G.txt M1_THREADS=1,4 M1_RUNS=3 M1_FERAL=3 M1_CT=1 M1_JIT=run M1_REFINE=$R
    echo "$G $R exit=$? $(tail -1 logs/main_${G}_${R}.log)" >> logs/sweep_status.txt
  done
done
echo SWEEP_DONE >> logs/sweep_status.txt
