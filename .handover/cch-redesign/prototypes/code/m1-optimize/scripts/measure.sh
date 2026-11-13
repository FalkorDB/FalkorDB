#!/bin/bash
# m1opt/measure.sh BIN NAME OPTS [GRAPH] [extra VAR=VAL...]: CA (default) at 1 and 4 threads, check run + 3 timed runs,
# equality hashes checked against the baseline's.
W=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1opt
D=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1gbdata
BIN=$1; NAME=$2; OPTS=$3; G=${4:-CA}; shift 4 2>/dev/null || shift $#
HB=$W/cache/$(echo $G | tr A-Z a-z)_base.hash
HM="check:$HB"; [ -f $HB ] || HM="write:$HB"
$W/run.sh $W/$BIN $W/logs/${G}_$NAME.log M1_GRAPH=$D/roadNet-$G.txt M1_THREADS=1,4 M1_RUNS=3 M1_DEPTHS=14 M1_FERAL=0 M1_PERM_CACHE=$W/cache/$G.perm M1_HASH=$HM M1O=$OPTS "$@"
rc=$?
echo "$NAME exit=$rc $(grep -o 'hash_compared: [0-9]*, hash_fail: [0-9]*' $W/logs/${G}_$NAME.log | tail -1) $(grep -o 'slow_borrow=[0-9]*' $W/logs/${G}_$NAME.log | tail -1)"
