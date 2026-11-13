#!/bin/bash
# Follow-ups with the verifier variant binary (vb): hpar break + control on PA, split-break probe, floor30 on CA.
S=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5
S=$S/scratchpad; V=$S/m1verify-work; L=$V/logs/fu; mkdir -p $L
F=hb2,hsplit,cpack,hcon,pk,book,hkey,hpar,gath,usel,zmatch,npar
PA="M1_GRAPH=$S/m1gbdata/roadNet-PA.txt M1_FERAL=0 M1_PERM_CACHE=$V/cache/PA.perm"
CA="M1_GRAPH=$S/m1gbdata/roadNet-CA.txt M1_FERAL=0 M1_PERM_CACHE=$V/cache/CA.perm"
r() { n=$1; shift; $V/run.sh "$@"; echo "$n rc=$? $(grep -o 'hash_compared: [0-9]*, hash_fail: [0-9]*' ${2} | tail -1) | $(grep -m1 -E 'CHECK FAILED|GraphBLAS:' ${2} | cut -c1-160)"; }
# split-break probe with the plain build (v): which host pass turns the break into a crash
r split_cpack $V/bin/graph-test-v $L/split_cpack.log $PA M1_THREADS=1 M1_RUNS=0 M1_DEPTHS=1 M1O=hsplit,cpack M1_BREAK=split
r split_cpack_hcon $V/bin/graph-test-v $L/split_cpack_hcon.log $PA M1_THREADS=1 M1_RUNS=0 M1_DEPTHS=1 M1O=hsplit,cpack,hcon M1_BREAK=split
# hpar: control and break, final config, PA depths 1-3, 1 and 4 threads, check run only
r hpar_control $V/bin/graph-test-vb $L/hpar_control.log $PA M1_THREADS=1,4 M1_RUNS=0 M1_DEPTHS=3 M1O=$F M1_HASH=check:$V/cache/pa_base.hash
r hpar_break $V/bin/graph-test-vb $L/hpar_break.log $PA M1_THREADS=1,4 M1_RUNS=0 M1_DEPTHS=3 M1O=$F M1_HASH=check:$V/cache/pa_base.hash M1V_BREAK=hpar
# floor30 vs the bench's rule, final config, CA, timed (no check run), paired A B B A
r f_ctl1 $V/bin/graph-test-vb $L/CA_ctl1.log $CA M1_THREADS=1,4 M1_RUNS=3 M1_DEPTHS=14 M1O=$F M1_NOCHECK=1
r f_f30_1 $V/bin/graph-test-vb $L/CA_f30_1.log $CA M1_THREADS=1,4 M1_RUNS=3 M1_DEPTHS=14 M1O=$F M1_NOCHECK=1 M1V_FLOOR30=1
r f_f30_2 $V/bin/graph-test-vb $L/CA_f30_2.log $CA M1_THREADS=1,4 M1_RUNS=3 M1_DEPTHS=14 M1O=$F M1_NOCHECK=1 M1V_FLOOR30=1
r f_ctl2 $V/bin/graph-test-vb $L/CA_ctl2.log $CA M1_THREADS=1,4 M1_RUNS=3 M1_DEPTHS=14 M1O=$F M1_NOCHECK=1
echo FU done
