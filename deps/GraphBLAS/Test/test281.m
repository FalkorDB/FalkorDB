function test281
%TEST281 test GrB_apply with user-defined idxunp

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n--- testing apply with user-defined idxunop (no JIT)\n') ;
rng ('default') ;

n = 10 ;
d = 0.5 ;
A.matrix = spones (sprand (n, n, 0.5)) ;
A.iso = true ;

GB_mex_burble (1) ;
C = GB_mex_apply_idxunop_user (A) ;
GB_mex_burble (0) ;

[i j x] = find (A.matrix) ;
x = (i-1) + (j-1) + 1 ;
C2 = sparse (i, j, x, n, n) ;

assert (isequal (C, C2)) ;

fprintf ('\ntest281: all tests passed\n') ;

