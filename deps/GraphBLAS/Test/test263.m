function test263
%TEST263 test JIT

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test263 ------------ jit tests\n') ;

rng ('default') ;

n = 10 ;
C = GB_spec_random (n, n, 0.5, 100, 'logical') ;
A = GB_spec_random (n, n, 0.5, 100, 'double') ;
A.class = 'double complex' ;
op.opname = 'abs' ;
op.optype = 'double complex' ;
C1 = GB_spec_apply (C, [ ], [ ], op, A, [ ]) ;
GB_mex_burble (1) ;
C2 = GB_mex_apply  (C, [ ], [ ], op, A, [ ]) ;
GB_spec_compare (C1, C2) ;
GB_mex_burble (0) ;
