function test110
%TEST110 test accum/mask (binary search of M(:,j))

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest110:  test accum/mask (binary search of M(:,j))\n') ;

rng ('default')

m = 100 ;
n = 100 ;
M = spones (rand (m,n)) ;
M (m,:) = 0 ;

C = sprand (m, n, 0.01) ;
A = sprand (m, n, 0.02) ;

C1 = GB_mex_apply  (C, M, 'plus', 'identity', A) ;
C2 = GB_spec_apply (C, M, 'plus', 'identity', A, [ ]) ;
GB_spec_compare (C1,C2) ;

d.mask = 'complement' ;

C1 = GB_mex_apply  (C, M, 'plus', 'identity', A, d) ;
C2 = GB_spec_apply (C, M, 'plus', 'identity', A, d) ;
GB_spec_compare (C1,C2) ;

fprintf ('\ntest110:  all tests passed\n') ;

