function test219
%TEST219 test reduce to scalar

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

GB_mex_burble (1) ;

n = 103 ;

A.matrix = sparse (rand (n,n)) ;
A.class = 'double' ;

save = nthreads_get ;

c1 = GB_mex_reduce_to_scalar  (0, [ ], 'max', A) ;
c2 = max (max (A.matrix)) ;
assert (c1 == c2) ;

nthreads_set (save) ;

GB_mex_burble (0) ;
fprintf ('\ntest219: all tests passed\n') ;

