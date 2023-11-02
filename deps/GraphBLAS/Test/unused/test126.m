function test126
%TEST126 test GrB_reduce to vector on a very sparse matrix 

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test126:  test GrB_reduce to vector on a very sparse matrix\n') ;
rng ('default') ;

n = 1000 ;
A = sprand (n, n, 0.1) ;
huge = 1e9 ;
A (huge, 2) = 42 ;

tic
row = sum (A,1) ;
tm1 = toc ;

nrm = full (sum (row)) ;

tic
col = sum (A,2) ;
tm2 = toc ;

s = sparse (huge, 1) ;
tic
w = GB_mex_reduce_to_vector (s, [ ], [ ], 'plus', A, [ ]) ;
tg1 = toc ;
assert (norm (col - w.matrix, 1) / nrm < 1e-12)

dtn.inp0 = 'tran' ;
s = sparse (n, 1) ;
tic
w = GB_mex_reduce_to_vector (s, [ ], [ ], 'plus', A, dtn) ;
tg2 = toc ;
assert (norm (row' - w.matrix, 1) / nrm < 1e-12)

fprintf ('reduce each vector: built-in %g GrB %g speedup %g\n', ...
    tm1, tg1, tm1/tg1) ;

fprintf ('reduce each index:  built-in %g GrB %g speedup %g\n', ...
    tm2, tg2, tm2/tg2) ;

fprintf ('test126: all tests passed\n') ;
