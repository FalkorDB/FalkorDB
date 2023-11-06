function test201
%TEST201 test iso reduce to vector and reduce to scalar

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

n = 10 ;
A.matrix = pi * sparse (ones (n)) ;
A.class = 'double' ;
A.iso = true ;
win = sparse (n,1) ;
w1 = GB_mex_reduce_to_vector  (win, [ ], [ ], 'plus', A, [ ]) ;
w2 = sum (A.matrix, 2) ;
err = norm (w1.matrix - w2, 1) ;
assert (err < 1e-12) 

c0 = 0 ;
c2 = GB_spec_reduce_to_scalar (c0, [ ], 'plus', A) ;
c1 = GB_mex_reduce_to_scalar  (c0, [ ], 'plus', A) ;
assert (abs (c1-c2) < 1e-12) ;
c3 = sum (sum (A.matrix)) ;
assert (abs (c2-c3) < 1e-12) ;

fprintf ('test201: all tests passed\n') ;
