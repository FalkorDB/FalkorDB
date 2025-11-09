function test297
%TEST297 test with plus_one semiring

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test297 --- C=A*B with the plus_one semiring\n') ;

rng ('default') ;

n = 10 ;
d = 0.2 ;
A = sprand (n, n, d) ;
A1 = spones (A) ; ;
B = sprand (n, n, d) ;
B1 = spones (B) ; ;
Cin = sparse (n, n) ;

C1 = A1*B1 ;
C2 = GB_mex_plusone (Cin, [ ], [ ], [ ], A, B, [ ]) ;

assert (isequal (C1, C2.matrix)) ;

fprintf ('test297: all tests passed\n') ;

