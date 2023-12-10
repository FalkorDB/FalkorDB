function gbtest6
%GBTEST6 test GrB.mxm

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
A = sparse (rand (2)) ;
B = sparse (rand (2)) ;

C = A*B ;

G = GrB.mxm ('+.*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12) ;

d.kind = 'sparse' ;
d.in0 = 'transpose' ;
G = GrB.mxm ('+.*', A, B, d) ;
C = A'*B ;

err = norm (C-G, 1) ;
assert (err < 1e-12) ;

d.kind = 'GrB' ;
G = GrB.mxm ('+.*', A, B, d) ;
err = norm (C-G, 1) ;
assert (err < 1e-12) ;

E = sparse (rand (2)) ;
C = E + A*B ;
G = GrB.mxm (E, '+', '+.*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12) ;

M = false (2,2) ;
Cin = rand (2) ;
M (1,1) = 1 ;
G = GrB.mxm (Cin, M, '+', '+.*', A, B) ;
T = Cin + A*B ;
C = Cin ;
C (M) = T (M) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

fprintf ('gbtest6: all tests passed\n') ;

