function gbtest14
%GBTEST14 test kron and GrB.kronecker

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
A = sparse (rand (2,3)) ;
B = sparse (rand (4,8)) ;

GA = GrB (A) ;
GB = GrB (B) ;

C = kron (A,B) ;
G = GrB.kronecker ('*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

G = kron (GA, GB) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

d.kind = 'sparse' ;
d.in0 = 'transpose' ;

G = GrB.kronecker ('*', A, B, d) ;
C = kron (A', B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)
G = kron (GA', GB) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)
d.kind = 'GrB' ;
G = GrB.kronecker ('*', A, B, d) ;
err = norm (C-G, 1) ;
assert (err < 1e-12) ;

d2 = d ;
d2.in1 = 'transpose' ;
G = GrB.kronecker ('*', A, B, d2) ;
C = kron (A', B') ;
err = norm (C-G, 1) ;
assert (err < 1e-12)
G = kron (GA', GB') ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

E = sparse (rand (8,24)) ;
C = E + kron (A,B) ;
G = GrB.kronecker (E, '+', '*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)
G = E + kron (GA, GB) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

[m, n] = size (G) ;
M = logical (sprand (m, n, 0.5)) ;
C = sprand (m, n, 0.5) ;
G = GrB (C) ;
T = C + kron (A,B) ;
C (M) = T (M) ;
G = GrB.kronecker (G, M, '+', '*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

C = sprand (m, n, 0.5) ;
G = GrB (C) ;
T = kron (A,B) ;
C (M) = T (M) ;
G2 = GrB.kronecker (G, M, '*', A, B) ;
err = norm (C-G2, 1) ;
assert (err < 1e-12)

fprintf ('gbtest14: all tests passed\n') ;

