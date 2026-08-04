function gbtest14 (ghb)
%GBTEST14 test kron, [GrB,GhB].kronecker

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = sparse (rand (2,3)) ;
B = sparse (rand (4,8)) ;

GA = gtb (ghb, A) ;
GB = gtb (ghb, B) ;

C = kron (A,B) ;
G = gtb_kronecker (ghb, '*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

G = kron (GA, GB) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

d.kind = 'sparse' ;
d.in0 = 'transpose' ;

G = gtb_kronecker (ghb, '*', A, B, d) ;
C = kron (A', B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)
G = kron (GA', GB) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)
d.kind = 'GrB' ;
G = gtb_kronecker (ghb, '*', A, B, d) ;
err = norm (C-G, 1) ;
assert (err < 1e-12) ;

d2 = d ;
d2.in1 = 'transpose' ;
G = gtb_kronecker (ghb, '*', A, B, d2) ;
C = kron (A', B') ;
err = norm (C-G, 1) ;
assert (err < 1e-12)
G = kron (GA', GB') ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

E = sparse (rand (8,24)) ;
C = E + kron (A,B) ;
G = gtb_kronecker (ghb, E, '+', '*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)
G = E + kron (GA, GB) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

[m, n] = size (G) ;
M = logical (sprand (m, n, 0.5)) ;
C = sprand (m, n, 0.5) ;
G = gtb (ghb, C) ;
T = C + kron (A,B) ;
C (M) = T (M) ;
G = gtb_kronecker (ghb, G, M, '+', '*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

C = sprand (m, n, 0.5) ;
G = gtb (ghb, C) ;
T = kron (A,B) ;
C (M) = T (M) ;
G2 = gtb_kronecker (ghb, G, M, '*', A, B) ;
err = norm (C-G2, 1) ;
assert (err < 1e-12)

fprintf ('gbtest14 (%d): all tests passed\n', ghb) ;

