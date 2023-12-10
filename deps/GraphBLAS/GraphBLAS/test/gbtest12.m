function gbtest12
%GBTEST12 test GrB.eadd, GrB.emult, GrB.eunion

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
A = sparse (rand (2)) ;
B = sparse (rand (2)) ;

C = A+B ;
D = A.*B ;
F = A-B ;

G = GrB.eadd ('+', A, B) ;
err = norm (C-G, 1) %#ok<*NOPRT>
assert (logical (err < 1e-12))

H = GrB.emult ('*', A, B) ;
err = norm (D-H, 1)
assert (logical (err < 1e-12))

G = GrB.eunion ('-', A, 0, B, 0) ;
err = norm (F-G, 1) %#ok<*NOPRT>
assert (logical (err < 1e-12))

d.kind = 'sparse' ;
d.in0 = 'transpose' ;
d

G = GrB.eadd ('+', A, B, d) ;
C = A'+B ;
err = norm (C-G, 1)
assert (logical (err < 1e-12))

H = GrB.emult ('*', A, B, d) ;
D = A'.*B ;
err = norm (H-D, 1)
assert (logical (err < 1e-12))

d.kind = 'GrB' ;
G = GrB.eadd ('+', A, B, d) ;
err = norm (C-G, 1) %#ok<*NASGU>

H = GrB.emult ('*', A, B, d) ;
err = norm (D-H, 1)

E = sparse (rand (2)) ;
C = E + A+B ;
G = GrB.eadd (E, '+', '+', A, B) ;
C-G %#ok<*MNEFF>

F = sparse (rand (2)) ;
D = F + A.*B ;
H = GrB.emult (F, '+', '*', A, B) ;
D-H
assert (gbtest_eq (D, H)) ;

G = GrB.eadd ('+', A, B)
C = A+B
assert (gbtest_eq (C, G)) ;

H = GrB.emult ('*', A, B)
D = A.*B
assert (gbtest_eq (D, H)) ;

m = 10 ;
n = 12 ;
A = sprand (m, n, 0.5) ;
B = sprand (m, n, 0.5) ;
M = logical (sprand (m, n, 0.5)) ;
Cin = sprand (m, n, 0.5) ;
G = GrB (Cin) ;
T = Cin + A .* B ;
C = Cin ;
C (M) = T (M) ;
G = GrB.emult (Cin, M, '+', '*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

G = GrB.eadd (Cin, M, '+', '+', A, B) ;
C = Cin ;
T = Cin + A + B ;
C (M) = T (M) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

G = GrB.eadd (Cin, M, '+', A, B) ;
C = Cin ;
T = A + B ;
C (M) = T (M) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

G = GrB.eunion ('-', A, 0, B, 0) ;
F = A-B ;
err = norm (F-G, 1) ;
assert (err < 1e-12)

C = sprand (m, n, 0.5) ;
G = GrB (C) ;
T = A .* B ;
C (M) = T (M) ;
G = GrB.emult (G, M, '*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

C1 = 2 - C ;
C2 = 2 - G ;
assert (isequal (C1, C2)) ;

C1 = 0 - C ;
C2 = 0 - G ;
assert (isequal (C1, C2)) ;

C1 = C - 2 ;
C2 = C - 2 ;
assert (isequal (C1, C2)) ;

fprintf ('gbtest12: all tests passed\n') ;

