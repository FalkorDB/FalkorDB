function gbtest12 (ghb)
%GBTEST12 test GrB.eadd, GrB.emult, GrB.eunion, and GhB variants.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

rng ('default') ;
A = sparse (rand (2)) ;
B = sparse (rand (2)) ;

C = A+B ;
D = A.*B ;
F = A-B ;

G = gtb_eadd (ghb, '+', A, B) ;
err = norm (C-G, 1) ;
assert (logical (err < 1e-12))

G = gzb_eadd (ghb, '+', GrB (A), GrB (B)) ;
err = norm (C-G, 1) ;
assert (logical (err < 1e-12))

G = gzb_eadd (ghb, GrB (A), '+', GrB (B)) ;
err = norm (C-G, 1) ;
assert (logical (err < 1e-12))

H = gtb_emult (ghb, '*', A, B) ;
err = norm (D-H, 1) ;
assert (logical (err < 1e-12))

H = gzb_emult (ghb, '*', GrB (A), GrB (B)) ;
err = norm (D-H, 1) ;
assert (logical (err < 1e-12))

H = gzb_emult (ghb, GrB (A), '*', GrB (B)) ;
err = norm (D-H, 1) ;
assert (logical (err < 1e-12))

G = gtb_eunion (ghb, '-', A, 0, B, 0) ;
err = norm (F-G, 1) ;
assert (logical (err < 1e-12))

G = gzb_eunion (ghb, GrB (A), '-', GrB (B)) ;
err = norm (F-G, 1) ;
assert (logical (err < 1e-12))

d.kind = 'sparse' ;
d.in0 = 'transpose' ;

G = gtb_eadd (ghb, '+', A, B, d) ;
C = A'+B ;
err = norm (C-G, 1) ;
assert (logical (err < 1e-12))

H = gtb_emult (ghb, '*', A, B, d) ;
D = A'.*B ;
err = norm (H-D, 1) ;
assert (logical (err < 1e-12))

d.kind = 'GrB' ;
G = gtb_eadd (ghb, '+', A, B, d) ;
err = norm (C-G, 1) ;
assert (logical (err < 1e-12)) ;

H = gtb_emult (ghb, '*', A, B, d) ;
err = norm (D-H, 1) ;
assert (logical (err < 1e-12)) ;

E = sparse (rand (2)) ;
C = E + A+B ;
G = gtb_eadd (ghb, E, '+', '+', A, B) ;
C_minus_G = C-G ;
err = norm (C_minus_G, 1) ;
assert (logical (err < 1e-12)) ;

F = sparse (rand (2)) ;
D = F + A.*B ;
H = gtb_emult (ghb, F, '+', '*', A, B) ;
D_minus_H = D-H ;
err = norm (D_minus_H, 1) ;
assert (logical (err < 1e-12)) ;
assert (gbtest_eq (D, H)) ;

G = gtb_eadd (ghb, '+', A, B) ;
C = A+B ;
assert (gbtest_eq (C, G)) ;

H = gtb_emult (ghb, '*', A, B) ;
D = A.*B ;
assert (gbtest_eq (D, H)) ;

m = 10 ;
n = 12 ;
A = sprand (m, n, 0.5) ;
B = sprand (m, n, 0.5) ;
M = logical (sprand (m, n, 0.5)) ;
Cin = sprand (m, n, 0.5) ;
G = gtb (ghb, Cin) ;
T = Cin + A .* B ;
C = Cin ;
C (M) = T (M) ;
G = gtb_emult (ghb, Cin, M, '+', '*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

G = gtb_eadd (ghb, Cin, M, '+', '+', A, B) ;
C = Cin ;
T = Cin + A + B ;
C (M) = T (M) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

G = gtb_eadd (ghb, Cin, M, '+', A, B) ;
C = Cin ;
T = A + B ;
C (M) = T (M) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

G = gtb_eunion (ghb, '-', A, 0, B, 0) ;
F = A-B ;
err = norm (F-G, 1) ;
assert (err < 1e-12)

C = sprand (m, n, 0.5) ;
G = gtb (ghb, C) ;
T = A .* B ;
C (M) = T (M) ;
G = gtb_emult (ghb, G, M, '*', A, B) ;
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

fprintf ('gbtest12 (%d): all tests passed\n', ghb) ;

