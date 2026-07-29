function gbtest16 (ghb)
%GBTEST16 test [GrB,GhB].extract

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

n = 6 ;
A = 100 * sprand (n, n, 0.5) ;
AT = A' ;
M = sparse (rand (n)) > 0.5 ;
Cin = sprand (n, n, 0.5) ;

Cout = gtb_extract (ghb, Cin, A) ;
assert (gbtest_eq (A, Cout)) ;

Cout = gtb_extract (ghb, Cin, A, { }, { }) ;
assert (gbtest_eq (A, Cout)) ;

Cout = gtb_extract (ghb, A, {n, -1, 1}, {n, -1, 1}) ;
assert (gbtest_eq (A (n:-1:1, n:-1:1), Cout)) ;

Cout = gtb_extract (ghb, Cin, M, A) ;
C2 = Cin ;
C2 (M) = A (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = gtb_extract (ghb, Cin, '+', A) ;
C2 = Cin + A ;
assert (gbtest_eq (C2, Cout)) ;

d.in0 = 'transpose' ;
Cout = gtb_extract (ghb, Cin, M, A, d) ;
C2 = Cin ;
C2 (M) = AT (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = gtb_extract (ghb, Cin, '+', A, d) ;
C2 = Cin + AT ;
assert (gbtest_eq (C2, Cout)) ;

d.mask = 'complement' ;
d2 = d ;
d2.kind = 'sparse' ;
Cout  = gtb_extract (ghb, Cin, M, A, d) ;
Cout2 = gtb_extract (ghb, Cin, M, A, d2) ;
C2 = Cin ;
C2 (~M) = AT (~M) ;
assert (gbtest_eq (C2, Cout)) ;
assert (gbtest_eq (C2, Cout2)) ;
assert (isequal (class (Cout2), 'double')) ;

I = [2 1 5] ;
J = [3 3 1 2] ;
% B = sprandn (length (I), length (J), 0.5) ;
Cout = gtb_extract (ghb, A, {I}, {J}) ;
C2 = A (I,J)  ;
assert (gbtest_eq (C2, Cout)) ;

desc.base = 'zero-based' ;
Cout = gtb_extract (ghb, A, { int64(I) - 1 }, { int64(J) - 1 }, desc) ;
assert (gbtest_eq (C2, Cout)) ;

G = gtb_random (ghb, 1, 10, inf) ;
A = double (G) ;
C0 = A (1:3) ;
C1 = gtb_extract (ghb, G, { 1, 3}) ;
assert (isequal (C0, C1)) ;

G = gtb_random (ghb, 10, 1, inf) ;
A = double (G) ;
C0 = A (1:3) ;
C1 = gtb_extract (ghb, G, { 1, 3}) ;
assert (isequal (C0, C1)) ;

fprintf ('gbtest16 (%d): all tests passed\n', ghb) ;

