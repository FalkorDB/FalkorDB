function gbtest16
%GBTEST16 test GrB.extract

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 6 ;
A = 100 * sprand (n, n, 0.5) ;
AT = A' ;
M = sparse (rand (n)) > 0.5 ;
Cin = sprand (n, n, 0.5) ;

Cout = GrB.extract (Cin, A) ;
assert (gbtest_eq (A, Cout)) ;

Cout = GrB.extract (Cin, A, { }, { }) ;
assert (gbtest_eq (A, Cout)) ;

Cout = GrB.extract (A, {n, -1, 1}, {n, -1, 1}) ;
assert (gbtest_eq (A (n:-1:1, n:-1:1), Cout)) ;

Cout = GrB.extract (Cin, M, A) ;
C2 = Cin ;
C2 (M) = A (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = GrB.extract (Cin, '+', A) ;
C2 = Cin + A ;
assert (gbtest_eq (C2, Cout)) ;

d.in0 = 'transpose' ;
Cout = GrB.extract (Cin, M, A, d) ;
C2 = Cin ;
C2 (M) = AT (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = GrB.extract (Cin, '+', A, d) ;
C2 = Cin + AT ;
assert (gbtest_eq (C2, Cout)) ;

d.mask = 'complement' ;
d2 = d ;
d2.kind = 'sparse' ;
Cout  = GrB.extract (Cin, M, A, d) ;
Cout2 = GrB.extract (Cin, M, A, d2) ;
C2 = Cin ;
C2 (~M) = AT (~M) ;
assert (gbtest_eq (C2, Cout)) ;
assert (gbtest_eq (C2, Cout2)) ;
assert (isequal (class (Cout2), 'double')) ;

I = [2 1 5] ;
J = [3 3 1 2] ;
% B = sprandn (length (I), length (J), 0.5) ;
Cout = GrB.extract (A, {I}, {J}) ;
C2 = A (I,J)  ;
assert (gbtest_eq (C2, Cout)) ;

desc.base = 'zero-based' ;
Cout = GrB.extract (A, { int64(I) - 1 }, { int64(J) - 1 }, desc) ;
assert (gbtest_eq (C2, Cout)) ;

G = GrB.random (1, 10, inf) ;
A = double (G) ;
C0 = A (1:3) ;
C1 = GrB.extract (G, { 1, 3}) ;
assert (isequal (C0, C1)) ;

G = GrB.random (10, 1, inf) ;
A = double (G) ;
C0 = A (1:3) ;
C1 = GrB.extract (G, { 1, 3}) ;
assert (isequal (C0, C1)) ;

fprintf ('gbtest16: all tests passed\n') ;

