function gbtest10
%GBTEST10 test GrB.assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 6 ;
A = 100 * sprand (n, n, 0.5) ;
AT = A' ;
M = sparse (rand (n)) > 0.5 ;
Cin = sprand (n, n, 0.5) ;


Cout = GrB.assign (Cin, A) ;
assert (gbtest_eq (A, Cout)) ;

Cout = GrB.assign (Cin, A, { }, { }) ;
assert (gbtest_eq (A, Cout)) ;

Cout = GrB.assign (Cin, M, A) ;
C2 = Cin ;
C2 (M) = A (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = GrB.assign (Cin, '+', A) ;
C2 = Cin + A ;
assert (gbtest_eq (C2, Cout)) ;

d.in0 = 'transpose' ;
Cout = GrB.assign (Cin, M, A, d) ;
C2 = Cin ;
C2 (M) = AT (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = GrB.assign (Cin, '+', A, d) ;
C2 = Cin + AT ;
assert (gbtest_eq (C2, Cout)) ;

d.mask = 'complement' ;
Cout = GrB.assign (Cin, M, A, d) ;
C2 = Cin ;
C2 (~M) = AT (~M) ;
assert (gbtest_eq (C2, Cout)) ;

I = [2 1 5] ;
J = [3 3 1 2] ;
B = sprandn (length (I), length (J), 0.5) ;
Cout = GrB.assign (Cin, B, {I}, {J}) ;
C2 = Cin ;
C2 (I,J) = B ;
assert (gbtest_eq (C2, Cout)) ;

A = rand (4) ;
G = GrB (A, 'by row') 
M = logical (eye (4)) ;
B = rand (4) ;
H = GrB (B, 'by row') 
A (M) = B (M) 
G (M) = H (M)
assert (isequal (A, G)) ;

A = sprand (4, 4, 0.5) %#ok<*NOPRT>
C1 = pi * spones (A)
C2 = GrB.expand (pi, A)
assert (isequal (C1, C2)) ;

fprintf ('gbtest10: all tests passed\n') ;

