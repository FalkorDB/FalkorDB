function gbtest17
%GBTEST17 test GrB.trans

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 6 ;
m = 7 ;
A = 100 * sprand (n, m, 0.5) ;
AT = A' ;
M = sparse (rand (m,n)) > 0.5 ;
Cin = sprand (m, n, 0.5) ;

Cout = GrB.trans (A) ;
assert (gbtest_eq (AT, Cout)) ;

Cout = GrB.trans (A) ;
assert (gbtest_eq (AT, Cout)) ;

Cout = GrB.trans (Cin, M, A) ;
C2 = Cin ;
C2 (M) = AT (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = GrB.trans (Cin, '+', A) ;
C2 = Cin + AT ;
assert (gbtest_eq (C2, Cout)) ;

M = logical (sprand (m, n, 0.5)) ;
Cout = GrB.trans (Cin, M, '+', A) ;
T = Cin + A' ;
C2 = Cin ;
C2 (M) = T (M) ;
assert (gbtest_eq (C2, Cout)) ;

d.in0 = 'transpose' ;
Cout = GrB.trans (Cin', M', A, d) ;
C2 = Cin' ;
C2 (M') = A (M') ;
assert (gbtest_eq (C2, Cout)) ;

Cout = GrB.trans (Cin', '+', A, d) ;
C2 = Cin' + A ;
assert (gbtest_eq (C2, Cout)) ;

d.mask = 'complement' ;
d2 = d ;
d2.kind = 'sparse' ;
Cout  = GrB.trans (Cin', M', A, d) ;
Cout2 = GrB.trans (Cin', M', A, d2) ;
C2 = Cin' ;
C2 (~M') = A (~M') ;
assert (gbtest_eq (C2, Cout)) ;
assert (gbtest_eq (C2, Cout2)) ;
assert (isequal (class (Cout2), 'double')) ;

fprintf ('gbtest17: all tests passed\n') ;

