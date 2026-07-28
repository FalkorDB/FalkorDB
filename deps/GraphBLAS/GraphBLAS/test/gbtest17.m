function gbtest17 (ghb)
%GBTEST17 test [GrB,GhB].trans

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

n = 6 ;
m = 7 ;
A = 100 * sprand (n, m, 0.5) ;
AT = A' ;
M = sparse (rand (m,n)) > 0.5 ;
Cin = sprand (m, n, 0.5) ;

Cout = gtb_trans (ghb, A) ;
assert (gbtest_eq (AT, Cout)) ;

Cout = gtb_trans (ghb, A) ;
assert (gbtest_eq (AT, Cout)) ;

Cout = gtb_trans (ghb, Cin, M, A) ;
C2 = Cin ;
C2 (M) = AT (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = gtb_trans (ghb, Cin, '+', A) ;
C2 = Cin + AT ;
assert (gbtest_eq (C2, Cout)) ;

M = logical (sprand (m, n, 0.5)) ;
Cout = gtb_trans (ghb, Cin, M, '+', A) ;
T = Cin + A' ;
C2 = Cin ;
C2 (M) = T (M) ;
assert (gbtest_eq (C2, Cout)) ;

d.in0 = 'transpose' ;
Cout = gtb_trans (ghb, Cin', M', A, d) ;
C2 = Cin' ;
C2 (M') = A (M') ;
assert (gbtest_eq (C2, Cout)) ;

Cout = gtb_trans (ghb, Cin', '+', A, d) ;
C2 = Cin' + A ;
assert (gbtest_eq (C2, Cout)) ;

d.mask = 'complement' ;
d2 = d ;
d2.kind = 'sparse' ;
Cout  = gtb_trans (ghb, Cin', M', A, d) ;
Cout2 = gtb_trans (ghb, Cin', M', A, d2) ;
C2 = Cin' ;
C2 (~M') = A (~M') ;
assert (gbtest_eq (C2, Cout)) ;
assert (gbtest_eq (C2, Cout2)) ;
assert (isequal (class (Cout2), 'double')) ;

fprintf ('gbtest17 (%d): all tests passed\n', ghb) ;

