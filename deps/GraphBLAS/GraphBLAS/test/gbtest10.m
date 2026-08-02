function gbtest10 (ghb)
%GBTEST10 test GrB.assign and GhB.assign

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

Cout = gtb_assign (ghb, Cin, A) ;
assert (gbtest_eq (A, Cout)) ;

Cout = gtb_assign (ghb, Cin, A, { }, { }) ;
assert (gbtest_eq (A, Cout)) ;

Cout = gtb_assign (ghb, Cin, M, A) ;
C2 = Cin ;
C2 (M) = A (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = gtb_assign (ghb, Cin, '+', A) ;
C2 = Cin + A ;
assert (gbtest_eq (C2, Cout)) ;

d.in0 = 'transpose' ;
Cout = gtb_assign (ghb, Cin, M, A, d) ;
C2 = Cin ;
C2 (M) = AT (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = gtb_assign (ghb, Cin, '+', A, d) ;
C2 = Cin + AT ;
assert (gbtest_eq (C2, Cout)) ;

d.mask = 'complement' ;
Cout = gtb_assign (ghb, Cin, M, A, d) ;
C2 = Cin ;
C2 (~M) = AT (~M) ;
assert (gbtest_eq (C2, Cout)) ;

I = [2 1 5] ;
J = [3 3 1 2] ;
B = sprandn (length (I), length (J), 0.5) ;
Cout = gtb_assign (ghb, Cin, B, {I}, {J}) ;
C2 = Cin ;
C2 (I,J) = B ;
assert (gbtest_eq (C2, Cout)) ;

A = rand (4) ;
G = gtb (ghb, A, 'by row') ;
M = logical (eye (4)) ;
B = rand (4) ;
H = gtb (ghb, B, 'by row') ;
A (M) = B (M) ;
G (M) = H (M) ;
assert (isequal (A, G)) ;

G = gtb (ghb, A, 'by row') ;
G (M) = gtb (ghb, H (M), 'bitmap') ;
assert (isequal (A, G)) ;

row_matlab = rand (1, 10) ;
row_gb = gtb (ghb, row_matlab) ;
J = [1 3 4] ;
row_matlab (J) = pi ;
row_gb ({J}) = pi ;
assert (isequal (row_matlab, row_gb)) ;

try
    G (M) = rand (2) ;
    ok = false ;
    msg = '' ;
catch me
    % error is expected
    msg = me.message ;
    ok = true ;
end
assert (ok) ;
assert (gb_contains (msg, ...
    'A must be a vector of length nnz(M) for logical indexing, C(M)=A')) ;

A = sprand (4, 4, 0.5) ;
C1 = pi * spones (A) ;
C2 = gtb_expand (ghb, pi, A) ;
assert (isequal (C1, C2)) ;

A = gtb (ghb, A) ;
C2 = gtb_expand (ghb, pi, A) ;
assert (isequal (C1, C2)) ;

C2 = gtb_expand (ghb, gtb (ghb, pi), A) ;
assert (isequal (C1, C2)) ;

fprintf ('gbtest10 (%d): all tests passed\n', ghb) ;

