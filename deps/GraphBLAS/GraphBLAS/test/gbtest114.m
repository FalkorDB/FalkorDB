function gbtest114 (ghb)
%GBTEST114 test kron with iso matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = gtb_random (ghb, 5, 10, 0.4) ;
B = gtb_ones (ghb, 3, 2) ;

C1 = kron (A, B) ;
C2 = kron (double (A), double (B)) ;
assert (isequal (C1, C2)) ;

C1 = kron (B, A) ;
C2 = kron (double (B), double (A)) ;
assert (isequal (C1, C2)) ;

C1 = kron (B, B) ;
C2 = kron (double (B), double (B)) ;
assert (isequal (C1, C2)) ;

fprintf ('\ngbtest114 (%d): all tests passed\n', ghb) ;

