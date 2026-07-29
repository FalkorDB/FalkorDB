function gbtest113 (ghb)
%GBTEST113 test ones and eq

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

ok = true ;
try
    C1 = gtb_ones (ghb, 5, 6, 'double', 'by row') ; %#ok<NASGU>
    ok = false ;
catch me
    fprintf ('error expected: %s\n', me.message) ;
end
assert (ok) ;

try
    C1 = gtb_ones (ghb, 5, 6, 'like') ; %#ok<NASGU>
    ok = false ;
catch me
    fprintf ('error expected: %s\n', me.message) ;
end
assert (ok) ;

A = magic (5) ;
A (1,2) = 0 ;
G = gtb (ghb, A, 'by row') ;
C1 = (0 == A) ;
C2 = (0 == G) ;
assert (isequal (C1, C2)) ;

fprintf ('\ngbtest113 (%d): all tests passed\n', ghb) ;

