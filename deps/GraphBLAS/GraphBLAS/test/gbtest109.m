function gbtest109 (ghb)
%GBTEST109 test num2cell

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

S = magic (5) ;
A = gtb (ghb, S) ;

ok = true ;
try
    C = num2cell (A, 3) %#ok<*NOPRT,*NASGU>
    ok = false ;
catch me
    fprintf ('error expected: %s\n', me.message) ;
end
assert (ok) ;

dim = 1 ;
C1 = num2cell (S, gtb (ghb, dim)) ;
C2 = num2cell (S, dim) ;
assert (isequal (C1, C2)) ;

fprintf ('\ngbtest109 (%d): all tests passed\n', ghb) ;

