function gbtest109
%GBTEST109 test num2cell

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

S = magic (5) ;
A = GrB (S) ;

ok = true ;
try
    C = num2cell (A, 3) %#ok<*NOPRT,*NASGU>
    ok = false ;
catch me
    fprintf ('error expected: %s\n', me.message) ;
end
assert (ok) ;

dim = 1 ;
C1 = num2cell (S, GrB (dim)) ;
C2 = num2cell (S, dim) ;
assert (isequal (C1, C2)) ;

fprintf ('\ngbtest109: all tests passed\n') ;

