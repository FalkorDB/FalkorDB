function gbtest108
%GBTEST108 test mat2cell

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

S = magic (5) ;
A = GrB (S) ;
C1 = mat2cell (A, [2 3]) ;
C2 = mat2cell (S, [2 3]) ;
assert (isequal (C1, C2)) ;
C3 = mat2cell (S, GrB ([2 3])) ;
assert (isequal (C1, C3)) ;

fprintf ('\ngbtest108: all tests passed\n') ;

