function test07b
%TEST07B test GrB_assign with a single pending tuple

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% adds a single pending tuple

rng ('default') ;
C = sparse (rand (5,4)) ;
C (2,3) = 0 ;
A = 100 * sparse (magic (2)) ;

I = [2 3] ;
J = [3 4] ;

C2 = C ;
C2 (I,J) = A ;

% full (C)
% full (C2)

I0 = uint64 (I-1) ;
J0 = uint64 (J-1) ;

C3 = GB_mex_assign (C, [ ], '', A, I0, J0, [ ]) 
% C3.matrix
% full (C3.matrix)
assert (isequal (C3.matrix, C2))

I0 = I0 (1) ;
I = I (1) ;
M = sparse ([0 1]) ;
% note that the Mask has the same size as C
C3 = GB_mex_assign (C, C, '', A(1,:), I0, J0, [ ]) ;
C2 = C ;
C2 (I,J(2)) = A (1,2) ;
assert (isequal (C3.matrix, C2))

fprintf ('\ntest07b: all tests passed\n') ;

