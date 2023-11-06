function test15
%TEST15 test AxB and AdotB internal functions

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n --------------------- GB_mex_AxB, GB_mex_AdotB tests\n') ;

rng ('default') ;
m = 8 ;
k = 6 ;
n = 10 ;
A = sprand (m,k,0.5) ;
B = sprand (k,n,0.5) ;
C1 = A*B ;
C = GB_mex_AxB (A, B) ;
assert (GB_spok (C) == 1) ;
assert (norm (C-C1,1) / norm (C,1)< 1e-12) ;

A = A' ;
C1 = A'*B ;
C = GB_mex_AdotB (A, B) ;
assert (GB_spok (C) == 1) ;
assert (isequal (C, C1)) ;

A = sprandn (10000,2,0.5) ;
B = sprandn (10000,2,0.0001) ;
A (5,1) = pi ;
B (5,2) = 42 ;
C1 = A'*B ;
C = GB_mex_AdotB (A, B) ;
assert (GB_spok (C) == 1) ;
assert (isequal (C, C1)) ;
C1 = B'*A ;
C = GB_mex_AdotB (B, A) ;
assert (GB_spok (C) == 1) ;
assert (isequal (C, C1)) ;

S = sparse (10000,2) ;
C1 = A.*B ;
C = GB_mex_Matrix_eWiseMult (S, [], [], 'times', A, B) ;
assert (GB_spok (C.matrix) == 1) ;
assert (isequal (C.matrix, C1)) ;
C1 = B.*A ;
C = GB_mex_Matrix_eWiseMult (S, [], [], 'times', B, A) ;
assert (GB_spok (C.matrix) == 1) ;
assert (isequal (C.matrix, C1)) ;

fprintf ('\ntest15: all tests passed\n') ;

