function test303
%TEST303 test C=A(I,J), method 6

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test303 ------------------  C=A(I,J)\n') ;

% construct the problem
n = 2000 ;
rng ('default') ;
A = sprand (n, n, 0.5) ;
nI = 4 ;
I = randi (2000, nI, 1) ;
I0 = uint64 (I-1) ;
I0

% test method 6 in GrB_extract
% C1 = A (I,:) ;
C1 = GB_mex_Matrix_subref (A, I0, [ ]) ;

% compare with MATLAB
B = double (A) ;
C2 = B (I,:) ;
assert (isequal (C1, C2)) ;

fprintf ('\ntest303: tests passed\n') ;

