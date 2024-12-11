function test78
%TEST78 test subref

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

n = 500 ;
I = speye (n) ;
X = sparse (rand (n)) ;
A = [X I ; I I] ;

I = 1:n ;
I0 = uint64 (I-1) ;

C1 = A(I,I) ;
C2 = GB_mex_Matrix_subref (A, I0, I0) ;
assert (isequal (C1, C2)) ;

Ahyper.matrix = A ;
Ahyper.is_hyper = true ;
% this requires a hyper realloc for C2
C2 = GB_mex_Matrix_subref (Ahyper, I0, I0) ;
assert (isequal (C1, C2)) ;

% C and A bitmap, with GB_J_KIND = GB_LIST (for JIT kernel, GB_macrofy_subref)
A0.matrix = sparse (A) ;
A0.sparsity = 4 ;
I = randperm (n) ;
I0 = uint64 (I-1) ;
C1 = A(I,I) ;
C2 = GB_mex_Matrix_subref (A0, I0, I0) ;
assert (isequal (C1, C2)) ;

% C and A bitmap, with GB_J_KIND = GB_STRIDE (for JIT kernel, GB_macrofy_subref)
clear I1
I1.begin = 0 ;
I1.inc = 2 ;
I1.end = n-1 ;
I = 1:2:n ;
C1 = A(I,I) ;
C2 = GB_mex_Matrix_subref (A0, I1, I1) ;
assert (isequal (C1, C2)) ;

fprintf ('\ntest78: all tests passed\n') ;

