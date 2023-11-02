function test78
%TEST78 test subref

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

n = 500 ;
I = speye (n) ;
X = sparse (rand (n)) ;
A = [X I ; I I] ;

I = 1:n ;
I0 = uint64 (I-1) ;

C1 = A(I,I) ;
C2 = GB_mex_Matrix_subref (A, I0, I0) ;

Ahyper.matrix = A ;
Ahyper.is_hyper = true ;

% this requires a hyper realloc for C2
C2 = GB_mex_Matrix_subref (Ahyper, I0, I0) ;

assert (isequal (C1, C2)) ;

fprintf ('\ntest78: all tests passed\n') ;
