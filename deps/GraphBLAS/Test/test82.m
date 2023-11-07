function test82
%TEST82 test GrB_Matrix_extract with index range (hypersparse)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test82: test GrB_Matrix_extract with index range (hypersparse)\n') ;

rng ('default') ;

n = 100 ;
A = GB_spec_random (n, n, 0.02, 100, 'double', true, true) ;

I.begin = 0 ;
I.inc = 0 ;
I.end = 0 ;

J.begin = 0 ;
J.end = 9 ;

C1 = A.matrix (1:0:1, 1:10) ;
S = sparse (0,10) ;
C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I, J, [ ]) ;

assert (isequal (C1, C2.matrix)) ;

C1 = A.matrix (1:10, 1:0:1) ;
S = sparse (10,0) ;
C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, J, I, [ ]) ;
assert (isequal (C1, C2.matrix)) ;

for k = 1:n
    K.begin = k-1 ;
    K.inc = 1 ;
    K.end = k-1 ;
    C1 = A.matrix (1:10, k) ;
    S = sparse (10,1) ;
    C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, J, K, [ ]) ;
    assert (isequal (C1, C2.matrix)) ;
end

B = GB_spec_random (n, n, 0.2, 100, 'double', true, true) ;

% I not contiguous, with duplicates, but no sort needed
I1 = [1 3 4 30 30 50 50 50 99 99 99 100] ;
I0 = uint64 (I1) - 1 ;
ni = length (I1) ;
S = sparse (ni,1) ;

for j = 1:n
    C1 = B.matrix (I1, j) ;
    J0 = uint64 (j)-1 ;
    C2 = GB_mex_Matrix_extract (S, [ ], [ ], B, I0, J0, [ ]) ;
    assert (isequal (C1, C2.matrix)) ;
end

% A hypersparse, but C is not.
d = sum (spones (A.matrix)) ;
j = find (d > 0, 1, 'first') ;

nJ = 1000 ;
J1 = j * ones (1, nJ) ;
J0 = uint64 (J1) - 1 ;

C1 = A.matrix (:, J1) ;
S = sparse (n, nJ) ;
C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, [ ], J0, [ ]) ;
assert (isequal (C1, C2.matrix)) ;

fprintf ('\ntest82: all tests passed\n') ;


