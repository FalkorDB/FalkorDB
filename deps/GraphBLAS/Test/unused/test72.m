function test72
%TEST72 special cases for mxm, ewise, ...

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n--------------test72: special cases\n') ;

rng ('default') ;
clear

dnt = struct ( 'inp1', 'tran' ) ;
dtn = struct ( 'inp0', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

n = 20 ;
p = randperm (n) ;
A = speye (n) ;
A = A (p,p) ;
B = sprand (n, n, 0.2) ;
Z = sparse (n, n) ;

semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'double' ;

Mask = sparse (ones (n)) ;
C0 = GB_spec_mxm (Z, Mask, [ ], semiring, A, B, dtt);
C1 = GB_mex_mxm  (Z, Mask, [ ], semiring, A, B, dtt);

C2 = (A'*B') .* Mask ;
GB_spec_compare (C0, C1, 0) ;
assert (isequal (C2, C0.matrix)) ;

M = GB_mex_Matrix_eWiseAdd (Z, [ ], [ ], 'minus', Mask, Mask, [ ]) ;
C0 = GB_spec_mxm (Z, M.matrix, [ ], semiring, A, B, dtn);
C1 = GB_mex_mxm  (Z, M.matrix, [ ], semiring, A, B, dtn);
GB_spec_compare (C0, C1, 0) ;
assert (isequal (Z, sparse (C0.matrix))) ;

n = 500 ;
% n = 4 ;
A = speye (n) ;
% A = sparse (rand (n)) ;
B = sparse (rand (n)) ;
Z = sparse (n,n) ;

C0 = GB_mex_Matrix_eWiseMult  (Z, [ ], [ ], 'times', A, B, [ ]) ;
C1 = A .* B ;
assert (isequal (C1, sparse (C0.matrix))) ;

C0 = GB_mex_Matrix_eWiseMult  (Z, [ ], [ ], 'times', B, A, [ ]) ;
C1 = B .* A ;
assert (isequal (C1, sparse (C0.matrix))) ;

A = logical (A) ;
C0 = GB_mex_Matrix_eWiseMult  (Z, [ ], [ ], 'times', A, B, [ ]) ;
C1 = double (A) .* B ;
assert (isequal (C1, sparse (C0.matrix))) ;

C0 = GB_mex_Matrix_eWiseMult  (Z, [ ], [ ], 'times', B, A, [ ]) ;
C1 = B .* double (A) ;
assert (isequal (C1, sparse (C0.matrix))) ;

M = sprand (n, n, 0.01) ;
C0 = GB_mex_mxm  (Z, M, [ ], semiring, A, B, dnt) ;
C1 = (A*B') .* spones (M) ;
assert (isequal (C1, sparse (C0.matrix))) ;

fprintf ('test72: all tests passed\n') ;
