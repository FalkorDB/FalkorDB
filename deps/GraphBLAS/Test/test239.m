function test239
%TEST239 test GxB_eWiseUnion

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 10 ;
A.matrix = 2 * spones (sprand (n, n, 0.5)) ;
A.iso = true ;
B.matrix = 3 * spones (sprand (n, n, 0.5)) ;
B.iso = true ;
C = sparse (n,n) ;
tol = 1e-12 ;

op.opname = 'plus' ;
op.optype = 'double' ;

GB_mex_burble (0) ;

% non-iso
alpha = 1 ;
beta = 2 ;
C1 = GB_mex_Matrix_eWiseUnion  (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
C2 = GB_spec_Matrix_eWiseUnion (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
GB_spec_compare (C1, C2, tol) ;

% iso
alpha = 2 ;
beta = 3 ;
C1 = GB_mex_Matrix_eWiseUnion  (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
C2 = GB_spec_Matrix_eWiseUnion (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
GB_spec_compare (C1, C2, tol) ;

% non-iso
alpha = 2 ;
beta = 4 ;
C1 = GB_mex_Matrix_eWiseUnion  (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
C2 = GB_spec_Matrix_eWiseUnion (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
GB_spec_compare (C1, C2, tol) ;

% non-iso
alpha = 1 ;
beta = 3 ;
C1 = GB_mex_Matrix_eWiseUnion  (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
C2 = GB_spec_Matrix_eWiseUnion (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
GB_spec_compare (C1, C2, tol) ;

op.opname = 'first' ;

% non-iso
alpha = 1 ;
C1 = GB_mex_Matrix_eWiseUnion  (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
C2 = GB_spec_Matrix_eWiseUnion (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
GB_spec_compare (C1, C2, tol) ;

alpha = 2 ;

% iso
C1 = GB_mex_Matrix_eWiseUnion  (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
C2 = GB_spec_Matrix_eWiseUnion (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
GB_spec_compare (C1, C2, tol) ;

op.opname = 'second' ;
alpha = 1 ;
beta = 2 ;

% non-iso
C1 = GB_mex_Matrix_eWiseUnion  (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
C2 = GB_spec_Matrix_eWiseUnion (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
GB_spec_compare (C1, C2, tol) ;

% iso
beta = 3 ;
C1 = GB_mex_Matrix_eWiseUnion  (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
C2 = GB_spec_Matrix_eWiseUnion (C, [ ], [ ], op, A, alpha, B, beta, [ ]) ;
GB_spec_compare (C1, C2, tol) ;

GB_mex_burble (0) ;
fprintf ('test239: all tests passed\n') ;

