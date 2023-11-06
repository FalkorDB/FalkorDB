function test235
%TEST235 test GxB_eWiseUnion and GrB_eWiseAdd

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

fprintf ('test235 -----------GxB_eWiseUnion and GrB_eWiseAdd\n') ;

m = 200 ;
n = 200 ;

rng ('default') ;

A = GB_spec_random (m, n, 0.1) ; A.sparsity_control = 1 ;
B = GB_spec_random (m, n, 0.1) ; B.sparsity_control = 1 ;
C = sparse (m,n) ;

plus.opname = 'plus' ;
plus.optype = 'double' ;
tol = 1e-12 ;

C1 = GB_mex_Matrix_eWiseAdd  (C, [ ], [ ], plus, A, B, [ ]) ;
C2 = GB_spec_Matrix_eWiseAdd (C, [ ], [ ], plus, A, B, [ ]) ;
GB_spec_compare (C1, C2, 0, tol) ;

C1 = GB_mex_Matrix_eWiseUnion  (C, [ ], [ ], plus, A, 1, B, 2, [ ]) ;
C2 = GB_spec_Matrix_eWiseUnion (C, [ ], [ ], plus, A, 1, B, 2, [ ]) ;
GB_spec_compare (C1, C2, 0, tol) ;

fprintf ('\ntest235: all tests passed\n') ;

