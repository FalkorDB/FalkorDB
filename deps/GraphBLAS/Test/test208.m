function test208
%TEST208 test iso apply, bind 1st and 2nd

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

% GB_mex_burble (1) ;
n = 100 ;
A.matrix = spones (sprand (n, n, 0.5)) ;
A.iso = true ;
A.class = 'double' ;

op.opname = 'minus' ;
op.optype = 'double' ;

op_single.opname = 'minus' ;
op_single.optype = 'single' ;

x.matrix = pi ;
x.class = 'double' ;

X.matrix = pi * spones (A.matrix) ;
X.class = 'double' ;

C = sparse (n, n) ;
tol = 0 ;

C1 = GB_mex_apply1 (C, [ ], [ ], op, 0, x, A, [ ]) ;
C2 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op, X, A, [ ]) ;
GB_spec_compare (C1, C2, 0, tol) ;

C1 = GB_mex_apply1 (C, [ ], [ ], op_single, 0, x, A, [ ]) ;
C2 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op_single, X, A, [ ]) ;
GB_spec_compare (C1, C2, 0, tol) ;

C1 = GB_mex_apply2 (C, [ ], [ ], op, 0, A, x, [ ]) ;
C2 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op, A, X, [ ]) ;
GB_spec_compare (C1, C2, 0, tol) ;

C1 = GB_mex_apply2 (C, [ ], [ ], op_single, 0, A, x, [ ]) ;
C2 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op_single, A, X, [ ]) ;
GB_spec_compare (C1, C2, 0, tol) ;

GB_mex_burble (0) ;
fprintf ('\ntest208: all tests passed\n') ;

