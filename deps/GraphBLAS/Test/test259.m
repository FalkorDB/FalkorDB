function test259
%TEST259 test with plus_plus semiring

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test259 ---C+=A''*B when C is dense, with dot4, plus_plus_fp32\n') ;

rng ('default') ;

A.matrix = sparse (rand (4)) ;
A.class = 'single' ;

B.matrix = sparse (rand (4)) ;
B.class = 'single' ;

C.matrix = sparse (rand (4)) ;
C.pattern = logical (ones (4)) ;
C.sparsity = 8 ;
C.iso = false ;
C.class = 'single' ;

AT.matrix = A.matrix' ;
AT.class = 'single' ;
BT.matrix = B.matrix' ;
BT.class = 'single' ;

semiring.add = 'plus' ;
semiring.multiply = 'plus' ;
semiring.class = 'single' ;
[mult_op add_op id] = GB_spec_semiring (semiring) ;

dnn = struct ('axb', 'dot') ;
dtn = struct ('axb', 'dot', 'inp0', 'tran') ;
dnt = struct ('axb', 'dot', 'inp1', 'tran') ;
dtt = struct ('axb', 'dot', 'inp0', 'tran', 'inp1', 'tran') ;

tol = 1e-5 ;

C2 = GB_mex_mxm  (C, [ ], add_op, semiring, A, B, dnn) ;
C1 = GB_spec_mxm (C, [ ], add_op, semiring, A, B, dnn) ;
C3 = GB_mex_plusplus (C, [ ], add_op, [ ], A, B, dnn) ;
GB_spec_compare (C1, C2, 0, tol) ;
GB_spec_compare (C1, C3, 0, tol) ;

C2 = GB_mex_mxm  (C, [ ], add_op, semiring, AT, B, dtn) ;
C1 = GB_spec_mxm (C, [ ], add_op, semiring, AT, B, dtn) ;
C3 = GB_mex_plusplus (C, [ ], add_op, [ ], AT, B, dtn) ;
GB_spec_compare (C1, C2, 0, tol) ;
GB_spec_compare (C1, C3, 0, tol) ;

C2 = GB_mex_mxm  (C, [ ], add_op, semiring, A, BT, dnt) ;
C1 = GB_spec_mxm (C, [ ], add_op, semiring, A, BT, dnt) ;
C3 = GB_mex_plusplus (C, [ ], add_op, [ ], A, BT, dnt) ;
GB_spec_compare (C1, C2, 0, tol) ;
GB_spec_compare (C1, C3, 0, tol) ;

C2 = GB_mex_mxm  (C, [ ], add_op, semiring, AT, BT, dtt) ;
C1 = GB_spec_mxm (C, [ ], add_op, semiring, AT, BT, dtt) ;
C3 = GB_mex_plusplus (C, [ ], add_op, [ ], AT, BT, dtt) ;
GB_spec_compare (C1, C2, 0, tol) ;
GB_spec_compare (C1, C3, 0, tol) ;

fprintf ('test259: all tests passed\n') ;

