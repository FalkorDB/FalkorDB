function test241
%TEST241 test GrB_mxm: swap_rule

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test241 -------- GrB_mxm swap_rule\n') ;

rng ('default') ;

monoid.opname = 'plus' ;
monoid.optype = 'double' ;
semiring.add = monoid ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

n = 10 ;
is_csc = true ;

A_csc = GB_spec_random (n, n, 0.5, 100, 'double', true) ;
B_csc = GB_spec_random (n, n, 0.5, 100, 'double', true) ;

A_csr = GB_spec_random (n, n, 0.5, 100, 'double', false) ;
B_csr = GB_spec_random (n, n, 0.5, 100, 'double', false) ;
C_csr = GB_spec_random (n, n, 0.5, 100, 'double', false) ;

% C' = A'*B'
C1 = GB_mex_mxm  (C_csr, [ ], [ ], semiring, A_csr, B_csr, [ ]) ;
C2 = GB_spec_mxm (C_csr, [ ], [ ], semiring, A_csr, B_csr, [ ]) ;
GB_spec_compare (C1, C2, 0, 1e-12) ;

% C' = A'*B
C1 = GB_mex_mxm  (C_csr, [ ], [ ], semiring, A_csr, B_csc, [ ]) ;
C2 = GB_spec_mxm (C_csr, [ ], [ ], semiring, A_csr, B_csc, [ ]) ;
GB_spec_compare (C1, C2, 0, 1e-12) ;

% C' = A*B'
C1 = GB_mex_mxm  (C_csr, [ ], [ ], semiring, A_csc, B_csr, [ ]) ;
C2 = GB_spec_mxm (C_csr, [ ], [ ], semiring, A_csc, B_csr, [ ]) ;
GB_spec_compare (C1, C2, 0, 1e-12) ;

% C' = A*B
C1 = GB_mex_mxm  (C_csr, [ ], [ ], semiring, A_csc, B_csc, [ ]) ;
C2 = GB_spec_mxm (C_csr, [ ], [ ], semiring, A_csc, B_csc, [ ]) ;
GB_spec_compare (C1, C2, 0, 1e-12) ;

% C' = A'*B, triggering the swap_rule test
n = 1000 ;
A_csr = GB_spec_random (n, n, 0.01, 100, 'double', false) ;
B_csc = GB_spec_random (n, n, 0.01, 100, 'double', true) ;
C_csr = GB_spec_random (n, n, 0.01, 100, 'double', false) ;
C1 = GB_mex_mxm  (C_csr, [ ], [ ], semiring, A_csr, B_csc, [ ]) ;
C2 = A_csr.matrix * B_csc.matrix ;
GB_spec_compare (C1, C2, 0, 1e-12) ;

GB_mex_burble (0) ;
fprintf ('\ntest241: all tests passed\n') ;

