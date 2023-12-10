function test98
%TEST98 test GrB_mxm, typecasting on the fly

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% This test is for the case when the auto AxB method selects the hash

rng ('default') ;

semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'double' ;

    n = 20 ;
    A.matrix = sprandn (n, n, 1) ;
    A.class = 'single' ;
    B.matrix = sprandn (n, n, 0.01) ;
    B.class = 'single' ;
    C.matrix = sparse (n, n) ;
    C.class = 'single' ;

    C1 = GB_mex_mxm  (C, [ ], [ ], semiring, A, B, [ ]) ;
    C2 = GB_spec_mxm (C, [ ], [ ], semiring, A, B, [ ]) ;
    GB_spec_compare (C1, C2) ;

fprintf ('\ntest98: all tests passed\n') ;

