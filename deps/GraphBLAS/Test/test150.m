function test150
%TEST150 test GrB_mxm with typecasting and zombies (dot3 and saxpy)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test150: ------- GrB_mxm with typecasting and zombies (dot3)\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;
identity = 0 ;

dnn_dot   = struct ( 'axb', 'dot') ;
dnn_saxpy = struct ( 'axb', 'saxpy') ;

m = 8 ;
n = 5 ;
s = 4 ;
density = 0.1 ;

for k6 = 1:length (types)
    atype = types {k6} ;
    fprintf ('%s ', atype) ;

    A = GB_spec_random (m, s, density, 100, atype) ;
    B = GB_spec_random (s, n, density, 100, atype) ;
    B.is_hyper = true ;
    M_matrix = GB_random_mask (m,n,0.2) ;
    M.matrix = M_matrix ;
    M.is_hyper = true ;
    M.class = 'logical' ;

    clear C
    C.matrix = sparse (m,n) ;
    C.class = 'int32' ;
    C.pattern = false (m,n) ;

    % dot3
    C0 = GB_spec_mxm (C, M, [ ], semiring, A, B, dnn_dot);
    C1 = GB_mex_mxm  (C, M, [ ], semiring, A, B, dnn_dot);
    GB_spec_compare (C0, C1, identity) ;

    % saxpy 
    C0 = GB_spec_mxm (C, M, [ ], semiring, A, B, dnn_saxpy);
    C1 = GB_mex_mxm  (C, M, [ ], semiring, A, B, dnn_saxpy);
    GB_spec_compare (C0, C1, identity) ;

end

fprintf ('\ntest150: all tests passed\n') ;

