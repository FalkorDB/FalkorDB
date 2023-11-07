function test237
%TEST237 test GrB_mxm (saxpy4)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

GB_mex_burble (1) ;
rng ('default') ;

n = 32 ;

for kind = 1:2
    
    if (kind == 1)
        add = 'plus' ;
        class = 'double' ;
    else
        add = 'times' ;
        class = 'single complex' ;
    end

    accum.opname = add ;
    accum.optype = class ;

    tol = 1e-6 ;

    semiring.multiply = 'times' ;
    semiring.add = add ;
    semiring.class = class ;
    semiring

    % k = 1 with a sparser A matrix

    A = GB_spec_random (n, n, 0.05, 1, class) ;
    B = GB_spec_random (n, 1, 0.3, 1, class) ;
    B.sparsity = 4 ;    % bitmap
    F = GB_spec_random (n, 1, inf, 1, class) ;
    F.sparsity = 8 ;    % full

    A.sparsity = 2 ;    % sparse
    C1 = GB_mex_mxm  (F, [ ], accum, semiring, A, B, [ ]) ;
    C2 = GB_spec_mxm (F, [ ], accum, semiring, A, B, [ ]) ;
    GB_spec_compare (C2, C1, 0, tol) ;

    % typecasting (to trigger the JIT)
    A.class = 'single' ;
    C1 = GB_mex_mxm  (F, [ ], accum, semiring, A, B, [ ]) ;
    C2 = GB_spec_mxm (F, [ ], accum, semiring, A, B, [ ]) ;
    GB_spec_compare (C2, C1, 0, tol) ;

end

GB_mex_burble (0) ;
fprintf ('test237: all tests passed\n') ;

