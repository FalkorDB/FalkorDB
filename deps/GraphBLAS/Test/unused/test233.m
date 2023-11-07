function test233
%TEST233 test bitmap saxpy C=A*B with A sparse and B bitmap

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'double' ;
tol = 1e-12 ;

for m = [1 10 100]
    for n = [1 4 8 10 100]
        for k = [1 10 100]

            A = GB_spec_random (m, k, 0.4) ;
            B = GB_spec_random (k, n, 0.4) ;
            C = GB_spec_random (m, n, 0) ;

            B.matrix = pi * spones (B.matrix) ;
            B.iso = true ;
            B.sparsity = 4 ;    % bitmap

            A.sparsity = 2 ;    % sparse

            C1 = GB_mex_mxm  (C, [ ], [ ], semiring, A, B, [ ]) ;
            C2 = GB_spec_mxm (C, [ ], [ ], semiring, A, B, [ ]) ;

            GB_spec_compare (C1, C2, 0, tol) ;
        end
    end
end

fprintf ('\ntest233: all tests passed\n') ;

