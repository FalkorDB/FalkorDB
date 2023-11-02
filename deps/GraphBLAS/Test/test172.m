function test172
%TEST172 eWiseMult with M bitmap/full

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

fprintf ('test172:\n') ;

n = 500 ;

    A = GB_spec_random (n, n, 0.05, 1, 'double') ;
    A.sparsity = 2 ;    % sparse

    B = GB_spec_random (n, n, 0.05, 1, 'double') ;
    B.sparsity = 2 ;    % sparse

    M = GB_spec_random (n, n, 0.1, 1, 'double') ;
    M.matrix (1,1) = 1 ;
    M.pattern (1,1) = true ;
    M.sparsity = 4 ;    % bitmap

    Cin = sparse (n,n) ;

    C1 = GB_spec_Matrix_eWiseMult (Cin, M, [ ], 'times', A, B, [ ]) ;
    C2 = GB_mex_Matrix_eWiseMult  (Cin, M, [ ], 'times', A, B, [ ]) ;
    C3 = spones (M.matrix) .* (A.matrix .* B.matrix) ;
    GB_spec_compare (C1, C2) ;
    GB_spec_compare (C1, C3) ;

    A2 = A ;
    B2 = B ;

    A2.matrix (:,1) = sprand (n, 1, 0.9) ;
    A2.pattern = logical (A2.matrix) ;

    B2.matrix (:,1) = sprand (n, 1, 0.01) ;
    B2.matrix (1,1) = 3 ;
    B2.pattern = logical (B2.matrix) ;

    C1 = GB_spec_Matrix_eWiseMult (Cin, M, [ ], 'times', A2, B2, [ ]) ;
    C2 = GB_mex_Matrix_eWiseMult  (Cin, M, [ ], 'times', A2, B2, [ ]) ;
    C3 = spones (M.matrix) .* (A2.matrix .* B2.matrix) ;
    GB_spec_compare (C1, C2) ;
    GB_spec_compare (C1, C3) ;

    C1 = GB_spec_Matrix_eWiseMult (Cin, M, [ ], 'times', B2, A2, [ ]) ;
    C2 = GB_mex_Matrix_eWiseMult  (Cin, M, [ ], 'times', B2, A2, [ ]) ;
    C3 = spones (M.matrix) .* (B2.matrix .* A2.matrix) ;
    GB_spec_compare (C1, C2) ;
    GB_spec_compare (C1, C3) ;

fprintf ('test172: all tests passed\n') ;

