function test169
%TEST169 C<M>=A+B with different sparsity formats

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

fprintf ('test169:\n') ;

n = 50 ;

desc = struct ('mask', 'complement') ;

for trial = 1:5

    C = GB_spec_random (n, n, 0.5, 1, 'double') ;
    M = GB_spec_random (n, n, 0.2, 1, 'double') ;
    A = GB_spec_random (n, n, 0.5, 1, 'double') ;
    B = GB_spec_random (n, n, 0.5, 1, 'double') ;

    % test the no_hyper_hash cases
    A.no_hyper_hash = true ;
    C.no_hyper_hash = true ;
    B.no_hyper_hash = true ;

    for C_sparsity = [1 2 4 8]
        C.sparsity = C_sparsity ;

        for M_sparsity = [1 2 4 8]
            M.sparsity = M_sparsity ;

            for A_sparsity = [1 2 4 8]
                A.sparsity = A_sparsity ;

                for B_sparsity = [1 2 4 8]
                    B.sparsity = B_sparsity ;

                    C1 = GB_spec_Matrix_eWiseAdd (C, M, [], 'plus', A, B, desc);
                    C2 = GB_mex_Matrix_eWiseAdd  (C, M, [], 'plus', A, B, desc);
                    GB_spec_compare (C1, C2) ;

                    C1 = GB_spec_Matrix_eWiseAdd (C, M, [], 'plus', A, B, [ ]) ;
                    C2 = GB_mex_Matrix_eWiseAdd  (C, M, [], 'plus', A, B, [ ]) ;
                    GB_spec_compare (C1, C2) ;
                end
            end
        end
        fprintf ('.') ;
    end
end

fprintf ('\ntest169: all tests passed\n') ;

