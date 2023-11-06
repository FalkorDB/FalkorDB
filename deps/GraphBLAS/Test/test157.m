function test157
%TEST157 test sparsity formats

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

for k1 = 1:length(types)
    in_type = types {k1} ;
    A = GB_spec_random (5, 5, 0.5, 10, in_type) ;

    for sparsity_control = 0:15
        A.sparsity = sparsity_control ;
        C = GB_mex_dump (A, 2) ;
        GB_spec_compare (C, A) ;
    end

    % try a full matrix
    A.matrix = 10 * rand (5, 5) ;
    A.pattern = true (5, 5) ;

    for sparsity_control = 0:15
        A.sparsity = sparsity_control ;
        C = GB_mex_dump (A, 2) ;
        GB_spec_compare (C, A) ;
    end

    % try a very sparse matrix
    A = GB_spec_random (50, 50, 0.002, 10, in_type) ;
    for is_hyper = 0:1
        A.is_hyper = is_hyper ;
        for sparsity_control = 0:15
            A.sparsity = sparsity_control ;
            C = GB_mex_dump (A, 2) ;
            GB_spec_compare (C, A) ;
        end
    end
end

% A = GrB (rand (40)) ;
% A (1,1) = sparse (0)

fprintf ('test157: all tests passed\n') ;

