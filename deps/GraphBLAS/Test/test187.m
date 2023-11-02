function test187
%TEST187 test dup/assign for all sparsity formats

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test187 ----------- C = A for all sparsity formats and all types\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

rng ('default') ;

for k1 = 1:length (types)
    atype = types {k1} ;
    fprintf ('\n%s', atype) ;
    for d = [0.5 inf]
        A = GB_spec_random (10, 10, d, 128, atype) ;
        for A_sparsity = 0:15
            fprintf ('.') ;
            A.sparsity = A_sparsity ;
            for C_sparsity = 0:15
                for method = 0:2
                    % no typecast, but do change the sparsity
                    C = GB_mex_dup (A, atype, method, C_sparsity) ;
                    GB_spec_compare (A, C) ;
                end
            end
        end
    end
end

fprintf ('\n') ;
GB_mex_burble (0) ;
fprintf ('test187: all tests passed\n') ;

