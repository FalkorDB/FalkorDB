function test193
%TEST193 test GxB_Matrix_diag and GrB_Matrix_diag

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test193 ----------- C = diag (v,k)\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

rng ('default') ;

n = 20 ;

for d = [1e-4 0.01 0.2 0.8 inf]
    fprintf ('\nd = %g\n', d) ;
    for ka = 1:length (types)
        vtype = types {ka} ;
        V = GB_spec_random (n, 1, d, 128, vtype) ;
        for sparsity_control = [2 4 8]
            fprintf ('.') ;
            V.sparsity = sparsity_control ;
            V.is_csc = true ;
            for kc = 1:length (types)
                ctype = types {kc} ;
                for k = [-10 -2 0 3 30]
                    for csc = 0:1
                        C2 = GB_spec_mdiag (V, k, ctype) ;
                        C1 = GB_mex_mdiag  (V, k, ctype, csc) ;
                        GB_spec_compare (C1, C2) ;
                    end
                end
            end
        end
    end
end

fprintf ('\n') ;
fprintf ('test193: all tests passed\n') ;

