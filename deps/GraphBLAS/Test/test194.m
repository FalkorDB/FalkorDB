function test194
%TEST194 test GxB_Vector_diag

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test194 ----------- V = diag (A,k)\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

rng ('default') ;
GB_builtin_complex_set (true) ;

ms = [10 20] ;
ns = [4 10] ;

for d = [1e-4 0.1 0.8 inf]
    fprintf ('\nd = %g\n', d) ;
    for ka = 1:length (types)
        atype = types {ka} ;
        for m = ms
            for n = ns
                A = GB_spec_random (m, n, d, 128, atype) ;
                for sparsity_control = [1 2 4 8]
                    fprintf ('.') ;
                    A.sparsity = sparsity_control ;
                    for csc = [1 0]
                        A.is_csc = csc ;
                        for kc = 1:length (types)
                            vtype = types {kc} ;
                            for k = [-10 -2 0 3 ]
                                V2 = GB_spec_vdiag (A, k, vtype) ;
                                V1 = GB_mex_vdiag  (A, k, vtype) ;
                                GB_spec_compare (V1, V2) ;
                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf ('\n') ;
fprintf ('test194: all tests passed\n') ;

