function test76
%TEST76 test GxB_resize

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

fprintf ('\n-------------- GrB_resize\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

for k1 = 1:length(types)
    type = types {k1} ;
    fprintf ('\n%-14s ', type) ;
    for nrows_old = [1 2 5 10 100]
        for ncols_old = [1 2 5 10 100]

            fprintf ('.') ;
            for A_is_hyper = 0:1
            for A_is_csc   = 0:1

            A = GB_spec_random (nrows_old, ncols_old, 0.5, 99, type, ...
                A_is_hyper, A_is_csc) ;
            for nrows_new = [1 2 5 10 ]
                for ncols_new = [1 2 5 100]
                    C1 = GB_spec_resize (A, nrows_new, ncols_new) ;
                    C2 = GB_mex_resize  (A, nrows_new, ncols_new) ;
                    GB_spec_compare (C1, C2, 0) ;
                end
            end
            end
            end
        end
    end
end

fprintf ('\ntest76: all tests passed\n') ;
