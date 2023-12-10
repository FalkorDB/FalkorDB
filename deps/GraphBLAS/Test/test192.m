function test192
%TEST192 test GrB_assign C<C,struct>=scalar

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

m = 10 ;
n = 14 ;

rng ('default') ;

desc.mask = 'structural' ;

for k = 1:length (types)

    ctype = types {k} ;
    fprintf ('%s, ', ctype) ;

    for d = [0.5 inf]

        C = GB_spec_random (m, n, d, 100, ctype) ;

        for k2 = 1:length (types)

            stype = types {k} ;

            S = GB_spec_random (1, 1, inf, 100, stype) ;
            S.matrix = sparse (S.matrix) ;

            for C_sparsity = 1:15
                C.sparsity = C_sparsity ;

                % C<C,struct> = scalar
                C1 = GB_mex_assign_alias_mask_scalar (C, S) ;
                C2 = GB_spec_assign (C, C, [ ], S, [ ], [ ], desc, true) ;
                GB_spec_compare (C1, C2) ;
            end
        end
    end
end

fprintf ('\ntest192: all tests passed\n') ;

