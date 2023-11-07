function test228
%TEST228 test serialize/deserialize for all sparsity formats

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test228 C = serialize (A) for all sparsity formats and all types\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

rng ('default') ;

for k1 = 1:length (types)
    atype = types {k1} ;
    fprintf ('%s ', atype) ;
    for d = [0.5 inf]
        for n = [1 10]
            A = GB_spec_random (10, n, d, 128, atype) ;
            for A_sparsity = 0:15
                A.sparsity = A_sparsity ;
                C = GB_mex_serialize (A, -2) ;      % GrB_serialize
                GB_spec_compare (A, C) ;
                for method = [-1 0 1000 2000:2009 3000:3019]
                    C = GB_mex_serialize (A, method) ;
                    GB_spec_compare (A, C) ;
                end
            end
        end
    end
end

[save_nthreads, save_chunk] = nthreads_get ;
nthreads_set (4, 1) ;
d = 0.5 ;
A = GB_spec_random (1000, 1000, d, 128, 'double') ;
C = GB_mex_serialize (A, 0) ;
nthreads_set (save_nthreads, save_chunk) ;

fprintf ('\n') ;
fprintf ('test228: all tests passed\n') ;

