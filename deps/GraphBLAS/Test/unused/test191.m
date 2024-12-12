function test191
%TEST191 test split

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test191 ----------- Tiles = split (A)\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

rng ('default') ;

m = 100 ;
n = 110 ;
ms = [10 1 89] ;
ns = [1 4 50 45 10] ;

for d = [1e-4 0.01 0.2 0.8 inf]
    fprintf ('\nd = %g\n', d) ;
    for ka = 1:length (types)
        atype = types {ka} ;
        A = GB_spec_random (m, n, d, 128, atype) ;
        for sparsity_control = [1 2 4 8]
            fprintf ('.') ;
            A.sparsity = sparsity_control ;
            for is_csc = [0 1]
                A.is_csc = is_csc ;
                C2 = GB_spec_split (A, ms, ns) ;
                C1 = GB_mex_split  (A, ms, ns) ;
                for i = 1:length(ms)
                    for j = 1:length(ns)
                        GB_spec_compare (C1 {i,j}, C2 {i,j}) ;
                    end
                end

                if (nnz (A.matrix) > 0)
                    % also try the iso case
                    B = A ;
                    B.matrix = spones (A.matrix) * pi ;
                    B.iso = true ;
                    C2 = GB_spec_split (B, ms, ns) ;
                    C1 = GB_mex_split  (B, ms, ns) ;
                    for i = 1:length(ms)
                        for j = 1:length(ns)
                            GB_spec_compare (C1 {i,j}, C2 {i,j}) ;
                        end
                    end
                end
            end
        end
    end
end

fprintf ('\n') ;
fprintf ('test191: all tests passed\n') ;

