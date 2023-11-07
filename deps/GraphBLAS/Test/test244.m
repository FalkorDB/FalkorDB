function test244
%TEST244 test reshape

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

for k1 = 1:length(types)
    type = types {k1} ;
    fprintf ('\n%-14s ', type) ;

    for m = [1 2 6] % 1:6
        for n = [1 2 6] % 1:6
            mn = m*n ;
            f = factor (mn) ;
            for d = [0.3 inf]
                A = GB_spec_random (m, n, d, 99, type) ;
                fprintf ('.') ;
                for sparsity = [1 2 4 8]
                    A.sparsity = sparsity ;
                    for is_csc = [0 1]
                        A.is_csc = is_csc ;
                        for iso = [false true]
                            A.iso = iso ;

                            for k = 1:length (f)
                                S = nchoosek (f, k) ;
                                for i = 1:size(S,1)

                                    m2 = prod (S (i,:)) ;
                                    n2 = mn / m2 ;

                                    % reshape by column
                                    C1 = A ;
                                    x = 1 ;
                                    if (iso)
                                        [i,j,x] = find (C1.matrix, 1,'first') ;
                                        C1.matrix (C1.pattern) = x ;
                                    end
                                    C1.matrix  = reshape (C1.matrix,  m2, n2) ;
                                    C1.pattern = reshape (C1.pattern, m2, n2) ;

                                    for inplace = [false true]
                                        C2 = GB_mex_reshape (A, m2, n2, ...
                                            true, inplace) ;
                                        GB_spec_compare (C1, C2, 0) ;
                                    end

                                    % reshape by row
                                    C1 = A ;
                                    if (iso)
                                        C1.matrix (C1.pattern) = x ;
                                    end
                                    C1.matrix  = reshape (C1.matrix', n2, m2)' ;
                                    C1.pattern = reshape (C1.pattern', n2, m2)';
                                    for inplace = [false true]
                                        C2 = GB_mex_reshape (A, m2, n2, ...
                                            false, inplace) ;
                                        GB_spec_compare (C1, C2, 0) ;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf ('\ntest244: all tests passed\n') ;
