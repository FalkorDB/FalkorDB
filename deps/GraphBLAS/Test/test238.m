function test238
%TEST238 test GrB_mxm (dot4 and dot2)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

desc.inp0 = 'tran' ;

n = 8 ;

for k = [1 2 4 8 32]
    fprintf ('\n%2d', k) ;
    for iso = [0 1]

        clear F
        if (iso)
            F.matrix = pi * ones (n,k) ;
        else
            F = GB_spec_random (n, k, inf) ;
        end
        F.sparsity = 8 ;    % full
        F.iso = iso ;
        C0 = sparse (n,k) ;

        for A_sparsity = [1 2 4 8]
            for B_sparsity = [1 2 4 8]

                fprintf ('.') ;
                clear A B
                if (A_sparsity == 8)
                    A = GB_spec_random (n, n, inf) ;
                else
                    A = GB_spec_random (n, n, 0.3) ;
                end
                A.sparsity = A_sparsity ;

                if (B_sparsity == 8)
                    B = GB_spec_random (n, k, inf) ;
                else
                    B = GB_spec_random (n, k, 0.3) ;
                end
                B.sparsity = B_sparsity ;

                for trial = 1:5

                    if (trial == 1)
                        % plus_times_double
                        accum.opname = 'plus' ;
                        accum.optype = 'double' ;
                        semiring.add = 'plus' ;
                        semiring.multiply = 'times' ;
                        semiring.class = 'double' ;
                        tol = 1e-12 ;
                        A.class = 'double' ;
                        B.class = 'double' ;
                        F.class = 'double' ;
                    elseif (trial == 2)
                        % max_firstj_int64
                        accum.opname = 'max' ;
                        accum.optype = 'int64' ;
                        semiring.add = 'max' ;
                        semiring.multiply = 'firstj' ;
                        semiring.class = 'int64' ;
                        tol = 0 ;
                        A.class = 'int64' ;
                        B.class = 'int64' ;
                        F.class = 'int64' ;
                    elseif (trial == 3)
                        % max_firstj1_int64
                        semiring.multiply = 'firstj1' ;
                    elseif (trial == 4)
                        % min_firstj_int64
                        accum.opname = 'min' ;
                        semiring.add = 'min' ;
                        semiring.multiply = 'firstj' ;
                    else
                        % min_firstj1_int64
                        semiring.multiply = 'firstj1' ;
                    end

                    % C = F ; C += A'*B, using dot4
                    C1 = GB_mex_mxm  (F, [ ], accum, semiring, A, B, desc) ;
                    C2 = GB_spec_mxm (F, [ ], accum, semiring, A, B, desc) ;
                    GB_spec_compare (C1, C2, tol) ;

                    % C = A'*B, using dot2
                    C1 = GB_mex_mxm  (C0, [ ], [ ], semiring, A, B, desc) ;
                    C2 = GB_spec_mxm (C0, [ ], [ ], semiring, A, B, desc) ;
                    GB_spec_compare (C1, C2, tol) ;

                end
            end
        end
    end
end

fprintf ('\ntest238: all tests passed\n') ;

