function test251
%TEST251 test dot4 for plus-pair semirings
% GB_AxB_dot4 computes C+=A'*B when C is dense.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test251 ------------ C+=A''*B when C is dense (plus-pair)\n') ;

rng ('default') ;
GB_mex_burble (0) ;

plus_pair.add = 'plus' ;
plus_pair.multiply = 'oneb' ;   % same as pair
[~, ~, ~, types, ~, ~, ~,] = GB_spec_opsall ;
types = types.all ;

add_op.opname = 'plus' ;
dtn_dot = struct ('axb', 'dot', 'inp0', 'tran') ;
dtn_sax = struct ('axb', 'saxpy', 'inp0', 'tran') ;

n = 20 ;
C = GB_spec_random (n, n, inf, 100, 'double') ;
C.sparsity = 8 ;
C0.matrix = sparse (n, n) ;

for A_sparsity = [1 2 4 8]
    if (A_sparsity == 8)
        A = GB_spec_random (n, n, inf, 100, 'double') ;
    else
        A = GB_spec_random (n, n, 0.1, 100, 'double') ;
    end
    A.sparsity = A_sparsity ;

    for B_sparsity = [1 2 4 8]
        if (B_sparsity == 8)
            B = GB_spec_random (n, n, inf, 100, 'double') ;
        else
            B = GB_spec_random (n, n, 0.1, 100, 'double') ;
        end
        B.sparsity = B_sparsity ;

        for k = 0:length(types)
            if (k == 0)
                type = 'logical' ;
                add_op.opname = 'xor' ;
                plus_pair.add = 'xor' ;
            else
                type = types {k} ;
                add_op.opname = 'plus' ;
                plus_pair.add = 'plus' ;
            end
            plus_pair.class = type ;
            add_op.optype = type ;
            if (test_contains (type, 'single'))
                tol = 1e-5 ;
            else
                tol = 1e-10 ;
            end
            fprintf ('.') ;

            for k2 = 1 % 1:2
                if (k2 == 1)
                    A.class = type ;
                    B.class = type ;
                    C0.class = type ;
                    C.class = type ;
                else
                    A.class = 'double' ;
                    B.class = 'double' ;
                    C0.class = 'double' ;
                    C.class = 'double' ;
                end

                % X = C + A'*B using dot4
                X2 = GB_mex_mxm  (C, [ ], add_op, plus_pair, A, B, dtn_dot) ;
                X1 = GB_spec_mxm (C, [ ], add_op, plus_pair, A, B, dtn_dot) ;
                GB_spec_compare (X1, X2, 0, tol) ;

                % X = A'*B using dot2/dot3
                X2 = GB_mex_mxm  (C0, [ ], [ ], plus_pair, A, B, dtn_dot) ;
                X1 = GB_spec_mxm (C0, [ ], [ ], plus_pair, A, B, dtn_dot) ;
                GB_spec_compare (X1, X2, 0, tol) ;

                % X = C + A'*B using saxpy
                X2 = GB_mex_mxm  (C, [ ], add_op, plus_pair, A, B, dtn_sax) ;
                X1 = GB_spec_mxm (C, [ ], add_op, plus_pair, A, B, dtn_sax) ;
                GB_spec_compare (X1, X2) ;
            end
        end
    end
end

fprintf ('\n') ;
GB_mex_burble (0) ;
fprintf ('test251: all tests passed\n') ;

