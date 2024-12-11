function test185
%TEST185 test dot4 for all sparsity formats
% GB_AxB_dot4 computes C+=A'*B when C is dense.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test185 -------------------- C+=A''*B when C is dense\n') ;

rng ('default') ;
GB_mex_burble (0) ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

plus_pair.add = 'plus' ;
plus_pair.multiply = 'oneb' ;   % same as pair
plus_pair.class = 'double' ;

add_op = 'plus' ;
dtn_dot = struct ('axb', 'dot', 'inp0', 'tran') ;
dtn_sax = struct ('axb', 'saxpy', 'inp0', 'tran') ;

n = 20 ;
C = GB_spec_random (n, n, inf, 1, 'double') ;
C.sparsity = 8 ;
C0 = sparse (n, n) ;
maxerr = 0 ;

M = sparse (rand (n, n) > 0.5) ;

for da = [0.01 0.1 .5 0.9 inf]
    A = GB_spec_random (n, n, da, 1, 'double') ;

    for db = [0.01 0.1 .5 0.9 inf]
        B = GB_spec_random (n, n, db, 1, 'double') ;

        for A_sparsity = [1 2 4 8]
            fprintf ('.') ;

            for B_sparsity = [1 2 4 8]
                A.sparsity = A_sparsity ;
                B.sparsity = B_sparsity ;

                % C2 = C + A'*B using dot4
                C2 = GB_mex_mxm  (C, [ ], add_op, semiring, A, B, dtn_dot) ;
                C1 = GB_spec_mxm (C, [ ], add_op, semiring, A, B, dtn_dot) ;
                GB_spec_compare (C1, C2) ;
                C3 = C.matrix + (A.matrix)'*B.matrix ;
                err = norm (C3 - C2.matrix, 1) ;
                maxerr = max (maxerr, err) ;
                assert (err < 1e-12) ;

                % C2 = A'*B using dot2/dot3
                C2 = GB_mex_mxm  (C0, [ ], [ ], semiring, A, B, dtn_dot) ;
                C1 = GB_spec_mxm (C0, [ ], [ ], semiring, A, B, dtn_dot) ;
                GB_spec_compare (C1, C2) ;
                C3 = (A.matrix)'*B.matrix ;
                err = norm (C3 - C2.matrix, 1) ;
                maxerr = max (maxerr, err) ;
                assert (err < 1e-12) ;

                % C2 = C + A'*B using saxpy
                C2 = GB_mex_mxm  (C, [ ], add_op, semiring, A, B, dtn_sax) ;
                C1 = GB_spec_mxm (C, [ ], add_op, semiring, A, B, dtn_sax) ;
                GB_spec_compare (C1, C2) ;
                C3 = C.matrix + (A.matrix)'*B.matrix ;
                err = norm (C3 - C2.matrix, 1) ;
                maxerr = max (maxerr, err) ;
                assert (err < 1e-12) ;

                % C2 = C + spones(A)'*spones(B) using dot4
                C2 = GB_mex_mxm  (C, [ ], add_op, plus_pair, A, B, dtn_dot) ;
                C1 = GB_spec_mxm (C, [ ], add_op, plus_pair, A, B, dtn_dot) ;
                GB_spec_compare (C1, C2) ;
                C3 = C.matrix + spones (A.matrix)' * spones (B.matrix) ;
                err = norm (C3 - C2.matrix, 1) ;
                maxerr = max (maxerr, err) ;
                assert (err < 1e-12) ;

                % C2 = spones(A)'*spones(B) using dot2/dot3
                C2 = GB_mex_mxm  (C0, [ ], [ ], plus_pair, A, B, dtn_dot) ;
                C1 = GB_spec_mxm (C0, [ ], [ ], plus_pair, A, B, dtn_dot) ;
                GB_spec_compare (C1, C2) ;
                C3 = spones (A.matrix)' * spones (B.matrix) ;
                err = norm (C3 - C2.matrix, 1) ;
                maxerr = max (maxerr, err) ;
                assert (err < 1e-12) ;
            end
        end
    end
end

fprintf ('\n') ;
GB_mex_burble (0) ;
fprintf ('maxerr: %g\n', maxerr) ;
fprintf ('test185: all tests passed\n') ;

