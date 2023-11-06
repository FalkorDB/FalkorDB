function test111
%TEST111 performance test for eWiseAdd

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest111 performance tests : eWiseAdd \n') ;
rng ('default') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;

n = 40e6 ;
% n = 1e6 ;
Empty = sparse (n, 1) ;

% for d = [0.001 0.01 0.1:.1:1]
for d = [0.001 0.01 0.1 0.4 1 2 3]

    if (d == 1)
        A = sparse (rand (n,1)) ;
        B = sparse (rand (n,1)) ;
        mds = 1 ;
    elseif (d == 2)
        A = sparse (rand (n,1)) ;
        B = sprand (n,1,0.1) ;
        mds = 0.1 ;
    elseif (d == 3)
        A = sprand (n,1,0.1) ;
        B = sparse (rand (n,1)) ;
        mds = 0.1 ;
    else
        A = sprand (n,1,d) ;
        B = sprand (n,1,d) ;
        mds = [d/36 d/8 d] ;
    end

    for md = mds

        if (md == 1)
            M = sparse (rand (n,1)) ;
        else
            M = sprand (n,1,md) ;
        end

        M = spones (M) ;
        M0 = logical (M) ;

        fprintf ('\n######################################################\n') ;
        fprintf ('d = %12.4f md = %12.4f\n', d, md) ;
        fprintf ('######################################################\n') ;
        fprintf ('nnz (A) = %d nnz (B) = %d nnz (M) = %d\n', ...
            nnz (A), nnz (B), nnz (M)) ;

        %----------------------------------------------------------------------
        % add
        %----------------------------------------------------------------------

        % warmup
        C1 = A + B ;
        tic
        C1 = A + B ;
        toc
        tm = toc ;
        fprintf ('nnz (C) = %d\n', nnz (C1));
        fprintf ('\nvia GB_add:\n') ;

        for nthreads = [1 2 4 8 20 40]
            nthreads_set (nthreads, chunk) ;
            tic ;
            C4 = GB_mex_AplusB (A, B, 'plus') ;
            tg = toc ;
            if (nthreads == 1)
                t1 = tg ;
            end
            fprintf ('nthreads %2d GraphBLAS time: %12.4f ', nthreads, tg) ;
            fprintf ('speedup %12.4f over built-in: %12.4f\n', t1/tg, tm/tg) ;
            assert (GB_spok (C4) == 1) ;
            assert (isequal (C1, C4)) ;
        end

        %----------------------------------------------------------------------
        % masked add
        %----------------------------------------------------------------------

        fprintf ('\nmasked add:\n') ;
        % warmup
        C1 = M.*(A + B) ;
        tic
        C1 = M.*(A + B) ;
        toc
        tm = toc ;
        fprintf ('nnz (C) = %d\n', nnz (C1));
        fprintf ('\nvia masked GB_add:\n') ;

        for nthreads = [1 2 4 8 20 40]
            nthreads_set (nthreads, chunk) ;
            % warmup
            C4 = GB_mex_Vector_eWiseAdd (Empty, M0, [ ], 'plus', A, B, [ ]) ;
            tic
            C4 = GB_mex_Vector_eWiseAdd (Empty, M0, [ ], 'plus', A, B, [ ]) ;
            tg = toc ;
            if (nthreads == 1)
                t1 = tg ;
            end
            fprintf ('nthreads %2d GraphBLAS time: %12.4f ', nthreads, tg) ;
            fprintf ('speedup %12.4f over built-in: %12.4f\n', t1/tg, tm/tg) ;
            assert (GB_spok (C4.matrix) == 1) ;
            assert (isequal (C1, C4.matrix)) ;
        end

        %-----------------------------------------------------------------------
        % unmasked add then ewise mult
        %-----------------------------------------------------------------------

        fprintf ('\nunmasked add then emult:\n') ;
        % warmup
        C1 = M.*(A + B) ;
        tic
        C1 = M.*(A + B) ;
        toc
        tm = toc ;
        fprintf ('nnz (C) = %d\n', nnz (C1));
        fprintf ('\nvia unmasked add then emult:\n') ;

        for nthreads = [1 2 4 8 20 40]
            nthreads_set (nthreads, chunk) ;
            % warmup
            C4 = GB_mex_Vector_eWiseAdd (Empty, [ ], [ ], 'plus', A, B, [ ]) ;
            C4 = GB_mex_Vector_eWiseMult(Empty, [ ], [ ], 'times',M, C4, [ ]) ;
            tic
            C4 = GB_mex_Vector_eWiseAdd (Empty, [ ], [ ], 'plus', A, B, [ ]) ;
            tg1 = toc ;
            tic
            C4 = GB_mex_Vector_eWiseMult(Empty, [ ], [ ], 'times',M, C4, [ ]) ;
            tg2 = toc ;
            tg = tg1 + tg2 ;
            if (nthreads == 1)
                t1 = tg ;
            end
            fprintf ('nthreads %2d GraphBLAS time: %12.4f ', nthreads, tg) ;
            fprintf ('speedup %12.4f over built-in: %12.4f\n', t1/tg, tm/tg) ;
            assert (GB_spok (C4.matrix) == 1) ;
            assert (isequal (C1, C4.matrix)) ;
        end

        %-----------------------------------------------------------------------
        % ewise multiply
        %-----------------------------------------------------------------------

        % warmup
        C1 = A .* B ;
        tic
        C1 = A .* B ;
        toc
        tm = toc ;
        fprintf ('nnz (C) = %d for A.*B\n', nnz (C1));
        fprintf ('\nvia GB_eWiseMult:\n') ;

        for nthreads = [1 2 4 8 20 40]
            nthreads_set (nthreads, chunk) ;
            % warmup
            C4 = GB_mex_Vector_eWiseMult (Empty, [ ], [ ], 'times', A,B, [ ]) ;
            tic
            C4 = GB_mex_Vector_eWiseMult (Empty, [ ], [ ], 'times', A,B, [ ]) ;
            tg = toc ;
            if (nthreads == 1)
                t1 = tg ;
            end
            fprintf ('nthreads %2d GraphBLAS time: %12.4f ', nthreads, tg) ;
            fprintf ('speedup %12.4f over built-in: %12.4f\n', t1/tg, tm/tg) ;
            assert (GB_spok (C4.matrix) == 1) ;
            assert (isequal (C1, C4.matrix)) ;
        end

        %-----------------------------------------------------------------------
        % masked ewise multiply
        %-----------------------------------------------------------------------

        fprintf ('\nmasked emult:\n') ;
        % warmup
        C1 = M.* (A .* B) ;
        tic
        C1 = M.* (A .* B) ;
        toc
        tm = toc ;
        fprintf ('nnz (C) = %d for A.*B\n', nnz (C1));
        fprintf ('\nvia GB_eWiseMult:\n') ;

        for nthreads = [1 2 4 8 20 40]
            nthreads_set (nthreads, chunk) ;
            % warmup
            C4 = GB_mex_Vector_eWiseMult (Empty, M0, [ ], 'times', A,B, [ ]) ;
            tic
            C4 = GB_mex_Vector_eWiseMult (Empty, M0, [ ], 'times', A,B, [ ]) ;
            tg = toc ;
            if (nthreads == 1)
                t1 = tg ;
            end
            fprintf ('nthreads %2d GraphBLAS time: %12.4f ', nthreads, tg) ;
            fprintf ('speedup %12.4f over built-in: %12.4f\n', t1/tg, tm/tg) ;
            assert (GB_spok (C4.matrix) == 1) ;
            assert (isequal (C1, C4.matrix)) ;
        end

    end
end

fprintf ('\ndense matrices:\n') ;

A = full (A) ;
B = full (B) ;

for trial = 1:4
    tic
    C1 = A + B ;
    toc
    tm = toc ;
end
nthreads_set (save, save_chunk) ;

fprintf ('test111: all tests passed\n') ;
