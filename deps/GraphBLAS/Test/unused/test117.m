function test117
%TEST117 performance tests for GrB_assign

% test C(:,:)<M> += A

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test117 ----------------------------------- C(:,:)<M> += A\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;

rng ('default') ;
n = 4000 ;

I.begin = 0 ;
I.inc = 1 ;
I.end = n-1 ;

ncores = feature_numcores ;

for dc = [1e-5 1e-4 1e-3 1e-2 1e-1 0.5]
    C0 = sprand (n, n, dc) ;

for da = [1e-5 1e-4 1e-3 1e-2 1e-1 0.5]
    A  = sprand (n, n, da) ;

for dm = [1e-5 1e-4 1e-3 1e-2 1e-1 0.5]

    M = spones (sprand (n, n, dm)) ;

    for subset = 0:1

        fprintf ('\n--------------------------------------\n') ;
        fprintf ('dc: %g, da: %g, dm: %g ', dc, da, dm) ;
        if (subset)
            fprintf ('M is a subset of C\n') ;
            M = spones (M.*C0) ;
        end
        fprintf ('\n') ;

        fprintf ('nnz(C): %g million, nnz(M): %g million, ', ...
            nnz (C0) / 1e6, nnz (M) / 1e6) ;
        fprintf ('nnz(A): %g million\n',  nnz (A) / 1e6) ;

        % warmup
        C1 = C0 + M.*A ;

        tic
        C1 = C0 + M.*A ;
        tm = toc ;

        for nthreads = [1 2 4 8 16 20 32 40 64]
            if (nthreads > 2*ncores)
                break ;
            end
            nthreads_set (nthreads, chunk) ;

            if (nthreads > 1 & t1 < 0.003)
                continue
            end

            % warmup:
            C2 = GB_mex_assign (C0, M, 'plus', A, I, I) ;
            tic
            C2 = GB_mex_assign (C0, M, 'plus', A, I, I) ;
            tg = toc ;
            assert (isequal (C1, C2.matrix)) ;
            if (nthreads == 1)
                t1 = tg ;
            end

            % ewise
            C2 = GB_mex_Matrix_eWiseMult (C0, [ ], 'plus', 'times', M, A) ;
            tic
            C2 = GB_mex_Matrix_eWiseMult (C0, [ ], 'plus', 'times', M, A) ;
            tg3 = toc ;

            assert (isequal (C1, C2.matrix)) ;

            fprintf (...
                '%3d : %8.4f GB: %8.4f %8.4f rel %8.2f %8.2f\n', ...
                nthreads, tm, tg, tg3, tm / tg, tm/tg3) ;
        end

    end
end
end
end

nthreads_set (save, save_chunk) ;
