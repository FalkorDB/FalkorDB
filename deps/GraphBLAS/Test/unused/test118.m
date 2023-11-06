function test118
%TEST118 performance tests for GrB_assign

% test C(:,:)<M> = A

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test118 ----------------------------------- C(:,:)<M> = A\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;

rng ('default') ;
n = 2000 ;
S = sparse (n,n) ;

I.begin = 0 ;
I.inc = 1 ;
I.end = n-1 ;

ncores = feature_numcores ;

for dc = [0 1e-5 1e-4 1e-3 1e-2 1e-1 0.5]

    C0 = sparse (n,n,dc) ;

    for da = [1e-5  1e-4 1e-3 1e-2 1e-1 0.5]
        A  = sprand (n, n, da) ;

        for dm = [1e-5 1e-4 1e-3 1e-2 1e-1 0.5]

            M = spones (sprand (n, n, dm)) ;
            Mbool = logical (M) ;

            fprintf ('\n--------------------------------------\n') ;
            fprintf ('dc: %g, da: %g, dm: %g ', dc, da, dm) ;
            fprintf ('\n') ;

            fprintf ('nnz(M): %g million, ',  nnz (M) / 1e6) ;
            fprintf ('nnz(A): %g million\n',  nnz (A) / 1e6) ;

            % warmup
            % C1 = C0 ;
            % C1 (Mbool) = A (Mbool) ;

            tic
            C1 = C0 ;
            C1 (Mbool) = A (Mbool) ;
            tm = toc ;

            t1 = 0 ;

            for nthreads = [1 2 4 8 16 20 32 40 64]
                if (nthreads > 2*ncores)
                    break ;
                end
                nthreads_set (nthreads, chunk) ;

                if (nthreads > 1 & t1 < 0.1)
                    continue
                end

                % if (nnz(A)<nnz(M)) use method 06s, else method 06n
                C2 = GB_mex_assign (C0, M, [ ], A, I, I) ;
                tic
                C2 = GB_mex_assign (C0, M, [ ], A, I, I) ;
                tg = toc ;
                assert (isequal (C1, C2.matrix)) ;
                if (nthreads == 1)
                    t1 = tg ;
                end

                fprintf ('%3d : %8.4f GB: %8.4f', nthreads, tm, tg) ;
                fprintf (' speedup: %8.2f  %8.2f\n', t1/tg, tm / tg) ;

            end
        end
    end
end

nthreads_set (save, save_chunk) ;
