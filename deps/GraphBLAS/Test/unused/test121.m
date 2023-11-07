function test121
%TEST121 performance tests for GrB_assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test121:---------------- C(I,J)+=A performance\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;

rng ('default') ;
n = 1e6 ;
k = n/10 ;

%   I.begin = 0 ;
%   I.inc = 1 ;
%   I.end = k-1 ;
    I1 = randperm (k) ;
    I0 = uint64 (I1) - 1 ;

ncores = feature_numcores ;

for dc = [ 0 1e-6 1e-5 1e-4 ]

    C0 = sprandn (n, n, dc) ;

    for da = [ 0 1e-6 1e-5 1e-4 1e-3 ]
    
        A = sprandn (k, k, da) ;

        % warmup
        C1 = C0 ;
        C1 (1:k,1:k) = C1 (1:k,1:k) + A ;

        fprintf ('\n--------------------------------------\n') ;
        fprintf ('dc = %g, da = %g\n', dc, da) ;
        tic
        C1 = C0 ;
        % C1 (1:k,1:k) = C1 (1:k,1:k) + A ;
        C1 (I1,I1) = C1 (I1,I1) + A ;
        tm = toc ;

        for nthreads = [1 2 4 8 16 20 32 40 64]
            if (nthreads > 2*ncores)
                break ;
            end
            nthreads_set (nthreads, chunk) ;
            if (nthreads > 1 && t1 < 0.01)
                continue ;
            end

            C2 = GB_mex_assign (C0, [ ], 'plus', A, I0, I0) ;
            tic
            C2 = GB_mex_assign (C0, [ ], 'plus', A, I0, I0) ;
            tg = toc ;
            assert (isequal (C1, C2.matrix)) ;
            if (nthreads == 1)
                t1 = tg ;
            end

            fprintf ('%3d : builtin: %10.4f GB: %10.4f', nthreads, tm, tg) ;
            fprintf ('  speedup %10.4f %10.4f\n', t1/tg, tm / tg) ;
        end
    end
end

nthreads_set (save, save_chunk) ;
