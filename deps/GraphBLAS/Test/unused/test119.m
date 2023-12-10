function test119
%TEST119 performance tests for GrB_assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test119:-------------------  C(I,J) += scalar:\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;

rng ('default') ;
n = 4000 ; ;

k = 3000 ;

%   I.begin = 0 ;
%   I.inc = 1 ;
%   I.end = k-1 ;
    I1 = randperm (k) ;
    I0 = uint64 (I1) - 1 ;

ncores = feature_numcores ;

for dc = [2 0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 1]

    if (dc == 2)
        C0 = sparse (rand (n)) ;
    else
        C0 = sprand (n, n, dc) ;
    end

    % warmup
    C1 = C0 ;
    % C1 (1:k,1:k) = C1 (1:k,1:k) + pi ;
    C1 (I1,I1) = C1 (I1,I1) + pi ;

    fprintf ('\n--------------------------------------\n') ;
    fprintf ('dc = %g  nnz(C) %8.4f  million\n', dc, nnz(C0)/1e6) ;
    tic
    C1 = C0 ;
    % C1 (1:k,1:k) = C1 (1:k,1:k) + pi ;
    C1 (I1,I1) = C1 (I1,I1) + pi ;
    tm = toc ;

    scalar = sparse (pi) ;

    for nthreads = [1 2 4 8 16 20 32 40 64]
        if (nthreads > 2*ncores)
            break ;
        end
        if (nthreads > 1 && t1 < 0.01)
            break ;
        end

        nthreads_set (nthreads, chunk) ;

        C2 = GB_mex_assign (C0, [ ], 'plus', scalar, I0, I0) ;
        tic
        C2 = GB_mex_assign (C0, [ ], 'plus', scalar, I0, I0) ;
        tg = toc ;
        assert (isequal (C1, C2.matrix)) ;
        if (nthreads == 1)
            t1 = tg ;
        end

        fprintf ('%3d : built-in: %10.4f GB: %8.4f ', nthreads, tm, tg) ;
        fprintf (' speedup %10.4f %10.4f\n', tm / tg, t1/tg) ;

    end
end

nthreads_set (save, save_chunk) ;
