function test120
%TEST120 performance tests for GrB_assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test120:-------------------  C(I,J)<!M> += scalar:\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;

rng ('default') ;
n = 4000 ; ;

k = 3000 ;

%   I0.begin = 0 ;
%   I0.inc = 1 ;
%   I0.end = k-1 ;
    I1 = randperm (k) ;
    I0 = uint64 (I1) - 1 ;

d.mask = 'complement' ;

ncores = feature_numcores ;

for dc = [2 0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 1]

    if (dc == 2)
        C0 = sparse (rand (n)) ;
    else
        C0 = sprand (n, n, dc) ;
    end

    for dm = [2 0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 1]

        if (dm == 2)
            M = sparse (ones (k)) ;
        else
            M = spones (sprand (k, k, dm)) ;
        end

        Mbool = logical (M) ;

        fprintf ('\n--------------------------------------\n') ;
        fprintf ('dc = %g  nnz(C) %8.4f  million\n', dc, nnz(C0)/1e6) ;
        fprintf ('dm = %g  nnz(M) %8.4f  million\n', dm, nnz(M)/1e6) ;

        tm = inf ;
        if (n < 500)
            % built-in methods are exceedingly slow for this case
            tic
            C1 = C0 ;
            % Csub = C1 (1:k, 1:k) ;
            Csub = C1 (I1, I1) ;
            Csub (~Mbool) = Csub (~Mbool) + pi ; 
            % C1 (1:k, 1:k) = Csub ;
            C1 (I1, I1) = Csub ;
            tm = toc ;
        end

        scalar = sparse (pi) ;

        for nthreads = [1 2 4 8 16 20 32 40 64]
            if (nthreads > 2*ncores)
                break ;
            end
            if (nthreads > 1 && t1 < 0.01)
                break ;
            end

            nthreads_set (nthreads, chunk) ;

            C2 = GB_mex_subassign (C0, M, 'plus', scalar, I0, I0, d) ;
            tic
            C2 = GB_mex_subassign (C0, M, 'plus', scalar, I0, I0, d) ;
            tg = toc ;
            if (n < 500)
                assert (isequal (C1, C2.matrix)) ;
            end
            if (nthreads == 1)
                t1 = tg ;
            end

            fprintf ('%3d : builtin: %10.4f GB: %8.4f ', nthreads, tm, tg) ;
            fprintf (' speedup %10.4f %10.4f\n', tm / tg, t1/tg) ;

        end
    end
end

nthreads_set (save, save_chunk) ;
