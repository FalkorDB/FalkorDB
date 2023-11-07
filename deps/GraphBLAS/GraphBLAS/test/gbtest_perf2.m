function gbtest_perf2
%GBTEST_PERF2 test A'*x performance

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

max_nthreads = GrB.threads ;
threads = [1 2 4 8 16 20 32 40 64] ;
desc = struct ('in0', 'transpose') ;
rng ('default') ;

n = 1e6 ; nz = 20e6 ;
% n = 1e5 ; nz = 1e6 ;
d = nz / n^2 ;
% same as A = sprand (n,n,d), but faster:
G = GrB.random (n,n,d) ;
A = double (G) ;
% warmup to make sure the GrB library is loaded
y = GrB (rand (2)) * GrB (rand (2)) ;

degree = sum (spones (G)) ;
nempty = length (find (degree == 0)) ;
fprintf ('matrix: n: %d nnz: %d  # empty cols: %d\n', n, nnz (A), nempty) ;

ntrials = 1 ;

for test = 1:4

    if (test == 1)
        X = 'sparse (rand (n,1))' ;
        x =  sparse (rand (n,1)) ;
    elseif (test == 2)
        X = 'rand (n,1)' ;
        x =  rand (n,1) ;
    elseif (test == 3)
        X = 'sprand (n,1,0.5)' ;
        x =  sprand (n,1,0.5) ;
    else
        X = 'sprand (n,1,0.05)' ;
        x =  sprand (n,1,0.05) ;
    end

    fprintf ('\n\n========================\n') ;
    fprintf ('built-in: y = A''*x where x = %s\n', X) ;

    tic
    for trial = 1:ntrials
        y = A'*x ;
    end
    tbuiltin = toc ;
    fprintf ('built-in time: %8.4f sec\n', tbuiltin) ;
    ybuiltin = y ;

    fprintf ('\nGrB: y = A''*x where x = %s\n', X) ;

    for nthreads = threads
        if (nthreads > max_nthreads)
            break ;
        end
        GrB.threads (nthreads) ;
        tic
        for trial = 1:ntrials
            % y = G'*x ;
            y = GrB.mxm (G, '+.*', x, desc) ;
        end
        t = toc ;
        if (nthreads == 1)
            t1 = t ;
        end
        fprintf (...
            'threads: %2d GrB time: %8.4f speedup vs built-in: %8.2f  vs: GrB(1 thread) %8.2f\n', ...
            nthreads, t, tbuiltin / t, t1 / t) ;
        assert (norm (y-ybuiltin, 1) / norm (ybuiltin,1) < 1e-12)
    end

    fprintf ('\nGrB: y = zeros(n,1) + A''*x where x = %s\n', X) ;

    for nthreads = threads
        if (nthreads > max_nthreads)
            break ;
        end
        GrB.threads (nthreads) ;
        tic
        for trial = 1:ntrials
            y = zeros (n,1) ;
            % y = y + G'*x
            y = GrB.mxm (y, '+', G, '+.*', x, desc) ;
        end
        t = toc ;
        if (nthreads == 1)
            t1 = t ;
        end
        fprintf (...
            'threads: %2d GrB time: %8.4f speedup vs built-in: %8.2f  vs: GrB(1 thread) %8.2f\n', ...
            nthreads, t, tbuiltin / t, t1 / t) ;
        assert (norm (y-ybuiltin, 1) / norm (ybuiltin,1) < 1e-12)
    end

end

GrB.burble (0) ;

