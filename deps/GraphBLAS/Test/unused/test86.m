function test86
%TEST86 performance test of of GrB_Matrix_extract

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test86: performance test of of GrB_Matrix_extract\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;

rng ('default') ;
nthreads_max = 2*GB_mex_omp_max_threads ;
nthread_list = [1 2 3 4 8 16 32 40 64 128 160 256] ;
nthread_list = nthread_list (nthread_list <= nthreads_max) ;

Prob = ssget (2662)

A = Prob.A ;
n = size (A,1) ;

fprintf ('\nrandperm==========================================================:\n') ;
J = randperm (n) ;
I = randperm (n) ;
for subset = [n 1e6 1e4 100]
    fprintf ('\nC = A (length %d randperm, length %d randperm)\n', subset, subset) ;
    I2 = I (1:subset) ;
    J2 = J (1:subset) ;
    tic ;
    C = A (I2,J2) ;
    t1 = toc ;
    fprintf ('    builtin %12.6f\n', t1) ;
    [cm cn] = size (C) ;
    S = sparse (cm, cn) ;
    I0 = uint64 (I2) - 1 ;
    J0 = uint64 (J2) - 1 ;
    for nthreads = nthread_list
        nthreads_set (nthreads, chunk) ;
        tic
        C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I0, J0) ;
        t2 = toc ;
        assert (isequal (C, C2.matrix)) ;
        fprintf ('    GraphBLAS nthreads %2d %12.6f speedup %8.2f\n', nthreads, t2, t1/t2) ;
    end
end

clear I0 J0 I2 J2

fprintf ('\nrandperm==========================================================:\n') ;
for subset = [n 1e6 1e4 100]
    fprintf ('\nC = A (length %d randperm, :)\n', subset) ;
    I2 = I (1:subset) ;
    tic ;
    C = A (I2,:) ;
    t1 = toc ;
    fprintf ('    builtin %12.6f\n', t1) ;
    [cm cn] = size (C) ;
    S = sparse (cm, cn) ;
    I0 = uint64 (I2) - 1 ;
    J0.begin = 0 ; J0.inc = 1   ; J0.end = n-1 ;
    for nthreads = nthread_list
        nthreads_set (nthreads, chunk) ;
        tic
        C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I0, J0) ;
        t2 = toc ;
        assert (isequal (C, C2.matrix)) ;
        fprintf ('    GraphBLAS nthreads %2d %12.6f speedup %8.2f\n', nthreads, t2, t1/t2) ;
    end
end

clear I J I0 J0 I2 J2

fprintf ('\nC = A (1:inc:n, :) ===============================================:\n') ;
for inc = [1:10 16 64 128 256 1024 100000 1e6 2e6]
    % fprintf ('\nC = A (1:%7d:n, 1:%7d:n)\n', inc, inc) ;
      fprintf ('\nC = A (1:%7d:n, :)\n', inc) ;
    tic
    % C = A (1:inc:n, 1:inc:n) ;
      C = A (1:inc:n, :) ;
    t1 = toc ;
    fprintf ('    builtin %12.6f\n', t1) ;
    clear I J
    I.begin = 0 ; I.inc = inc ; I.end = n-1 ;
    J.begin = 0 ; J.inc = 1   ; J.end = n-1 ;
    [cm cn] = size (C) ;
    S = sparse (cm, cn) ;
    for nthreads = nthread_list
        nthreads_set (nthreads, chunk) ;
        tic
        C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I, J) ;
        t2 = toc ;
        GB_spok (C2.matrix) ;
        assert (isequal (C, C2.matrix)) ;
        fprintf ('    GraphBLAS nthreads %2d %12.6f speedup %8.2f\n', nthreads, t2, t1/t2) ;
    end
end

fprintf ('\nC = A (1:k, 1:k) =================================================:\n') ;

for hi = [1:10 16 64 128 256 1024 100000 1e6 2e6]
    fprintf ('\nC = A (1:%7d, 1:%7d)\n', hi, hi) ;
    tic
    C = A (1:hi, 1:hi) ;
    t1 = toc ;
    fprintf ('    builtin %12.6f\n', t1) ;
    I.begin = 0 ;
    I.inc = 1 ;
    I.end = hi-1 ;
    [cm cn] = size (C) ;
    S = sparse (cm, cn) ;
    for nthreads = nthread_list
        nthreads_set (nthreads, chunk) ;
        tic
        C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I, I) ;
        t2 = toc ;
        assert (isequal (C, C2.matrix)) ;
        fprintf ('    GraphBLAS nthreads %2d %12.6f speedup %8.2f\n', nthreads, t2, t1/t2) ;
    end
end

fprintf ('\nC = A (lo:hi, lo:hi) =============================================:\n') ;

for lo = [1:10 16 64 128 256 1024 100000 1e6 2e6]
    for delta = [1 10000 1e6]
        hi = lo + delta ;
        hi = min (n, hi) ;
        fprintf ('\nC = A (%7d:%7d, %7d:%7d)\n', lo, hi, lo, hi) ;
        tic
        C = A (lo:hi, lo:hi) ;
        t1 = toc ;
        fprintf ('    builtin %12.6f\n', t1) ;
        I.begin = lo-1 ;
        I.inc = 1 ;
        I.end = hi-1 ;
        [cm cn] = size (C) ;
        S = sparse (cm, cn) ;
        for nthreads = nthread_list
            nthreads_set (nthreads, chunk) ;
            tic
            C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I, I) ;
            t2 = toc ;
            assert (isequal (C, C2.matrix)) ;
            fprintf ('    GraphBLAS nthreads %2d %12.6f speedup %8.2f\n', nthreads, t2, t1/t2) ;
        end
    end
end

fprintf ('\nC = A (hi:-1:lo, hi:-1:lo) =======================================:\n') ;

for lo = [1:10 16 64 128 256 1024 100000 1e6 2e6]
    for delta = [1 10000 1e6]
        hi = lo + delta ;
        hi = min (n, hi) ;
        fprintf ('\nC = A (%7d:-1:%7d, %7d:-1:%7d)\n', hi, lo, hi, lo) ;
        tic
        C = A (hi:-1:lo, hi:-1:lo) ;
        t1 = toc ;
        fprintf ('    builtin %12.6f\n', t1) ;
        I.begin = hi-1 ;
        I.inc = -1 ;
        I.end = lo-1 ;
        [cm cn] = size (C) ;
        S = sparse (cm, cn) ;
        for nthreads = nthread_list
            nthreads_set (nthreads, chunk) ;
            tic
            C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I, I) ;
            t2 = toc ;
            assert (isequal (C, C2.matrix)) ;
            fprintf ('    GraphBLAS nthreads %2d %12.6f speedup %8.2f\n', nthreads, t2, t1/t2) ;
        end
    end
end

fprintf ('\nC = A (n:-inc:1, n:-inc:1) =======================================:\n') ;

for inc = [1:10 16 64 128 256 1024 100000 1e6 2e6]
    fprintf ('\nC = A (n:%7d:1, n:%7d:1)\n', -inc, -inc) ;
    tic
    C = A (n:(-inc):1, n:(-inc):1) ;
    t1 = toc ;
    fprintf ('    builtin %12.6f\n', t1) ;
    I.begin = n-1 ;
    I.inc = -inc ;
    I.end = 0 ;
    [cm cn] = size (C) ;
    S = sparse (cm, cn) ;
    for nthreads = nthread_list
        nthreads_set (nthreads, chunk) ;
        tic
        C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I, I) ;
        t2 = toc ;
        assert (isequal (C, C2.matrix)) ;
        fprintf ('    GraphBLAS nthreads %2d %12.6f speedup %8.2f\n', nthreads, t2, t1/t2) ;
    end
end

fprintf ('test86: all tests passed\n') ;

nthreads_set (save, save_chunk) ;
