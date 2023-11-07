function test116
%TEST116 performance tests for GrB_assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test116:---------------- C(I,J)=A and C=A(I,J) performance\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;

million = 1e6 ;

rng ('default') ;
n = million ;
nz = 100 * million ;
d = nz / n^2 ;
C0 = sprand (n, n, d) ;

k = n/10 ;
nz = 10 * million ;
d = nz / k^2 ;
A = sprand (k, k, d) ;

I.begin = 0 ;
I.inc = 1 ;
I.end = k-1 ;

ncores = feature_numcores ;

% warmup
C1 = C0 ;
C1 (1:k,1:k) = A ;

fprintf ('\n--------------------------------------\n') ;
fprintf ('C(I,J) = A:\n') ;
tic
C1 = C0 ;
C1 (1:k,1:k) = A ;
tm = toc ;

for nthreads = [1 2 4 8 16 20 32 40 64]
    if (nthreads > 2*ncores)
        break ;
    end
    nthreads_set (nthreads, chunk) ;

    % warmup
    C2 = GB_mex_assign (C0, [ ], [ ], A, I, I) ;
    tic
    C2 = GB_mex_assign (C0, [ ], [ ], A, I, I) ;
    tg = toc ;

    if (nthreads == 1)
        t1 = tg ;
    end

    fprintf ('%3d : built-in: %10.4f GB: %10.4f speedup %10.4f %10.4f\n', ...
        nthreads, tm, tg, tm / tg, t1/tg) ;

    assert (isequal (C1, C2.matrix)) ;
end


fprintf ('\n--------------------------------------\n') ;
fprintf ('B = C(I,J):\n') ;

% warmup
B1 = C1 (1:k,1:k) ;

tic
B1 = C1 (1:k,1:k) ;
tm = toc ;
S = sparse (k,k) ;

for nthreads = [1 2 4 8 16 20 32 40 64]
    if (nthreads > 2*ncores)
        break ;
    end
    nthreads_set (nthreads, chunk) ;

    % warmup
    B2 = GB_mex_Matrix_extract (S, [ ], [ ], C1, I, I) ;
    tic
    B2 = GB_mex_Matrix_extract (S, [ ], [ ], C1, I, I) ;
    tg = toc ;

    if (nthreads == 1)
        t1 = tg ;
    end

    fprintf ('%3d : built-in: %10.4f GB: %10.4f speedup %10.4f %10.4f\n', ...
        nthreads, tm, tg, tm / tg, t1/tg) ;

    assert (isequal (B1, B2.matrix)) ;
end

nthreads_set (save, save_chunk) ;
