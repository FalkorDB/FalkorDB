function test36
%TEST36 performance test of matrix subref

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest36 --------------------- performance of GB_Matrix_subref\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

rng ('default') ;
n = 100e6 ;
fprintf ('-------------------------- column vector (%d-by-1):\n', n) ;
V = sprand (n, 1, 0.01) ;
J = uint64 (0) ;
[i j x] = find (V) ;
nz = length (i) ;


fprintf (' V(:,1): nnz (V) = %d\n', nnz (V)) ;
    tic
    C0 = V (:,1) ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_subref (V, [ ], [ ]) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin %0.6f GrB: %0.6f  speedup %g\n', t0, t1, t0/t1) ;

fprintf (' V(50e6:80e6,1) explicit list:\n') ;
I = uint64 (50e6:80e6) ;
I1 = I+1 ;
    tic
    C0 = V (I1,1) ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_subref (V, I, [ ]) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin %0.6f GrB: %0.6f  speedup %g\n', t0, t1, t0/t1) ;

fprintf (' V(50e6:80e6,1) colon:\n') ;
clear I
I.begin = 50e6-1 ; I.inc = 1 ; I.end = 80e6-1  ;
% I1 = I+1 ;
    tic
    C0 = V (50e6:80e6,1) ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_subref (V, I, [ ]) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin %0.6f GrB: %0.6f  speedup %g\n', t0, t1, t0/t1) ;

I1 = i (floor (nz/2)) ;
I = uint64 (I1)-1 ;
fprintf (' V(%d,1):\n', I1) ;

    C0 = V (I1,1) ;
    tic
    C0 = V (I1,1) ;
    t0 = toc ;
    C1 = GB_mex_Matrix_subref (V, I, J) ;
    tic
    C1 = GB_mex_Matrix_subref (V, I, J) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin %0.6f GrB: %0.6f  speedup %g\n', t0, t1, t0/t1) ;

    tic
    C0 = V (I1,1) ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_subref (V, I, J) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin %0.6f GrB: %0.6f  speedup %g\n', t0, t1, t0/t1) ;


fprintf (' V( 100 entries ,1):\n') ;
p = randperm (nz) ;
I1 = i (p (1:100)) ;
I = uint64 (I1)-1 ;

    tic
    C0 = V (I1,1) ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_subref (V, I, [ ] ) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin %0.6f GrB: %0.6f  speedup %g\n', t0, t1, t0/t1) ;


fprintf (' V( 100 entries ,1:4):\n') ;
V = [V V V V] ;

    tic
    C0 = V (I1,:) ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_subref (V, I, [ ]) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin %0.6f GrB: %0.6f  speedup %g\n', t0, t1, t0/t1) ;

nthreads_set (save, save_chunk) ;

fprintf ('\ntest36: all tests passed\n') ;

