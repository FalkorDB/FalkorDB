function gbtest99
%GBTEST99 test performance of C=A'*B and C=A'

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('# of threads in @GrB: %d\n', GrB.threads) ;
n = 10 * 1e6 ;
kset = [1    2    10  32 100 120 150 1000] ;
nset = [1000 1000 100 10 10  10  10  10  ] ;
nset = nset/10 ;

for kk = 1:length (kset)
    ntrials = nset (kk) ;
    k = kset (kk) ;

    fprintf ('\n======================== k = %d\n', k) ;
    A = sprand (n, k, 0.001) ;
    B = sprand (n, k, 0.001) ;

    % built-in, with warmup:
    C1 = A'*B ;
    tic
    for trial = 1:ntrials
        C1 = A'*B ;
    end
    t1 = toc / ntrials ;
    fprintf ('built-in time: %g sec\n', t1) ;

    % GrB, with warmup, using the descriptor transpose
    A = GrB (A) ;
    B = GrB (B) ;
    d.in0 = 'transpose' ;
    C2 = GrB.mxm (A, '+.*', B, d) ;
    tic
    for trial = 1:ntrials
        C2 = GrB.mxm (A, '+.*', B, d) ;
    end
    t2 = toc / ntrials ;
    err = norm (C1-C2, 1) ;
    fprintf ('@GrB default time: %g sec, speedup %g error: %g\n', ...
        t2, t1/t2, err) ;
    assert (err <= 1e-12 * norm (C1,1)) ;

    % GrB, with warmup, using the explicit transpose
    C2 = A'*B ;
    tic
    for trial = 1:ntrials
        C2 = A'*B ;
    end
    t3 = toc / ntrials ;
    err = norm (C1-C2, 1) ;
    fprintf ('@GrB saxpy/transpose time: %g sec, speedup %g, error: %g\n', ...
        t3, t1/t3, err) ;
    assert (err <= 1e-12 * norm (C1,1)) ;

    % with burble, to see what GraphBLAS is doing
    GrB.burble (1) ;
    fprintf ('\nGrB with mxm and descriptor transpose:\n') ;
    C2 = GrB.mxm (A, '+.*', B, d) ; %#ok<NASGU>
    fprintf ('\nGrB with A''*B syntax and explicit transpose:\n') ;
    C2 = A'*B ; %#ok<NASGU>
    GrB.burble (0) ;

    % built-in transpose time
    A = double (A) ;
    C1 = A' ;
    tic
    for trial = 1:ntrials
        C1 = A' ;
    end
    t1 = toc / ntrials ;
    fprintf ('\nbuilt-in transpose time: %g sec\n', t1) ;

    % GrB transpose time
    A = GrB (A) ;
    C2 = A' ;
    tic
    for trial = 1:ntrials
        C2 = A' ;
    end
    t2 = toc / ntrials ;
    assert (isequal (C1, C2)) ;
    fprintf ('@GrB transpose time: %g sec, speedup %g\n', t2, t1/t2) ;

end

fprintf ('gbtest99: all tests passed\n') ;

