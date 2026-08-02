function gbtest99 (ghb)
%GBTEST99 test performance of C=A'*B and C=A'

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

fprintf ('# of threads in GraphBLAS: %d\n', gtb_threads (ghb)) ;
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

    % GraphBLAS, with warmup, using the descriptor transpose
    A = gtb (ghb, A) ;
    B = gtb (ghb, B) ;
    d.in0 = 'transpose' ;
    C2 = gtb_mxm (ghb, A, '+.*', B, d) ;
    tic
    for trial = 1:ntrials
        C2 = gtb_mxm (ghb, A, '+.*', B, d) ;
    end
    t2 = toc / ntrials ;
    err = norm (C1-C2, 1) ;
    fprintf ('@%s default time: %g sec, speedup %g error: %g\n', ...
        gtb_name, t2, t1/t2, err) ;
    assert (err <= 1e-12 * norm (C1,1)) ;

    % GraphBLAS, with warmup, using the explicit transpose
    C2 = A'*B ;
    tic
    for trial = 1:ntrials
        C2 = A'*B ;
    end
    t3 = toc / ntrials ;
    err = norm (C1-C2, 1) ;
    fprintf ('@%s saxpy/transpose time: %g sec, speedup %g, error: %g\n', ...
        gtb_name, t3, t1/t3, err) ;
    assert (err <= 1e-12 * norm (C1,1)) ;

    % with burble, to see what GraphBLAS is doing
    gtb_burble (ghb, 1) ;
    fprintf ('\n%s with mxm and descriptor transpose:\n', gtb_name) ;
    C2 = gtb_mxm (ghb, A, '+.*', B, d) ; %#ok<NASGU>
    fprintf ('\n%s with A''*B syntax and explicit transpose:\n', gtb_name) ;
    C2 = A'*B ; %#ok<NASGU>
    gtb_burble (ghb, 0) ;

    % built-in transpose time
    A = double (A) ;
    C1 = A' ;
    tic
    for trial = 1:ntrials
        C1 = A' ;
    end
    t1 = toc / ntrials ;
    fprintf ('\nbuilt-in transpose time: %g sec\n', t1) ;

    % GraphBLAS transpose time
    A = gtb (ghb, A) ;
    C2 = A' ;
    tic
    for trial = 1:ntrials
        C2 = A' ;
    end
    t2 = toc / ntrials ;
    assert (isequal (C1, C2)) ;
    fprintf ('@%s transpose time: %g sec, speedup %g\n', gtb_name, t2, t1/t2) ;

end

fprintf ('gbtest99 (%d): all tests passed\n', ghb) ;

