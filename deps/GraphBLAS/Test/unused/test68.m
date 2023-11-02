function test68(n)
%TEST68 performance tests for eWiseMult

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest68 --------------------------- quick test of GrB_eWiseMult\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

rng ('default') ;

if (nargin < 1)
    n = 3000 ;
end

A = sparse (rand (n)) ;
B = sparse (rand (n)) ;
C = sparse (n,n) ;

for trial = 1:2
    % C = A.*B, no mask
    Afull = full (A) ;
    Bfull = full (B) ;

    tic
    C0 = Afull .* Bfull ;
    tf = toc ;
    fprintf ('built-in, full: %0.4f\n', tf) ;

    tic
    C0 = A .* B ;
    t0 = toc ;

    tic
    C1 = GB_mex_Matrix_eWiseMult  (C, [ ], [ ], 'times', A, B, [ ]);
    t1 = toc ;
    fprintf ('built-in %0.4f  GB %0.4f speedup %g\n', t0, t1, t0/t1) ;
    assert (isequal (C0, C1.matrix)) ;
end

A = sprand (n, n, 0.001) ;

    % C = A.*B, no mask
    tic
    C0 = A .* B ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_eWiseMult  (C, [ ], [ ], 'times', A, B, [ ]);
    t1 = toc ;
    fprintf ('built-in %0.4f  GB %0.4f speedup %g\n', t0, t1, t0/t1) ;
    assert (isequal (C0, C1.matrix)) ;

A = sparse (n, n) ;
A (n,:) = 1 ;

    % C = A.*B, no mask
    tic
    C0 = A .* B ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_eWiseMult  (C, [ ], [ ], 'times', A, B, [ ]);
    t1 = toc ;
    fprintf ('built-in %0.4f  GB %0.4f speedup %g\n', t0, t1, t0/t1) ;
    assert (isequal (C0, C1.matrix)) ;

A = sparse (n, n) ;
A (1,:) = 1 ;

    % C = A.*B, no mask
    tic
    C0 = A .* B ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_eWiseMult  (C, [ ], [ ], 'times', A, B, [ ]);
    t1 = toc ;
    fprintf ('built-in %0.4f  GB %0.4f speedup %g\n', t0, t1, t0/t1) ;
    assert (isequal (C0, C1.matrix)) ;

for d =  [0.000:0.002:0.1]
    A = sprand (n, n, d) ;

    % C = A.*B, no mask
    tic
    C0 = A .* B ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_eWiseMult  (C, [ ], [ ], 'times', A, B, [ ]);
    t1 = toc ;
    fprintf ('d %8.3f built-in %0.4f  GB %0.4f speedup %g\n', d, t0, t1, t0/t1) ;
    assert (isequal (C0, C1.matrix)) ;
end

nthreads_set (save, save_chunk) ;
