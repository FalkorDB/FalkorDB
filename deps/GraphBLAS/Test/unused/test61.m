function test61
%TEST61 performance test of GrB_eWiseMult

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n----------------------------- eWiseMult performance tests\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

Prob = ssget (2662)
A = Prob.A ;
[m n] = size (A) ;

B = A ;
A (:, 1:5) = 44 ;

S = sparse (m,n) ;
fprintf ('\n\nm: %d n %d nnz(A) %d\n', m, n,  nnz (A)) ;

d = nnz (A) / prod (size (A)) ;

    tic
    C = A .*B ;
    t1 = toc ;

    tic
    C2 = GB_mex_Matrix_eWiseMult (S, [ ], [ ], 'times', A, B, [ ]) ;
    t2 = toc ;

    fprintf (...
    'd %10.6g nnz(C) %8d built-in %10.6f GB %10.6f  speedup %10.4f\n',...
        d, nnz (C), t1, t2, t1/t2) ;


A = sparse (rand (5000)) ;
[m n] = size (A) ;
S = sparse (m,n) ;
fprintf ('\n\nm: %d n %d nnz(A) %d\n', m, n,  nnz (A)) ;

for d = [0.00001:0.00001:0.0001 0.0002:0.0001: 0.001 0.002:.001:0.01 0.02:0.01:.1 1]

    B = sprandn (m, n, d) ;

    tic
    C = A .*B ;
    t1 = toc ;

    tic
    C2 = GB_mex_Matrix_eWiseMult (S, [ ], [ ], 'times', A, B, [ ]) ;
    t2 = toc ;

    fprintf (...
    'd %10.6g nnz(C) %8d built-in %10.6f GB %10.6f  speedup %10.4f\n',...
        d, nnz (C), t1, t2, t1/t2) ;

    assert (isequal (C, C2.matrix)) ;
end

fprintf ('\ntest61: all tests passed\n') ;

nthreads_set (save, save_chunk) ;
