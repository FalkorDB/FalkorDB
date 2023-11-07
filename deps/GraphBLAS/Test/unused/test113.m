function test113
%TEST113 performance tests for GrB_kron

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test113: performance tests for GrB_kron\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
ncores = feature_numcores ;

A = sprand (310, 302, 0.1) ;
B = sprand (300, 301, 0.1) ;
fprintf ('nnz(A) %g\n', nnz (A)) ;
fprintf ('nnz(B) %g\n', nnz (B)) ;
fprintf ('nnz(C) %g\n', nnz (A) * nnz (B)) ;

tic
C = kron (A,B) ;
tm = toc ;
fprintf ('built-in: %g sec\n', tm) ;

[m n] = size (C) ;
Empty = sparse (m,n) ;

for nthreads = [1 2 4 8 16 20 40]

    if (nthreads > 2*ncores)
        break ;
    end

    nthreads_set (nthreads,chunk) ;

    tic
    C1 = GB_mex_kron (Empty, [ ], [ ], 'times', A, B) ;
    t (nthreads) = toc ;

    assert (isequal (C, C1.matrix)) ;

    fprintf ('GB: %12.4f sec speedup: %12.4f  vs built-in: %12.4f\n', ...
        t (nthreads), t (1) / t (nthreads), tm / t (nthreads)) ;

end

nthreads_set (save, save_chunk) ;
