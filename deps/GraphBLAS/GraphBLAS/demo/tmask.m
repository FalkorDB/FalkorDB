% TMASK performance tests for logical indexing

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all
GrB (1) ;

for k = [10:22]
    n = 2^k ;
    nz = 10 * n ;
    d = nz / n^2 ;
    d2 = d/10 ;

    tic
    C0 = sprand (n, n, d) ;
    M = logical (sprand (n, n, d2)) ;
    mnz = nnz (M) ;
    A = rand (mnz, 1) ;
    t1 = toc ;
    fprintf ('\nn: %d nnz(C): %d nnz(M) %d\n', n, nnz (C0), mnz) ;
    fprintf ('create C: %g sec\n', t1) ;

    C1 = GrB (C0) ;
    tic
    C1 (M) = A ;
    t3 = toc ;
    fprintf ('C(M)=A in GrB     : %g sec\n', t3) ;

    C2 = C0 ;
    tic
    C2 (M) = A ;
    t2 = toc ;
    fprintf ('C(M)=A in builtin : %g sec  GrB speedup: %g\n', t2, t2/t3) ;

    assert (isequal (C1, C2)) ;
end

