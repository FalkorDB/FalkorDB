function test112
%TEST112 test row/col scale

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test112: row/col scale\n') ;

rng ('default') ;

n = 2000 ;

    D = spdiags (rand (n,1), 0, n, n) ;
    A = sprand (n, n, 0.1) ;
    B = sprand (n, n, 0.1) ;
    p = randperm (n) ;
    P = D (p,:) ;

    % both diag
    fprintf ('\nboth diag:\n') ;
    C1 = D*D ;
    C2 = GB_mex_AxB (D, D) ;
    assert (norm (C1-C2,1) < 1e-14) ;

    % row scale
    fprintf ('\nA is diag:\n') ;
    C1 = D*B ;
    C2 = GB_mex_AxB (D, B) ;
    assert (norm (C1-C2,1) < 1e-14) ;

    % col scale
    fprintf ('\nB is diag:\n') ;
    C1 = A*D ;
    C2 = GB_mex_AxB (A, D) ;
    assert (norm (C1-C2,1) < 1e-14) ;

    % regular
    fprintf ('\nneither diag:\n') ;
    C1 = A*B ;
    C2 = GB_mex_AxB (A, B) ;
    assert (norm (C1-C2,1) < 1e-14) ;

    % permute
    fprintf ('\ncol permutation:\n') ;
    C1 = A*P ;
    C2 = GB_mex_AxB (A, P) ;
    assert (norm (C1-C2,1) < 1e-14) ;

    % permute
    fprintf ('\nrow permutation:\n') ;
    C1 = P*B ;
    C2 = GB_mex_AxB (P, B) ;
    assert (norm (C1-C2,1) < 1e-14) ;

fprintf ('test112: all tests passed\n') ;
