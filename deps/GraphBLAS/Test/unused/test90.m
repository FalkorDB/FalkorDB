function test90
%TEST90 test AxB with user-defined semirings: plus_rdiv and plus_rdiv2

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n -------------- A*B plus_rdiv (user-defined semiring)\n') ;
GB_mex_burble (1) ;

    % 7081: Gustavson
    % 7083: dot
    % 7084: hash
    % 7085: saxpy

rng ('default') ;

for N = [10 100] % 1000]

    % create the problem
    A = sprand (4*N, 5*N, 0.01) ;
    B = sprand (5*N, 3*N, 0.01) ;

    [i j x] = find (A) ;
    [m n] = size (A) ;
    X = sparse (i, j, 1./x, m, n) ;
    clear i j x

    C1 = X*B ;

    C2 = GB_mex_rdiv (A, B) ;
    assert (norm (C1-C2,1) / norm (C1,1) < 1e-10) ;

    for method = [7081 7083 7084 7085]
        fprintf ('method: %d\n', method) ;
        cprint = (N <= 10) ;
        C2 = GB_mex_rdiv (A, B, method, cprint) ;
        assert (norm (C1-C2,1) / norm (C1,1) < 1e-10) ;
    end

    % try rdiv2, which typecasts B to single precision first
    C3 = GB_mex_rdiv2 (A, B) ;
    assert (norm (C1-C3,1) / norm (C1,1) < 1e-5) ;

    [i j x] = find (B) ;
    [m n] = size (B) ;
    Y = sparse (i, j, 1./x, m, n) ;

    C4 = A*Y ;

    % try rdiv2, with flipxy, which inverts A instead of B
    C5 = GB_mex_rdiv2 (A, B, false, false, 0, 1) ;
    assert (norm (C4-C5,1) / norm (C4,1) < 1e-5) ;

    %--------------------------------------------------------------------------
    % fprintf ('\nextensive tests:\n') ;

    A = sprand (n, n, 0.01) ;
    B = sprand (n, n, 0.01) ;
    [i j x] = find (A) ;
    X = sparse (i, j, 1./x, n, n) ;
    [i j x] = find (B) ;
    Y = sparse (i, j, 1./x, n, n) ;

    flipxy = 0 ;
    at = 0 ; 
    bt = 0 ;
    C0 = X*B ;
    for method = [7081 7083 7084 7085]
        fprintf ('method: %d\n', method) ;
        C5 = GB_mex_rdiv2 (A, B, at, bt, method, flipxy) ;
        assert (norm (C0-C5,1) / norm (C5,1) < 1e-5) ;
    end

    flipxy = 0 ;
    at = 1 ; 
    bt = 0 ;
    C0 = X'*B ;
    for method = [7081 7083 7084 7085]
        fprintf ('method: %d\n', method) ;
        C5 = GB_mex_rdiv2 (A, B, at, bt, method, flipxy) ;
        assert (norm (C0-C5,1) / norm (C5,1) < 1e-5) ;
    end

    flipxy = 0 ;
    at = 0 ; 
    bt = 1 ;
    C0 = X*B' ;
    for method = [7081 7083 7084 7085]
        fprintf ('method: %d\n', method) ;
        C5 = GB_mex_rdiv2 (A, B, at, bt, method, flipxy) ;
        assert (norm (C0-C5,1) / norm (C5,1) < 1e-5) ;
    end

    flipxy = 0 ;
    at = 1 ; 
    bt = 1 ;
    C0 = X'*B' ;
    for method = [7081 7083 7084 7085]
        fprintf ('method: %d\n', method) ;
        C5 = GB_mex_rdiv2 (A, B, at, bt, method, flipxy) ;
        assert (norm (C0-C5,1) / norm (C5,1) < 1e-5) ;
    end

    flipxy = 1 ;
    at = 0 ; 
    bt = 0 ;
    C0 = A*Y ;
    for method = [7081 7083 7084 7085]
        fprintf ('method: %d\n', method) ;
        C5 = GB_mex_rdiv2 (A, B, at, bt, method, flipxy) ;
        assert (norm (C0-C5,1) / norm (C5,1) < 1e-5) ;
    end

    flipxy = 1 ;
    at = 1 ; 
    bt = 0 ;
    C0 = A'*Y ;
    for method = [7081 7083 7084 7085]
        fprintf ('method %d\n', method) ;
        C5 = GB_mex_rdiv2 (A, B, at, bt, method, flipxy) ;
        assert (norm (C0-C5,1) / norm (C5,1) < 1e-5) ;
    end

    flipxy = 1 ;
    at = 0 ; 
    bt = 1 ;
    C0 = A*Y' ;
    for method = [7081 7083 7084 7085]
        fprintf ('method %d\n', method) ;
        C5 = GB_mex_rdiv2 (A, B, at, bt, method, flipxy) ;
        assert (norm (C0-C5,1) / norm (C5,1) < 1e-5) ;
    end

    flipxy = 1 ;
    at = 1 ; 
    bt = 1 ;
    C0 = A'*Y' ;
    for method = [7081 7083 7084 7085]
        fprintf ('method %d\n', method) ;
        C5 = GB_mex_rdiv2 (A, B, at, bt, method, flipxy) ;
        assert (norm (C0-C5,1) / norm (C5,1) < 1e-5) ;
    end

end

GB_mex_burble (0) ;
fprintf ('\ntest90: all tests passed\n') ;

