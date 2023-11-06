function test45(use_ssget)
%TEST45 test GrB_*_setElement and GrB_*_*build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest45\n------------------ testing GrB_setElement and _build\n') ;

if (nargin < 1)
    use_ssget = true ;
end

rng ('default') ;
A = sparse (rand (3,2)) ;

C = GB_mex_setElement (A, uint64(0), uint64(0), 42.1) ;

A = rand (3,2) ;
A (2,2) = 0 ;
A = sparse (A)  ;

C = GB_mex_setElement (A, uint64(1), uint64(1), 99) ;
GB_spok (C.matrix) ;

if (use_ssget)
    Prob = ssget ('HB/west0067') ;
    A = Prob.A ;
else
    A = sprand (67, 67, 0.1) ;
end
[m n] = size (A) ;

ntuples = 1000 ;
A1 = A ;
I = 1 + floor (m * rand (ntuples, 1)) ;
J = 1 + floor (n * rand (ntuples, 1)) ;
X = 100 * rand (ntuples, 1) ;
I0 = uint64 (I)-1 ;
J0 = uint64 (J)-1 ;

for k = 1:ntuples
    A1 (I (k), J (k)) =  X (k) ;
end

A2 = A ;
A3 = GB_mex_setElement (A2, I0, J0, X) ;
assert (GB_spok (A3.matrix) == 1)

assert (isequal (A3.matrix, A1)) ;
% nnz (A)
% ntuples
% nnz (A1)
% nnz (A3.matrix)
% nnz (A) + ntuples

if (use_ssget)
    Prob = ssget (2662)
    A = Prob.A ;
else
    n = 2999349 ;
    nz = 14.3e6 ;
    density = nz / (n^2) ;
    A = sprandn (n, n, density) ;
end
[m n] = size (A) ;
fprintf ('nnz(A) = %g\n', nnz (A)) ;

for trial = 1:3

    if (trial == 1)
        fprintf ('\n---------------------- with I,J,X in sorted order\n') ;
    elseif (trial == 2)
        fprintf ('\n---------------------- with I,J,X in randomized order\n') ;
    elseif (trial == 3)
        fprintf ('\n---------------------- with I,J,X in randomized order') ;
        fprintf (' and duplicates\n') ;
    end

    ntuples = 100 ;
    A1 = A ;
    I = 1 + floor (m * rand (ntuples, 1)) ;
    J = 1 + floor (n * rand (ntuples, 1)) ;
    X = 100 * rand (ntuples, 1) ;
    I0 = uint64 (I)-1 ;
    J0 = uint64 (J)-1 ;

    fprintf ('starting built-in method... please wait\n') ;
    tic
    for k = 1:ntuples
        A1 (I (k), J (k)) =  X (k) ;
    end
    t = toc ;
    fprintf ('built-in set element:   %g sec\n', t) ;

    tic
    A2 = GB_mex_setElement (A, I0, J0, X) ;
    t2 = toc ;
    fprintf ('GraphBLAS setElement: %g seconds speedup %g\n', t2, t/t2) ;

    assert (isequal (A1, A2.matrix))

    tic
    [I,J,X]=find(A) ;
    t = toc ;
    fprintf ('built-in find:          %g sec\n', t) ;

    if (trial >= 2)
        p = randperm (length (X)) ;
        X = X (p) ;
        I = I (p) ;
        J = J (p) ;
    end

    tic
    G=sparse(I,J,X) ;
    t3 = toc ;
    fprintf ('built-in sparse:        %g sec\n', t3) ;

    I0 = uint64 (I)-1 ;
    J0 = uint64 (J)-1 ;
    S = sparse (m,n) ;
    tic
    S = GB_mex_setElement (S, I0, J0, X) ;
    t5 = toc ;
    fprintf ('GraphBLAS setElement: %g sec from scratch, nnz %d\n', ...
        t5, nnz (S.matrix)) ;

    % fprintf ('GB_spok it 1\n') ;
    assert (GB_spok (S.matrix*1) == 1) ;
    assert (isequal (G, S.matrix)) ;

    if (trial == 3)
        X = [X ; X] ;
        I = [I ; I] ;
        J = [J ; J] ;
        I0 = uint64 (I)-1 ;
        J0 = uint64 (J)-1 ;
        fprintf ('\nnow with lots of duplicates\n') ;
    end

    tic
    G=sparse(I,J,X) ;
    t3 = toc ;
    fprintf ('built-in sparse:        %g sec\n', t3) ;

    tic
    T = GB_mex_Matrix_build (I0, J0, X, m, n) ;
    t4 = toc ;
    fprintf ('GraphBLAS build:      %g sec from scratch, nnz %d\n', ...
        t4, nnz (T.matrix)) ;

    % fprintf ('GB_spok it 2\n') ;
    assert (GB_spok (T.matrix*1) == 1) ;
    assert (isequal (G, T.matrix)) ;

    fprintf ('\n------------------- now try a vector B = A(:)\n') ;

    B = A (:) ;
    blen = size (B,1) ;
    fprintf ('vector B has length %d with %d nonzeros\n', blen, nnz (B)) ;

    tic
    [I,J,X]=find(B) ;
    t = toc ;
    fprintf ('built-in find:          %g sec\n', t) ;

    if (trial == 2)
        p = randperm (length (X)) ;
        X = X (p) ;
        I = I (p) ;
        J = J (p) ;
    end

    tic
    G=sparse(I,J,X,blen,1) ;
    t3 = toc ;
    fprintf ('built-in sparse:        %g sec\n', t3) ;

    I0 = uint64 (I)-1 ;
    J0 = uint64 (J)-1 ;
    S = sparse (blen,1) ;

    tic
    S = GB_mex_setElement (S, I0, J0, X) ;
    t5 = toc ;
    fprintf ('GraphBLAS setElement: %g sec from scratch, nnz %d\n', ...
        t5, nnz (S.matrix)) ;

    % fprintf ('GB_spok it 3\n') ;
    assert (GB_spok (S.matrix*1) == 1) ;
    assert (isequal (G, S.matrix)) ;

    if (trial == 3)
        X = [X ; X] ;
        I = [I ; I] ;
        J = [J ; J] ;
        I0 = uint64 (I)-1 ;
        J0 = uint64 (J)-1 ;
        fprintf ('\nnow with lots of duplicates\n') ;
    end

    tic
    G=sparse(I,J,X,blen,1) ;
    t3 = toc ;
    fprintf ('built-in sparse:        %g sec\n', t3) ;

    tic
    T = GB_mex_Matrix_build (I0, J0, X, blen, 1) ;
    t4 = toc ;
    fprintf ('GraphBLAS mtx: build: %g sec from scratch, nnz %d\n', ...
        t4, nnz (T.matrix)) ;

    % fprintf ('GB_spok it 4\n') ;
    T_matrix = T.matrix * 1 ;
    assert (GB_spok (T_matrix) == 1) ;
    % assert (isequal (G, T.matrix)) ;
    assert (norm (G -  T_matrix, 1) / norm (G,1) < 1e-12) ;

    tic
    T = GB_mex_Vector_build (I0, X, blen) ;
    t4 = toc ;
    fprintf ('GraphBLAS vec: build: %g sec from scratch, nnz %d\n', ...
        t4, nnz (T.matrix)) ;

    % fprintf ('GB_spok it 4\n') ;
    T_matrix = T.matrix * 1 ;
    assert (GB_spok (T_matrix) == 1) ;
    assert (norm (G -  T_matrix, 1) / norm (G,1) < 1e-12) ;

end


fprintf ('\ntest45: all tests passed\n') ;

