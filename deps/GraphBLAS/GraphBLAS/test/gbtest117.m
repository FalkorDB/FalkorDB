function gbtest117
%GBTEST117 test idxunop in GrB.apply2

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

m = 6 ;
n = 4 ;
A = GrB.random (m, n, 0.5) ;
[i,j,x] = find (A) ;

for k = -6:6

    %   tril            j <= (i + thunk)
    C1 = GrB.apply2 ('tril', A, k) ;
    x = (j <= (i + k)) ;
    C2 = GrB.build (i, j, x, m, n) ;
    assert (isequal (C1, C2))

    %   triu            j >= (i + thunk)
    C1 = GrB.apply2 ('triu', A, k) ;
    x = (j >= (i + k)) ;
    C2 = GrB.build (i, j, x, m, n) ;
    assert (isequal (C1, C2))

    %   diag            j == (i + thunk)
    C1 = GrB.apply2 ('diag', A, k) ;
    x = (j == (i + k)) ;
    C2 = GrB.build (i, j, x, m, n) ;
    assert (isequal (C1, C2))

    %   offdiag         j != (i + thunk)
    C1 = GrB.apply2 ('offdiag', A, k) ;
    x = (j ~= (i + k)) ;
    C2 = GrB.build (i, j, x, m, n) ;
    assert (isequal (C1, C2))

    %   diagindex       j - (i + thunk)
    C1 = GrB.apply2 ('diagindex', A, k) ;
    x = int64 (j - (i + k)) ;
    C2 = GrB.build (i, j, x, m, n) ;
    assert (isequal (C1, C2))

    %   rowindex        i + thunk
    C1 = GrB.apply2 ('rowindex', A, k) ;
    x = int64 (i + k) ;
    C2 = GrB.build (i, j, x, m, n) ;
    assert (isequal (C1, C2))

    %   rowle           i <= thunk
    C1 = GrB.apply2 ('rowle', A, k) ;
    x = i <= k ;
    C2 = GrB.build (i, j, x, m, n) ;
    assert (isequal (C1, C2))

    %   rowgt           i > thunk
    C1 = GrB.apply2 ('rowgt', A, k) ;
    x = i > k ;
    C2 = GrB.build (i, j, x, m, n) ;
    assert (isequal (C1, C2))

    %   colindex        j + thunk
    C1 = GrB.apply2 ('colindex', A, k) ;
    x = int64 (j + k) ;
    C2 = GrB.build (i, j, x, m, n) ;
    assert (isequal (C1, C2))

    %   colle           j <= thunk
    C1 = GrB.apply2 ('colle', A, k) ;
    x = j <= k ;
    C2 = GrB.build (i, j, x, m, n) ;
    assert (isequal (C1, C2))

    %   colgt           j > thunk
    C1 = GrB.apply2 ('colgt', A, k) ;
    x = j > k ;
    C2 = GrB.build (i, j, x, m, n) ;
    assert (isequal (C1, C2))

end

%   diagindex       j - (i + thunk)
C1 = GrB.apply2 ('diagindex', 0, 3) ;
assert (C1 == -3) ;
C1 = GrB.apply2 ('diagindex', 1, 0) ;
assert (C1 == 0) ;

fprintf ('gbtest117: all tests passed\n') ;
