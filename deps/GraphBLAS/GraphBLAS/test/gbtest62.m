function gbtest62
%GBTEST62 test ldivide, rdivide, mldivide, mrdivide

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 10 ;
for trial = 1:40

    fprintf ('.') ;

    A = 100 * rand (n) ;
    B = 100 * rand (n) ;
    b = rand (n, 1) ;

    r = rand ;
    s = GrB (r) ;

    GA = GrB (A) ;
    GB = GrB (B) ;

    C0 = A ./ r ;
    C1 = GA ./ s ;
    assert (isequal (C0, C1)) ;

    C0 = A / r ;
    C1 = GA / s ;
    assert (isequal (C0, C1)) ;

    C0 = A ./ 0 ;
    C1 = GA ./ 0 ;
    assert (isequal (C0, C1)) ;

    C0 = A / 0 ;
    C1 = GA / 0 ;
    assert (isequal (C0, C1)) ;

    C0 = 0 .\ A ;
    C1 = 0 .\ GA ;
    assert (isequal (C0, C1)) ;

    C0 = 0 \ A ;
    C1 = 0 \ GA ;
    assert (isequal (C0, C1)) ;

    C0 = 2 ./ r ;
    C1 = GrB (2) ./ s ;
    assert (isequal (C0, C1)) ;

    C0 = 2 ./ A ;
    C1 = 2 ./ GA ;
    assert (isequal (C0, C1)) ;

    C0 = 2 .\ r ;
    C1 = GrB (2) .\ s ;
    assert (isequal (C0, C1)) ;

    C0 = 2 \ r ;
    C1 = GrB (2) \ s ;
    assert (isequal (C0, C1)) ;

    C0 = A ./ B ;
    C1 = GA ./ GB ;
    assert (isequal (C0, C1)) ;

    C0 = A .\ B ;
    C1 = GA .\ GB ;
    assert (isequal (C0, C1)) ;

    x = A \ b ;
    y = GA \ b ;
    assert (norm (x - y) < 1e-12) ;

    x = b' / A ;
    y = b' / GA ;
    assert (norm (x - y) < 1e-12) ;

    A = sprand (n, n, 0.5) ;
    B = rand * A ;
    GA = GrB (A) ;
    GB = GrB (B) ;

    C0 = A ./ B ;
    C1 = GA ./ GB ;
    assert (isequal (GrB.prune (C0, nan), GrB.prune (C1, nan))) ;

    C0 = A .\ B ;
    C1 = GA .\ GB ;
    assert (isequal (GrB.prune (C0, nan), GrB.prune (C1, nan))) ;

end

fprintf ('\ngbtest62: all tests passed\n') ;

