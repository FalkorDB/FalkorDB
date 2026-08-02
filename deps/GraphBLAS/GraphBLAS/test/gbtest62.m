function gbtest62 (ghb)
%GBTEST62 test ldivide, rdivide, mldivide, mrdivide

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

n = 10 ;
for trial = 1:40

    fprintf ('.') ;

    A = 100 * rand (n) ;
    B = 100 * rand (n) ;
    b = rand (n, 1) ;

    r = rand ;
    s = gtb (ghb, r) ;

    GA = gtb (ghb, A) ;
    GB = gtb (ghb, B) ;

    C0 = A ./ r ;
    C1 = GA ./ s ;
    err = norm (C0-C1,1) ;
    assert (err < 1e-12) ;

    C0 = A / r ;
    C1 = GA / s ;
    err = norm (C0-C1,1) ;
    assert (err < 1e-12) ;

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
    C1 = gtb (ghb, 2) ./ s ;
    assert (isequal (C0, C1)) ;

    C0 = 2 ./ A ;
    C1 = 2 ./ GA ;
    assert (isequal (C0, C1)) ;

    C0 = 2 .\ r ;
    C1 = gtb (ghb, 2) .\ s ;
    assert (isequal (C0, C1)) ;

    C0 = 2 \ r ;
    C1 = gtb (ghb, 2) \ s ;
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
    GA = gtb (ghb, A) ;
    GB = gtb (ghb, B) ;

    C0 = A ./ B ;
    C1 = GA ./ GB ;
    assert (isequal (gtb_prune (ghb, C0, nan), gtb_prune (ghb, C1, nan))) ;

    C0 = A .\ B ;
    C1 = GA .\ GB ;
    assert (isequal (gtb_prune (ghb, C0, nan), gtb_prune (ghb, C1, nan))) ;

end

fprintf ('\ngbtest62 (%d): all tests passed\n', ghb) ;

