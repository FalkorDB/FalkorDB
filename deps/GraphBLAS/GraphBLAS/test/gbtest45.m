function gbtest45
%GBTEST45 test GrB.vreduce

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
d.kind = 'sparse' ;

for trial = 1:40

    A = rand (4) ;
    G = GrB (A) ;
    x = GrB.vreduce ('+', A) ;
    y = GrB.vreduce ('+', G) ;
    t = GrB.vreduce ('+', G, d) ;
    z = sum (G, 2) ;
    w = sum (A, 2) ;

    assert (isequal (w, x)) ;
    assert (isequal (w, y)) ;
    assert (isequal (w, z)) ;
    assert (isequal (w, t)) ;

    assert (isequal (class (t), 'double')) ;

    cin = rand (4,1) ;
    x = GrB.vreduce (cin, '+', '+', A) ;
    y = cin + sum (A, 2) ;
    assert (isequal (x, y)) ;

    m = logical (sprand (4, 1, 0.5)) ;
    x = GrB.vreduce (cin, m, '+', '+', A) ;
    t = cin + sum (A, 2) ;
    y = cin ;
    y (m) = t (m) ;
    assert (isequal (x, y)) ;

    x = GrB.vreduce (cin, m, '+', A) ;
    t = sum (A, 2) ;
    y = cin ;
    y (m) = t (m) ;
    assert (isequal (x, y)) ;

end

fprintf ('gbtest45: all tests passed\n') ;

