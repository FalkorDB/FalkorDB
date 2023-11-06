function gbtest66
%GBTEST66 test graph

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 32 ;
for trial = 1:40
    fprintf ('.') ;

    A = sprand (n, n, 0.5) ;
    A = A + A' ;
    G = GrB (A) ;

    D1 = graph (A) ;
    D2 = graph (G) ;
    assert (isequal (D1, D2)) ;

    D1 = graph (A, 'upper') ;
    D2 = graph (G, 'upper') ;
    D3 = graph (triu (A), 'upper') ;
    D4 = graph (triu (G), 'upper') ;
    assert (isequal (D1, D2)) ;
    assert (isequal (D1, D3)) ;
    assert (isequal (D1, D4)) ;

    D1 = graph (A, 'lower') ;
    D2 = graph (G, 'lower') ;
    D3 = graph (tril (A), 'lower') ;
    D4 = graph (tril (G), 'lower') ;
    assert (isequal (D1, D2)) ;
    assert (isequal (D1, D3)) ;
    assert (isequal (D1, D4)) ;

    D1 = graph (A, 'omitselfloops') ;
    D2 = graph (G, 'omitselfloops') ;
    assert (isequal (D1, D2)) ;

    D1 = graph (A, 'lower', 'omitselfloops') ;
    D2 = graph (G, 'lower', 'omitselfloops') ;
    D3 = graph (tril (A), 'lower', 'omitselfloops') ;
    D4 = graph (tril (G), 'lower', 'omitselfloops') ;
    assert (isequal (D1, D2)) ;
    assert (isequal (D1, D3)) ;
    assert (isequal (D1, D4)) ;

    D1 = graph (A, 'upper', 'omitselfloops') ;
    D2 = graph (G, 'upper', 'omitselfloops') ;
    D3 = graph (triu (A), 'upper', 'omitselfloops') ;
    D4 = graph (triu (G), 'upper', 'omitselfloops') ;
    assert (isequal (D1, D2)) ;
    assert (isequal (D1, D3)) ;
    assert (isequal (D1, D4)) ;

    D1 = graph (logical (A)) ;
    D2 = graph (GrB (A, 'logical')) ;
    assert (isequal (D1, D2)) ;

    D1 = graph (logical (A), 'omitselfloops') ;
    D2 = graph (GrB (A, 'logical'), 'omitselfloops') ;
    assert (isequal (D1, D2)) ;
end

types = gbtest_types ;

for k = 1:length (types)
    type = types {k} ;

    A = gbtest_cast (rand (4), type) ;
    A = A + A' ;
    G = GrB (A) ;

    if (isequal (type, 'double') || isequal (type, 'single') || ...
        isequal (type, 'logical'))
        D1 = graph (A) ;
    else
        D1 = graph (double (A)) ;
    end

    D2 = graph (G) ;
    assert (isequal (D1, D2)) ;
end

fprintf ('\ngbtest66: all tests passed\n') ;

