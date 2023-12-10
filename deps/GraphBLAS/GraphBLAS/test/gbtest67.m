function gbtest67
%GBTEST67 test digraph

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 32 ;
for trial = 1:40
    fprintf ('.') ;

    A = sprand (n, n, 0.5) ;
    G = GrB (A) ;

    D1 = digraph (A) ;
    D2 = digraph (G) ;
    assert (isequal (D1, D2)) ;

    D1 = digraph (A, 'omitselfloops') ;
    D2 = digraph (G, 'omitselfloops') ;
    assert (isequal (D1, D2)) ;

    D1 = digraph (logical (A)) ;
    D2 = digraph (GrB (A, 'logical')) ;
    assert (isequal (D1, D2)) ;

    D1 = digraph (logical (A), 'omitselfloops') ;
    D2 = digraph (GrB (A, 'logical'), 'omitselfloops') ;
    assert (isequal (D1, D2)) ;

end

types = gbtest_types ;

for k = 1:length (types)
    type = types {k} ;

    A = gbtest_cast (rand (4), type) ;
    G = GrB (A) ;

    if (isequal (type, 'double') || isequal (type, 'single') || ...
        isequal (type, 'logical'))
        D1 = digraph (A) ;
    else
        D1 = digraph (double (A)) ;
    end

    D2 = digraph (G) ;
    assert (isequal (D1, D2)) ;
end

fprintf ('\ngbtest67: all tests passed\n') ;

