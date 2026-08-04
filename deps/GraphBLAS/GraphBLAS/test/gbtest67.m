function gbtest67 (ghb)
%GBTEST67 test digraph

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

n = 32 ;
for trial = 1:40
    fprintf ('.') ;

    A = sprand (n, n, 0.5) ;
    G = gtb (ghb, A) ;

    D1 = digraph (A) ;
    D2 = digraph (G) ;
    assert (isequal (D1, D2)) ;

    D1 = digraph (A, 'omitselfloops') ;
    D2 = digraph (G, 'omitselfloops') ;
    assert (isequal (D1, D2)) ;

    D1 = digraph (logical (A)) ;
    D2 = digraph (gtb (ghb, A, 'logical')) ;
    assert (isequal (D1, D2)) ;

    D1 = digraph (logical (A), 'omitselfloops') ;
    D2 = digraph (gtb (ghb, A, 'logical'), 'omitselfloops') ;
    assert (isequal (D1, D2)) ;

end

types = gbtest_types ;

for k = 1:length (types)
    type = types {k} ;

    A = gbtest_cast (rand (4), type) ;
    G = gtb (ghb, A) ;

    if (isequal (type, 'double') || isequal (type, 'single') || ...
        isequal (type, 'logical'))
        D1 = digraph (A) ;
    else
        A2 = real (double (A)) ;
        D1 = digraph (A2) ;
    end

    D2 = digraph (G) ;
    assert (isequal (D1, D2)) ;
end

fprintf ('\ngbtest67 (%d): all tests passed\n', ghb) ;

