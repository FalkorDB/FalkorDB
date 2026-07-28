function gbtest36 (ghb)
%GBTEST36 test abs, sign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

types = gbtest_types ;
for k = 1:length (types)
    type = types {k} ;

    A = floor (100 * (rand (3, 3) - 0.5)) ;
    A (1,1) = 0 ;

    if (type (1) == 'u')
        A = max (A, 0) ;
    end
    G = gtb (ghb, A, type) ;
    B = gbtest_cast (A, type) ;
    assert (gbtest_eq (B, G))

    H = abs (G) ;
    C = abs (B) ;
    assert (gbtest_eq (double (C), double (H)))

    H = sign (G) ;
    if (isequal (type, 'logical'))
        C = double (B) ;
    else
        C = sign (B) ;
    end
    err = gbtest_err (C, H) ;
    if (gb_contains (type, 'single'))
        assert (err < 1e-6)
    else
        assert (err < 1e-12)
    end

end

fprintf ('gbtest36 (%d): all tests passed\n', ghb) ;

