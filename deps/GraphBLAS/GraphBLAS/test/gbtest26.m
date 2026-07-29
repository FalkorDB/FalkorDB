function gbtest26 (ghb)
%GBTEST26 test typecasting

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

types = gbtest_types ;

for k1 = 1:length (types)

    atype = types {k1} ;
    fprintf ('\n================================================ %s\n', atype) ;
    A = gbtest_cast (100 * rand (3), atype) %#ok<*NOPRT>
    H = gtb (ghb, A) ;
    B = gtb (ghb, H, atype) ;
    assert (gbtest_eq (A, B)) ;

    for k2 = 1:length (types)

        gtype = types {k2} ;
        fprintf ('\n------------ %s:\n', gtype) ;
        G = gtb (ghb, H, gtype)
        K = gtb (ghb, G, atype) %#ok<*NASGU>
    end
end

fprintf ('gbtest26 (%d): all tests passed\n', ghb) ;

