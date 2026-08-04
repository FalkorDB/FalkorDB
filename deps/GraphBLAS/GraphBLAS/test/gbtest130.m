function gbtest130 (ghb)
%GBTEST130 test argmin and argmax

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

types = gbtest_types ;
for k = 1:length (types)
    type = types {k} ;
    if (~gb_contains (type, 'complex'))
        fprintf ('%s ', type) ;
        if (gb_contains (type, 'uint') || isequal (type, 'logical'))
            lo = 0 ;
        else
            lo = -10 ;
        end
        if (isequal (type, 'logical'))
            hi = 1 ;
        else
            hi = 10 ;
        end
        for fmt = {'by row', 'by col'} ;
            G = gtb_random (ghb, 10, 10, 0.3, 'range', ...
                gtb (ghb, [lo hi], type)) ;
            G = gtb (ghb, G, fmt {1}) ;
            for dim = 0:2
                [x1, i1] = gbtest_argminmax (ghb, G, true, dim) ;
                [x2, i2] = gtb_argmin (ghb, G, dim) ;
                assert (isequal (x1, x2)) ;
                assert (isequal (i1, i2)) ;
                [x1, i1] = gbtest_argminmax (ghb, G, false, dim) ;
                [x2, i2] = gtb_argmax (ghb, G, dim) ;
                assert (isequal (x1, x2)) ;
                assert (isequal (i1, i2)) ;
            end
        end
    end
end

fprintf ('\ngbtest130 (%d): all tests passed\n', ghb) ;

