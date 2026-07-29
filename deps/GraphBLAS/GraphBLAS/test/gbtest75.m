function gbtest75 (ghb)
%GBTEST75 test bitshift

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

fprintf ('\ngbtest75: bitshift\n') ;

for b = -8:8
    fprintf ('.') ;
    for a = intmin ('int8') : intmax ('int8')
        c = bitshift (a, b) ;
        c2 = bitshift (gtb (ghb, a), gtb (ghb, b)) ;
        assert (isequal (c, c2)) ;
    end
end

fprintf ('\ngbtest75 (%d): all tests passed\n', ghb) ;

