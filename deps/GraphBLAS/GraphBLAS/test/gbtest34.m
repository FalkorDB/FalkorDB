function gbtest34 (ghb)
%GBTEST34 test repmat

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

for m1 = 0:5
    for n1 = 0:5
        fprintf ('.') ;
        for m2 = 0:5
            for n2 = 0:5
                A = rand (m1, n1) ;
                C = repmat (A, m2, n2) ;
                G = gtb (ghb, A) ;
                H = repmat (G, m2, n2) ;
                assert (gbtest_eq (C, H)) ;

                C = repmat (A, m2) ;
                H = repmat (G, m2) ;
                assert (gbtest_eq (C, H)) ;
            end
        end
    end
end

fprintf ('\ngbtest34 (%d): all tests passed\n', ghb) ;

