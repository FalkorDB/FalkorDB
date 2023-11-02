function gbtest34
%GBTEST34 test repmat

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

for m1 = 0:5
    for n1 = 0:5
        fprintf ('.') ;
        for m2 = 0:5
            for n2 = 0:5
                A = rand (m1, n1) ;
                C = repmat (A, m2, n2) ;
                G = GrB (A) ;
                H = repmat (G, m2, n2) ;
                assert (gbtest_eq (C, H)) ;

                C = repmat (A, m2) ;
                H = repmat (G, m2) ;
                assert (gbtest_eq (C, H)) ;
            end
        end
    end
end

fprintf ('\ngbtest34: all tests passed\n') ;

