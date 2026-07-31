function gbtest122 (ghb)
%GBTEST122 test reshape (extended methods in GrB, GhB)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

for m = 1:12
    fprintf ('.') ;
    for n = 1:12
        for kind = [0 1]
            if (kind == 0)
                A = rand (m, n) ;
            else
                A = sprand (m, n, 0.3) ;
            end
            G = gtb (ghb, A) ;
            mn = m*n ;
            H = gtb (ghb, A, 'by row') ;

            f = factor (mn) ;

            for k = 1:length (f)
                S = nchoosek (f, k) ;
                for i = 1:size(S,1)

                    % reshape by column
                    m2 = prod (S (i,:)) ;
                    n2 = mn / m2 ;
                    C1 = reshape (A, m2, n2) ;
                    C2 = reshape (G, m2, n2) ;
                    assert (gbtest_eq (C1, C2)) ;

                    C3 = reshape (H, m2, n2) ;
                    assert (gbtest_eq (C1, C3)) ;

                    C1 = reshape (A, [m2 n2]) ;
                    C2 = reshape (G, [m2 n2]) ;
                    assert (gbtest_eq (C1, C2)) ;

                    C3 = reshape (H, [m2 n2]) ;
                    assert (gbtest_eq (C1, C3)) ;

                    % reshape by row
                    C1 = reshape (A', n2, m2)' ;
                    C2 = reshape (G, m2, n2, 'by row') ;
                    assert (gbtest_eq (C1, C2)) ;

                    C3 = reshape (H, m2, n2, 'by row') ;
                    assert (gbtest_eq (C1, C3)) ;

                    C2 = reshape (G, [m2 n2], 'by row') ;
                    assert (gbtest_eq (C1, C2)) ;

                    C3 = reshape (H, [m2 n2], 'by row') ;
                    assert (gbtest_eq (C1, C3)) ;

                end
            end
        end
    end
end

fprintf ('\ngbtest122 (%d): all tests passed\n', ghb) ;

