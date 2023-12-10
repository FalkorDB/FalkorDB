function gbtest25
%GBTEST25 test diag, tril, triu

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
for trials = 1:10
    fprintf ('.') ;

    for m = 2:6
        for n = 2:6
            A = sprand (m, n, 0.5) ;
            G = GrB (A) ;
            for k = -m:n
                B = diag (A, k) ;
                C = diag (G, k) ;
                assert (gbtest_eq (B, C)) ;
                B = tril (A, k) ;
                C = tril (A, k) ;
                assert (gbtest_eq (B, C)) ;
                B = triu (A, k) ;
                C = triu (A, k) ;
                assert (gbtest_eq (B, C)) ;
            end
            B = diag (A) ;
            C = diag (G) ;
            assert (gbtest_eq (B, C)) ;
        end
    end

    for m = 1:6
        A = sprandn (m, 1, 0.5) ;
        G = GrB (A) ;
        for k = -6:6
            B = diag (A, k) ;
            C = diag (G, k) ;
            assert (gbtest_eq (B, C)) ;
            B = tril (A, k) ;
            C = tril (G, k) ;
            assert (gbtest_eq (B, C)) ;
            B = triu (A, k) ;
            C = triu (G, k) ;
            assert (gbtest_eq (B, C)) ;
        end

        B = diag (A) ;
        C = diag (G) ;
        assert (gbtest_eq (B, C)) ;
        B = tril (A) ;
        C = tril (G) ;
        assert (gbtest_eq (B, C)) ;
        B = triu (A) ;
        C = triu (G) ;
        assert (gbtest_eq (B, C)) ;
    end
end

fprintf ('\ngbtest25: all tests passed\n') ;

