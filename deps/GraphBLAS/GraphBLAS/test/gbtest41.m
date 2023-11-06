function gbtest41
%GBTEST41 test ones, zeros, false

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

types = gbtest_types ;

for trial = 1:40
    fprintf ('.') ;

    for k = 1:length(types)
        type = types {k} ;
        G = GrB (rand (2), type) ;

        G2 = GrB.ones (3, 4, 'like', G) ;
        G3 = GrB (ones (3, 4), type) ;
        assert (gbtest_eq (G2, G3)) ;

        G1 = GrB.zeros ([3, 4], 'like', G) ;
        G2 = GrB.zeros (3, 4, 'like', G) ;
        G3 = GrB (zeros (3, 4), type) ;

        assert (isequal (G1, G2)) ;
        assert (isequal (GrB.type (G2), GrB.type (G3))) ;
        assert (isequal (type, GrB.type (G3))) ;
        assert (norm (double (G2) - double (G3), 1) == 0) ;

        if (isequal (type, 'logical'))
            G2 = GrB.false (3, 4, 'like', G) ;
            G3 = GrB (false (3, 4)) ;
            assert (isequal (GrB.type (G2), GrB.type (G3))) ;
            assert (isequal (type, GrB.type (G3))) ;
            assert (norm (double (G2) - double (G3), 1) == 0) ;
            assert (gbtest_eq (G2, G3)) ;

            G2 = GrB.true (3, 4, 'like', G) ;
            G3 = GrB (true (3, 4), type) ;
            assert (isequal (GrB.type (G2), GrB.type (G3))) ;
            assert (isequal (type, GrB.type (G3))) ;
            assert (norm (double (G2) - double (G3), 1) == 0) ;
            assert (gbtest_eq (G2, G3)) ;
        end
    end
end

fprintf ('\ngbtest41: all tests passed\n') ;

