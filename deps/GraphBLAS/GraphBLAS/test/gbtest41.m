function gbtest41 (ghb)
%GBTEST41 test ones, zeros, false

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

types = gbtest_types ;

for trial = 1:40
    fprintf ('.') ;

    for k = 1:length(types)
        type = types {k} ;
        G = gtb (ghb, rand (2), type) ;

        G2 = gtb_ones (ghb, 3, 4, 'like', G) ;
        G3 = gtb (ghb, ones (3, 4), type) ;
        G4 = gb_scalar_to_full (ghb, 3, 4, type, 'by col', GrB (1)) ;
        assert (gbtest_eq (G2, G3)) ;
        assert (gbtest_eq (G2, G4)) ;

        G1 = gtb_zeros (ghb, [3, 4], 'like', G) ;
        G2 = gtb_zeros (ghb, 3, 4, 'like', G) ;
        G3 = gtb (ghb, zeros (3, 4), type) ;

        assert (isequal (G1, G2)) ;
        assert (isequal (gtb_type (ghb, G2), gtb_type (ghb, G3))) ;
        assert (isequal (type, gtb_type (ghb, G3))) ;
        assert (norm (double (G2) - double (G3), 1) == 0) ;

        if (isequal (type, 'logical'))
            G2 = gtb_false (ghb, 3, 4, 'like', G) ;
            G3 = gtb (ghb, false (3, 4)) ;
            assert (isequal (gtb_type (ghb, G2), gtb_type (ghb, G3))) ;
            assert (isequal (type, gtb_type (ghb, G3))) ;
            assert (norm (double (G2) - double (G3), 1) == 0) ;
            assert (gbtest_eq (G2, G3)) ;

            G2 = gtb_true (ghb, 3, 4, 'like', G) ;
            G3 = gtb (ghb, true (3, 4), type) ;
            assert (isequal (gtb_type (ghb, G2), gtb_type (ghb, G3))) ;
            assert (isequal (type, gtb_type (ghb, G3))) ;
            assert (norm (double (G2) - double (G3), 1) == 0) ;
            assert (gbtest_eq (G2, G3)) ;
        end
    end
end

fprintf ('\ngbtest41 (%d): all tests passed\n', ghb) ;

