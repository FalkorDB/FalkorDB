function gbtest9 (ghb)
%GBTEST9 test eye and speye

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

types = gbtest_types ;

A = eye ;
G = gtb_eye (ghb) ;
assert (gbtest_eq (A, G)) ;
G = gtb_speye (ghb) ;
assert (gbtest_eq (A, G)) ;

for m = -1:10
    fprintf ('.') ;

    A = eye (m) ;
    G = gtb_eye (ghb, m) ;
    assert (gbtest_eq (A, G)) ;
    G = gtb_speye (ghb, m) ;
    assert (gbtest_eq (A, G)) ;

    for n = -1:10

        A = eye (m, n) ;
        G = gtb_eye (ghb, m, n) ;
        assert (gbtest_eq (A, G)) ;
        G = gtb_speye (ghb, m, n) ;
        assert (gbtest_eq (A, G)) ;

        for k = 1:length (types)
            type = types {k} ;

            A = gbtest_cast (eye (m, n), type) ;

            G = gtb_eye (ghb, m, n, type) ;
            assert (gbtest_eq (A, G)) ;
            G = gtb_speye (ghb, m, n, type) ;
            assert (gbtest_eq (A, G)) ;

            G = gtb_eye (ghb, [m n], type) ;
            assert (gbtest_eq (A, G)) ;
            G = gtb_speye (ghb, [m n], type) ;
            assert (gbtest_eq (A, G)) ;

            A = gbtest_cast (eye (m, m), type) ;

            G = gtb_eye (ghb, m, type) ;
            assert (gbtest_eq (A, G)) ;
            G = gtb_speye (ghb, m, type) ;
            assert (gbtest_eq (A, G)) ;

        end
    end
end

fprintf ('\ngbtest9 (%d): all tests passed\n', ghb) ;

