function gbtest56 (ghb)
%GBTEST56 test [GrB,GhB].empty

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

for m1 = -1:5
    for n1 = -1:5

        m = max (m1, 0) ;
        n = max (n1, 0) ;

        if (~ ((m == 0) || (n == 0)))
            continue
        end

        C1 = gtb_empty (ghb, m1, n1) ;
        C2 = gtb_empty (ghb, [m1, n1]) ;
        C3 = gtb (ghb, m, n) ;
        C0 = sparse (m, n) ;
        C4 = gzb (0, m, n, 'double', 'by col') ;
        C5 = gzb (1, m, n, 'double', 'by col') ;

        assert (isequal (C0, C1)) ;
        assert (isequal (C0, C2)) ;
        assert (isequal (C0, C3)) ;
        assert (isequal (C0, C4)) ;
        assert (isequal (C0, C5)) ;
    end
end

C1 = gtb_empty (ghb, 0) ;
C2 = gtb_empty (ghb, -1) ;
C3 = gtb (ghb, 0,0) ;
C0 = sparse (0,0) ;

assert (isequal (C0, C1)) ;
assert (isequal (C0, C2)) ;
assert (isequal (C0, C3)) ;

assert (length (C0) == 0) ; %#ok<*ISMT>
assert (length (C1) == 0) ;
assert (length (C2) == 0) ;
assert (length (C3) == 0) ;

C1 = gtb_empty (ghb, 0,5) ;
C2 = gtb_empty (ghb, 0,5) ;
C3 = gtb (ghb, 0,5) ;
C0 = sparse (0,5) ;

assert (isequal (C0, C1)) ;
assert (isequal (C0, C2)) ;
assert (isequal (C0, C3)) ;

assert (length (C0) == 0) ;
assert (length (C1) == 0) ;
assert (length (C2) == 0) ;
assert (length (C3) == 0) ;

C1 = gtb_empty (ghb, 5,0) ;
C2 = gtb_empty (ghb, 5,0) ;
C3 = gtb (ghb, 5,0) ;
C0 = sparse (5,0) ;

assert (isequal (C0, C1)) ;
assert (isequal (C0, C2)) ;
assert (isequal (C0, C3)) ;

assert (length (C0) == 0) ;
assert (length (C1) == 0) ;
assert (length (C2) == 0) ;
assert (length (C3) == 0) ;

fprintf ('gbtest56 (%d): all tests passed\n', ghb) ;

