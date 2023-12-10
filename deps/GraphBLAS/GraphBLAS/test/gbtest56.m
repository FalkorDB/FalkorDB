function gbtest56
%GBTEST56 test GrB.empty

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

for m1 = -1:5
    for n1 = -1:5

        m = max (m1, 0) ;
        n = max (n1, 0) ;

        if (~ ((m == 0) || (n == 0)))
            continue
        end

        C1 = GrB.empty (m1, n1) ;
        C2 = GrB.empty ([m1, n1]) ;
        C3 = GrB (m, n) ;
        C0 = sparse (m, n) ;

        assert (isequal (C0, C1)) ;
        assert (isequal (C0, C2)) ;
        assert (isequal (C0, C3)) ;
    end
end

C1 = GrB.empty (0) ;
C2 = GrB.empty (-1) ;
C3 = GrB (0,0) ;
C0 = sparse (0,0) ;

assert (isequal (C0, C1)) ;
assert (isequal (C0, C2)) ;
assert (isequal (C0, C3)) ;

assert (length (C0) == 0) ; %#ok<*ISMT>
assert (length (C1) == 0) ;
assert (length (C2) == 0) ;
assert (length (C3) == 0) ;

C1 = GrB.empty (0,5) ;
C2 = GrB.empty (0,5) ;
C3 = GrB (0,5) ;
C0 = sparse (0,5) ;

assert (isequal (C0, C1)) ;
assert (isequal (C0, C2)) ;
assert (isequal (C0, C3)) ;

assert (length (C0) == 0) ;
assert (length (C1) == 0) ;
assert (length (C2) == 0) ;
assert (length (C3) == 0) ;

C1 = GrB.empty (5,0) ;
C2 = GrB.empty (5,0) ;
C3 = GrB (5,0) ;
C0 = sparse (5,0) ;

assert (isequal (C0, C1)) ;
assert (isequal (C0, C2)) ;
assert (isequal (C0, C3)) ;

assert (length (C0) == 0) ;
assert (length (C1) == 0) ;
assert (length (C2) == 0) ;
assert (length (C3) == 0) ;

fprintf ('gbtest56: all tests passed\n') ;

