function gbtest1 (ghb)
%GBTEST1 test GrB and GhB constructors

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

X = 100 * sprand (3, 4, 0.4) %#ok<*NOPRT>

% types = { 'double' } ;

types = gbtest_types ;

m = 2 ;
n = 3 ;

for k = 1:length (types)
    type = types {k} ;

    fprintf ('\n---- A = %s (X) :\n', gtb_name) ;
    A = gtb (ghb, X)
    Z = double (A)
    assert (gbtest_eq (Z, X)) ;

    fprintf ('\n---- A = %s (X, ''%s'') :\n', gtb_name, type) ;
    A = gtb (ghb, X, type)
    Z = logical (A)
    if (isequal (type, 'logical'))
        assert (islogical (Z)) ;
        assert (gbtest_eq (Z, logical (X))) ;
    end

    fprintf ('\n---- A = %s (%d, %d) :\n', gtb_name, m, n) ;
    A = gtb (ghb, m, n)
    Z = sparse (m, n)
    assert (isequal (A, Z)) ;
    A = gtb (ghb, m, n, 'by row') ;
    Z
    A
    assert (isequal (A, Z)) ;

    B = gb_dup (ghb, GrB (Z)) ;
    assert (isequal (B, Z)) ;

    B = gb_dup (ghb, GhB (Z)) ;
    assert (isequal (B, Z)) ;

    fprintf ('\n---- A = %s (%d, %d, ''%s'') :\n', gtb_name, m, n, type) ;
    A = gtb (ghb, m, n, type)
    Z = logical (A)
    if (isequal (type, 'logical'))
        assert (islogical (Z)) ;
        assert (gbtest_eq (Z, logical (sparse (m,n)))) ;
    end

    Z = full (fix (X)) ;
    A = gtb (ghb, Z, 'by row', type) ;
    Y = gbtest_cast (Z, type) ;
    assert (gbtest_eq (A, Y)) ;

end

X = [ ] ;
A = gtb (ghb, X, 'by row', 'double') ;
assert (isequal (A, X)) ;

A = gtb (ghb, m, n, 'by row', 'double') ;
X = sparse (m, n) ;
assert (isequal (A, X)) ;

% GrB or GhB with no inputs:
gtb (ghb) ;

fprintf ('gbtest1 (%d): all tests passed\n', ghb) ;

