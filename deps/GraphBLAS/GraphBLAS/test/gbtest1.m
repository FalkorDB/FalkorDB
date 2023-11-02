function gbtest1
%GBTEST1 test GrB

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
X = 100 * sprand (3, 4, 0.4) %#ok<*NOPRT>

% types = { 'double' } ;

types = gbtest_types ;

m = 2 ;
n = 3 ;

for k = 1:length (types)
    type = types {k} ;

    fprintf ('\n---- A = GrB (X) :\n') ;
    A = GrB (X)
    Z = double (A)
    assert (gbtest_eq (Z, X)) ;

    fprintf ('\n---- A = GrB (X, ''%s'') :\n', type) ;
    A = GrB (X, type)
    Z = logical (A)
    if (isequal (type, 'logical'))
        assert (islogical (Z)) ;
        assert (gbtest_eq (Z, logical (X))) ;
    end

    fprintf ('\n---- A = GrB (%d, %d) :\n', m, n) ;
    A = GrB (m, n)
    Z = sparse (m, n)
    assert (isequal (A, Z)) ;
    A = GrB (m, n, 'by row') ;
    assert (isequal (A, Z)) ;

    fprintf ('\n---- A = GrB (%d, %d, ''%s'') :\n', m, n, type) ;
    A = GrB (m, n, type)
    Z = logical (A)
    if (isequal (type, 'logical'))
        assert (islogical (Z)) ;
        assert (gbtest_eq (Z, logical (sparse (m,n)))) ;
    end

    Z = full (fix (X)) ;
    A = GrB (Z, 'by row', type) ;
    Y = gbtest_cast (Z, type) ;
    assert (gbtest_eq (A, Y)) ;

end

X = [ ] ;
A = GrB (X, 'by row', 'double') ;
assert (isequal (A, X)) ;

A = GrB (m, n, 'by row', 'double') ;
X = sparse (m, n) ;
assert (isequal (A, X)) ;

fprintf ('gbtest1: all tests passed\n') ;

