function gbtest13
%GBTEST13 test find and GrB.extracttuples

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

list = gbtest_types ;

A = 100 * rand (3) ;
[I, J, X] = find (A) ; %#ok<*ASGLU>
I_0 = int64 (I) - 1 ;
J_0 = int64 (J) - 1 ;
A (1,1) = 0 ;

desc_default.base = 'default' ;
desc0.base = 'zero-based' ;
desc1.base = 'one-based' ;
desc1_int.base = 'one-based int' ;

for k = 1:length(list)
    xtype = list {k} ;
    fprintf ('%s ', xtype) ;
    C = gbtest_cast (A, xtype) ;
    G = GrB (C) ;

    [I1, J1, X1] = find (G) ;
    nz = find (C (:) ~= 0) ;
    assert (isequal (C (nz), X1)) ;
    assert (isequal (I (nz), I1)) ;
    assert (isequal (J (nz), J1)) ;

    [I1, J1] = find (G) ;
    assert (isequal (I (nz), I1)) ;
    assert (isequal (J (nz), J1)) ;

    [I1] = find (G) ;
    [I0] = find (C) ;
    assert (isequal (I0, I1)) ;

    [I0, J0, X0] = GrB.extracttuples (G, desc0)  ;
    assert (isequal (C (:), X0)) ;
    assert (isequal (I_0, I0)) ;
    assert (isequal (J_0, J0)) ;

    [I1, J1, X0] = GrB.extracttuples (G, desc1) ;
    assert (isequal (C (:), X0)) ;
    assert (isequal (double (I_0+1), I1)) ;
    assert (isequal (double (J_0+1), J1)) ;

    [I1, J1, X0] = GrB.extracttuples (G, desc1_int) ;
    assert (isequal (C (:), X0)) ;
    assert (isequal (I_0+1, I1)) ;
    assert (isequal (J_0+1, J1)) ;

    [I1, J1, X0] = GrB.extracttuples (G, desc_default) ;
    assert (isequal (C (:), X0)) ;
    assert (isequal (double (I_0+1), I1)) ;
    assert (isequal (double (J_0+1), J1)) ;

    [I1, J1, X0] = GrB.extracttuples (G) ;
    assert (isequal (C (:), X0)) ;
    assert (isequal (double (I_0+1), I1)) ;
    assert (isequal (double (J_0+1), J1)) ;

    [I1, J1] = GrB.extracttuples (G) ;
    assert (isequal (double (I_0+1), I1)) ;
    assert (isequal (double (J_0+1), J1)) ;

    [I1] = GrB.extracttuples (G, desc0) ;
    assert (isequal (I1, I0)) ;
end

v = rand (1,3) ;
[i1, j1, x1] = find (v) ;
[i2, j2, x2] = find (GrB (v)) ;
assert (isequal (x1, x2)) ;
assert (isequal (i1, i2)) ;
assert (isequal (j1, j2)) ;

[i2, j2] = find (GrB (v)) ;
assert (isequal (i1, i2)) ;
assert (isequal (j1, j2)) ;

j1 = find (v) ;
j2 = find (GrB (v)) ;
assert (isequal (j1, j2)) ;

G = GrB.prune (GrB (A, 'by row')) ;
[i1, j1, x1] = find (A, 4) ;
[i2, j2, x2] = find (G, 4) ;
assert (isequal (x1, x2)) ;
assert (isequal (i1, i2)) ;
assert (isequal (j1, j2)) ;

[i1, j1, x1] = find (A, 4, 'last') ;
[i2, j2, x2] = find (G, 4, 'last') ;
assert (isequal (x1, x2)) ;
assert (isequal (i1, i2)) ;
assert (isequal (j1, j2)) ;

fprintf ('\ngbtest13: all tests passed\n') ;

