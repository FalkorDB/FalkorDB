function gbtest46
%GBTEST46 test GrB.subassign and GrB.assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
d.kind = 'sparse' ;
d0.kind = 'sparse' ;
d0.base = 'zero-based' ;

types = gbtest_types ;
for k = 1:length (types)
    type = types {k} ;
    A = gbtest_cast (rand (4) * 100, type) ;
    C = GrB.subassign (A, {1}, {1}, gbtest_cast (pi, type)) ;
    A (1,1) = pi ;
    assert (gbtest_err (A, C) == 0) ;
end

for trial = 1:40

    A = rand (4) ;
    G = GrB (A) ;
    pg = GrB (pi) ;

    C1 = A ;
    C1 (1:3,1) = pi ;

    C2 = GrB.subassign (A, pi, { 1:3}, { 1 }) ;
    C3 = GrB.subassign (G, pi, { 1:3}, { 1 }) ;
    C4 = GrB.subassign (G, pg, { 1:3}, { 1 }) ;
    C5 = GrB.subassign (G, pg, { 1:3}, { 1 }, d) ;
    assert (isequal (C1, C2)) ;
    assert (isequal (C1, C3)) ;
    assert (isequal (C1, C4)) ;
    assert (isequal (C1, C5)) ;
    assert (isequal (class (C5), 'double')) ;

    C2 = GrB.assign (A, pi, { 1:3}, { 1 }) ;
    C3 = GrB.assign (G, pi, { 1:3}, { 1 }) ;
    C4 = GrB.assign (G, pg, { 1:3}, { 1 }) ;
    C5 = GrB.assign (G, pg, { 1:3}, { 1 }, d) ;
    C6 = GrB.assign (G, pg, { int64(1:3)-1 }, { int64(0) }, d0) ;
    C7 = GrB.assign (G, pg, { int64(0), int64(2) }, { int64(0) }, d0) ;
    C8 = GrB.assign (G, pg, { int64(0), int64(1), int64(2) }, { int64(0) }, ...
        d0) ;
    assert (isequal (C1, C2)) ;
    assert (isequal (C1, C3)) ;
    assert (isequal (C1, C4)) ;
    assert (isequal (C1, C5)) ;
    assert (isequal (C1, C6)) ;
    assert (isequal (C1, C7)) ;
    assert (isequal (C1, C8)) ;
    assert (isequal (class (C5), 'double')) ;

    x = [ 1 2 3 4 5 ]' ;
    C1 = A ;
    C1 (5:-1:1,1) = x ;
    G = GrB (A) ;
    C8 = GrB.assign (G, x, { int64(4), int64(-1), int64(0) }, { int64(0) }, ...
        d0) ;
    assert (isequal (C1, C8)) ;

end

fprintf ('gbtest46: all tests passed\n') ;

