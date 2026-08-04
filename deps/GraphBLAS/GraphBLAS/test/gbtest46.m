function gbtest46 (ghb)
%GBTEST46 test GrB.subassign and GrB.assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

d.kind = 'sparse' ;
d0.kind = 'sparse' ;
d0.base = 'zero-based' ;

types = gbtest_types ;
for k = 1:length (types)
    type = types {k} ;
    A = gbtest_cast (rand (4) * 100, type) ;
    C = gtb_subassign (ghb, A, {1}, {1}, gbtest_cast (pi, type)) ;
    A (1,1) = pi ;
    assert (gbtest_err (A, C) == 0) ;
end

for trial = 1:40

    A = rand (4) ;
    G = gtb (ghb, A) ;
    pg = gtb (ghb, pi) ;

    C1 = A ;
    C1 (1:3,1:2) = pi ;

    C2 = gtb_subassign (ghb, A, pi, { 1:3}, { 1:2 }) ;
    C3 = gtb_subassign (ghb, G, pi, { 1:3}, { 1:2 }) ;
    C4 = gtb_subassign (ghb, G, pg, { 1:3}, { 1:2 }) ;
    C5 = gtb_subassign (ghb, G, pg, { 1:3}, { 1:2 }, d) ;
    assert (isequal (C1, C2)) ;
    assert (isequal (C1, C3)) ;
    assert (isequal (C1, C4)) ;
    assert (isequal (C1, C5)) ;
    assert (isequal (class (C5), 'double')) ;

    C2 = gtb_assign (ghb, A, pi, { 1:3}, { 1:2 }) ;
    C3 = gtb_assign (ghb, G, pi, { 1:3}, { 1:2 }) ;
    C4 = gtb_assign (ghb, G, pg, { 1:3}, { 1:2 }) ;
    C5 = gtb_assign (ghb, G, pg, { 1:3}, { 1:2 }, d) ;
    C6 = gtb_assign (ghb, G, pg, { int64(1:3)-1 }, { int64(0), int64(1) }, d0) ;
    C7 = gtb_assign (ghb, G, pg, { int64(0), int64(2) }, { int64(1:2)-1 }, d0) ;
    C8 = gtb_assign (ghb, G, pg, { int64(0), int64(1), int64(2) }, ...
        { int64(1:2)-1 }, d0) ;
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
    G = gtb (ghb, A) ;
    C8 = gtb_assign (ghb, G, x, { int64(4), int64(-1), int64(0) }, ...
        { int64(0) }, d0) ;
    assert (isequal (C1, C8)) ;

end

fprintf ('gbtest46 (%d): all tests passed\n', ghb) ;

