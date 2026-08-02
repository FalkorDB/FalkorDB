function gbtest7 (ghb)
%GBTEST7 test GrB.build and GhB.build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

n = 5 ;
A = sprand (n, n, 0.5) ;
A (n,n) = 5 ;

[i, j, x] = find (A) ;
[m, n] = size (A) ;

G = gtb_build (ghb, i, j, x, m, n) ;
S = sparse   (i, j, x, m, n) ;
assert (gbtest_eq (S, G)) ;

G = gtb_build (ghb, GrB (i), GrB (j), GrB (x), m, n) ;
assert (gbtest_eq (S, G)) ;

G = gtb_build (ghb, i, j, x, m, n, '') ;
assert (gbtest_eq (S, G)) ;

G = gtb_build (ghb, i, j, x, m, n, 'ignore') ;
assert (gbtest_eq (S, G)) ;

% add some duplicates
ii = [i ; i] ;
jj = [j ; j] ;
xx = [x ; 2*x] ;
ok = false ;
try
    % no duplicates are tolerated
    fprintf ('an error is expected here:\n') ;
    G = gtb_build (ghb, ii, jj, xx, m, n, '') ;
catch
    ok = true ;
%   fprintf ('OK: error was expected\n') ;
end
assert (ok) ;

% duplicates are ignored
G = gtb_build (ghb, ii, jj, xx, m, n, 'ignore') ;
assert (gbtest_eq (2*S, G)) ;

% duplicates are summed
G = gtb_build (ghb, ii, jj, xx, m, n) ;
assert (gbtest_eq (3*S, G)) ;

d.kind = 'GrB' ;
G = gtb_build (ghb, i, j, x, m, n, d) ;
assert (gbtest_eq (S, G)) ;

d.kind = 'sparse' ;
G = gtb_build (ghb, i, j, x, m, n, d) ;
assert (gbtest_eq (S, G))

I = gtb (ghb, i', 'by row') ;
J = gtb (ghb, j', 'by row') ;
X = gtb (ghb, x) ;
G = gtb_build (ghb, I, J, X, m, n, d) ;
assert (gbtest_eq (S, G))

i0 = int64 (i) - 1 ;
j0 = int64 (j) - 1 ;

G = gtb_build (ghb, i0, j0, x, struct ('base', 'zero-based')) ;
assert (gbtest_eq (S, G)) ;

G = gtb_build (ghb, 1:3, 1:3, [1 1 1]) ;
assert (gbtest_eq (speye (3), G)) ;

G = gtb_build (ghb, 1, 1, [1 2 3]) ;
assert (isequal (sparse (6), G)) ;

G = gtb_build (ghb, 1:3, 1:3, 1) ;
assert (isequal (speye (3), G)) ;

types = gbtest_types ;
for k = 1: length(types)
    type = types {k} ;
    X = gbtest_cast (1, type) ;
    G = gtb_build (ghb, 1:3, 1:3, X) ;
    S = gbtest_cast (eye (3, 3), type) ;
    assert (gbtest_eq (S, G)) ;
    assert (isequal (gtb_type (ghb, G), type)) ;

    % build an iso matrix
    G = gtb_build (ghb, 1:3, 1:3, X, 3, 3, '1st') ;
    assert (gbtest_eq (S, G)) ;
    assert (isequal (gtb_type (ghb, G), type)) ;
end

fprintf ('gbtest7 (%d): all tests passed\n', ghb) ;

