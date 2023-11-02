function gbtest110
%GBTEST110 test argmax

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

A = magic (5) ;
G = GrB (A) ;

[x1, i1] = max (A, [ ], 1) ;
[x2, i2] = GrB.argmax (G, 1) ;
assert (isequal (x1, x2')) ;
assert (isequal (i1, double (i2'))) ;

[x1, i1] = max (A, [ ], 2) ;
[x2, i2] = GrB.argmax (G, 2) ;
assert (isequal (x1, x2)) ;
assert (isequal (i1, double (i2))) ;

[x1, i1] = max (A (:)) ;
[x2, i2] = GrB.argmax (G, 0) ;
assert (isequal (x1, x2)) ;
s = double (size (G)) ;
i = double (i2 (1)) ;
j = double (i2 (2)) ;
assert (isequal (i1, sub2ind (s, i, j))) ;

[x2, i2] = GrB.argmax (G) ;
assert (isequal (x1, x2)) ;
s = double (size (G)) ;
i = double (i2 (1)) ;
j = double (i2 (2)) ;
assert (isequal (i1, sub2ind (s, i, j))) ;

% max and GrB.argmax differ since A has an empty row and column
A (:,1) = 0 ;
A (2,:) = 0 ;
G = GrB.prune (A) ;
[x1,p1] = max (A, [ ], 2) ;
[x2,p2] = GrB.argmax (G, 2) ;
assert (isequal (GrB.prune (x1), GrB.prune (x2))) ;
p1 (2) = 0 ;
assert (isequal (p1, double (p2))) ;

[x1, p1] = max (A, [ ], 1) ;
[x2, p2] = GrB.argmax (G, 1) ;
assert (isequal (GrB.prune (x1), GrB.prune (x2'))) ;
p1 (1) = 0 ;
assert (isequal (p1, double (p2'))) ;

fprintf ('\ngbtest110: all tests passed\n') ;

