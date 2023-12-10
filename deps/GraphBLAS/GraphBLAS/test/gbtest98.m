function gbtest98
%GBTEST98 test row/col degree for hypersparse matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 2^12 ;
G = GrB (n, n) ;
I = randperm (n, 8) ;
G (I,I) = magic (8) ;

d = double (GrB.entries (G, 'row', 'degree')) ;
A = double (G) ;
d2 = sum (spones (A))' ;
assert (isequal (d, d2)) ;

G = GrB (G, 'by row') ;
d = double (GrB.entries (G, 'col', 'degree')) ;
assert (isequal (d, d2)) ;

G = G + GrB.eye (n) ;
A = double (G) ;
d2 = sum (spones (A))' ;
d = double (GrB.entries (G, 'col', 'degree')) ;
assert (isequal (d, d2)) ;

n = 2 * flintmax ;
G = GrB (n, n) ;
m = flintmax / 2 ;
I = sort (randperm (m, 8)) ;
A = magic (8) ;
G (I,I) = A ;
x1 = nonzeros (A) ;
x2 = nonzeros (G) ;
assert (isequal (x1, x2)) ;

[i1,j1,x1] = GrB.extracttuples (G) ;
[~ ,~ ,x2] = GrB.extracttuples (A) ;
assert (isequal (x1, x2)) ;

assert (isequal (class (i1), 'int64')) ;
assert (isequal (class (j1), 'int64')) ;

G = GrB.random (8, 8, 0.5) ;
A = double (G) ;
G = full (G, 'double', 1) ;
A (A == 0) = 1 ;
assert (isequal (A, G)) ;

fprintf ('\ngbtest98: all tests passed\n') ;
