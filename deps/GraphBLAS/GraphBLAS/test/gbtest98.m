function gbtest98 (ghb)
%GBTEST98 test row/col degree for hypersparse matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

n = 2^12 ;
G = gtb (ghb, n, n) ;
I = randperm (n, 8) ;
G (I,I) = magic (8) ;

d = double (gtb_entries (ghb, G, 'row', 'degree')) ;
A = double (G) ;
d2 = sum (spones (A))' ;
assert (isequal (d, d2)) ;

d3 = double (gzb_degree (ghb, G, 'row')) ;
assert (isequal (d, d3)) ;

G = gtb (ghb, G, 'by row') ;
d = double (gtb_entries (ghb, G, 'col', 'degree')) ;
assert (isequal (d, d2)) ;

d3 = double (gzb_degree (ghb, G, 'col')) ;
assert (isequal (d, d3)) ;

G = G + gtb_eye (ghb, n) ;
A = double (G) ;
d2 = sum (spones (A))' ;
d = double (gtb_entries (ghb, G, 'col', 'degree')) ;
assert (isequal (d, d2)) ;

n = 2 * flintmax ;
G = gtb (ghb, n, n) ;
m = flintmax / 2 ;
I = sort (randperm (m, 8)) ;
A = magic (8) ;
G (I,I) = A ;
x1 = nonzeros (A) ;
x2 = nonzeros (G) ;
assert (isequal (x1, x2)) ;

[i1,j1,x1] = gtb_extracttuples (ghb, G) ;
[~ ,~ ,x2] = gtb_extracttuples (ghb, A) ;
assert (isequal (x1, x2)) ;

assert (isequal (class (i1), 'int32') || isequal (class (i1), 'int64')) ;
assert (isequal (class (j1), 'int32') || isequal (class (j1), 'int64')) ;

G = gtb_random (ghb, 8, 8, 0.5) ;
A = double (G) ;
G = full (G, 'double', 1) ;
A (A == 0) = 1 ;
assert (isequal (A, G)) ;

fprintf ('\ngbtest98 (%d): all tests passed\n', ghb) ;

