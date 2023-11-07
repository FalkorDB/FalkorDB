function gbtest101
%GBTEST101 test loading of v3 GraphBLAS objects

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

load gbtestv3 %#ok<LOAD>
whos

G
fprintf ('================== v3 sparse struct:\n') ;
G_struct = struct (G) %#ok<*NOPRT>
G2 = GrB (G, 'sparse') ;
fprintf ('================== v7_3 sparse struct:\n') ;
G2_struct = struct (G2)
assert (isequal (G, A)) ;
assert (isequal (G2, A)) ;

assert (isfield (G_struct, 'GraphBLAS')) ;
assert (isfield (G2_struct, 'GraphBLASv7_3')) ;
G3 = GrB (G) ;
G3_struct = struct (G3) ;
assert (isfield (G3_struct, 'GraphBLASv7_3')) ;
assert (isequal (G3, A)) ;

[m1, n1] = size (G) ;
[m2, n2] = size (A) ;
assert (m1 == m2) ;
assert (n1 == n2) ;

t1 = GrB.type (G) ;
t2 = GrB.type (A) ;
assert (isequal (t1, t2)) ;

[s1, f1] = GrB.format (G) ;
[s2, f2, iso] = GrB.format (G2) ;
assert (isequal (s1, s2)) ;
assert (isequal (f1, f2)) ;
iso

H2 = GrB (H, 'hyper') ;
fprintf ('================== v3 hypersparse struct:\n') ;
H_struct = struct (H)
fprintf ('================== v7_3 hypersparse struct:\n') ;
H2_struct = struct (H2)

assert (isfield (H_struct, 'GraphBLAS')) ;
assert (isfield (H2_struct, 'GraphBLASv7_3')) ;
H3 = GrB (H) ;
H3_struct = struct (H3) ;
assert (isfield (H3_struct, 'GraphBLASv7_3')) ;
assert (isequal (H3, H)) ;

H3 = GrB (n,n) ;
H3 (1:4, 1:4) = magic (4) ;
assert (isequal (H2, H)) ;
assert (isequal (H3, H)) ;

[s1, f1] = GrB.format (H) ;
[s2, f2] = GrB.format (H2) ;
assert (isequal (s1, s2)) ;
assert (isequal (f1, f2)) ;

t1 = GrB.type (H2) ;
t2 = GrB.type (H) ;
assert (isequal (t1, t2)) ;

R2 = GrB (R) ;
assert (isequal (R2, R)) ;
assert (isequal (R2, A')) ;

assert (isfield (struct (R), 'GraphBLAS')) ;
assert (isfield (struct (R2), 'GraphBLASv7_3')) ;

X2 = GrB (X) ;
assert (isequal (magic (4), X)) ;
assert (isequal (magic (4), X2)) ;

assert (isfield (struct (X), 'GraphBLAS')) ;
assert (isfield (struct (X2), 'GraphBLASv7_3')) ;

fprintf ('================== v3 dense struct (held in sparse format):\n') ;
X_struct = struct (X) %#ok<*NASGU>
fprintf ('================== v7_3 dense struct (no integers in struct):\n') ;
X2_struct = struct (X2)

fprintf ('gbtest101: all tests passed\n') ;

