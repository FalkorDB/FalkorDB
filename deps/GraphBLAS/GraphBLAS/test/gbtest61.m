function gbtest61
%GBTEST61 test GrB.laplacian

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
n = 10 ;
A = sprand (n, n, 0.4) ;

S = tril (A, -1) ;
S = S+S' ;
G = GrB (S) ;

L0 = laplacian (graph (S, 'OmitSelfLoops')) ;

% GrB.laplacian places explicit zeros on the diagonal
L1 = GrB.laplacian (S) ;
L2 = GrB.laplacian (G) ;
L3 = GrB.laplacian (G, 'double', 'check') ;

assert (norm (L0-L1,1) == 0) ;
assert (isequal (GrB.offdiag (L0), GrB.offdiag (L1))) ;
assert (isequal (L0, double (L1))) ;

assert (isequal (L1, L2)) ;
assert (isequal (L1, L3)) ;

G = GrB (G, 'by row') ;

L2 = GrB.laplacian (G) ;
L3 = GrB.laplacian (G, 'double', 'check') ;

assert (norm (L0-L2,1) == 0) ;
assert (isequal (GrB.offdiag (L0), GrB.offdiag (L2))) ;
assert (isequal (L2, L3)) ;

types = { 'double', 'single', 'int8', 'int16', 'int32', 'int64' } ;
for k = 1:6
    type = types {k} ;
    L2 = GrB.laplacian (G, type) ;
    assert (isequal (GrB.type (L2), type)) ;
    assert (isequal (L0, double (L2))) ;
end


fprintf ('gbtest61: all tests passed\n') ;

