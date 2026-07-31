function gbtest61 (ghb)
%GBTEST61 test GrB.laplacian

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

n = 10 ;
A = sprand (n, n, 0.4) ;

S = tril (A, -1) ;
S = S+S' ;
G = gtb (ghb, S) ;

L0 = laplacian (graph (S, 'OmitSelfLoops')) ;

% GrB.laplacian places explicit zeros on the diagonal
L1 = gtb_laplacian (ghb, S) ;
L2 = gtb_laplacian (ghb, G) ;
L3 = gtb_laplacian (ghb, G, 'double', 'check') ;
L4 = gtb_laplacian (ghb, gtb (ghb, G, 'by row')) ;

assert (norm (L0-L1,1) == 0) ;
assert (isequal (gtb_offdiag (ghb, L0), gtb_offdiag (ghb, L1))) ;
assert (isequal (L0, double (L1))) ;

assert (isequal (L1, L2)) ;
assert (isequal (L1, L3)) ;
assert (isequal (L1, L4)) ;

G = gtb (ghb, G, 'by row') ;

L2 = gtb_laplacian (ghb, G) ;
L3 = gtb_laplacian (ghb, G, 'double', 'check') ;

assert (norm (L0-L2,1) == 0) ;
assert (isequal (gtb_offdiag (ghb, L0), gtb_offdiag (ghb, L2))) ;
assert (isequal (L2, L3)) ;

types = { 'double', 'single', 'int8', 'int16', 'int32', 'int64' } ;
for k = 1:6
    type = types {k} ;
    L2 = gtb_laplacian (ghb, G, type) ;
    assert (isequal (gtb_type (ghb, L2), type)) ;
    assert (isequal (L0, double (L2))) ;
end

fprintf ('gbtest61 (%d): all tests passed\n', ghb) ;

