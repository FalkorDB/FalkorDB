function gbtest111 (ghb)
%GBTEST111 test argmin

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = magic (5) ;
G = gtb (ghb, A) ;

[x1, i1] = min (A, [ ], 1) ;
[x2, i2] = gtb_argmin (ghb, G, 1) ;
assert (isequal (x1, x2')) ;
assert (isequal (i1, double (i2'))) ;

[x1, i1] = min (A, [ ], 2) ;
[x2, i2] = gtb_argmin (ghb, G, 2) ;
assert (isequal (x1, x2)) ;
assert (isequal (i1, double (i2))) ;

[x1, i1] = min (A (:)) ;
[x2, i2] = gtb_argmin (ghb, G, 0) ;
assert (isequal (x1, x2)) ;
s = double (size (G)) ;
i = double (i2 (1)) ;
j = double (i2 (2)) ;
assert (isequal (i1, sub2ind (s, i, j))) ;

[x2, i2] = gtb_argmin (ghb, G) ;
assert (isequal (x1, x2)) ;
s = double (size (G)) ;
i = double (i2 (1)) ;
j = double (i2 (2)) ;
assert (isequal (i1, sub2ind (s, i, j))) ;

% min and [GrB,GhB].argmin differ since A has an empty row and column
A = -A ;
A (:,1) = 0 ;
A (2,:) = 0 ;
G = gtb_prune (ghb, A) ;
[x1,p1] = min (A, [ ], 2) ;
[x2,p2] = gtb_argmin (ghb, G, 2) ;
assert (isequal (gtb_prune (ghb, x1), gtb_prune (ghb, x2))) ;
p1 (2) = 0 ;
assert (isequal (p1, double (p2))) ;

[x1, p1] = min (A, [ ], 1) ;
[x2, p2] = gtb_argmin (ghb, G, 1) ;
assert (isequal (gtb_prune (ghb, x1), gtb_prune (ghb, x2'))) ;
p1 (1) = 0 ;
assert (isequal (p1, double (p2'))) ;

fprintf ('\ngbtest111 (%d): all tests passed\n', ghb) ;

