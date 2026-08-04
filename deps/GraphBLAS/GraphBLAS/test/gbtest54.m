function gbtest54 (ghb)
%GBTEST54 test [GrB,GhB].compact

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

n = 32 ;
H = gtb (ghb, n,n) ;
I = sort (randperm (n, 4)) ;
J = sort (randperm (n, 4)) ;
A = magic (4) ;
H (I,J) = A ;
[C, I, J] = gtb_compact (ghb, H) ; %#ok<*ASGLU>
H (I, J(1)) = 0 ;
[C, I, J] = gtb_compact (ghb, H, 0) ;
assert (isequal (C, A (:,2:end))) ;

A = sprand (n, n, 0.02) ;
[C, I, J] = gtb_compact (ghb, A, [ ], 'symmetric') ;
assert (isequal (I, J)) ;
C2 = A (I, I) ;
assert (isequal (C, C2)) ;

[C, I, J] = gtb_compact (ghb, A, 0, 'symmetric') ;
assert (isequal (I, J)) ;
C2 = A (I, I) ;
assert (isequal (C, C2)) ;

[C, I, J] = gtb_compact (ghb, A, [ ]) ;
assert (~isequal (I, J)) ;
C2 = A (I, J) ;
assert (isequal (C, C2)) ;

A = ones (4) ;
A (1,1) = 2 ;
G = gtb_compact (ghb, A, 2) ;
assert (nnz (G) == 15) ;
A = ones (4) ;
A (1,1) = 0 ;
A = sparse (A) ;
assert (isequal (G, A)) ;

A = sprand (n, n/2, 0.5) ;
try
    [C, I, J] = gtb_compact (ghb, A, [ ], 'symmetric') ;
    ok = 0 ;
catch expected_error
    fprintf ('expected: %s\n', expected_error.message) ;
    ok = 1 ;
end
assert (ok) ;

fprintf ('gbtest54 (%d): all tests passed\n', ghb) ;

