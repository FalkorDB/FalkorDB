function gbtest54
%GBTEST54 test GrB.compact

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
n = 32 ;
H = GrB (n,n) ;
I = sort (randperm (n, 4)) ;
J = sort (randperm (n, 4)) ;
A = magic (4) ;
H (I,J) = A ;
[C, I, J] = GrB.compact (H) ; %#ok<*ASGLU>
H (I, J(1)) = 0 ;
[C, I, J] = GrB.compact (H, 0) ;
assert (isequal (C, A (:,2:end))) ;

A = sprand (n, n, 0.02) ;
[C, I, J] = GrB.compact (A, [ ], 'symmetric') ;
assert (isequal (I, J)) ;
C2 = A (I, I) ;
assert (isequal (C, C2)) ;

[C, I, J] = GrB.compact (A, [ ]) ;
assert (~isequal (I, J)) ;
C2 = A (I, J) ;
assert (isequal (C, C2)) ;

A = ones (4) ;
A (1,1) = 2 ;
G = GrB.compact (A, 2) ;
assert (nnz (G) == 15) ;
A = ones (4) ;
A (1,1) = 0 ;
A = sparse (A) ;
assert (isequal (G, A)) ;

A = sprand (n, n/2, 0.5) ;
try
    [C, I, J] = GrB.compact (A, [ ], 'symmetric') ;
    ok = 0 ;
catch expected_error
    expected_error
    ok = 1 ;
end
assert (ok) ;

fprintf ('gbtest54: all tests passed\n') ;

