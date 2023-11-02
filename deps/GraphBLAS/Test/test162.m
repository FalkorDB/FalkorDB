function test162
%TEST162 test C<M>=A*B with very sparse M

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

rng ('default') ;

d = 0.02 ;
n = 1000 ;
A = sprand (n, n, d) ;
A (1:257,1) = rand (257, 1) ;
B = sprand (n, n, d) ;
B (1,1) = 1 ;
M = logical (sprand (n, n, 0.002)) ;
Cin = sparse (n, n) ;

C1 = double (M) .* (A*B) ;
C2 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, [ ]) ;
GB_spec_compare (C1, C2) ;

clear A B M
n = 80 ;
A.matrix = sparse (rand (n)) ; A.sparsity = 4 ;    % A is bitmap
B.matrix = sparse (rand (n)) ; B.sparsity = 4 ;    % B is bitmap
M.matrix = logical (sprand (n, n, 0.5)) ; M.sparsity = 2 ;  % M is sparse
desc.mask = 'complement' ;

Cin = sparse (n, n) ;
C1 = double (~(M.matrix)) .* (A.matrix*B.matrix) ;
C2 = GB_mex_mxm  (Cin, M, [ ], semiring, A, B, desc) ;
GB_spec_compare (C1, C2) ;

M.sparsity = 4 ; % make M bitmap
M.matrix (1:64, 1:64) = 0 ; % clear the leading 64-by-64 tile
C1 = double (M.matrix) .* (A.matrix*B.matrix) ;
C2 = GB_mex_mxm  (Cin, M, [ ], semiring, A, B, [ ]) ;
GB_spec_compare (C1, C2) ;

fprintf ('test162: all tests passed\n') ;

