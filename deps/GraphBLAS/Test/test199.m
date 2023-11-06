function test199
%TEST199 test dot2 with hypersparse

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

GB_mex_burble (1) ;
n = 10 ;
A.matrix = sprand (n, n, 0.5) ;
A.matrix (:,3) = sparse (n,1) ;
A.sparsity = 1 ;    % hypersparse
B.matrix = sprand (n, n, 0.5) ;
B.matrix (:,8) = sparse (n,1) ;
B.sparsity = 1 ;    % hypersparse
% GB_mex_dump (A,3)
% GB_mex_dump (B,3)
M.matrix = spones (sprand (n, n, 0.5)) ;
M.sparsity = 4 ;    % bitmap
% GB_mex_dump (M,3)

dtn = struct ('inp0', 'tran', 'axb', 'dot') ;
semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

Cin = sparse (n,n) ;
C1 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, dtn) ;
C2 = (A.matrix' * B.matrix) .* M.matrix ;
err = norm (C1.matrix - C2, 1) ;
assert (err < 1e-12) ;

GB_mex_burble (0) ;
fprintf ('test199: all tests passed\n') ;

