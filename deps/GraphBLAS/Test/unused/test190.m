function test190
%TEST190 test dense matrix for C<!M>=A*B

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test190 ----------- dense matrix for C<!M>=A*B\n') ;

rng ('default') ;
n = 1000 ;
M.matrix = eye (n) ;
M.sparsity = 2 ;
desc.mask = 'structural complement' ;
A = sprand (n, n, 0.001) ;
B = sprand (n, n, 0.001) ;
C = sparse (n, n) ;

semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'double' ;

C1 = GB_mex_mxm (C, M, [ ], semiring, A, B, desc) ;

assert (nnz (C1.matrix) == 0) ;

fprintf ('test190 all tests passed\n') ;
