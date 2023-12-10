function test149
%TEST149 test fine hash method for C<!M>=A*B

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test149: --------- fine hash method for C<!M>=A*B\n') ;

rng ('default') ;

nthreads_set (4, 1) ;

desc.axb = 'hash' ;
desc.mask = 'complement' ;

n = 1000 ;
m = 1e8 ;
A = sparse (m, n) ;
A (1:n, 1:n) = rand (n) ;
B = sparse (rand (n,1)) ;
C = sparse (m, 1) ;

M = logical (sparse (m, 1)) ;
M (1:n, 1) = sparse (rand (n,1) > 0.5) ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

tic
C1 = GB_mex_mxm (C, M, [ ], semiring, A, B, desc) ;
toc
tic
C2 = (A*B) .* double (~M) ;
toc

cnorm = norm (C2,1) ;
assert (norm (C1.matrix - C2, 1) / cnorm < 1e-12)

fprintf ('test149: all tests passed\n') ;

