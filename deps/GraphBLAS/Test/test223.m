function test223
%TEST223 test matrix multiply, C<!M>=A*B

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% tests the Coarse Gustavson method for C<!M>=A*B, for the case when
% C(:,j) is very sparse compared with the # of rows of C.

rng ('default') ;

GB_mex_burble (1) ;
n = 100 ;
m = 10000 ;

clear desc
desc.axb = 'gustavson' ;
desc.mask = 'complement' ;

[save1, save2] = nthreads_get ;
nthreads_set (2,1) ;

A = sprand (n, n, 0.9) ;
B = sprand (n, n, 0.9) ;
A (m,m) = 1 ;
B (m,m) = 1 ;
M = sparse (m,m) ;
M (1,1) = 1 ;

Ain.matrix = A ;
Ain.sparsity = 2 ;  % sparse

Bin.matrix = B ;
Bin.sparsity = 2 ;  % sparse

Cin = sparse (m,m) ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

% C<!M> = A*B
C1 = GB_mex_mxm (Cin, M, [ ], semiring, Ain, Bin, desc) ;
C1.matrix (1,1) = 0 ;
C2 = A*B ;
C2 (1,1) = 0 ;
assert (isequal (C1.matrix, C2)) ;

GB_mex_burble (0) ;
nthreads_set (save1, save2) ;
fprintf ('\ntest223: all tests passed\n') ;

