function test218
%TEST218 test C=A+B, C and A are full, B is bitmap

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

GB_mex_burble (1) ;

n = 10 ;

A.matrix = sparse (rand (n,n)) ;
A.class = 'double' ;
A.sparsity = 8 ;

B.matrix = sprand (n,n, 0.5) ;
B.class = 'double' ;
B.sparsity = 4 ;

op.opname = 'plus' ;
op.optype = 'double' ;

Cin = sparse (n,n) ;

C1 = GB_mex_Matrix_eWiseAdd  (Cin, [ ], [ ], op, A, B, [ ]) ;
C2 = A.matrix + B.matrix ;
assert (isequal (C1.matrix, C2)) ;

GB_mex_burble (0) ;
fprintf ('\ntest218: all tests passed\n') ;

