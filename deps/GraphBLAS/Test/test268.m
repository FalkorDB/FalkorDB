function test268
%TEST268 test sparse masker, C<M>=Z

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

% test Method04e, Factories/GB_sparse_masker_template.c
% when M(:,j) is much denser than Z(:,j)

n = 1000 ;
C = sprand (n, n, 0.01) ;
M = logical (sprand (n, n, 0.01)) ;
M (600:1000, 1:10) = true ;
A = sprand (n, n, 0.01) ;
B = sprand (n, n, 0.01) ;
op.opname = 'plus' ;
op.optype = 'double' ;

GB_mex_burble (1) ;
C1 = GB_mex_Matrix_eWiseAdd (C, M, [ ], op, A, B, [ ]) ;
Z = A+B ;
C2 = C ;
C2 (M) = Z (M) ;
err = norm (C1.matrix - C2, 1) ;
assert (err < 1e-12) ;
GB_mex_burble (0) ;

fprintf ('test268: all tests passed.\n') ;


