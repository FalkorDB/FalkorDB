function test203
%TEST203 test iso subref

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
% GB_mex_burble (1) ;
n = 10 ;
A.matrix = pi * spones (sprandn (n, n, 0.5)) ;
A.class = 'double' ;
A.iso = true ;

m = 5 ;
I = 1:m ;
I0 = uint64 (I) - 1 ;

Cin = sparse (m,m) ;
C1 = GB_mex_Matrix_extract (Cin, [ ], [ ], A, I0, I0, [ ]) ;
C2 = A.matrix (I,I) ;
assert (isequal (C1.matrix, C2)) ;

A.sparsity = 4 ;
C1 = GB_mex_Matrix_extract (Cin, [ ], [ ], A, I0, I0, [ ]) ;
C2 = A.matrix (I,I) ;
assert (isequal (C1.matrix, C2)) ;

A.matrix = pi * sparse (ones (n,n)) ;
A.sparsity = 8 ;
C1 = GB_mex_Matrix_extract (Cin, [ ], [ ], A, I0, I0, [ ]) ;
C2 = A.matrix (I,I) ;
assert (isequal (C1.matrix, C2)) ;

GB_mex_burble (0) ;
fprintf ('test203: all tests passed\n') ;

