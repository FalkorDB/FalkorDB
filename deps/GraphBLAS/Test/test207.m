function test207
%TEST207 test iso subref

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

% GB_mex_burble (1) ;
n = 1000 ;
A.matrix = spones (sprand (n, 1, 0.5)) ;
A.iso = true ;
m = 500 ;
I = randperm (n, m) ;
I0 = uint64 (I) - 1 ;
J0 = uint64 (0) ;

Cin = sparse (m, 1) ;
C1 = A.matrix (I, 1) ;
C2 = GB_mex_Matrix_extract (Cin, [ ], [ ], A, I0, J0, [ ]) ;
assert (isequal (C2.matrix, C1)) ;

GB_mex_burble (0) ;
fprintf ('\ntest207: all tests passed\n') ;

