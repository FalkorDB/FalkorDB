function test221
%TEST221 test C += A where C is bitmap and A is full

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

GB_mex_burble (1) ;
n = 10 ;
Cin.matrix = spones (sprand (n, n, 0.5)) ;
Cin.sparsity = 4 ;
Cin.iso = true ;

A.matrix = sparse (rand (n, n)) ;
A.sparsity = 8 ;

accum.opname = 'plus' ;
accum.optype = 'double' ;

C1 = GB_mex_assign  (Cin, [ ], accum, A, [ ], [ ], [ ]) ;
C2 = GB_spec_assign (Cin, [ ], accum, A, [ ], [ ], [ ], false) ;
GB_spec_compare (C1, C2) ;

GB_mex_burble (0) ;
fprintf ('\ntest221: all tests passed\n') ;



