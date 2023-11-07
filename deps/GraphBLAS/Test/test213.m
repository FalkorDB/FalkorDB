function test213
%TEST213 test iso assign (method 05d)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

GB_mex_burble (1) ;

n = 10 ;
Cin.matrix = pi * sparse (ones (n,n)) ;
Cin.class = 'double' ;
Cin.iso = true ;

M = logical (sprand (n, n, 0.5)) ;

scalar.matrix = sparse (pi) ;
scalar.class = 'double' ;

C1 = GB_mex_assign  (Cin, M, [ ], scalar, [ ], [ ], [ ]) ;
C2 = GB_spec_assign (Cin, M, [ ], scalar, [ ], [ ], [ ], true) ;
GB_spec_compare (C1, C2) ;

GB_mex_burble (0) ;
fprintf ('\ntest213: all tests passed\n') ;

