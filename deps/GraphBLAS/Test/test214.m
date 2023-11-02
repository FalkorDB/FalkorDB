function test214
%TEST214 test C<M>=A'*B (tricount)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

GB_mex_burble (1) ;

n = 10 ;
Cin = sparse (n, n) ;

M = logical (sprand (n, n, 0.5)) ;
desc = struct ('mask', 'structural', 'inp0', 'tran') ;

A.matrix = sprandn (n, n, 0.5) ;
A.class  = 'double' ;

B.matrix = sprandn (n, n, 0.5) ;
B.class  = 'double' ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

C1 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, desc) ;
C2 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, desc) ;
GB_spec_compare (C1, C2) ;

semiring.multiply = 'pair' ;
C1 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, desc) ;
C2 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, desc) ;
GB_spec_compare (C1, C2) ;

semiring.multiply = 'oneb' ;
C1 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, desc) ;
C2 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, desc) ;
GB_spec_compare (C1, C2) ;

GB_mex_burble (0) ;
fprintf ('\ntest214: all tests passed\n') ;

