function test165
%TEST165 test C=A*B' where A is diagonal and B becomes bitmap

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 10 ;
D.matrix = sparse (1:n, 1:n, rand (n,1)) ;
D.pattern = logical (speye (n)) ;
D.class = 'double' ;

d = 0.5 ;
B = GB_spec_random (n, n, d, 1, 'double') ;
Cin = sparse (n, n) ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;
dnt = struct ('inp1', 'tran') ;

C1 = D.matrix*B.matrix' ;
C2 = GB_mex_mxm (Cin, [ ], [ ], semiring, D, B, dnt) ;
GB_spec_compare (C1, C2) ;

fprintf ('test165: all tests passed\n') ;

