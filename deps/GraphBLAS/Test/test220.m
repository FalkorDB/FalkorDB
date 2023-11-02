function test220
%TEST220 test mask C<M>=Z, iso case

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

GB_mex_burble (0) ;
n = 100 ;
Cin.matrix = spones (sprand (n, n, 0.01)) ;
Cin.sparsity = 1 ;
Cin.iso = true ;

m = 10 ;
A.matrix = spones (sprand (m, m, 0.5)) ;
A.iso = true ;

B.matrix = spones (sprand (m, m, 0.5)) ;
B.iso = true ;

op.opname = 'times' ;
op.optype = 'double' ;

accum.opname = 'second' ;
accum.optype = 'double' ;

M = logical (sprand (n, n, 0.5)) ;

C1 = GB_mex_kron  (Cin, M, accum, op, A, B, [ ]) ;
C2 = GB_spec_kron (Cin, M, accum, op, A, B, [ ]) ;
GB_spec_compare (C1, C2) ;
assert (C1.iso) ;

GB_mex_burble (0) ;
fprintf ('\ntest220: all tests passed\n') ;



