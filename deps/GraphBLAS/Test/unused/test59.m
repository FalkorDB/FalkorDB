function test59
%TEST59 test GrB_mxm

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n----- quick test for GB_mex_mxm\n') ;

A = sparse (rand (4,3)) ;
B = sparse (rand (3,5)) ;
Cin = sparse (rand (4,5)) ;

semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'double' ;
accum = 'plus' ;
Mask = [ ] ;
accum = 'plus' ;

GB_mex_semiring (semiring)

C0 = Cin + A*B ;

C = GB_mex_mxm (Cin, Mask, accum, semiring, A, B, [ ]) ;
assert (isequal (C.matrix,  C0))

C3 = GB_spec_mxm (Cin, Mask, accum, semiring, A, B, [ ]) ;
assert (isequal (C3.matrix,  C0))

fprintf ('\ntest59: all tests passed\n') ;

