function test34
%TEST34 test GrB_eWiseAdd

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n----- quick test for GB_mex_Matrix_eWiseAdd\n') ;

A = sparse (rand (4,3)) ;
B = sparse (rand (4,3)) ;
Cin = sparse (rand (4,3)) ;

add = 'plus' ;
Mask = [ ] ;
accum = 'plus' ;

C0 = Cin + (A+B) ;

C = GB_mex_Matrix_eWiseAdd (Cin, Mask, accum, add, A, B, [ ]) ;
assert (isequal (C.matrix,  C0))

C3 = GB_spec_Matrix_eWiseAdd (Cin, Mask, accum, add, A, B, [ ]) ;
assert (isequal (C3.matrix,  C0))

try
    % compare with CSparse
    C4 = cs_add (Cin, cs_add (A, B)) ;
    ok = isequal (C4, C0) ;
catch
    % CSparse not available
    ok = true ;
end
assert (ok) ;

fprintf ('\ntest34: all tests passed\n') ;

