function test09
%TEST09 test GxB_subassign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n-----------duplicate I,J test of GB_mex_subassign\n') ;

I = [2 2 3 3] ; J = [2 2 4 4 ] ;
I0 = uint64 (I) ;
J0 = uint64 (J);

C = sparse (magic (5)) ;
A = sparse (77 * ones (4,4)) ;

C2 = GB_mex_subassign(C, [ ], [ ], A, I0, J0) ;

% check erroneous I and J

fprintf ('testing error handling, errors expected:\n') ;
A = sparse (1) ;
try
    K = uint64 (99) ;
    Crud = GB_mex_subassign (C, [], 'plus', A, K, K) ;
    ok = false ;
catch me
    me
    ok = true ;
end
assert (ok) ;

fprintf ('testing more:\n') ;
A = sparse (rand (2)) ;
try
    I = uint64 ([0 0]) ;
    K = uint64 ([99 100]) ;
    Crud = GB_mex_subassign (C, [], 'plus', A, I, K) ;
    ok = false ;
catch me
    me
    ok = true ;
end
assert (ok) ;

fprintf ('\nAll tests passed (errors expected)\n') ;
fprintf ('\ntest09: all tests passed\n') ;

