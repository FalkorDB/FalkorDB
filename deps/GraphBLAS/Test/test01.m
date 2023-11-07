function test01
%TEST01 test GraphBLAS error handling

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

GB_mex_test2 ;
GB_mex_test1 ;
GB_mex_test0 ;
GB_mex_test3 ;
GB_mex_test4 ;
GB_mex_test5 ;
GB_mex_test6 ;
GB_mex_test7 ;
GB_mex_test8 ;
GB_mex_test10 ;
if (~ispc)
    GB_mex_test9 ;
end

fprintf ('\ntest01: all tests passed\n') ;

