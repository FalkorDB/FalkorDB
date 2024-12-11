function test283
%TEST283 test index binary op

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n--- testing ewise with user-defined index binary op\n') ;
rng ('default') ;

GB_mex_test37

fprintf ('\ntest283: all tests passed\n') ;

