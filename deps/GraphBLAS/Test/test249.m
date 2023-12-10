function test249
%TEST249 GxB_Context object tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

nthreads_set (2 * maxNumCompThreads) ;
GB_mex_context_test ;
fprintf ('test249: all tests pass\n') ;

