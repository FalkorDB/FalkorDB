function test310
%TEST310 test GxB_Matrix_reshape

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% This should be tested with 0:3 pji controls in testall.m, even though it
% is no longer requried for test coverage.  The pji controls trigger a bug in
% v10.5.1 which was fixed in v10.5.2.

GB_mex_test52 ;
fprintf ('test310: reshape tests passed\n') ;

