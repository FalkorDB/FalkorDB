%T74 run test20 and test74

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all ; make
threads {1} = [4 1] ;
t = threads ;
logstat ('test20',t) ;  % quick test of GB_mex_mxm on a few semirings
logstat ('test74',t) ;  % test GrB_mxm on all semirings

