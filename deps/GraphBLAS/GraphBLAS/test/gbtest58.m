function gbtest58
%GBTEST58 test uplus

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

A = 1 - 2 * rand (3) ;
G = GrB (A) ;
G = +G ;
A = +A ;

assert (isequal (A, G)) ;

fprintf ('gbtest58: all tests passed\n') ;

