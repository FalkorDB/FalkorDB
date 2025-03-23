function gbtest128
%GBTEST128 test GrB.unops

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

help GrB.unops ;
list = GrB.unops
GrB.unops ;

fprintf ('\ngbtest128: all tests passed\n') ;
