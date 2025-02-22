function gbtest124
%GBTEST124 test GrB.binops

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

help GrB.binops ;
list = GrB.binops
GrB.binops ;

fprintf ('\ngbtest124: all tests passed\n') ;
