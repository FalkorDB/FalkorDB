function gbtest127
%GBTEST127 test GrB.semirings

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

help GrB.semirings ;
list = GrB.semirings
GrB.semirings ;

fprintf ('\ngbtest127: all tests passed\n') ;
