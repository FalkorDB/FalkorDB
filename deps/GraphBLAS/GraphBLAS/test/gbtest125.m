function gbtest125
%GBTEST125 test GrB.monoids

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

list = GrB.monoids
GrB.monoids ;
help GrB.monoids ;
fprintf ('\ngbtest125: all tests passed\n') ;
