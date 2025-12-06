function gbtest126
%GBTEST126 test GrB.selectops

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

list = GrB.selectops
GrB.selectops ;
help GrB.selectops ;
fprintf ('\ngbtest126: all tests passed\n') ;
