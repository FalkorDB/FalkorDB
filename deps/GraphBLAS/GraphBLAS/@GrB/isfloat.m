function s = isfloat (G)
%ISFLOAT true for floating-point matrices.
% isfloat (G) is true if the matrix G has a type of 'double', 'single', 'single
% complex', or 'double complex'.
%
% See also GrB/isnumeric, GrB/isreal, GrB/isinteger, GrB/islogical, GrB.type,
% GrB/isa, GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = gb_isfloat (gb_type (G)) ;

