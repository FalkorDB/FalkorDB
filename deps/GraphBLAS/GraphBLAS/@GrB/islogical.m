function s = islogical (G)
%ISLOGICAL true for logical matrices.
% islogical (G) is true if the matrix G has the logical type.
%
% See also GrB/isnumeric, GrB/isfloat, GrB/isreal, GrB/isinteger, GrB.type,
% GrB/isa, GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = isequal (gb_type (G), 'logical') ;

