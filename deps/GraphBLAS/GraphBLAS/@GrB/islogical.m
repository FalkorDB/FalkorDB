function s = islogical (G)
%ISLOGICAL true for logical matrices.
% islogical (G) is true if the matrix G has the logical type.
%
% See also GrB/isnumeric, GrB/isfloat, GrB/isreal, GrB/isinteger,
% GrB.type, GrB/isa, GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
s = isequal (gbtype (G), 'logical') ;

