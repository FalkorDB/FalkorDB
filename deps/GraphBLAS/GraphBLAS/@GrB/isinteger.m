function s = isinteger (G)
%ISINTEGER true for integer matrices.
% isinteger (G) is true if the matrix G has an integer type
% (int8, int16, int32, int64, uint8, uint16, uint32, or uint64).
%
% See also GrB/isnumeric, GrB/isfloat, GrB/isreal, GrB/islogical,
% GrB.type, GrB/isa, GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
s = gb_contains (gbtype (G), 'int') ;

