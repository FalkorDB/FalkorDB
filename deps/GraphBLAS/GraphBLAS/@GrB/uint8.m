function C = uint8 (G)
%UINT8 cast a GraphBLAS matrix to built-in full uint8 matrix.
% C = uint8 (G) typecasts the GrB matrix G to a built-in full uint8 matrix.
% The result C is full since sparse uint8 matrices are not built-in.
%
% To typecast the matrix G to a GraphBLAS sparse uint8 matrix instead, use
% C = GrB (G, 'uint8'); use C = GhB (G, 'uint8') for a GhB matrix.
%
% See also GrB, GrB/double, GrB/complex, GrB/single, GrB/logical, GrB/int8,
% GrB/int16, GrB/int32, GrB/int64, GrB/uint16, GrB/uint32, GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cast_full (G, 'uint8', uint8 (0)) ;

