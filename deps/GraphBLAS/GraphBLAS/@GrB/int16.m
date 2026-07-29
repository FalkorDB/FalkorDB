function C = int16 (G)
%INT16 cast a GraphBLAS matrix to built-in full int16 matrix.
% C = int16 (G) typecasts the GrB matrix G to a built-in full int16 matrix.
% The result C is full since sparse int16 matrices are not built-in.
%
% To typecast the matrix G to a GraphBLAS GrB int16 matrix instead, use
% C = GrB (G, 'int16'); use C = GhB (G, 'int16') for a GhB matrix.
%
% See also GrB, GrB/double, GrB/complex, GrB/single, GrB/logical, GrB/int8,
% GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32, GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cast_full (G, 'int16', int16 (0)) ;

