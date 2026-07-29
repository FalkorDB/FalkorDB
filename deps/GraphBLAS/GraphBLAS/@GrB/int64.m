function C = int64 (G)
%INT64 cast a GraphBLAS matrix to built-in full int64 matrix.
% C = int64 (G) typecasts the GrB matrix G to a full int64 matrix.  The
% result C is full since sparse int64 matrices are not built-in.
%
% To typecast the matrix G to a GraphBLAS sparse int64 matrix instead,
% C = GrB (G, 'int64'); use C = GhB (G, 'int64') for a GhB matrix.
%
% See also GrB, GrB/double, GrB/complex, GrB/single, GrB/logical, GrB/int8,
% GrB/int16, GrB/int32, GrB/uint8, GrB/uint16, GrB/uint32, GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cast_full (G, 'int64', int64 (0)) ;

