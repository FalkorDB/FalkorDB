function C = int32 (G)
%INT32 cast a GraphBLAS matrix to built-in full int32 matrix.
% C = int32 (G) typecasts the GrB matrix G to a built-in full int32 matrix.
% The result C is full since sparse int32 matrices are not built-in.
%
% To typecast the matrix G to a GraphBLAS sparse int32 matrix instead,
% use C = GrB (G, 'int32').
%
% See also GrB, GrB/double, GrB/complex, GrB/single, GrB/logical, GrB/int8,
% GrB/int16, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32, GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
desc.kind = 'full' ;
C = gbfull (G, 'int32', int32 (0), desc) ;      % export as a full matrix

