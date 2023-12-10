function C = uint32 (G)
%UINT32 cast a GraphBLAS matrix to built-in full uint32 matrix.
% C = uint32 (G) typecasts the GrB matrix G to a built-in full uint32
% matrix.  The result C is full since sparse uint32 matrices are not
% built-in.
%
% To typecast the matrix G to a GraphBLAS sparse uint32 matrix instead,
% use C = GrB (G, 'uint32').
%
% See also GrB, GrB/double, GrB/complex, GrB/single, GrB/logical,
% GrB/int8, GrB/int16, GrB/int32, GrB/int64, GrB/uint8, GrB/uint16,
% GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
desc.kind = 'full' ;
C = gbfull (G, 'uint32', uint32 (0), desc) ;    % export as a full matrix

