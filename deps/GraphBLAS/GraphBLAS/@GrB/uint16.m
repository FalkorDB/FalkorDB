function C = uint16 (G)
%UINT16 cast a GraphBLAS matrix to built-in full uint16 matrix.
% C = uint16 (G) typecasts the GrB matrix G to a built-in full uint16
% matrix.  The result C is full since sparse uint16 matrices are not
% built-in.
%
% To typecast the matrix G to a GraphBLAS sparse uint16 matrix instead,
% use C = GrB (G, 'uint16').
%
% See also GrB, GrB/double, GrB/complex, GrB/single, GrB/logical,
% GrB/int8, GrB/int16, GrB/int32, GrB/int64, GrB/uint8, GrB/uint32,
% GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
desc.kind = 'full' ;
C = gbfull (G, 'uint16', uint16 (0), desc) ;    % export as a full matrix

