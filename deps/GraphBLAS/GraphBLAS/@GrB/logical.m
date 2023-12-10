function C = logical (G)
%LOGICAL typecast a GraphBLAS matrix to built-in logical matrix.
% C = logical (G) typecasts the GraphBLAS matrix G to into a built-in
% logical matrix.  C is full if all entries in G are present, and
% sparse otherwise.
%
% To typecast the matrix G to a GraphBLAS logical matrix instead,
% use C = GrB (G, 'logical').
%
% See also cast, GrB, GrB/double, GrB/complex, GrB/single, GrB/int8,
% GrB/int16, GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32,
% GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
C = gbbuiltin (G, 'logical') ;

