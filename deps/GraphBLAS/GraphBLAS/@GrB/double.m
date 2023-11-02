function C = double (G)
%DOUBLE cast a GraphBLAS matrix to a built-in double matrix.
% C = double (G) typecasts the GraphBLAS matrix G into a built-in
% double matrix C, either real or complex.  C is full if all
% entries in G are present, and sparse otherwise.
%
% To typecast the matrix G to a GraphBLAS double (real) matrix
% instead, use C = GrB (G, 'double').  Explicit zeros are kept in C.
%
% See also GrB/cast, GrB, GrB/complex, GrB/single, GrB/logical, GrB/int8,
% GrB/int16, GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32,
% GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;

if (gb_contains (gbtype (G), 'complex'))
    C = gbbuiltin (G, 'double complex') ;
else
    C = gbbuiltin (G, 'double') ;
end

