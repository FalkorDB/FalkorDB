function C = complex (A, B)
%COMPLEX cast to a built-in double complex matrix.
% C = complex (G) typecasts the GraphBLAS matrix G to into a built-in double
% complex matrix.  C is full if all entries in G are present, or sparse
% otherwse.
%
% With two inputs, C = complex (A,B) returns a matrix C = A + 1i*B, where A or
% B are real matrices.  If A or B are nonzero scalars and the other input is a
% matrix, or if both A and B are scalars, C is full.
%
% To typecast the matrix G to a GraphBLAS GrB double complex matrix instead,
% use C = GrB (G, 'complex') or C = GrB (G, 'double complex'); use GhB to
% create a GraphBLAS GhB matrix.  .  To typecast the matrix G to a GraphBLAS
% GrB single complex matrix, use C = GrB (G, 'single complex'); use GhB to
% create a GraphBLAS GhB matrix.
%
% To construct a complex GraphBLAS matrix from real GraphBLAS matrices A and B,
% use C = A + 1i*B instead.
%
% Since sparse single complex matrices are not built-in, C is always returned
% as a double complex matrix (sparse or full).
%
% See also cast, GrB, GrB/double, GrB/single, GrB/logical, GrB/int8, GrB/int16,
% GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32, GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    C = gb_complex (A) ;
else
    C = gb_complex (A, B) ;
end

