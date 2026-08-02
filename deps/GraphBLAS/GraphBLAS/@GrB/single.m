function C = single (G)
%SINGLE cast a GraphBLAS matrix to built-in full single matrix.
% C = single (G) typecasts the GrB or GhB matrix G to a built-in full single
% matrix.  The result C is full since sparse single matrices are not built-in.
% C is real if G is real, and complex if G is complex.
%
% To typecast the matrix G to a GraphBLAS sparse single matrix instead, use
% C = GrB (G, 'single'); use C = GhB (G, 'single') for a GhB matrix.  To
% typecast to a sparse single complex matrix, use
% G = GrB (G, 'single complex'); or G = GhB (G, 'single complex') for a GhB
% matrix.
%
% See also GrB, GrB/double, GrB/complex, GrB/logical, GrB/int8, GrB/int16,
% GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32, GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_single (G) ;

