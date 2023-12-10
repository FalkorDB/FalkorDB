function s = ismatrix (G)    %#ok<INUSD>
%ISMATRIX always true for any GraphBLAS matrix.
% ismatrix (G) is always true for any GraphBLAS matrix G.
%
% See also GrB/issparse, GrB/isvector, GrB/isscalar, GrB/full, GrB/isa,
% GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = true ;

