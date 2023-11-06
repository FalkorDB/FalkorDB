function s = isvector (G)
%ISVECTOR determine if a matrix is a row or column vector.
% isvector (G) is true for an m-by-n matrix G if m or n is 1.
%
% See also GrB/issparse, GrB/ismatrix, GrB/isscalar, GrB/issparse,
% GrB/isfull, GrB/isa, GrB, GrB/size.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
s = gb_isvector (G) ;

