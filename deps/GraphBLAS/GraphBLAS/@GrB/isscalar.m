function s = isscalar (G)
%ISSCALAR determine if a matrix is a scalar.
% isscalar (G) is true for an m-by-n GraphBLAS matrix if m and n are 1.
%
% See also GrB/issparse, GrB/ismatrix, GrB/isvector, GrB/issparse,
% GrB/isfull, GrB/isa, GrB, GrB/size.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
s = gb_isscalar (G) ;
