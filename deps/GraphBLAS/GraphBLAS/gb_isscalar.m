function s = gb_isscalar (G)
%GB_ISSCALAR determine if the matrix is a scalar.  Not user-callable.
% isscalar (G) is true for an m-by-n GraphBLAS matrix if m and n are 1.
% G is an opaque GraphBLAS struct or a built-in matrix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = gb_size (G) ;
s = (m == 1) && (n == 1) ;

