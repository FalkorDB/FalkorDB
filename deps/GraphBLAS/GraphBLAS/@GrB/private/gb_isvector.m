function s = gb_isvector (G)
%GB_ISVECTOR determine if the GraphBLAS matrix is a row or column vector,
% where G is the opaque struct of the GraphBLAS matrix.
% gb_isvector (G) is true for an m-by-n GraphBLAS matrix if m or n is 1.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = gbsize (G) ;
s = (m == 1) || (n == 1) ;

