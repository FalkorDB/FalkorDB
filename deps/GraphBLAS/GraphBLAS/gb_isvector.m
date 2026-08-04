function s = gb_isvector (G)
%GB_ISVECTOR determine if matrix is a row or column vector.  Not user-callable.
% gb_isvector (G) is true for an m-by-n GraphBLAS matrix if m or n is 1.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = gb_size (G) ;
s = (m == 1) || (n == 1) ;

