function s = isempty (G)
%ISEMPTY true for an empty matrix.
% isempty (G) is true if any dimension of G is zero.
%
% See also GrB/size.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = size (G) ;
s = (m == 0) | (n == 0) ;

