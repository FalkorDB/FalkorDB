function C = eps (G)
%EPS spacing of numbers in a GraphBLAS matrix.
% C = eps (G) returns the spacing of numbers in a floating-point GraphBLAS
% matrix.
%
% See also GrB/isfloat, realmax, realmin.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_eps (0, G) ;

