function C = asinh (G)
%ASINH inverse hyperbolic sine.
% C = asinh (G) is the inverse hyberbolic sine of each entry G.
%
% See also GhB/sin, GhB/asin, GhB/sinh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_asinh (1, G) ;

