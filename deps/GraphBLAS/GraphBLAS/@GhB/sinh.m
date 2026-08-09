function C = sinh (G)
%SINH hyperbolic sine.
% C = sinh (G) is the hyperbolic sine of each entry of G.
%
% See also GhB/sin, GhB/asin, GhB/asinh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_sinh (1, G) ;

