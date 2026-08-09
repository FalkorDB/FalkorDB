function C = asin (G)
%ASIN inverse sine.
% C = asin (G) is the inverse sine of each entry of G.  C is complex if
% any (abs(G) > 1).
%
% See also GrB/sin, GrB/sinh, GrB/asinh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_trig (0, 'asin', G) ;

