function C = acoth (G)
%ACOTH inverse hyperbolic cotangent.
% C = acoth (G) is the inverse hyberbolic cotangent of each entry of G.  Since
% acoth (0) is nonozero, C is a full matrix.  C is complex if G is complex, or
% if any (abs (G) < 1).
%
% See also GrB/cot, GrB/acot, GrB/coth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_acoth (0, G) ;

