function C = atanh (G)
%ATANH inverse hyperbolic tangent.
% C = atanh (G) is the inverse hyberbolic tangent of each entry G.  C is
% complex if G is complex, or if any (abs (G) > 1).
%
% See also GrB/tan, GrB/atan, GrB/tanh, GrB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_trig (0, 'atanh', G) ;

