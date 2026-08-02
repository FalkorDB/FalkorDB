function C = tanh (G)
%TANH hyperbolic tangent.
% C = tanh (G) is the hyperbolic tangent of each entry of G.
%
% See also GrB/tan, GrB/atan, GrB/atanh, GrB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_tanh (0, G) ;

