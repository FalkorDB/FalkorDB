function C = tan (G)
%TAN tangent.
% C = tan (G) is the tangent of each entry of G.
%
% See also GrB/tanh, GrB/atan, GrB/atanh, GrB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_tan (0, G) ;

