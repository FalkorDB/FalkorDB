function C = atan (G)
%ATAN inverse tangent.
% C = atan (G) is the inverse tangent of each entry of G.
%
% See also GrB/tan, GrB/tanh, GrB/atanh, GrB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_atan (0, G) ;

