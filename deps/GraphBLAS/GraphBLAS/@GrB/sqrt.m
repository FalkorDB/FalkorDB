function C = sqrt (G)
%SQRT square root.
% C = sqrt (G) is the square root of the entries of G.
%
% See also GrB.apply, GrB/hypot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_sqrt (1, G) ;

