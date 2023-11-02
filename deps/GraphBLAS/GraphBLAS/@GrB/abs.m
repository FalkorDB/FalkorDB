function C = abs (G)
%ABS absolute value.
% C = abs (G) is the absolute value of each entry of G.
% C is always real, even if C is complex.
%
% See also GrB/sign.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
C = GrB (gb_abs (G)) ;

