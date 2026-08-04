function C = sin (G)
%SIN sine.
% C = sin (G) is the sine of each entry of G.
%
% See also GrB/asin, GrB/sinh, GrB/asinh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_sin (0, G) ;

