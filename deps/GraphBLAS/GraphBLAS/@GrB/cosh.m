function C = cosh (G)
%COSH hyperbolic cosine.
% C = cosh (G) is the hyperbolic cosine of each entry of G.  Since cosh
% (0) = 1, the result is a full matrix.
%
% See also GrB/cos, GrB/acos, GrB/acosh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cosh (0, G) ;

