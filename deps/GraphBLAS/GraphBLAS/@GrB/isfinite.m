function C = isfinite (G)
%ISFINITE true for finite elements.
% C = isfinite (G) is a logical matrix where C(i,j) = true if G(i,j) is finite.
% C is a full matrix.
%
% See also GrB/isnan, GrB/isinf.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_isfinite (0, G) ;

