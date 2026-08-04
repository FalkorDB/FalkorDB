function C = isnan (G)
%ISNAN true for NaN elements.
% C = isnan (G) is a logical C matrix with C(i,j)=true if G(i,j) is NaN.
%
% See also GhB/isinf, GhB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_isnan (1, G) ;

