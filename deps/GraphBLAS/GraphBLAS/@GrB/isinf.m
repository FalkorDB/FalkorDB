function C = isinf (G)
%ISINF true for infinite elements.
% C = isinf (G) returns a logical matrix C where C(i,j) = true if G(i,j) is
% infinite.
%
% See also GrB/isnan, GrB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_isinf (0, G) ;

