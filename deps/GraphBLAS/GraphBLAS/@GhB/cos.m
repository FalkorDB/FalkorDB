function C = cos (G)
%COS cosine.
% C = cos (G) is the cosine of each entry of G.  Since cos (0) = 1, the
% result is a full matrix.
%
% See also GhB/acos, GhB/cosh, GhB/acosh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cos (1, G) ;

