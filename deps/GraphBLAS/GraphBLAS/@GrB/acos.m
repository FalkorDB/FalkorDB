function C = acos (G)
%ACOS inverse cosine.
% C = acos (G) is the inverse cosine of each entry of G.  Since acos (0) is
% nonzero, the result is a full matrix.  C is complex if any (abs(G) > 1).
%
% See also GrB/cos, GrB/cosh, GrB/acosh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_trig (0, 'acos', G) ;

