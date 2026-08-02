function C = csch (G)
%CSCH hyperbolic cosecant.
% C = csch (G) is the hyperbolic cosecant of each entry of G.  Since
% csch(0) is nonzero, C is a full matrix.
%
% See also GhB/csc, GhB/acsc, GhB/acsch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_csch (1, G) ;

