function C = acsch (G)
%ACSCH inverse hyperbolic cosecant.
% C = acsch (G) is the inverse hyberbolic cosecant of each entry G.  Since
% acsch (0) is nonzero, C is a full matrix.
%
% See also GhB/csc, GhB/acsc, GhB/csch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_acsch (1, G) ;

