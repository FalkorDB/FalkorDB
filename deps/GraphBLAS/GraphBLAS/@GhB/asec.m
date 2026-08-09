function C = asec (G)
%ASEC inverse secant.
% C = asec (G) is the inverse secant of each entry of G.  Since asec (0) is
% nonzero, the result is a full matrix.  C is complex if any (abs(G) < 1).
%
% See also GhB/sec, GhB/sech, GhB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_asec (1, G) ;

