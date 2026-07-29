function C = sec (G)
%SEC secant.
% C = sec (G) is the secant of each entry of G.  Since sec (0) = 1, the result
% is a full matrix.
%
% See also GhB/asec, GhB/sech, GhB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_sec (1, G) ;

