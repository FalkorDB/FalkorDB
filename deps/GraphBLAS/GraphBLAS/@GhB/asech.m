function C = asech (G)
%ASECH inverse hyperbolic secant.
% C = asech (G) is the inverse hyperbolic secant of each entry of G.  Since
% asech (0) is nonzero, the result is a full matrix.  C is complex if G is
% complex, or if any real entries are outside of the range [0,1].
%
% See also GhB/sec, GhB/asec, GhB/sech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_asech (1, G) ;

