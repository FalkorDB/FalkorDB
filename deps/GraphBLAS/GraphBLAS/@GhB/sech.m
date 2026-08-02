function C = sech (G)
%SECH hyperbolic secant.
% C = sech (G) is the hyperbolic secant of each entry of G.  Since sech(0) is
% nonzero, C is a full matrix.
%
% See also GhB/sec, GhB/asec, GhB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_sech (1, G) ;

