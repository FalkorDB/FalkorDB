function C = erfc (G)
%ERFC complementary error function.
% C = erfc (G) is the complementary error function of each entry of G.
% Since erfc (0) = 1, the result is a full matrix.  G must be real.
%
% See also GhB/erf, erfcx, erfinv, erfcinv.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_erfc (1, G) ;

