function C = erf (G)
%ERF error function.
% C = erf (G) computes the error function of each entry of G.  G must be real.
%
% See also GhB/erfc, erfcx, erfinv, erfcinv.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_erf (1, G) ;

