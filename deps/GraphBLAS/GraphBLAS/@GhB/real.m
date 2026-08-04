function C = real (G)
%REAL complex real part.
% C = real (G) returns the real part of G.
%
% See also GhB/conj, GhB/imag.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_real (1, G) ;

